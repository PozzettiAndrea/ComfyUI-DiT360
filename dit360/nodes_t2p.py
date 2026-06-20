"""DiT360 seamless panorama sampler (text-to-panorama).

A FLUX KSampler variant that makes the output seamless across the 0°/360° seam
by wrap-padding the latent (and its RoPE position ids) before denoising and
cropping afterwards — the native equivalent of DiT360's circular padding.

Use with a 2048x1024 (W x H) empty latent and the FLUX guidance you'd normally
use for FLUX.1-dev (the paper uses 28 steps, guidance 2.8).
"""

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling
import latent_preview

from .padding import circular_pad_width, crop_width, make_wrap_ids_patch

# FLUX dynamic-shift constants (diffusers calculate_shift / ComfyUI ModelSamplingFlux):
# mu is linear in the packed-token count between (256 tokens -> 0.5) and (4096 -> 1.15).
_BASE_TOKENS, _MAX_TOKENS = 256, 4096
_BASE_SHIFT, _MAX_SHIFT = 0.5, 1.15


def _flux_resolution_shift(width, height, pad_columns):
    """mu for the *padded* panorama, matching the original diffusers pipeline.

    The original computes calculate_shift on the padded packed-token count
    (n_h * (n_w + 2)). ComfyUI's default FLUX shift is a fixed 1.15 (mu), which
    is ~half the correct value at 2048x1024 -> a different denoising trajectory.
    One padded token column = 16 px wide, so the effective width is widened.
    """
    eff_w = width + pad_columns * 2 * 16
    tokens = (eff_w * height) / (16 * 16)  # packed FLUX token count
    mm = (_MAX_SHIFT - _BASE_SHIFT) / (_MAX_TOKENS - _BASE_TOKENS)
    return tokens * mm + (_BASE_SHIFT - mm * _BASE_TOKENS)


def _patch_flux_shift(m, shift):
    base = comfy.model_sampling.ModelSamplingFlux
    const = comfy.model_sampling.CONST

    class _MS(base, const):
        pass

    ms = _MS(m.model.model_config)
    ms.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", ms)


def diffusers_flux_sigmas(steps, width, height, pad_columns):
    """The EXACT sigma schedule diffusers' FlowMatchEulerDiscreteScheduler uses for
    FLUX: sigmas = linspace(1, 1/N, N), then the dynamic flux shift with mu from
    the (padded) resolution, then a trailing 0. ComfyUI's `flux_time_shift` is the
    same formula, and ModelSamplingFlux.timestep(sigma)=sigma, so feeding these
    sigmas to the sampler reproduces the diffusers trajectory step-for-step.
    """
    import math
    mu = _flux_resolution_shift(width, height, pad_columns)  # == diffusers calculate_shift
    t = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float32)
    em = math.exp(mu)
    sig = em / (em + (1.0 / t - 1.0))            # time_shift(mu, 1.0, t)
    return torch.cat([sig, sig.new_zeros(1)])    # trailing sigma=0


class DiT360PanoramaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "Sampler CFG. For FLUX.1-dev keep at 1.0 and "
                                             "use FluxGuidance (paper uses 2.8)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "wrap_position_ids": ("BOOLEAN", {"default": True,
                    "tooltip": "Give padded boundary columns the position ids of the "
                               "opposite edge (faithful to DiT360 training)."}),
                "pad_columns": ("INT", {"default": 1, "min": 1, "max": 8,
                    "tooltip": "Token columns of circular padding per side (1 = 16 px)."}),
                "flux_resolution_shift": ("BOOLEAN", {"default": True,
                    "tooltip": "Set FLUX's dynamic timestep shift from the (padded) panorama "
                               "resolution, matching the original DiT360 schedule. ComfyUI's "
                               "default (shift=1.15) is ~half the correct value at 2048x1024 "
                               "and gives a different composition. Turn off for ComfyUI default."}),
                "match_diffusers_sigmas": ("BOOLEAN", {"default": True,
                    "tooltip": "Use the EXACT diffusers sigma schedule (linspace(1,1/N,N) + flux "
                               "shift) the original DiT360 runs, overriding the 'scheduler' "
                               "dropdown. The most faithful setting."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DiT360"
    TITLE = "DiT360 Panorama Sampler"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, wrap_position_ids=True, pad_columns=1,
               flux_resolution_shift=True, match_diffusers_sigmas=True):
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        h, w = latent.shape[-2] * 8, latent.shape[-1] * 8  # latent px -> image px

        m = model.clone()
        # Faithful wrap position-ids on the padded boundary.
        if wrap_position_ids:
            m.set_model_post_input_patch(make_wrap_ids_patch(pad_columns))

        # Exact diffusers sigma schedule (overrides scheduler + encodes the shift).
        sigmas = None
        if match_diffusers_sigmas:
            sigmas = diffusers_flux_sigmas(steps, w, h, pad_columns).to(latent.device)
            print(f"[DiT360] exact diffusers sigmas: {sigmas[0]:.3f}..{sigmas[-2]:.3f} ({steps} steps)")
        elif flux_resolution_shift:
            # FLUX dynamic timestep shift from the padded resolution (ComfyUI's
            # default 1.15 is ~half the correct value at 2048x1024).
            shift = _flux_resolution_shift(w, h, pad_columns)
            _patch_flux_shift(m, shift)
            print(f"[DiT360] FLUX resolution shift mu={shift:.3f} for {w}x{h} +{pad_columns}col")

        # Wrap-pad latent + matching noise so the padded columns are copies of the
        # opposite edge (persistent across the whole denoise, then cropped).
        noise = comfy.sample.prepare_noise(latent, seed, latent_image.get("batch_index"))
        latent_p = circular_pad_width(latent, pad_columns)
        noise_p = circular_pad_width(noise, pad_columns)

        noise_mask = latent_image.get("noise_mask")
        if noise_mask is not None:
            noise_mask = circular_pad_width(noise_mask, pad_columns)

        callback = latent_preview.prepare_callback(m, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            m, noise_p, steps, cfg, sampler_name, scheduler, positive, negative,
            latent_p, denoise=denoise, noise_mask=noise_mask, sigmas=sigmas,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        samples = crop_width(samples, pad_columns)
        out = latent_image.copy()
        out["samples"] = samples
        return (out,)
