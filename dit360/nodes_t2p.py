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
import latent_preview

from .padding import circular_pad_width, crop_width, make_wrap_ids_patch


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
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DiT360"
    TITLE = "DiT360 Panorama Sampler"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, wrap_position_ids=True, pad_columns=1):
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        # Patch the model: faithful wrap position-ids on the padded boundary.
        m = model
        if wrap_position_ids:
            m = model.clone()
            m.set_model_post_input_patch(make_wrap_ids_patch(pad_columns))

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
            latent_p, denoise=denoise, noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed,
        )

        samples = crop_width(samples, pad_columns)
        out = latent_image.copy()
        out["samples"] = samples
        return (out,)
