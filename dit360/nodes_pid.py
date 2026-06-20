"""(down)Load NVIDIA PiD (Pixel Diffusion decoder) — zero extra dependencies.

PiD replaces the VAE decode with a pixel-diffusion model that decodes *and*
upscales a source latent (4x) in one pass. ComfyUI core already ships the PiD
model/text-encoder support and the `PiDConditioning` node, so this just fetches
the weights and loads them natively — no requirements.txt, no third-party pack.

Pipeline (with core nodes):
    DiT360 latent ─┐
                   ├─ PiDConditioning(positive, latent, "flux") ─► KSampler(pid_model) ─► image
    caption ► CLIPTextEncode(pid_clip) ─┘

For a 1024x2048 DiT360 panorama the ``flux1_1024_to_4096`` variant matches the
1024 short side (4x -> 4096x8192). Use ``flux1_512_to_2048`` for a lighter
2048x4096 result.
"""

import time

import torch
import folder_paths
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.latent_formats
import comfy.model_management
import node_helpers
import latent_preview

from .download import ensure_file

# PiD design constants (comfy/ldm/pixeldit/pid.py): 4x super-res, source latent
# downscaled 8x from the source image. So output pixels = lq_latent * 8 * 4 = *32.
SR_SCALE = 4
LATENT_DOWN = 8

PID_REPO = "Comfy-Org/PixelDiT"

# label -> diffusion_models filename in the repo (4-step distilled, FLUX.1 16ch)
PID_VARIANTS = {
    "flux1_1024_to_4096 (4x -> 8K pano)": "pid_flux1_1024_to_4096_4step_{prec}.safetensors",
    "flux1_512_to_2048 (4x -> 4K pano)": "pid_flux1_512_to_2048_4step_{prec}.safetensors",
}
GEMMA_TE = {
    "bf16": "gemma_2_2b_it_elm_bf16.safetensors",
    "fp8_scaled": "gemma_2_2b_it_elm_fp8_scaled.safetensors",
}


class DiT360PiDLoader:
    """Download + load the PiD decoder model and its Gemma2-2B text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_variant": (list(PID_VARIANTS.keys()),),
                "pid_precision": (["bf16", "mxfp8"], {
                    "tooltip": "Weight precision of the PiD model file (mxfp8 = smaller, needs fp8 support)."}),
                "text_encoder_precision": (["bf16", "fp8_scaled"],),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("pid_model", "pid_clip")
    FUNCTION = "load"
    CATEGORY = "DiT360"
    TITLE = "(down)Load PiD decoder"

    def load(self, pid_variant, pid_precision, text_encoder_precision):
        model_file = PID_VARIANTS[pid_variant].format(prec=pid_precision)
        te_file = GEMMA_TE[text_encoder_precision]

        # PiD diffusion model -> models/diffusion_models/
        model_path = ensure_file(
            "diffusion_models", model_file, PID_REPO,
            repo_filename=model_file, subfolder="diffusion_models")
        # Gemma2-2B text encoder -> models/text_encoders/
        te_path = ensure_file(
            "text_encoders", te_file, PID_REPO,
            repo_filename=te_file, subfolder="text_encoders")

        model = comfy.sd.load_diffusion_model(model_path)
        clip = comfy.sd.load_clip(
            ckpt_paths=[te_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.PIXELDIT,
        )
        print(f"[DiT360] PiD loaded: {model_file} + {te_file}")
        return (model, clip)


def _log(msg):
    print(f"[DiT360] {msg}", flush=True)


def _encode(clip, text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


class DiT360PiDDecode:
    """Seam-safe PiD decode: 4x pixel-diffusion decode of a panorama latent.

    Wraps ComfyUI core's native PiD sampling (lq_latent conditioning + a
    pixel-space FLOW sampler) but applies circular padding on the longitude axis
    so the 0/360 seam stays continuous through the high-res decode, then crops.

    EXPERIMENTAL: pixel-space schedule/cfg may need tuning; 4x of a 1024x2048
    panorama is 4096x8192 — heavy on VRAM (consider a smaller upscale or expect
    to need a big GPU).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("MODEL",),
                "pid_clip": ("CLIP",),
                "latent": ("LATENT", {"tooltip": "DiT360 panorama latent (16-ch FLUX.1)."}),
                "caption": ("STRING", {"multiline": True, "default": "This is a panorama image."}),
                "upscale": ("INT", {"default": 4, "min": 1, "max": 4,
                    "tooltip": "PiD is distilled for 4x; lower only to fit VRAM."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = clean latent; raise to denoise a corrupted/low-quality latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
            "optional": {
                "pad_columns": ("INT", {"default": 1, "min": 0, "max": 8,
                    "tooltip": "Circular padding columns for the seam (0 = off)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "DiT360"
    TITLE = "DiT360 PiD Decode (seam-safe)"

    @torch.no_grad()
    def decode(self, pid_model, pid_clip, latent, caption, upscale, steps, cfg,
               degrade_sigma, seed, sampler_name, scheduler, pad_columns=1):
        t0 = time.time()
        samples = latent["samples"]
        b, _, h, w = samples.shape

        # lq latent, matching PiDConditioning (Flux format process_in), then wrap-pad
        # the longitude axis by `pad_columns` latent pixels (explicit, so the crop
        # in output pixels is exactly pad_columns * LATENT_DOWN * upscale per side).
        lq = comfy.latent_formats.Flux().process_in(samples)
        if pad_columns > 0:
            lq = torch.cat([lq[..., -pad_columns:], lq, lq[..., :pad_columns]], dim=-1)
        wp = lq.shape[-1]
        out_h = h * LATENT_DOWN * upscale
        out_w = wp * LATENT_DOWN * upscale
        _log(f"PiD Decode: latent {tuple(samples.shape)} -> lq(padded) {tuple(lq.shape)}; "
             f"canvas {b}x3x{out_h}x{out_w} ({upscale}x); steps={steps} cfg={cfg}")

        # caption conditioning + attach lq_latent / degrade_sigma (== PiDConditioning)
        positive = _encode(pid_clip, caption)
        negative = _encode(pid_clip, "")
        sigma_t = torch.tensor([float(degrade_sigma)], dtype=torch.float32)
        positive = node_helpers.conditioning_set_values(
            positive, {"lq_latent": lq, "degrade_sigma": sigma_t})

        # pixel-space canvas (ChromaRadiance: 3-ch, spatial = output pixels)
        canvas = torch.zeros((b, 3, out_h, out_w), dtype=torch.float32)
        noise = comfy.sample.prepare_noise(canvas, seed)
        _log(f"PiD Decode: sampling on {comfy.model_management.get_torch_device()} ...")

        callback = latent_preview.prepare_callback(pid_model, steps)
        out = comfy.sample.sample(
            pid_model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, canvas, denoise=1.0, disable_noise=False,
            callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)

        # crop the wrap-padding back off (in output pixels)
        if pad_columns > 0:
            crop_px = pad_columns * LATENT_DOWN * upscale
            out = out[..., crop_px:-crop_px]
        # pixel latent (-1..1, BCHW) -> IMAGE (0..1, BHWC)
        image = (out.float() * 0.5 + 0.5).clamp(0.0, 1.0).movedim(1, -1)
        _log(f"PiD Decode: done -> image {tuple(image.shape)} in {time.time() - t0:.1f}s")
        return (image,)


PID_NODE_CLASS_MAPPINGS = {
    "DiT360PiDLoader": DiT360PiDLoader,
    "DiT360PiDDecode": DiT360PiDDecode,
}
PID_NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360PiDLoader": "(down)Load PiD decoder",
    "DiT360PiDDecode": "DiT360 PiD Decode (seam-safe)",
}
