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

import folder_paths
import comfy.sd

from .download import ensure_file

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


PID_NODE_CLASS_MAPPINGS = {"DiT360PiDLoader": DiT360PiDLoader}
PID_NODE_DISPLAY_NAME_MAPPINGS = {"DiT360PiDLoader": "(down)Load PiD decoder"}
