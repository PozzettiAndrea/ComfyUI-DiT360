"""(down)Load DiT360 model(s).

A single node that produces a ComfyUI ``MODEL`` / ``CLIP`` / ``VAE`` ready for
panorama generation: it loads a FLUX.1-dev checkpoint (using an existing file or
auto-downloading an fp8 build) and applies the DiT360 LoRA on top.

Everything is ComfyUI-native — no diffusers / transformers / peft.
"""

import folder_paths
import comfy.sd
import comfy.utils

from .download import ensure_file

# Defaults. The base FLUX repo/filename are overridable text inputs because the
# canonical fp8 build's exact location varies; testers who already have FLUX can
# just pick their existing checkpoint from the dropdown instead of downloading.
DEFAULT_FLUX_REPO = "Comfy-Org/flux1-dev"
DEFAULT_FLUX_FILE = "flux1-dev-fp8.safetensors"
DIT360_LORA_REPO = "Insta360-Research/DiT360-Panorama-Image-Generation"
DIT360_LORA_FILE = "pytorch_lora_weights.safetensors"
DOWNLOAD_SENTINEL = "⤓ download fp8"


class DiT360ModelLoader:
    """Load (and optionally download) FLUX.1-dev + the DiT360 LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = folder_paths.get_filename_list("checkpoints")
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "base_checkpoint": ([DOWNLOAD_SENTINEL] + ckpts, {
                    "tooltip": "FLUX.1-dev checkpoint. Pick an existing file, or "
                               "choose download to fetch the fp8 build."}),
                "dit360_lora": ([DOWNLOAD_SENTINEL] + loras, {
                    "tooltip": "DiT360 LoRA. Download fetches it from the official HF repo."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "flux_repo_id": ("STRING", {"default": DEFAULT_FLUX_REPO}),
                "flux_filename": ("STRING", {"default": DEFAULT_FLUX_FILE}),
                "lora_repo_id": ("STRING", {"default": DIT360_LORA_REPO}),
                "lora_filename": ("STRING", {"default": DIT360_LORA_FILE}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "DiT360"
    TITLE = "(down)Load DiT360 model(s)"

    def load(self, base_checkpoint, dit360_lora, lora_strength,
             flux_repo_id=DEFAULT_FLUX_REPO, flux_filename=DEFAULT_FLUX_FILE,
             lora_repo_id=DIT360_LORA_REPO, lora_filename=DIT360_LORA_FILE):

        # --- resolve / fetch base checkpoint ---
        if base_checkpoint == DOWNLOAD_SENTINEL:
            ckpt_path = ensure_file("checkpoints", flux_filename, flux_repo_id)
        else:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", base_checkpoint)

        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )[:3]
        if model is None or clip is None or vae is None:
            raise RuntimeError(
                "DiT360: base checkpoint did not yield MODEL+CLIP+VAE. Use an "
                "all-in-one FLUX.1-dev checkpoint, or wire CLIP/VAE separately.")

        # --- resolve / fetch LoRA ---
        if dit360_lora == DOWNLOAD_SENTINEL:
            lora_path = ensure_file("loras", lora_filename, lora_repo_id)
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", dit360_lora)

        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(
            model, clip, lora_sd, lora_strength, 0.0)

        return (model, clip, vae)
