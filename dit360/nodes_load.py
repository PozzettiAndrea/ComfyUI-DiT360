"""(down)Load DiT360 model(s).

A single node that produces a ComfyUI ``MODEL`` / ``CLIP`` / ``VAE`` ready for
panorama generation: it loads a FLUX.1-dev checkpoint (using an existing file or
auto-downloading an fp8 build) and applies the DiT360 LoRA on top.

Everything is ComfyUI-native — no diffusers / transformers / peft.
"""

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.lora

from .download import ensure_file

# Runtime weight-dtype casts, mirroring ComfyUI's UNETLoader. Applied on load
# regardless of the file's on-disk precision, so e.g. a bf16 FLUX can be run at
# fp8 to save VRAM.
QUANTIZATIONS = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _quant_model_options(quantization):
    opts = {}
    if quantization == "fp8_e4m3fn":
        opts["dtype"] = torch.float8_e4m3fn
    elif quantization == "fp8_e4m3fn_fast":
        opts["dtype"] = torch.float8_e4m3fn
        opts["fp8_optimizations"] = True
    elif quantization == "fp8_e5m2":
        opts["dtype"] = torch.float8_e5m2
    return opts

# Download sources (all non-gated).
DEFAULT_FLUX_REPO = "Comfy-Org/flux1-dev"
DEFAULT_FLUX_FILE = "flux1-dev-fp8.safetensors"   # all-in-one fp8 (~17GB)
BF16_UNET_FILE = "flux1-dev.safetensors"          # bf16 diffusion model only (~24GB)
TE_REPO = "comfyanonymous/flux_text_encoders"
T5_FILE = "t5xxl_fp16.safetensors"
CLIPL_FILE = "clip_l.safetensors"
VAE_REPO = "Kijai/flux-fp8"
VAE_FILE = "flux-vae-bf16.safetensors"
DIT360_LORA_REPO = "Insta360-Research/DiT360-Panorama-Image-Generation"
DIT360_LORA_FILE = "adapter_model.safetensors"

DL_FP8 = "⤓ download fp8 (all-in-one, ~17GB)"
DL_BF16 = "⤓ download bf16 (full, ~34GB)"
DL_LORA = "⤓ download"


class DiT360ModelLoader:
    """Load (and optionally download) FLUX.1-dev + the DiT360 LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = folder_paths.get_filename_list("checkpoints")
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "base_checkpoint": ([DL_FP8, DL_BF16] + ckpts, {
                    "tooltip": "FLUX.1-dev base. Pick an existing all-in-one checkpoint, or "
                               "download: fp8 (smaller/faster, slightly softer) or bf16 (full "
                               "precision = closest to the original DiT360, needs offload on 24GB)."}),
                "dit360_lora": ([DL_LORA] + loras, {
                    "tooltip": "DiT360 LoRA. Download fetches it from the official HF repo."}),
                "quantization": (QUANTIZATIONS, {
                    "tooltip": "Runtime weight cast. 'default' keeps the file's precision; "
                               "fp8 options lower VRAM (works on any base file)."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "DiT360"
    TITLE = "(down)Load DiT360 model(s)"

    def _load_bf16_full(self, quantization):
        """bf16 diffusion model + separate fp16 text encoders + VAE (the bf16 file
        is not an all-in-one checkpoint)."""
        unet = ensure_file("diffusion_models", BF16_UNET_FILE, DEFAULT_FLUX_REPO)
        t5 = ensure_file("text_encoders", T5_FILE, TE_REPO)
        clip_l = ensure_file("text_encoders", CLIPL_FILE, TE_REPO)
        vae_p = ensure_file("vae", VAE_FILE, VAE_REPO)
        model = comfy.sd.load_diffusion_model(unet, model_options=_quant_model_options(quantization))
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_l, t5],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.FLUX)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_p))
        return model, clip, vae

    def load(self, base_checkpoint, dit360_lora, quantization, lora_strength):
        # --- resolve / fetch base model ---
        if base_checkpoint == DL_BF16:
            model, clip, vae = self._load_bf16_full(quantization)
        else:
            if base_checkpoint == DL_FP8:
                ckpt_path = ensure_file("checkpoints", DEFAULT_FLUX_FILE, DEFAULT_FLUX_REPO)
            else:
                ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", base_checkpoint)
            model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, output_vae=True, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options=_quant_model_options(quantization))[:3]
            if model is None or clip is None or vae is None:
                raise RuntimeError(
                    "DiT360: base checkpoint did not yield MODEL+CLIP+VAE. Use an "
                    "all-in-one FLUX.1-dev checkpoint, or the bf16 download option.")

        # --- resolve / fetch LoRA ---
        if dit360_lora == DL_LORA:
            lora_path = ensure_file("loras", DIT360_LORA_FILE, DIT360_LORA_REPO)
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", dit360_lora)

        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        # Report how many LoRA keys actually map onto the model, so a silent
        # no-op (wrong format / wrong base) is visible instead of mysterious.
        try:
            key_map = comfy.lora.model_lora_keys_unet(model.model, {})
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
            loaded = comfy.lora.load_lora(lora_sd, key_map)
            print(f"[DiT360] LoRA: {len(loaded)}/{len(lora_sd)} tensors mapped to the model.")
            if len(loaded) == 0:
                print("[DiT360] WARNING: no LoRA keys matched — panoramas will look "
                      "like plain FLUX. Check the base model is FLUX.1-dev.")
        except Exception as e:
            print(f"[DiT360] LoRA key-map check skipped: {e}")

        model, clip = comfy.sd.load_lora_for_models(
            model, clip, lora_sd, lora_strength, 0.0)

        return (model, clip, vae)
