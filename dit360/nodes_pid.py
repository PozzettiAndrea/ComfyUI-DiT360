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


def _vram_report(tag, device):
    """Dump the real VRAM picture: physical, torch, aimdo buffers, tracked models."""
    mm = comfy.model_management
    try:
        free, total = torch.cuda.mem_get_info(device)
        ta = torch.cuda.memory_allocated(device)
        tr = torch.cuda.memory_reserved(device)
        GB = 1024 ** 3
        _log(f"VRAM[{tag}] physical free={free/GB:.2f}G / total={total/GB:.2f}G | "
             f"torch alloc={ta/GB:.2f}G reserved={tr/GB:.2f}G")
        # aimdo cast-buffer reservation + live buffers
        rsv = getattr(mm, "DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE", None)
        if rsv is not None:
            _log(f"VRAM[{tag}] aimdo cast-buffer reservation={rsv/GB:.2f}G; "
                 f"largest_casted_weight={getattr(mm,'LARGEST_AIMDO_CASTED_WEIGHT',('',0))[1]/1024/1024:.1f}MB")
        bufs = getattr(mm, "STREAM_AIMDO_CAST_BUFFERS", {})
        for k, vb in list(bufs.items()):
            _log(f"VRAM[{tag}] aimdo cast buf: max={getattr(vb,'max_size',0)/GB:.2f}G alloc={vb.size()/GB:.2f}G")
        # every tracked loaded model
        for lm in getattr(mm, "current_loaded_models", []):
            name = type(getattr(lm.model, "model", lm.model)).__name__
            mem = lm.model_memory() / GB if hasattr(lm, "model_memory") else 0
            loaded = lm.model_loaded_memory() / GB if hasattr(lm, "model_loaded_memory") else 0
            off = lm.model_offloaded_memory() / GB if hasattr(lm, "model_offloaded_memory") else 0
            dyn = lm.model.is_dynamic() if hasattr(lm.model, "is_dynamic") else "?"
            _log(f"VRAM[{tag}]  model {name}: total={mem:.2f}G on-gpu={loaded:.2f}G "
                 f"offloaded={off:.2f}G dynamic={dyn}")
    except Exception as e:
        _log(f"VRAM[{tag}] report failed: {e}")


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
                "mlp_chunks": ("INT", {"default": 0, "min": 0, "max": 64,
                    "tooltip": "PiD's own VRAM dial: split each pixel-block MLP into N chunks "
                               "(lower peak VRAM, marginally slower). 0 = auto-scale with output "
                               "resolution."}),
                "tile_strips": ("INT", {"default": 0, "min": 0, "max": 32,
                    "tooltip": "Decode in N circular width strips to fit VRAM (8K needs it on "
                               "24GB). 0 = auto (~2048px/strip). 1 = single canvas (needs a big "
                               "GPU). More strips = less VRAM, more time."}),
                "tile_overlap": ("INT", {"default": 8, "min": 0, "max": 32,
                    "tooltip": "Strip overlap in latent columns (1 col = upscale*16 px). "
                               "Feathered-blended; also keeps the 0/360 seam continuous."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "DiT360"
    TITLE = "DiT360 PiD Decode (seam-safe)"

    @torch.no_grad()
    def decode(self, pid_model, pid_clip, latent, caption, upscale, steps, cfg,
               degrade_sigma, seed, sampler_name, scheduler, pad_columns=1, mlp_chunks=0,
               tile_strips=0, tile_overlap=8):
        t0 = time.time()
        samples = latent["samples"]
        b, _, h, w = samples.shape
        lq_full = comfy.latent_formats.Flux().process_in(samples)
        px = LATENT_DOWN * upscale            # output px per lq column (=32 at 4x)
        out_h, out_w = h * px, w * px

        # The full 8K pixel-space forward needs ~16-20GB of activations (measured)
        # and OOMs a 24GB card. Decode in circular, overlapping width strips instead:
        # at ~2048px/strip the peak is ~8GB. The panorama wraps, so strips wrap too,
        # and the feathered blend across the 0/360 boundary keeps the seam continuous.
        if tile_strips <= 0:
            tile_strips = max(1, -(-out_w // 2048))   # ~2048px per strip

        # mlp_chunks sized for the per-strip canvas (the model's own MLP-peak dial).
        strip_area = out_h * (out_w // max(1, tile_strips))
        chunks = mlp_chunks if mlp_chunks > 0 else max(2, -(-strip_area // 3_000_000))
        blocks = getattr(pid_model.model.diffusion_model, "pixel_blocks", None)
        if blocks is not None:
            for blk in blocks:
                blk.mlp_chunks = int(chunks)

        # Free FLUX/TE from VRAM (native, tracking-aware offload to CPU).
        dev = comfy.model_management.get_torch_device()
        comfy.model_management.free_memory(
            pid_model.model.memory_required([b * 2, 3, out_h, out_w // max(1, tile_strips)]), dev)

        positive = _encode(pid_clip, caption)
        negative = _encode(pid_clip, "")
        sigma_t = torch.tensor([float(degrade_sigma)], dtype=torch.float32)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # PiD is distilled for a SPECIFIC 4-step sigma schedule. The stock schedulers
        # only approximate it -- "simple" gives [1,.931,.818,.599,0], so the last step
        # jumps from 0.599->0 and the image is left UNDER-denoised = grainy. Feed the
        # exact distilled sigmas (per the model's own sampling_settings comment).
        pid_sigmas = None
        if steps == 4:
            pid_sigmas = torch.tensor([0.999, 0.866, 0.634, 0.342, 0.0])
            _log("PiD Decode: using exact distilled 4-step sigmas [.999,.866,.634,.342,0]")

        def _decode(lq, ow):
            p = node_helpers.conditioning_set_values(
                positive, {"lq_latent": lq, "degrade_sigma": sigma_t})
            canvas = torch.zeros((b, 3, out_h, ow), dtype=torch.bfloat16)
            noise = comfy.sample.prepare_noise(canvas, seed)
            s = comfy.sample.sample(pid_model, noise, steps, cfg, sampler_name, scheduler,
                                    p, negative, canvas, denoise=1.0, seed=seed,
                                    sigmas=pid_sigmas, disable_pbar=disable_pbar)
            return (s.float() * 0.5 + 0.5).clamp(0.0, 1.0)  # [b,3,out_h,ow] in 0..1

        if tile_strips <= 1:
            # single full canvas (small outputs): keep the explicit circular pad.
            lq = lq_full
            if pad_columns > 0:
                lq = torch.cat([lq[..., -pad_columns:], lq, lq[..., :pad_columns]], dim=-1)
            _log(f"PiD Decode: single canvas {out_h}x{lq.shape[-1]*px} (chunks={chunks})")
            img = _decode(lq, lq.shape[-1] * px)
            if pad_columns > 0:
                cp = pad_columns * px
                img = img[..., cp:-cp]
            image = img.movedim(1, -1)
        else:
            ov = int(tile_overlap)
            _log(f"PiD Decode: {tile_strips} circular strips, overlap {ov} cols, "
                 f"chunks={chunks}, out {out_h}x{out_w}")
            acc = torch.zeros((b, 3, out_h, out_w), dtype=torch.float32)   # CPU
            wsum = torch.zeros((1, 1, 1, out_w), dtype=torch.float32)
            for i in range(tile_strips):
                c0 = (i * w) // tile_strips - ov
                c1 = ((i + 1) * w) // tile_strips + ov
                sw = c1 - c0
                idx = torch.arange(c0, c1) % w
                strip = _decode(lq_full[..., idx], sw * px).cpu()  # [b,3,out_h,sw*px]
                # feather: linear ramp across the overlap on each side
                spx, ovpx = sw * px, ov * px
                f = torch.ones(spx)
                if ovpx > 0:
                    r = torch.linspace(0.0, 1.0, ovpx + 2)[1:-1]
                    f[:ovpx], f[-ovpx:] = r, r.flip(0)
                oidx = torch.arange(c0 * px, c1 * px) % out_w
                acc.index_add_(3, oidx, strip * f.view(1, 1, 1, -1))
                wsum.index_add_(3, oidx, f.view(1, 1, 1, -1).expand(1, 1, 1, spx).contiguous())
                comfy.model_management.soft_empty_cache()
                _log(f"  strip {i+1}/{tile_strips} done ({time.time()-t0:.0f}s)")
            image = (acc / wsum.clamp_min(1e-6)).movedim(1, -1)

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
