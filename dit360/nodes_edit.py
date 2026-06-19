"""Inpainting / outpainting via RF-inversion + masked feature sharing.

EXPERIMENTAL — the math is ported faithfully from the upstream pipeline
(`pa_src/pipeline.py`: `invert` + the editing `__call__` controller) but drives
ComfyUI's FLUX model directly and has not yet been validated on real weights.
The sign/scale conventions of the flow schedule and the eta/gamma windows are
the most likely things to need tuning on-GPU.

Pipeline:
  DiT360 RF Invert   : clean image latent -> inverted noise (forward ODE)
  DiT360 Inpaint/Outpaint : reverse ODE with a per-branch controller; a 2-item
      [source, edit] batch denoises together while the attn1 patch copies the
      source branch's Q/K/V into the edit branch at the masked tokens (t > tau).
"""

import time

import torch
import comfy.samplers
import comfy.utils
import comfy.model_management

from .padding import circular_pad_width, crop_width, make_wrap_ids_patch
from .masks import mask_to_token_mask
from .attention import MaskedShareState, make_attn1_share_patch
from .flow import stack_conds, get_sigmas, load_for_direct_call, velocity

PAD_COLS = 1


def _log(msg):
    print(f"[DiT360] {msg}", flush=True)


def _pad(latent):
    return circular_pad_width(latent, PAD_COLS)


@torch.no_grad()
def rf_invert(model, source_cond, clean_latent, steps, scheduler, gamma, seed):
    """Algorithm 1 (controlled forward ODE): clean latent -> inverted noise."""
    t0 = time.time()
    _log("RF Invert: cloning model + registering wrap-ids patch ...")
    m = model.clone()
    m.set_model_post_input_patch(make_wrap_ids_patch(PAD_COLS))

    _log("RF Invert: loading model weights to GPU (load_models_gpu) ...")
    base, topts = load_for_direct_call(m)
    device = comfy.model_management.get_torch_device()
    _log(f"RF Invert: model ready on {device} ({time.time() - t0:.1f}s)")

    context, pooled, guidance = stack_conds(source_cond, source_cond)
    context, pooled = context[:1], (pooled[:1] if pooled is not None else None)
    guidance = guidance[:1]
    _log(f"RF Invert: context={tuple(context.shape)} "
         f"pooled={tuple(pooled.shape) if pooled is not None else None} "
         f"guidance={guidance.tolist()}")

    sigmas = get_sigmas(m, scheduler, steps).tolist()  # ~1 .. 0
    y0 = _pad(clean_latent).to(device)
    Yt = y0.clone()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    y1 = torch.randn(Yt.shape, generator=gen).to(Yt.device, Yt.dtype)
    _log(f"RF Invert: latent(padded)={tuple(Yt.shape)} on {Yt.device}; "
         f"{len(sigmas) - 1} ODE steps (sigmas {sigmas[0]:.3f}..{sigmas[-1]:.3f})")

    # integrate from t=0 (clean) to t=1 (noise): walk sigmas in reverse
    asc = list(reversed(sigmas))  # ~0 .. 1
    n = len(asc) - 1
    pbar = comfy.utils.ProgressBar(n)
    import tqdm as _tqdm
    for i in _tqdm.trange(n, desc="[DiT360] RF Invert", dynamic_ncols=True):
        t_i = asc[i]
        dt = asc[i + 1] - asc[i]  # > 0
        ts = time.time()
        v = velocity(base, Yt, t_i, context, pooled, guidance, topts)
        if i == 0:
            _log(f"RF Invert: first forward OK, v={tuple(v.shape)} on {v.device} "
                 f"({time.time() - ts:.1f}s/step)")
        denom = max(1.0 - t_i, 1e-3)
        v_cond = (y1 - Yt) / denom
        v_hat = v + gamma * (v_cond - v)
        Yt = Yt + v_hat * dt
        pbar.update_absolute(i + 1, n)
    _log(f"RF Invert: done in {time.time() - t0:.1f}s")
    return Yt, y0, base, topts


class DiT360RFInvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "source_conditioning": ("CONDITIONING", {"tooltip": "Empty/source prompt."}),
                "latent_image": ("LATENT", {"tooltip": "VAE-encoded source panorama."}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 1000}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "gamma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("DIT360_INVERSION",)
    RETURN_NAMES = ("inversion",)
    FUNCTION = "invert"
    CATEGORY = "DiT360"
    TITLE = "DiT360 RF Invert"

    def invert(self, model, source_conditioning, latent_image, steps, scheduler, gamma, seed):
        clean = latent_image["samples"]
        inverted, y0, _base, _topts = rf_invert(
            model, source_conditioning, clean, steps, scheduler, gamma, seed)
        bundle = {
            "inverted": inverted, "source_padded": y0,
            "source_cond": source_conditioning,
            "steps": steps, "scheduler": scheduler,
            "latent_meta": {k: v for k, v in latent_image.items() if k != "samples"},
        }
        return (bundle,)


@torch.no_grad()
def rf_edit(model, inversion, edit_cond, token_mask, tau, eta, start, stop):
    """Reverse ODE with controller on the source branch + masked attention sharing."""
    steps, scheduler = inversion["steps"], inversion["scheduler"]
    inverted, y0 = inversion["inverted"], inversion["source_padded"]
    src_cond = inversion["source_cond"]

    t0 = time.time()
    _log("RF Edit: cloning model + registering wrap-ids/attn-share patches ...")
    m = model.clone()
    m.set_model_post_input_patch(make_wrap_ids_patch(PAD_COLS))
    state = MaskedShareState(token_mask, tau)
    m.set_model_attn1_patch(make_attn1_share_patch(state))

    _log("RF Edit: loading model weights to GPU ...")
    base, topts = load_for_direct_call(m)
    device = comfy.model_management.get_torch_device()

    context, pooled, guidance = stack_conds(src_cond, edit_cond)
    sigmas = get_sigmas(m, scheduler, steps).tolist()

    latents = torch.cat([inverted, inverted.clone()], dim=0).to(device)  # [source, edit]
    y0 = y0.to(device)
    mask_true = int(token_mask.sum().item()) if token_mask is not None else 0
    _log(f"RF Edit: latents={tuple(latents.shape)} on {device}; tau={tau} eta={eta}; "
         f"mask injects {mask_true}/{token_mask.numel() if token_mask is not None else 0} tokens; "
         f"{len(sigmas) - 1} steps")

    start_i = int(start * steps)
    stop_i = int(stop * steps)

    n = len(sigmas) - 1
    pbar = comfy.utils.ProgressBar(n)
    import tqdm as _tqdm
    for i in _tqdm.trange(n, desc="[DiT360] RF Edit", dynamic_ncols=True):
        t_i = sigmas[i]
        dt = sigmas[i] - sigmas[i + 1]  # > 0
        state.t = float(t_i)
        ts = time.time()
        v = velocity(base, latents, t_i, context, pooled, guidance, topts)
        if i == 0:
            _log(f"RF Edit: first forward OK, v={tuple(v.shape)} on {v.device} "
                 f"({time.time() - ts:.1f}s/step)")

        # edit branch (1): plain euler step
        edit_next = latents[1:2] - v[1:2] * dt

        # source branch (0): controller toward the clean source
        denom = max(1.0 - t_i, 1e-3)
        v_t = v[0:1]
        v_cond = (y0 - latents[0:1]) / denom
        eta_t = eta if (start_i <= i < stop_i) else 0.0
        v_hat = v_t + eta_t * (v_cond - v_t)
        src_next = latents[0:1] - v_hat * dt

        latents = torch.cat([src_next, edit_next], dim=0)
        pbar.update_absolute(i + 1, n)

    _log(f"RF Edit: done in {time.time() - t0:.1f}s")
    edited = crop_width(latents[1:2], PAD_COLS)
    out = dict(inversion["latent_meta"])
    out["samples"] = edited
    return out


def _run_edit(model, inversion, edit_conditioning, mask, image_h, image_w,
              tau, eta, start_step, stop_step, invert_mask):
    token_mask = mask_to_token_mask(mask, image_h, image_w, PAD_COLS, invert=invert_mask)
    out = rf_edit(model, inversion, edit_conditioning, token_mask,
                  tau, eta, start_step, stop_step)
    return (out,)


_EDIT_INPUTS = {
    "required": {
        "model": ("MODEL",),
        "inversion": ("DIT360_INVERSION",),
        "edit_conditioning": ("CONDITIONING",),
        "mask": ("MASK",),
        "image_width": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 16}),
        "image_height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 16}),
        "tau": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "Share source features while flow time t > tau."}),
        "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "Controller strength: higher = more faithful to source."}),
        "start_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "stop_step": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
    }
}


class DiT360Outpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return _EDIT_INPUTS

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "DiT360"
    TITLE = "DiT360 Outpaint"

    def run(self, model, inversion, edit_conditioning, mask, image_width, image_height,
            tau, eta, start_step, stop_step):
        # Outpaint: mask marks the KNOWN view to keep -> inject source there.
        return _run_edit(model, inversion, edit_conditioning, mask, image_height,
                         image_width, tau, eta, start_step, stop_step, invert_mask=False)


class DiT360Inpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return _EDIT_INPUTS

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "DiT360"
    TITLE = "DiT360 Inpaint"

    def run(self, model, inversion, edit_conditioning, mask, image_width, image_height,
            tau, eta, start_step, stop_step):
        # Inpaint: mask marks the hole to regenerate -> inject source everywhere else.
        return _run_edit(model, inversion, edit_conditioning, mask, image_height,
                         image_width, tau, eta, start_step, stop_step, invert_mask=True)


EDIT_NODE_CLASS_MAPPINGS = {
    "DiT360RFInvert": DiT360RFInvert,
    "DiT360Inpaint": DiT360Inpaint,
    "DiT360Outpaint": DiT360Outpaint,
}
EDIT_NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360RFInvert": "DiT360 RF Invert",
    "DiT360Inpaint": "DiT360 Inpaint",
    "DiT360Outpaint": "DiT360 Outpaint",
}
