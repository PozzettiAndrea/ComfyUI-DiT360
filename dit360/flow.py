"""Low-level helpers to drive ComfyUI's native FLUX model for the RF-inversion /
editing loops (which need per-batch-item update rules that the stock sampler
doesn't express).

``model_base.BaseModel.apply_model`` returns the **flow velocity** for flow
models (``v = noise - x0``), which is exactly upstream's ``noise_pred``, so we
can reproduce DiT360's editing math directly.
"""

import torch
import comfy.model_management
import comfy.samplers


def cond_parts(conditioning):
    """Extract (context, pooled, guidance) from a ComfyUI CONDITIONING."""
    cond, meta = conditioning[0][0], conditioning[0][1]
    pooled = meta.get("pooled_output")
    guidance = meta.get("guidance", 3.5)
    return cond, pooled, guidance


def stack_conds(src_cond, edit_cond):
    """Build a 2-item [source, edit] batch of (context, pooled, guidance)."""
    c0, p0, g0 = cond_parts(src_cond)
    c1, p1, g1 = cond_parts(edit_cond)
    # pad T5 context to equal length if needed
    L = max(c0.shape[1], c1.shape[1])

    def padc(c):
        if c.shape[1] < L:
            pad = c.new_zeros((c.shape[0], L - c.shape[1], c.shape[2]))
            c = torch.cat([c, pad], dim=1)
        return c

    context = torch.cat([padc(c0), padc(c1)], dim=0)
    pooled = torch.cat([p0, p1], dim=0) if p0 is not None else None
    guidance = torch.tensor([g0, g1], dtype=torch.float32)
    return context, pooled, guidance


def get_sigmas(model, scheduler, steps):
    ms = model.get_model_object("model_sampling")
    return comfy.samplers.calculate_sigmas(ms, scheduler, steps)


def load_for_direct_call(model):
    """Move the patched model (weights + LoRA) onto the GPU and return the inner
    BaseModel plus the transformer_options carrying our attn patches."""
    comfy.model_management.load_models_gpu([model])
    base = model.model
    transformer_options = model.model_options.get("transformer_options", {}).copy()
    return base, transformer_options


def velocity(base, x, sigma_scalar, context, pooled, guidance, transformer_options):
    """One FLUX forward → flow velocity, for a whole batch at scalar time."""
    b = x.shape[0]
    device = x.device
    sigma = torch.full((b,), float(sigma_scalar), device=device, dtype=torch.float32)
    kwargs = {}
    if pooled is not None:
        kwargs["y"] = pooled.to(device)
    if guidance is not None:
        kwargs["guidance"] = guidance.to(device)
    return base.apply_model(
        x, sigma,
        c_crossattn=context.to(device),
        transformer_options=transformer_options,
        **kwargs,
    )
