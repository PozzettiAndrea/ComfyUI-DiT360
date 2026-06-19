"""Masked feature sharing for inpaint / outpaint (Personalize-Anything style).

DiT360's editing reuses the source image by, at every FLUX attention layer and
while the diffusion timestep is still noisy (``t > tau``), replacing the *edit*
branch's Q/K/V at the masked tokens with the *source* branch's Q/K/V. Upstream
does this inside a custom diffusers attention processor (``r_q/r_k/r_v`` from
``r_hidden_states``).

ComfyUI exposes the exact same point natively: the ``attn1_patch`` hook (in both
double- and single-stream FLUX blocks) receives ``q, k, v`` *after* modulation
and projection and lets us return modified tensors. ``extra_options["img_slice"]``
gives the text-token offset so we can index image tokens. So we don't have to
reimplement any block — we just swap the masked rows for batch item 1 (edit)
from batch item 0 (source).
"""

import torch


class MaskedShareState:
    """Mutable state shared between the edit sampler and the attention patch."""

    def __init__(self, token_mask: torch.Tensor, tau: float):
        # token_mask: bool [n_img_tokens] over the (circular-padded) latent grid.
        self.token_mask = token_mask
        self.tau = float(tau)
        self.t = 1.0          # current flow time in [0, 1]; updated each step
        self.enabled = True
        self._idx_cache = {}

    def masked_indices(self, img_len: int, txt_len: int, device) -> torch.Tensor:
        key = (img_len, txt_len, str(device))
        idx = self._idx_cache.get(key)
        if idx is None:
            m = self.token_mask
            if m is None or m.numel() != img_len:
                return None
            idx = torch.nonzero(m.to(device), as_tuple=False).squeeze(-1) + txt_len
            self._idx_cache[key] = idx
        return idx


def make_attn1_share_patch(state: MaskedShareState):
    """Build an ``attn1_patch`` that injects source Q/K/V into the edit branch."""

    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        out = {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
        if not state.enabled or state.t <= state.tau:
            return out
        if q.shape[0] < 2:  # need a [source, edit] batch
            return out
        sl = (extra_options or {}).get("img_slice")
        if sl is None:
            return out
        txt_len, total = int(sl[0]), int(sl[1])
        img_len = total - txt_len
        idx = state.masked_indices(img_len, txt_len, q.device)
        if idx is None or idx.numel() == 0:
            return out

        # q, k, v: [B, heads, seq, head_dim]. Copy source (0) -> edit (1) at mask.
        for t_ in (q, k, v):
            t_[1, :, idx, :] = t_[0, :, idx, :]
        return out

    return patch
