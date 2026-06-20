"""Circular padding utilities for seamless 360° panorama generation.

DiT360's only inference-time change to FLUX is *circular padding*: the packed
latent grid is wrap-padded along the width (longitude) axis so the 0°/360° seam
is continuous, and the RoPE position ids of the padded boundary columns are set
to those of the opposite real edge. We reproduce this natively:

  * the latent tensor is wrap-padded in the sampler (persistent across all steps,
    matching the upstream pipeline), then cropped before VAE decode;
  * a ``post_input`` patch rewrites ``img_ids`` so the padded columns carry the
    position ids of the opposite edge (faithful to how the LoRA was trained).

FLUX uses a patch size of 2, so one "column" in packed-token space corresponds
to ``PATCH`` (=2) latent pixels.
"""

import torch

PATCH = 2  # FLUX spatial patch size; 1 token column == PATCH latent pixels


def circular_pad_width(x: torch.Tensor, patch_cols: int = 1) -> torch.Tensor:
    """Wrap-pad latent ``[B, C, H, W]`` along width by ``patch_cols`` token columns.

    The left pad is a copy of the right edge and vice-versa, matching
    ``torch.cat([X[..., -p:], X, X[..., :p]])`` in the upstream pipeline.
    """
    p = patch_cols * PATCH
    if p <= 0:
        return x
    left = x[..., -p:]
    right = x[..., :p]
    return torch.cat([left, x, right], dim=-1)


def crop_width(x: torch.Tensor, patch_cols: int = 1) -> torch.Tensor:
    """Inverse of :func:`circular_pad_width` — strip the padded columns."""
    p = patch_cols * PATCH
    if p <= 0:
        return x
    return x[..., p:-p]


def make_wrap_ids_patch(patch_cols: int = 1):
    """Build a ``post_input`` patch that gives padded boundary columns the
    position ids of the opposite real edge.

    The patch is shape-agnostic: it infers the token grid (n_h, n_w) directly
    from ``img_ids`` (FLUX stores the column index in channel 2, row in channel
    1), so it works regardless of resolution. If the layout is unexpected it
    leaves ``img_ids`` untouched.
    """

    def patch(args):
        img_ids = args.get("img_ids")
        if img_ids is None:
            return args

        ids = img_ids
        squeezed = False
        if ids.dim() == 2:  # [seq, 3]
            ids = ids.unsqueeze(0)
            squeezed = True

        col = ids[..., 2]
        n_w = int(col.max().item()) + 1
        seq = ids.shape[1]
        pc = patch_cols
        if n_w <= 2 * pc or seq % n_w != 0:
            return args
        n_h = seq // n_w
        n_real = n_w - 2 * pc

        # Rebuild the width-id channel EXACTLY as the original DiT360 pipeline:
        # real columns numbered 0..n_real-1, and each pad column duplicates the
        # opposite real edge's id. This is critical, not cosmetic: ComfyUI's
        # native grid numbers the padded width 0..n_w-1, which leaves the real
        # columns at 1..n_w-2 and pushes the duplicated id to n_real (an absolute
        # position the model never saw at this resolution -> RoPE garbage at the
        # seam). Keeping real cols at 0..n_real-1 stays in-distribution.
        real = torch.arange(n_real, device=ids.device, dtype=ids.dtype)
        width = torch.cat([real[-pc:], real, real[:pc]])  # [n_real-pc.., 0..n_real-1, ..pc-1]
        g = ids.view(ids.shape[0], n_h, n_w, ids.shape[-1]).clone()
        g[..., 2] = width.view(1, 1, n_w).expand(g.shape[0], n_h, n_w)
        ids = g.view(ids.shape[0], seq, ids.shape[-1])

        if squeezed:
            ids = ids.squeeze(0)
        args["img_ids"] = ids
        return args

    return patch
