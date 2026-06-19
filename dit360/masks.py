"""Mask helpers for inpainting / outpainting.

The editing path operates on a *token-level* boolean mask over the packed FLUX
latent grid (after circular padding). Upstream (`pa_src.utils.create_mask` +
`editing.py`) downsamples the pixel mask to the latent-token grid with NEAREST,
then wrap-pads the width to match the padded latents.

ComfyUI masks come in as a float ``MASK`` tensor in ``[0, 1]`` shaped ``[H, W]``
or ``[B, H, W]`` at image resolution. We convert to a boolean token mask of
length ``n_h * (n_w + 2*patch_cols)``.
"""

import torch
import torch.nn.functional as F

# FLUX: VAE downscale 8x, patch size 2  ->  one token == 16 image pixels.
TOKEN_PX = 16


def mask_to_token_mask(mask: torch.Tensor, image_h: int, image_w: int,
                       patch_cols: int = 1, invert: bool = False) -> torch.Tensor:
    """Downsample a pixel ``MASK`` to a wrap-padded boolean token mask.

    Args:
        mask: ``[H, W]`` or ``[B, H, W]`` float mask in ``[0, 1]``.
        image_h, image_w: panorama pixel dimensions (e.g. 1024 x 2048).
        patch_cols: number of token columns padded on each side.
        invert: upstream convention is mask==1 -> region *preserved* from the
            source. Set ``invert=True`` to flip (for the inpainting example
            ``mask = 1 - mask``).

    Returns:
        Bool tensor ``[n_h * (n_w + 2*patch_cols)]`` (row-major), True where the
        source features should be replaced into the edit branch.
    """
    if mask.dim() == 3:
        mask = mask[0]
    n_h = image_h // TOKEN_PX
    n_w = image_w // TOKEN_PX

    m = mask[None, None].float()
    m = F.interpolate(m, size=(n_h, n_w), mode="nearest")[0, 0]
    m = (m >= 0.5)
    if invert:
        m = ~m

    # wrap-pad width to match circular-padded latents
    if patch_cols > 0:
        left = m[:, -patch_cols:]
        right = m[:, :patch_cols]
        m = torch.cat([left, m, right], dim=1)

    return m.reshape(-1).contiguous()
