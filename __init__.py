"""ComfyUI-DiT360 — seamless 360° panorama generation, fully ComfyUI-native.

No third-party Python dependencies: nodes import only torch, numpy and ComfyUI's
own modules. The DiT360 method is FLUX.1-dev + a LoRA plus circular padding;
this pack reuses ComfyUI's native FLUX model/sampler/LoRA and ports only the
panorama-specific logic (circular padding, and — for editing — RF-inversion and
masked feature sharing).
"""

from .dit360.nodes_load import DiT360ModelLoader
from .dit360.nodes_t2p import DiT360PanoramaSampler

NODE_CLASS_MAPPINGS = {
    "DiT360ModelLoader": DiT360ModelLoader,
    "DiT360PanoramaSampler": DiT360PanoramaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360ModelLoader": "(down)Load DiT360 model(s)",
    "DiT360PanoramaSampler": "DiT360 Panorama Sampler",
}

# Editing nodes (inpaint / outpaint via RF-inversion + masked feature sharing).
# Registered when the module is present; kept optional so the core pack always
# imports cleanly.
try:
    from .dit360.nodes_edit import EDIT_NODE_CLASS_MAPPINGS, EDIT_NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(EDIT_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(EDIT_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:  # pragma: no cover
    print(f"[DiT360] editing nodes not loaded yet: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
