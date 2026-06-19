"""Make the bundled example panorama(s) available in ComfyUI on startup.

ComfyUI runs each custom node's prestartup_script.py before nodes load. We copy
the images in this pack's assets/ into ComfyUI's input/ directory so the editing
workflow's LoadImage finds them out of the box. Existing files are never
overwritten.
"""

import os
import shutil

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _input_dir(node_dir):
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        # custom_nodes/<pack>/ -> ComfyUI/ -> ComfyUI/input
        return os.path.join(os.path.dirname(os.path.dirname(node_dir)), "input")


def _copy_assets_to_input():
    node_dir = os.path.dirname(os.path.realpath(__file__))
    assets = os.path.join(node_dir, "assets")
    if not os.path.isdir(assets):
        return

    input_dir = _input_dir(node_dir)
    os.makedirs(input_dir, exist_ok=True)

    for name in sorted(os.listdir(assets)):
        if not name.lower().endswith(_IMG_EXTS):
            continue
        src = os.path.join(assets, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(input_dir, name)
        if os.path.exists(dst):
            continue
        try:
            shutil.copy2(src, dst)
            print(f"[DiT360] copied example asset -> input/{name}")
        except Exception as e:
            print(f"[DiT360] could not copy {name}: {e}")


_copy_assets_to_input()
