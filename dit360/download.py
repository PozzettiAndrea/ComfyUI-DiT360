"""Model download helpers (uses huggingface_hub, which ships with ComfyUI).

No third-party packages beyond what ComfyUI already provides.
"""

import os
import folder_paths


def ensure_file(folder_key: str, filename: str, repo_id: str,
                repo_filename: str = None, subfolder: str = None) -> str:
    """Return a local path to ``filename`` under ComfyUI's ``folder_key`` model
    directory, downloading it from ``repo_id`` on Hugging Face if missing.

    If a file named ``filename`` already exists in any registered path for
    ``folder_key`` (e.g. the user already has FLUX), that path is returned and
    nothing is downloaded.
    """
    existing = folder_paths.get_full_path(folder_key, filename)
    if existing and os.path.exists(existing):
        return existing

    dest_dir = folder_paths.get_folder_paths(folder_key)[0]
    os.makedirs(dest_dir, exist_ok=True)

    from huggingface_hub import hf_hub_download

    print(f"[DiT360] Downloading {repo_id}/{repo_filename or filename} -> {dest_dir}")
    local = hf_hub_download(
        repo_id=repo_id,
        filename=repo_filename or filename,
        subfolder=subfolder,
        local_dir=dest_dir,
    )
    # hf_hub_download may nest under subfolder; expose a flat name in dest_dir
    flat = os.path.join(dest_dir, filename)
    if os.path.abspath(local) != os.path.abspath(flat):
        try:
            if not os.path.exists(flat):
                os.symlink(local, flat)
        except OSError:
            flat = local
    # refresh ComfyUI's cache so dropdowns see the new file
    try:
        folder_paths.get_filename_list.cache_clear()
    except Exception:
        pass
    return flat if os.path.exists(flat) else local
