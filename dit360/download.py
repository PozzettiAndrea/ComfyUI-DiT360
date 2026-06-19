"""Model download helpers with dual progress: console (tqdm) + ComfyUI pbar.

Uses huggingface_hub for URL resolution and requests for streaming — both ship
with ComfyUI, so no new dependencies.
"""

import os
import folder_paths


def _streamed_download(url, dest_path, token=None):
    """Stream ``url`` to ``dest_path`` showing a console tqdm bar AND a ComfyUI
    progress bar (the node shows a bar in the UI while downloading)."""
    import requests
    from tqdm import tqdm
    import comfy.utils

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        pbar = comfy.utils.ProgressBar(total if total > 0 else 1)
        done = 0
        tmp = dest_path + ".part"
        chunk = 1 << 20  # 1 MiB
        name = os.path.basename(dest_path)
        with open(tmp, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"[DiT360] {name}", dynamic_ncols=True,
        ) as bar:
            for data in r.iter_content(chunk_size=chunk):
                if not data:
                    continue
                f.write(data)
                done += len(data)
                bar.update(len(data))
                pbar.update_absolute(done, total if total > 0 else done)
        os.replace(tmp, dest_path)
    return dest_path


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
    dest_path = os.path.join(dest_dir, filename)

    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=repo_id, filename=repo_filename or filename, subfolder=subfolder)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[DiT360] Downloading {repo_id}/{repo_filename or filename} -> {dest_path}")
    try:
        _streamed_download(url, dest_path, token=token)
    except Exception as e:
        # Fall back to hf_hub_download (still shows hub's own tqdm) on any issue.
        print(f"[DiT360] streamed download failed ({e}); falling back to hf_hub_download")
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo_id, filename=repo_filename or filename,
                                subfolder=subfolder, local_dir=dest_dir)
        if os.path.abspath(local) != os.path.abspath(dest_path) and not os.path.exists(dest_path):
            try:
                os.symlink(local, dest_path)
            except OSError:
                dest_path = local

    # refresh ComfyUI's cache so dropdowns see the new file
    try:
        folder_paths.get_filename_list.cache_clear()
    except Exception:
        pass
    return dest_path
