import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from get_S2ORC.utils import ensure_dirs, json_dump, json_load


def _filename_from_url(url: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    return name or "part.jsonl.gz"


def download_with_resume(
    url: str,
    target_path: str,
    chunk_size: int = 1024 * 1024,
    timeout: int = 120,
    logger=None,
    max_retries: int = 5,
) -> dict[str, Any]:
    ensure_dirs(os.path.dirname(target_path) or ".")
    tmp_path = f"{target_path}.part"
    last_err = None
    for attempt in range(max_retries):
        try:
            headers = {}
            downloaded = 0
            if os.path.exists(tmp_path):
                downloaded = os.path.getsize(tmp_path)
                if downloaded > 0:
                    headers["Range"] = f"bytes={downloaded}-"

            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                mode = "ab" if r.status_code == 206 and downloaded > 0 else "wb"
                if mode == "wb":
                    downloaded = 0
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
            break
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            sleep_s = min(2 ** attempt, 16) + random.random()
            if logger:
                logger.warning(
                    "Download failed for %s (attempt %d/%d): %s; retry in %.2fs",
                    os.path.basename(target_path),
                    attempt + 1,
                    max_retries,
                    e,
                    sleep_s,
                )
            time.sleep(sleep_s)

    os.replace(tmp_path, target_path)
    size = os.path.getsize(target_path)
    if logger:
        logger.info("Downloaded %s (%d bytes)", target_path, size)
    return {"path": target_path, "size": size}


def download_dataset_files(
    dataset_name: str,
    urls: list[str],
    raw_dir: str,
    max_files: int,
    manifest_path: str,
    logger=None,
) -> dict[str, Any]:
    ensure_dirs(raw_dir)
    manifest = json_load(manifest_path, default={}) or {}
    manifest.setdefault("datasets", {})
    manifest["datasets"].setdefault(dataset_name, [])

    selected = urls[: max(1, min(max_files, len(urls)))]
    existing = {item.get("url"): item for item in manifest["datasets"][dataset_name]}
    out_items = []

    for idx, url in enumerate(selected):
        fname = f"{dataset_name}__{idx:03d}__{_filename_from_url(url)}"
        target_path = str(Path(raw_dir) / fname)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            info = {"url": url, "path": target_path, "size": os.path.getsize(target_path), "status": "exists"}
            out_items.append(info)
            continue

        info = download_with_resume(url, target_path, logger=logger)
        out_items.append({"url": url, "path": info["path"], "size": info["size"], "status": "downloaded"})

    manifest["datasets"][dataset_name] = out_items
    json_dump(manifest_path, manifest)
    return manifest
