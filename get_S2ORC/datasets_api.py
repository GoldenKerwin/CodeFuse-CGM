import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from get_S2ORC.utils import RateLimiter, ensure_dirs, json_load, json_dump, stable_hash

_GLOBAL_RATE_LIMITER = RateLimiter(rate_per_sec=1.0)


class S2DatasetsClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.semanticscholar.org",
        cache_dir: str = ".cache/api",
        max_retries: int = 5,
        rate_per_sec: float = 1.0,
        timeout: int = 60,
        logger=None,
    ) -> None:
        self.api_key = api_key or os.getenv("S2_API_KEY")
        if not self.api_key:
            raise ValueError("Missing S2_API_KEY environment variable")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        ensure_dirs(str(self.cache_dir))
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key})
        self.rate_limiter = _GLOBAL_RATE_LIMITER
        self.logger = logger

    def _cache_path(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None = None,
    ) -> Path:
        key = stable_hash(
            {"method": method, "endpoint": endpoint, "params": params or {}, "json": json_body or {}}
        )
        return self.cache_dir / f"{key}.json"

    def _request_json(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        cache_path = self._cache_path(method, endpoint, params, json_body=json_body)
        if use_cache and cache_path.exists():
            return json_load(str(cache_path))

        url = f"{self.base_url}{endpoint}"
        last_err = None
        for i in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                resp = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                if use_cache:
                    json_dump(str(cache_path), data)
                return data
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** i, 16) + random.random()
                if self.logger:
                    self.logger.warning("API request failed (%s), retry in %.2fs", e, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"API request failed after retries: {url} {params}, error={last_err}")

    def list_releases(self) -> list[dict[str, Any]]:
        data = self._request_json("GET", "/datasets/v1/release/")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "releases" in data and isinstance(data["releases"], list):
                return data["releases"]
        return []

    def get_release_datasets(self, release_id: str) -> Any:
        return self._request_json("GET", f"/datasets/v1/release/{release_id}")

    def get_dataset_files(self, release_id: str, dataset_name: str) -> Any:
        return self._request_json(
            "GET", f"/datasets/v1/release/{release_id}/dataset/{dataset_name}", use_cache=False
        )

    def get_diff_files(self, start_release_id: str, end_release_id: str, dataset_name: str) -> Any:
        return self._request_json(
            "GET",
            f"/datasets/v1/diffs/{start_release_id}/to/{end_release_id}/{dataset_name}",
            use_cache=False,
        )

    def graph_paper_batch(self, ids: list[str], fields: str, use_cache: bool = True) -> Any:
        return self._request_json(
            "POST",
            "/graph/v1/paper/batch",
            params={"fields": fields},
            json_body={"ids": ids},
            use_cache=use_cache,
        )


def normalize_release_ids(releases: list[dict[str, Any]] | list[str]) -> tuple[str, str | None]:
    ids: list[str] = []
    for r in releases:
        if isinstance(r, str):
            ids.append(r)
        elif isinstance(r, dict):
            rid = r.get("release_id") or r.get("releaseId") or r.get("id")
            if rid:
                ids.append(str(rid))
    ids = sorted(set(ids), reverse=True)
    latest = ids[0] if ids else ""
    previous = ids[1] if len(ids) > 1 else None
    return latest, previous


def extract_dataset_names(release_payload: Any) -> list[str]:
    if isinstance(release_payload, list):
        return [str(x) for x in release_payload]
    if isinstance(release_payload, dict):
        datasets = release_payload.get("datasets")
        if isinstance(datasets, list):
            names = []
            for item in datasets:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    n = item.get("name") or item.get("dataset")
                    if n:
                        names.append(str(n))
            return names
        # some releases may return mapping
        names = []
        for k, v in release_payload.items():
            if isinstance(v, (list, dict)) and k.lower() in {"datasets", "dataset"}:
                continue
            if isinstance(k, str) and any(x in k.lower() for x in ["papers", "abstract", "citation", "reference"]):
                names.append(k)
        return sorted(set(names))
    return []


def extract_file_urls(dataset_payload: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(dataset_payload, dict):
        for key in ["files", "file_urls", "urls", "download_urls"]:
            v = dataset_payload.get(key)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        urls.append(item)
                    elif isinstance(item, dict):
                        u = item.get("url") or item.get("download_url")
                        if u:
                            urls.append(str(u))
        if not urls:
            for v in dataset_payload.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.startswith("http"):
                            urls.append(item)
    elif isinstance(dataset_payload, list):
        urls = [x for x in dataset_payload if isinstance(x, str) and x.startswith("http")]
    return urls
