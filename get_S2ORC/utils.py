import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: str = "logs/pipeline.log") -> logging.Logger:
    ensure_dirs(os.path.dirname(log_path) or ".")
    logger = logging.getLogger("s2_pipeline")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def now_year() -> int:
    return datetime.utcnow().year


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clip_text(text: str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def json_dump(path: str, data: Any) -> None:
    ensure_dirs(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def json_load(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=True, sort_keys=True)
    return sha1_text(payload)


class RateLimiter:
    """Thread-safe global rate limiter."""

    def __init__(self, rate_per_sec: float = 1.0):
        self.min_interval = 1.0 / max(rate_per_sec, 1e-9)
        self._lock = threading.Lock()
        self._next_ts = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            if now < self._next_ts:
                time.sleep(self._next_ts - now)
                now = time.time()
            self._next_ts = now + self.min_interval


def estimate_token_len(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return len(text.split())
