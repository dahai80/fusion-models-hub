from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# E-E8: one canonical file-hashing helper. Prior copies lived in
# cache/manager.py, repo/downloader.py, server/routers/inference.py,
# server/routers/sync.py, storage/local_store.py — five re-implementations
# with inconsistent chunk sizes (8KB in sync.py, 64KB elsewhere). A 64KB
# chunk is the right default: large enough to amortize per-call overhead on
# multi-GB model files, small enough to keep memory flat regardless of file
# size. Callers that streamed hashes during upload (versions.py,
# local_store chunked assemble, downloads streaming, minio put) are NOT
# touched — those hash in-flight bytes, not a re-opened file.
DEFAULT_CHUNK_SIZE = 65536


def compute_sha256(file_path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    # E-E8: stream the file in fixed chunks so memory stays flat for multi-GB
    # model weights. Returns the lowercase hex digest.
    path = Path(file_path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_and_size(file_path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    # E-E8: sync.py's _disk_hash_and_size hashed then re-stated; unify into one
    # helper that hashes and reports the os stat size in a single pass.
    path = Path(file_path)
    size = os.stat(path).st_size
    digest = compute_sha256(path, chunk_size=chunk_size)
    return digest, size


def verify_sha256(file_path: str | Path, expected_hash: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
    # E-E8: local_store.verify_hash reimplemented this with its own chunk loop
    # and a mismatch log. Centralize so the comparison + log is identical
    # everywhere; callers that want silent comparison can call compute_sha256
    # directly.
    path = Path(file_path)
    actual = compute_sha256(path, chunk_size=chunk_size)
    ok = actual == expected_hash.lower()
    if not ok:
        logger.warning(
            "Hash mismatch: file=%s expected=%s actual=%s",
            path, expected_hash[:16], actual[:16],
        )
    return ok
