from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .types import CacheEntry, CacheLevel, CacheStats

logger = logging.getLogger(__name__)

# E-R4: minimum seconds between two index.json writes caused by cache hits
# refreshing last_accessed. Without throttling, a hot cache would rewrite the
# full index on every get(); without persisting at all, the LRU/age signal seen
# by gc after a restart was stale. 30s is a coarse but safe middle ground.
_ACCESS_PERSIST_INTERVAL = 30.0


class CacheManager:
    def __init__(self, cache_root: str = ""):
        if not cache_root:
            cache_root = str(Path.home() / "Library" / "Fusion" / "Cache")
        self.cache_root = Path(cache_root)
        self.raw_dir = self.cache_root / "raw"
        self.converted_dir = self.cache_root / "converted"
        self.quantized_dir = self.cache_root / "quantized"
        for d in (self.raw_dir, self.converted_dir, self.quantized_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_root / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        # E-R1: serialize read-modify-write of the index so concurrent put/remove/gc
        # calls cannot interleave and corrupt index.json.
        self._lock = threading.Lock()
        # E-R4: tracks the last index.json write so cache-hit last_accessed
        # bumps persist at most once per _ACCESS_PERSIST_INTERVAL instead of on
        # every get().
        self._last_index_save: float = 0.0
        self._load_index()

    def _load_index(self) -> None:
        if self._index_file.exists():
            try:
                self._index = json.loads(self._index_file.read_text(encoding="utf-8"))
                logger.info("Loaded cache index: %d entries", len(self._index))
            except (json.JSONDecodeError, ValueError) as e:
                # E-R1: corrupt index must not silently wipe and pretend empty —
                # that hides data loss. Quarantine the bad file and start fresh,
                # but log loudly so the operator can recover the old entries.
                corrupt_path = self._index_file.with_suffix(f".corrupt.{uuid.uuid4().hex[:8]}.json")
                with contextlib.suppress(OSError):
                    shutil.move(str(self._index_file), str(corrupt_path))
                logger.error(
                    "Cache index corrupted, quarantined to %s: %s",
                    corrupt_path,
                    e,
                )
                self._index = {}
            except Exception as e:
                logger.error("Failed to load cache index: %s", e)
                self._index = {}

    def _save_index(self) -> None:
        # E-R1: atomic write — stage to a temp file, fsync, os.replace. A crash
        # or a concurrent writer can no longer leave a half-written index.json
        # that would zero the cache on next load.
        staging = self._index_file.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with open(staging, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(staging, self._index_file)
        finally:
            if staging.exists():
                staging.unlink(missing_ok=True)
        # E-R4: every persisted write refreshes the throttle clock so a cache
        # hit right after a put/remove/gc doesn't immediately re-save.
        self._last_index_save = time.time()

    def _level_dir(self, level: CacheLevel) -> Path:
        return {
            CacheLevel.RAW: self.raw_dir,
            CacheLevel.CONVERTED: self.converted_dir,
            CacheLevel.QUANTIZED: self.quantized_dir,
        }[level]

    def put(
        self,
        model_id: str,
        level: CacheLevel,
        source_path: str | Path,
        quant_bits: int = 0,
        mlx_version: str = "",
        source_version_id: str = "",
    ) -> CacheEntry:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        dest_dir = self._level_dir(level)
        if level == CacheLevel.QUANTIZED and quant_bits > 0:
            dest_dir = dest_dir / f"{quant_bits}bit"
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f"{model_id}{src.suffix}"
        shutil.copy2(src, dest_path)

        sha256 = self._hash_file(dest_path)
        size_bytes = dest_path.stat().st_size
        now = time.time()

        key = self._cache_key(model_id, level, quant_bits, source_version_id)
        entry_data = {
            "model_id": model_id,
            "level": level.value,
            "path": str(dest_path),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "quant_bits": quant_bits,
            "mlx_version": mlx_version,
            "source_version_id": source_version_id,
            "created_at": now,
            "last_accessed": now,
        }
        with self._lock:
            self._index[key] = entry_data
            self._save_index()

        logger.info(
            "Cached %s at level=%s quant=%d size=%.2fGB mlx=%s",
            model_id,
            level.value,
            quant_bits,
            size_bytes / 1e9,
            mlx_version,
        )
        return CacheEntry(**entry_data)

    def get(
        self,
        model_id: str,
        level: CacheLevel,
        quant_bits: int = 0,
        source_version_id: str = "",
    ) -> CacheEntry | None:
        key = self._cache_key(model_id, level, quant_bits, source_version_id)
        data = self._index.get(key)
        if not data:
            return None

        path = Path(data["path"])
        if not path.exists():
            logger.warning("Cache entry %s path missing, removing", key)
            with self._lock:
                self._index.pop(key, None)
                self._save_index()
            return None

        data["last_accessed"] = time.time()
        self._index[key] = data
        # E-R4: ref_count was bumped here but never persisted and never decremented
        # anywhere, so after a reload every entry read back as 0. The bump fed a
        # fictional GC "pin" that never protected a live entry. Removed the
        # ref_count machinery entirely (see gc() + remove_model() + put()); the
        # cache is now evicted purely by age and by last_accessed LRU under a
        # size cap. last_accessed is the real recency signal. Persist it
        # opportunistically — only when the last save is older than the throttle
        # window — so a cache hit still refreshes the LRU/age signal without
        # rewriting index.json on every single hit.
        now = time.time()
        if now - self._last_index_save > _ACCESS_PERSIST_INTERVAL:
            with self._lock:
                self._save_index()
                self._last_index_save = now
        return CacheEntry(**data)

    def has(
        self,
        model_id: str,
        level: CacheLevel,
        quant_bits: int = 0,
        source_version_id: str = "",
    ) -> bool:
        key = self._cache_key(model_id, level, quant_bits, source_version_id)
        data = self._index.get(key)
        if not data:
            return False
        return Path(data["path"]).exists()

    def remove(
        self,
        model_id: str,
        level: CacheLevel,
        quant_bits: int = 0,
        source_version_id: str = "",
    ) -> bool:
        key = self._cache_key(model_id, level, quant_bits, source_version_id)
        with self._lock:
            data = self._index.pop(key, None)
            if not data:
                return False
            path = Path(data["path"])
            if path.exists():
                path.unlink()
                logger.info("Removed cache entry: %s", key)
            self._save_index()
        return True

    def remove_model(self, model_id: str) -> int:
        with self._lock:
            removed = 0
            # E-R5: the cache key is "{model_id}:{level}[:{bits}bit][:{ver}]".
            # The prior `k.startswith(f"{model_id}:")` matched by raw prefix, so
            # deleting model "abc" also matched keys for model "abcxyz" — a
            # cross-model purge. Match only keys whose first colon-segment equals
            # model_id exactly.
            keys_to_remove = [k for k in self._index if k.split(":", 1)[0] == model_id]
            for key in keys_to_remove:
                data = self._index.pop(key)
                path = Path(data["path"])
                if path.exists():
                    path.unlink()
                removed += 1
            if removed:
                self._save_index()
                logger.info("Removed %d cache entries for model %s", removed, model_id)
        return removed

    def stats(self) -> CacheStats:
        total_bytes = 0
        counts = {CacheLevel.RAW: 0, CacheLevel.CONVERTED: 0, CacheLevel.QUANTIZED: 0}
        for data in self._index.values():
            path = Path(data["path"])
            if not path.exists():
                continue
            level = CacheLevel(data["level"])
            counts[level] += 1
            total_bytes += path.stat().st_size
        return CacheStats(
            total_entries=sum(counts.values()),
            total_size_bytes=total_bytes,
            raw_count=counts[CacheLevel.RAW],
            converted_count=counts[CacheLevel.CONVERTED],
            quantized_count=counts[CacheLevel.QUANTIZED],
            levels={k.value: v for k, v in counts.items()},
        )

    def _reconcile_orphans(self) -> int:
        # E-R6: gc walked self._index, never the filesystem. A crash, a manual
        # rm of index.json, or a corrupt index quarantined by _load_index (which
        # starts fresh with self._index = {}) left real cache files on disk with
        # no index entry — silently leaked forever, still eating space. Scan the
        # three level dirs and delete any file whose absolute path is not
        # referenced by any current index entry. Must run under self._lock.
        indexed_paths = {Path(d["path"]).resolve() for d in self._index.values() if d.get("path")}
        orphan_removed = 0
        for level_dir in (self.raw_dir, self.converted_dir, self.quantized_dir):
            if not level_dir.exists():
                continue
            for entry_path in level_dir.rglob("*"):
                if not entry_path.is_file():
                    continue
                # Skip our own atomic-write / quarantine sidecars (.tmp, .corrupt).
                if entry_path.suffix in (".tmp",) or ".corrupt." in entry_path.name:
                    continue
                if entry_path.resolve() not in indexed_paths:
                    try:
                        size = entry_path.stat().st_size
                        entry_path.unlink()
                        orphan_removed += 1
                        logger.warning(
                            "GC reconciler removed orphan cache file: %s (%d bytes)",
                            entry_path,
                            size,
                        )
                    except OSError as e:
                        logger.warning("GC reconciler could not remove orphan %s: %s", entry_path, e)
        return orphan_removed

    def gc(self, max_size_gb: float = 0, max_age_days: float = 30) -> int:
        with self._lock:
            now = time.time()
            removed = 0
            keys_to_remove = []

            # E-R4: eviction is by age (last_accessed, falling back to created_at)
            # and, if a size cap is set, by last_accessed LRU. The prior
            # `ref_count <= 0 and age > max_age_days/2` clause was a fictional
            # "pin" — ref_count was never persisted and never decremented, so
            # every entry read 0 after a reload and the clause merely evicted at
            # half the age threshold under a false "in use" excuse. A live
            # ModelVersion whose backend cache was freshly written (ref_count 0
            # because inference never touches the cache) could be evicted, forcing
            # the next quantize to fully re-run. Dropping the clause makes GC
            # honestly age+LRU based; nothing claims to protect in-use entries.
            for key, data in self._index.items():
                age_days = (now - data.get("last_accessed", data.get("created_at", 0))) / 86400
                if max_age_days > 0 and age_days > max_age_days:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                data = self._index.pop(key)
                path = Path(data["path"])
                if path.exists():
                    path.unlink()
                removed += 1

            if max_size_gb > 0:
                total_bytes = sum(
                    Path(d["path"]).stat().st_size for d in self._index.values() if Path(d["path"]).exists()
                )
                if total_bytes > max_size_gb * 1e9:
                    sorted_keys = sorted(
                        self._index.items(),
                        key=lambda x: x[1].get("last_accessed", 0),
                    )
                    for key, data in sorted_keys:
                        if total_bytes <= max_size_gb * 1e9:
                            break
                        path = Path(data["path"])
                        if path.exists():
                            total_bytes -= path.stat().st_size
                            path.unlink()
                        del self._index[key]
                        removed += 1

            # E-R6: after index-driven eviction, reconcile filesystem against the
            # surviving index so leaked/orphaned files (crash, manual index
            # deletion, or a corrupt index quarantined to a fresh empty state)
            # are reclaimed, not silently retained forever.
            orphan_removed = self._reconcile_orphans()
            removed += orphan_removed

            if removed:
                self._save_index()
                logger.info("GC removed %d entries (%d orphans)", removed, orphan_removed)
        return removed

    def validate(self, current_mlx_version: str = "") -> dict[str, Any]:
        with self._lock:
            missing = []
            hash_mismatch = []
            version_stale = []
            for key, data in self._index.items():
                path = Path(data["path"])
                if not path.exists():
                    missing.append(key)
                    continue
                if data.get("sha256"):
                    actual = self._hash_file(path)
                    if actual != data["sha256"]:
                        hash_mismatch.append(key)
                if current_mlx_version and data.get("mlx_version") and data["mlx_version"] != current_mlx_version:
                    level = data.get("level", "")
                    if level in (CacheLevel.CONVERTED.value, CacheLevel.QUANTIZED.value):
                        version_stale.append(key)
            result = {
                "missing": len(missing),
                "hash_mismatch": len(hash_mismatch),
                "version_stale": len(version_stale),
                "valid": len(self._index) - len(missing) - len(hash_mismatch) - len(version_stale),
                "stale_keys": version_stale,
            }
            for key in missing:
                self._index.pop(key, None)
            if missing:
                self._save_index()
        logger.info("Cache validate: %s", result)
        return result

    @staticmethod
    def _cache_key(
        model_id: str,
        level: CacheLevel,
        quant_bits: int = 0,
        source_version_id: str = "",
    ) -> str:
        # H9/R2: the cache key MUST include the source version id. Before, a
        # 4bit quantize of version A and a 4bit quantize of version B of the
        # same model shared one key "model:quantized:4bit" — the second quantize
        # hit the first's stale cache and served the wrong weights. Keying by
        # source_version_id makes each (version, bits) pair distinct; old
        # keyless entries simply never match a versioned lookup, so they are
        # implicitly invalidated and age out via gc.
        if level == CacheLevel.QUANTIZED and quant_bits > 0:
            base = f"{model_id}:{level.value}:{quant_bits}bit"
        else:
            base = f"{model_id}:{level.value}"
        if source_version_id:
            return f"{base}:{source_version_id}"
        return base

    @staticmethod
    def _hash_file(path: Path) -> str:
        # E-E8: delegate to the shared utils helper so chunk size and behavior
        # are identical across cache, downloader, inference, sync, storage.
        from ..utils.hashing import compute_sha256

        return compute_sha256(path)

    # P1-6/P1-8: the sync methods above are correct but block the event loop
    # when called from an async context (tasks.py does cache.put() with no
    # await, running multi-GB copy+SHA256 on the loop). These async wrappers
    # offload the blocking work to a worker thread; the index dict is still
    # guarded by the threading.Lock so concurrent sync + async callers stay
    # serialized. The cache router (sync endpoints) keeps using the sync
    # methods; the quantize hot path uses these async wrappers.
    async def put_async(self, *args: Any, **kwargs: Any) -> CacheEntry:
        from functools import partial

        import anyio

        # anyio.to_thread.run_sync only forwards positional args to the func
        # (its **kwargs are its own: abandon_on_cancel/cancellable/limiter).
        # Bind kwargs via partial so callers using keyword args reach self.put.
        return await anyio.to_thread.run_sync(partial(self.put, *args, **kwargs))

    async def gc_async(self, *args: Any, **kwargs: Any) -> int:
        from functools import partial

        import anyio

        return await anyio.to_thread.run_sync(partial(self.gc, *args, **kwargs))

    async def validate_async(self, *args: Any, **kwargs: Any) -> int:
        from functools import partial

        import anyio

        return await anyio.to_thread.run_sync(partial(self.validate, *args, **kwargs))
