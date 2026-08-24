from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .types import CacheEntry, CacheLevel, CacheStats

logger = logging.getLogger(__name__)


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
        self._load_index()

    def _load_index(self) -> None:
        if self._index_file.exists():
            try:
                self._index = json.loads(self._index_file.read_text(encoding="utf-8"))
                logger.info("Loaded cache index: %d entries", len(self._index))
            except Exception as e:
                logger.error("Failed to load cache index: %s", e)
                self._index = {}

    def _save_index(self) -> None:
        self._index_file.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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

        key = self._cache_key(model_id, level, quant_bits)
        entry_data = {
            "model_id": model_id,
            "level": level.value,
            "path": str(dest_path),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "quant_bits": quant_bits,
            "mlx_version": mlx_version,
            "created_at": now,
            "last_accessed": now,
            "ref_count": 0,
        }
        self._index[key] = entry_data
        self._save_index()

        logger.info(
            "Cached %s at level=%s quant=%d size=%.2fGB mlx=%s",
            model_id, level.value, quant_bits, size_bytes / 1e9, mlx_version,
        )
        return CacheEntry(**entry_data)

    def get(
        self, model_id: str, level: CacheLevel, quant_bits: int = 0
    ) -> CacheEntry | None:
        key = self._cache_key(model_id, level, quant_bits)
        data = self._index.get(key)
        if not data:
            return None

        path = Path(data["path"])
        if not path.exists():
            logger.warning("Cache entry %s path missing, removing", key)
            del self._index[key]
            self._save_index()
            return None

        data["last_accessed"] = time.time()
        data["ref_count"] = data.get("ref_count", 0) + 1
        self._index[key] = data
        # F-11: access-time/ref_count bump is in-memory only; avoid rewriting the
        # full index.json on every cache hit. Persisted on structural changes
        # (put/remove/gc/validate) and on next _save_index() call.
        return CacheEntry(**data)

    def has(self, model_id: str, level: CacheLevel, quant_bits: int = 0) -> bool:
        key = self._cache_key(model_id, level, quant_bits)
        data = self._index.get(key)
        if not data:
            return False
        return Path(data["path"]).exists()

    def remove(self, model_id: str, level: CacheLevel, quant_bits: int = 0) -> bool:
        key = self._cache_key(model_id, level, quant_bits)
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
        removed = 0
        keys_to_remove = [k for k in self._index if k.startswith(f"{model_id}:")]
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

    def gc(self, max_size_gb: float = 0, max_age_days: float = 30) -> int:
        now = time.time()
        removed = 0
        keys_to_remove = []

        for key, data in self._index.items():
            age_days = (now - data.get("last_accessed", data.get("created_at", 0))) / 86400
            if max_age_days > 0 and age_days > max_age_days:
                keys_to_remove.append(key)
                continue
            if data.get("ref_count", 0) <= 0 and age_days > max_age_days / 2:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            data = self._index.pop(key)
            path = Path(data["path"])
            if path.exists():
                path.unlink()
            removed += 1

        if max_size_gb > 0:
            total_bytes = sum(
                Path(d["path"]).stat().st_size
                for d in self._index.values()
                if Path(d["path"]).exists()
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

        if removed:
            self._save_index()
            logger.info("GC removed %d entries", removed)
        return removed

    def validate(self, current_mlx_version: str = "") -> dict[str, Any]:
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
            del self._index[key]
        if missing:
            self._save_index()
        logger.info("Cache validate: %s", result)
        return result

    @staticmethod
    def _cache_key(model_id: str, level: CacheLevel, quant_bits: int = 0) -> str:
        if level == CacheLevel.QUANTIZED and quant_bits > 0:
            return f"{model_id}:{level.value}:{quant_bits}bit"
        return f"{model_id}:{level.value}"

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
