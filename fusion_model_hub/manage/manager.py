"""Local model manager — manages locally installed .mlx models."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalModelManager:
    """Manages locally installed Fusion-MLX models.

    All model operations go through fusion-mlx API.
    """

    def __init__(self, models_dir: str = ""):
        if not models_dir:
            models_dir = str(Path.home() / "Library" / "Fusion" / "Models")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._meta_file = self.models_dir / "models.json"
        self._load_meta()

    def _load_meta(self) -> None:
        self._models: dict[str, dict] = {}
        if self._meta_file.exists():
            try:
                self._models = json.loads(self._meta_file.read_text(encoding="utf-8"))
            except Exception:
                self._models = {}

    def _save_meta(self) -> None:
        self._meta_file.write_text(json.dumps(self._models, indent=2, ensure_ascii=False), encoding="utf-8")

    def register(self, model_id: str, name: str, path: str,
                 quant: str = "4bit", version: str = "1.0.0",
                 mlx_version: str = ">=0.5.0") -> None:
        """Register a locally installed model."""
        self._models[model_id] = {
            "id": model_id,
            "name": name,
            "path": path,
            "quantization": quant,
            "version": version,
            "mlx_version": mlx_version,
            "active": False,
            "last_used": "",
            "created_at": time.time(),
        }
        self._save_meta()

    def unregister(self, model_id: str) -> bool:
        """Remove a model from local registry."""
        removed = self._models.pop(model_id, None) is not None
        if removed:
            self._save_meta()
        return removed

    def list(self) -> list[dict[str, Any]]:
        """List all locally installed models."""
        return list(self._models.values())

    def get(self, model_id: str) -> dict | None:
        return self._models.get(model_id)

    def set_active(self, model_id: str, active: bool = True) -> bool:
        """Mark a model as active (running)."""
        if model_id not in self._models:
            return False
        for mid in self._models:
            self._models[mid]["active"] = False
        self._models[model_id]["active"] = active
        self._models[model_id]["last_used"] = time.time()
        self._save_meta()
        return True

    def delete_model(self, model_id: str) -> dict[str, Any]:
        """Delete a model file and its metadata."""
        model = self._models.get(model_id)
        if not model:
            return {"status": "not_found"}
        path = Path(model["path"])
        if path.exists():
            path.unlink()
        self.unregister(model_id)
        return {"status": "deleted", "path": str(path)}

    def get_stats(self) -> dict[str, Any]:
        """Get local model statistics."""
        total_models = len(self._models)
        total_size = 0
        active_count = 0
        for m in self._models.values():
            p = Path(m.get("path", ""))
            if p.exists():
                total_size += p.stat().st_size
            if m.get("active"):
                active_count += 1
        return {
            "total_models": total_models,
            "active_models": active_count,
            "total_size_gb": round(total_size / (1024**3), 2),
        }