"""Fusion-MLX base binding — detects, verifies, and manages the Fusion-MLX base dependency.

All model operations depend on Fusion-MLX being installed. This module handles
detection, version checking, and compatibility verification.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class FusionMLXBase:
    """Manages the Fusion-MLX base dependency.

    All model operations require Fusion-MLX to be installed and running.
    This module provides detection, version checking, and compatibility verification.
    """

    def __init__(self, mlx_url: str = "http://localhost:11434"):
        self.mlx_url = mlx_url.rstrip("/")

    async def detect(self) -> dict[str, Any]:
        """Detect if Fusion-MLX is installed and running.

        Returns:
            Dict with status, version, and capabilities.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.mlx_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    version = self._extract_version(data)
                    return {
                        "installed": True,
                        "running": True,
                        "version": version,
                        "models_available": len(data.get("data", [])),
                    }
        except Exception:
            pass

        # Check if fusion-mlx command exists
        import shutil

        if shutil.which("fusion-mlx"):
            return {"installed": True, "running": False, "version": "detected", "models_available": 0}

        return {"installed": False, "running": False, "version": "", "models_available": 0}

    async def get_version(self) -> str:
        """Get Fusion-MLX version string, or empty string if unavailable."""
        info = await self.detect()
        return info.get("version", "")

    async def check_compatibility(self, required_version: str = ">=0.5.0") -> dict[str, bool]:
        """Check if installed Fusion-MLX version meets requirements.

        Args:
            required_version: Version requirement string (e.g. ">=0.5.0").

        Returns:
            Dict with compatible flag and details.
        """
        info = await self.detect()
        if not info["installed"]:
            return {"compatible": False, "reason": "Fusion-MLX not installed"}

        if not info["running"]:
            return {"compatible": False, "reason": "Fusion-MLX not running"}

        version = info.get("version", "")
        # H10: the prior implementation ignored required_version entirely and
        # returned compatible=True unconditionally once MLX was running — so an
        # incompatible older build passed as compatible. Actually compare now.
        if not version or not _version_satisfies(version, required_version):
            return {
                "compatible": False,
                "version": version,
                "required": required_version,
                "reason": f"version {version or 'unknown'} does not satisfy {required_version}",
            }
        return {"compatible": True, "version": version, "required": required_version}

    async def get_capabilities(self) -> dict[str, Any]:
        """Get Fusion-MLX capabilities (Metal, KV Cache, etc.).

        Returns only what Fusion-MLX actually reports. The prior version
        fabricated metal_available/kv_cache/quantization/max_context from a
        bare /v1/models 200 (which carries none of that) — a hardcoded lie that
        made the hub claim support Fusion-MLX may not have.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.mlx_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    # Surface only fields Fusion-MLX actually returns; unknown
                    # capabilities are reported as unknown, not invented.
                    caps = data.get("capabilities") or {}
                    return {
                        "metal_available": caps.get("metal_available"),
                        "kv_cache": caps.get("kv_cache"),
                        "quantization": caps.get("quantization", []),
                        "max_context": caps.get("max_context", 0),
                        "models_available": len(data.get("data", [])),
                    }
        except Exception:
            pass
        return {
            "metal_available": None,
            "kv_cache": None,
            "quantization": [],
            "max_context": 0,
            "models_available": 0,
        }

    @staticmethod
    def _extract_version(data: dict) -> str:
        """Extract version from API response. Empty string when unknown — never
        invent a version, which would let an incompatible build pass checks."""
        return data.get("version", "")


def _parse_version(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for tok in v.split("."):
        num = ""
        for ch in tok:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _version_satisfies(version: str, requirement: str) -> bool:
    # Minimal spec parser supporting ">=X.Y.Z" (the only form the hub uses).
    # Returns False on any unparseable input rather than guessing compatible.
    req = requirement.strip()
    if not req.startswith(">="):
        logger.warning("Unsupported version requirement %r — treating as incompatible", req)
        return False
    threshold = _parse_version(req[2:].strip())
    actual = _parse_version(version)
    if not actual or not threshold:
        return False
    # Pad shorter tuple with zeros for comparison.
    n = max(len(actual), len(threshold))
    actual = actual + (0,) * (n - len(actual))
    threshold = threshold + (0,) * (n - len(threshold))
    return actual >= threshold
