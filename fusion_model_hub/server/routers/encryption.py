from __future__ import annotations

import base64
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["encryption"])

# NFR-002: Static encryption for model files (Fernet: AES-128-CBC + HMAC).
# E-S5: the prior default key was a source-public constant, so an unset
# FMH_ENCRYPTION_KEY silently produced ciphertext anyone with the source could
# decrypt — "encrypted: true" was a false guarantee. Now refuse to operate
# unless a non-default key is configured. Fail loud, not false security.
_DEFAULT_KEY = "fusion-model-hub-default-encryption-key-32b"
_MAX_INMEM_BYTES = 512 * 1024 * 1024  # 512MB guard against whole-file OOM (P2: stream)


class EncryptRequest(BaseModel):
    version_id: str


class DecryptRequest(BaseModel):
    version_id: str


def _resolve_fernet() -> Any:
    from cryptography.fernet import Fernet
    key = os.environ.get("FMH_ENCRYPTION_KEY", "")
    if not key or key == _DEFAULT_KEY:
        raise HTTPException(
            status_code=503,
            detail="Encryption disabled: set a non-default FMH_ENCRYPTION_KEY env "
                   "(>=32 bytes, high entropy) before encrypting/decrypting model files",
        )
    key_bytes = key.encode()[:32].ljust(32, b"\0")
    return Fernet(base64.urlsafe_b64encode(key_bytes))


@router.post("/encryption/encrypt")
async def encrypt_version(body: EncryptRequest, session: SessionDep, store: StoreDep):
    v = await crud.get_version(session, body.version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    if v.encrypted:
        raise HTTPException(status_code=409, detail="Version already encrypted")
    fernet = _resolve_fernet()
    if v.file_path:
        file_path = store.get_file(v.file_path)
        if file_path and file_path.exists():
            size = file_path.stat().st_size
            if size > _MAX_INMEM_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large to encrypt in memory ({size} bytes); "
                           "streaming encryption not yet implemented (P2)",
                )
            raw = file_path.read_bytes()
            encrypted = fernet.encrypt(raw)
            file_path.write_bytes(encrypted)
            logger.info("Encrypted version file: id=%s path=%s bytes=%d", v.id, v.file_path, size)
    v = await crud.update_version(session, body.version_id, encrypted=True)
    return {"version_id": v.id, "encrypted": v.encrypted}


@router.post("/encryption/decrypt")
async def decrypt_version(body: DecryptRequest, session: SessionDep, store: StoreDep):
    v = await crud.get_version(session, body.version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    if not v.encrypted:
        raise HTTPException(status_code=409, detail="Version not encrypted")
    fernet = _resolve_fernet()
    if v.file_path:
        file_path = store.get_file(v.file_path)
        if file_path and file_path.exists():
            size = file_path.stat().st_size
            if size > _MAX_INMEM_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large to decrypt in memory ({size} bytes); "
                           "streaming decryption not yet implemented (P2)",
                )
            encrypted = file_path.read_bytes()
            decrypted = fernet.decrypt(encrypted)
            file_path.write_bytes(decrypted)
            logger.info("Decrypted version file: id=%s path=%s bytes=%d", v.id, v.file_path, size)
    v = await crud.update_version(session, body.version_id, encrypted=False)
    return {"version_id": v.id, "encrypted": v.encrypted}


@router.get("/encryption/status/{version_id}")
async def encryption_status(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"version_id": v.id, "encrypted": v.encrypted}

