from __future__ import annotations

import base64
import logging
import os
import struct
from pathlib import Path
from typing import Any

import anyio
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
# P1-4: chunked streaming so GB-scale model files no longer hit a 512MB in-mem
# cap. Each chunk is Fernet-encrypted independently and framed with a 4-byte
# big-endian length prefix; a magic header distinguishes the chunked format
# from the legacy whole-file format so old ciphertext still decrypts.
_CHUNK = 64 * 1024 * 1024  # 64MB per encrypted chunk
_MAGIC = b"FMH1"  # chunked-encryption format marker


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


def _stream_encrypt(path: Path, fernet: Any) -> int:
    # P1-4: read + encrypt in _CHUNK-sized pieces, framing each Fernet token
    # with a 4-byte length prefix under a magic header. Memory stays bounded
    # regardless of file size; the staging-then-replace keeps the original
    # intact if encryption dies mid-way.
    import uuid

    staging = path.parent / f".{path.name}.{uuid.uuid4().hex}.enc.tmp"
    total = 0
    try:
        with open(path, "rb") as src, open(staging, "wb") as out:
            out.write(_MAGIC)
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                token = fernet.encrypt(chunk)
                out.write(struct.pack(">I", len(token)))
                out.write(token)
                total += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)
    return total


def _stream_decrypt(path: Path, fernet: Any) -> int:
    import uuid

    staging = path.parent / f".{path.name}.{uuid.uuid4().hex}.dec.tmp"
    total = 0
    try:
        with open(path, "rb") as src, open(staging, "wb") as out:
            head = src.read(len(_MAGIC))
            if head == _MAGIC:
                # chunked format: framed tokens
                while True:
                    lp = src.read(4)
                    if not lp:
                        break
                    if len(lp) < 4:
                        raise ValueError("Truncated length prefix in encrypted file")
                    (tlen,) = struct.unpack(">I", lp)
                    token = src.read(tlen)
                    if len(token) < tlen:
                        raise ValueError("Truncated Fernet token in encrypted file")
                    out.write(fernet.decrypt(token))
                    total += 1
            else:
                # legacy whole-file format: rest of file is one Fernet token
                rest = head + src.read()
                out.write(fernet.decrypt(rest))
                total = 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)
    return total


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
            # P1-4/P1-10: offload the blocking read/encrypt/write to a thread so
            # the event loop is not blocked by GB-scale file IO.
            size = await anyio.to_thread.run_sync(_stream_encrypt, file_path, fernet)
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
            size = await anyio.to_thread.run_sync(_stream_decrypt, file_path, fernet)
            logger.info("Decrypted version file: id=%s path=%s bytes=%d", v.id, v.file_path, size)
    v = await crud.update_version(session, body.version_id, encrypted=False)
    return {"version_id": v.id, "encrypted": v.encrypted}


@router.get("/encryption/status/{version_id}")
async def encryption_status(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"version_id": v.id, "encrypted": v.encrypted}
