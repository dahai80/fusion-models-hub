import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["encryption"])

# NFR-002: Static encryption for model files (AES-256 via Fernet)
# Called by: app.py include_router, tests/test_api.py
# Depends: cryptography package, ModelVersion.encrypted field


class EncryptRequest(BaseModel):
    version_id: str


class DecryptRequest(BaseModel):
    version_id: str


@router.post("/encryption/encrypt")
async def encrypt_version(body: EncryptRequest, session: SessionDep, store: StoreDep):
    v = await crud.get_version(session, body.version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    if v.encrypted:
        raise HTTPException(status_code=409, detail="Version already encrypted")
    key = os.environ.get("FMH_ENCRYPTION_KEY", "fusion-model-hub-default-encryption-key-32b")
    key_bytes = key.encode()[:32].ljust(32, b'\0')
    if v.file_path:
        file_path = store.get_file(v.file_path)
        if file_path and file_path.exists():
            import base64

            from cryptography.fernet import Fernet
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)
            raw = file_path.read_bytes()
            encrypted = fernet.encrypt(raw)
            file_path.write_bytes(encrypted)
            logger.info("Encrypted version file: id=%s path=%s", v.id, v.file_path)
    v = await crud.update_version(session, body.version_id, encrypted=True)
    return {"version_id": v.id, "encrypted": v.encrypted}


@router.post("/encryption/decrypt")
async def decrypt_version(body: DecryptRequest, session: SessionDep, store: StoreDep):
    v = await crud.get_version(session, body.version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    if not v.encrypted:
        raise HTTPException(status_code=409, detail="Version not encrypted")
    key = os.environ.get("FMH_ENCRYPTION_KEY", "fusion-model-hub-default-encryption-key-32b")
    key_bytes = key.encode()[:32].ljust(32, b'\0')
    if v.file_path:
        file_path = store.get_file(v.file_path)
        if file_path and file_path.exists():
            import base64

            from cryptography.fernet import Fernet
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)
            encrypted = file_path.read_bytes()
            decrypted = fernet.decrypt(encrypted)
            file_path.write_bytes(decrypted)
            logger.info("Decrypted version file: id=%s path=%s", v.id, v.file_path)
    v = await crud.update_version(session, body.version_id, encrypted=False)
    return {"version_id": v.id, "encrypted": v.encrypted}


@router.get("/encryption/status/{version_id}")
async def encryption_status(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"version_id": v.id, "encrypted": v.encrypted}
