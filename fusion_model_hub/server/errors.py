from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def safe_http_error(
    status_code: int,
    public_detail: str,
    *,
    exc: BaseException | None = None,
    context: str = "",
) -> HTTPException:
    # E-E5: endpoints raised HTTPException(detail=str(e)) / detail=resp.text,
    # which the global handler passes straight to the client (app.py
    # exception_handler: content={"detail": exc.detail}). That leaked the
    # SQLAlchemy db_url with creds, "(sqlite3.OperationalError) database is
    # locked", and MLX error bodies containing absolute filesystem paths to
    # the caller. Centralize: log the raw internal message + a short trace_id
    # at ERROR/DEBUG, and raise an HTTPException whose detail is a fixed,
    # non-revealing string the client can act on. The trace_id lets an
    # operator correlate a user report to the log line without exposing
    # internals in the response.
    trace_id = uuid.uuid4().hex[:12]
    where = f" [{context}]" if context else ""
    if exc is not None:
        logger.error(
            "safe_http_error%s trace_id=%s: %s: %s",
            where,
            trace_id,
            type(exc).__name__,
            exc,
        )
        logger.debug("safe_http_error%s detail trace_id=%s: %r", where, trace_id, exc)
    else:
        logger.error("safe_http_error%s trace_id=%s", where, trace_id)
    return HTTPException(
        status_code=status_code,
        detail=f"{public_detail} (trace_id={trace_id})",
    )
