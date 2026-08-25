"""add download integrity columns (expected_sha256, file_hash)

Revision ID: b3c4d5e6f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-08-24 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b3c4d5e6f7a8"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "download_tasks",
        sa.Column("expected_sha256", sa.String(64), nullable=False, server_default=""),
    )
    op.add_column(
        "download_tasks",
        sa.Column("file_hash", sa.String(64), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("download_tasks", "file_hash")
    op.drop_column("download_tasks", "expected_sha256")
