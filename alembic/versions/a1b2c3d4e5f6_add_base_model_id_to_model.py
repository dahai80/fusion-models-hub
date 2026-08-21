"""add base_model_id to model

Revision ID: a1b2c3d4e5f6
Revises: c7b2f3a91d04
Create Date: 2026-08-21 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "c7b2f3a91d04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column(
            "base_model_id",
            sa.String(16),
            sa.ForeignKey("models.id"),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("models", "base_model_id")
