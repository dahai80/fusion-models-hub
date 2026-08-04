"""add model_status to model

Revision ID: c7b2f3a91d04
Revises: 4a71a09a8f21
Create Date: 2026-08-04 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c7b2f3a91d04"
down_revision: str | None = "4a71a09a8f21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column(
            "model_status",
            sa.Enum("DRAFT", "PUBLISHED", "DEPRECATED", name="modelstatus"),
            nullable=False,
            server_default="draft",
        ),
    )


def downgrade() -> None:
    op.drop_column("models", "model_status")
