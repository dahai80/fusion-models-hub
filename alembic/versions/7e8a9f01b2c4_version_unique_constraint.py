"""version_unique_constraint

Revision ID: 7e8a9f01b2c4
Revises: 0f2330f0ac47
Create Date: 2026-08-25 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '7e8a9f01b2c4'
down_revision: Union[str, Sequence[str], None] = '0f2330f0ac47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add uq_model_version (model_id, version).

    P1-D: two concurrent uploads of the same (model_id, version) created two
    rows (TOCTOU in create_version). The ORM now declares this unique; this
    migration adds the matching DB constraint. Before adding it, collapse any
    pre-existing duplicates so the constraint does not fail on a brownfield DB
    (keep the earliest created_at row, delete the rest). SQLite and PG both
    support the raw DELETE + create_unique_constraint used here.
    """
    bind = op.get_bind()
    # Collapse duplicates: for each (model_id, version) with >1 row, keep the
    # one with the smallest created_at (or smallest id as a stable tiebreaker)
    # and delete the rest. Pure SQL, dialect-agnostic.
    bind.execute(sa.text("""
        DELETE FROM model_versions
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_id, version
                           ORDER BY created_at ASC NULLS FIRST, id ASC
                       ) AS rn
                FROM model_versions
            ) ranked
            WHERE ranked.rn = 1
        )
    """))
    # batch_alter_table so SQLite (no ALTER CONSTRAINT) uses copy-and-move;
    # PG adds the constraint inline. Both produce uq_model_version.
    with op.batch_alter_table('model_versions', schema=None) as batch_op:
        batch_op.create_unique_constraint('uq_model_version', ['model_id', 'version'])


def downgrade() -> None:
    with op.batch_alter_table('model_versions', schema=None) as batch_op:
        batch_op.drop_constraint('uq_model_version', type_='unique')

