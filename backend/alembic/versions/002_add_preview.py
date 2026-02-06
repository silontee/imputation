"""Add imputation_preview column

Revision ID: 002
Revises: 001_initial
Create Date: 2024-01-05
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '002_add_preview'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('imputation_jobs', sa.Column('imputation_preview', JSONB, nullable=True, default=dict))


def downgrade():
    op.drop_column('imputation_jobs', 'imputation_preview')
