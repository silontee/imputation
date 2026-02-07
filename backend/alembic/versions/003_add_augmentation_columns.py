"""Add augmentation columns to imputation_jobs

Revision ID: 003
Revises: 002_add_preview
Create Date: 2026-02-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002_add_preview'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('imputation_jobs', sa.Column('augment_status', sa.String(20), nullable=True, server_default='IDLE'))
    op.add_column('imputation_jobs', sa.Column('augment_params', JSONB, nullable=True))
    op.add_column('imputation_jobs', sa.Column('augment_output_path', sa.Text, nullable=True))
    op.add_column('imputation_jobs', sa.Column('augment_progress', sa.Integer, nullable=True, server_default='0'))
    op.add_column('imputation_jobs', sa.Column('augment_stage', sa.String(50), nullable=True, server_default=''))
    op.add_column('imputation_jobs', sa.Column('augment_error', sa.Text, nullable=True))
    op.add_column('imputation_jobs', sa.Column('augment_preview', JSONB, nullable=True))


def downgrade():
    op.drop_column('imputation_jobs', 'augment_preview')
    op.drop_column('imputation_jobs', 'augment_error')
    op.drop_column('imputation_jobs', 'augment_stage')
    op.drop_column('imputation_jobs', 'augment_progress')
    op.drop_column('imputation_jobs', 'augment_output_path')
    op.drop_column('imputation_jobs', 'augment_params')
    op.drop_column('imputation_jobs', 'augment_status')
