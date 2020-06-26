"""add starting pitcher flag

Revision ID: ca4b241e053d
Revises: 
Create Date: 2020-06-25 11:20:52.602732

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ca4b241e053d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('run', sa.Column('is_sp', sa.Boolean))


def downgrade():
    op.drop_column('run', 'is_sp')
