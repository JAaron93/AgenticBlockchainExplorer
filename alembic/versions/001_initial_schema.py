"""Initial database schema.

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-11

Creates the initial database schema with tables:
- users: Authenticated user records
- agent_runs: Data collection run tracking
- run_results: Collection results and summaries
- audit_logs: User action audit trail
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create run_status_enum type
    run_status_enum = postgresql.ENUM(
        'pending', 'running', 'completed', 'failed',
        name='run_status_enum',
        create_type=True
    )
    run_status_enum.create(op.get_bind(), checkfirst=True)
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('user_id', sa.String(255), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    
    # Create agent_runs table
    op.create_table(
        'agent_runs',
        sa.Column('run_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), sa.ForeignKey('users.user_id', ondelete='RESTRICT'), nullable=False),
        sa.Column(
            'status',
            sa.Enum('pending', 'running', 'completed', 'failed', name='run_status_enum', create_type=False),
            nullable=False,
            server_default='pending'
        ),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('config', postgresql.JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('progress_message', sa.String(500), nullable=True),
    )
    op.create_index('ix_agent_runs_user_id', 'agent_runs', ['user_id'])
    op.create_index('ix_agent_runs_status', 'agent_runs', ['status'])
    op.create_index('ix_agent_runs_started_at', 'agent_runs', ['started_at'])
    
    # Create run_results table
    op.create_table(
        'run_results',
        sa.Column('result_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'run_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('agent_runs.run_id', ondelete='CASCADE'),
            nullable=False,
            unique=True
        ),
        sa.Column('total_records', sa.Integer(), nullable=True),
        sa.Column('explorers_queried', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('output_file_path', sa.Text(), nullable=True),
        sa.Column('summary', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_run_results_run_id', 'run_results', ['run_id'])
    op.create_index('ix_run_results_created_at', 'run_results', ['created_at'])
    
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('log_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), sa.ForeignKey('users.user_id', ondelete='SET NULL'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('ip_address', postgresql.INET(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(), nullable=True),
    )
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('ix_audit_logs_resource', 'audit_logs', ['resource_type', 'resource_id'])


def downgrade() -> None:
    """Drop all tables and types."""
    
    # Drop tables in reverse order of creation (respecting foreign keys)
    op.drop_index('ix_audit_logs_resource', table_name='audit_logs')
    op.drop_index('ix_audit_logs_timestamp', table_name='audit_logs')
    op.drop_index('ix_audit_logs_action', table_name='audit_logs')
    op.drop_index('ix_audit_logs_user_id', table_name='audit_logs')
    op.drop_table('audit_logs')
    
    op.drop_index('ix_run_results_created_at', table_name='run_results')
    op.drop_index('ix_run_results_run_id', table_name='run_results')
    op.drop_table('run_results')
    
    op.drop_index('ix_agent_runs_started_at', table_name='agent_runs')
    op.drop_index('ix_agent_runs_status', table_name='agent_runs')
    op.drop_index('ix_agent_runs_user_id', table_name='agent_runs')
    op.drop_table('agent_runs')
    
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
    
    # Drop the enum type
    run_status_enum = postgresql.ENUM(
        'pending', 'running', 'completed', 'failed',
        name='run_status_enum'
    )
    run_status_enum.drop(op.get_bind(), checkfirst=True)
