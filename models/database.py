"""SQLAlchemy models for database tables.

Defines the database schema for users, agent_runs, run_results, and audit_logs tables.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Enum,
    ARRAY,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class RunStatus(str, PyEnum):
    """Status of an agent run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    """User model for storing authenticated users."""
    
    __tablename__ = "users"
    
    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    
    # Relationships
    agent_runs: Mapped[List["AgentRun"]] = relationship(
        "AgentRun", back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog", back_populates="user", cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("ix_users_email", "email"),
    )
    
    def __repr__(self) -> str:
        return f"<User(user_id='{self.user_id}')>"


class AgentRun(Base):
    """Agent run model for tracking data collection runs."""
    
    __tablename__ = "agent_runs"
    
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.user_id"), nullable=False
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status_enum"),
        nullable=False,
        default=RunStatus.PENDING
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    progress: Mapped[Optional[float]] = mapped_column(nullable=True)
    progress_message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="agent_runs")
    run_result: Mapped[Optional["RunResult"]] = relationship(
        "RunResult", back_populates="agent_run", uselist=False, cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("ix_agent_runs_user_id", "user_id"),
        Index("ix_agent_runs_status", "status"),
        Index("ix_agent_runs_started_at", "started_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AgentRun(run_id='{self.run_id}', status='{self.status}')>"


class RunResult(Base):
    """Run result model for storing collection results."""
    
    __tablename__ = "run_results"
    
    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.run_id"), nullable=False, unique=True
    )
    total_records: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    explorers_queried: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(Text), nullable=True
    )
    output_file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    
    # Relationships
    agent_run: Mapped["AgentRun"] = relationship("AgentRun", back_populates="run_result")
    
    __table_args__ = (
        Index("ix_run_results_run_id", "run_id"),
        Index("ix_run_results_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<RunResult(result_id='{self.result_id}', total_records={self.total_records})>"


class AuditLog(Base):
    """Audit log model for tracking user actions."""
    
    __tablename__ = "audit_logs"
    
    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.user_id"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
    )
    
    def __repr__(self) -> str:
        return f"<AuditLog(log_id='{self.log_id}', action='{self.action}')>"
