import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Text,
    DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.db.session import Base


class ImputationJob(Base):
    __tablename__ = "imputation_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(20), nullable=False, default="UPLOADED", index=True)
    original_filename = Column(String(255), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    input_path = Column(Text, nullable=False)
    output_path = Column(Text, nullable=True)
    model_type = Column(String(30), nullable=True)
    model_params = Column(JSONB, nullable=True, default=dict)
    column_config = Column(JSONB, nullable=True, default=list)
    progress = Column(Integer, nullable=False, default=0)
    stage = Column(String(50), nullable=True, default="")
    error_message = Column(Text, nullable=True)
    logs = Column(JSONB, nullable=True, default=list)
    imputation_preview = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ImputationJob(id={self.id}, status={self.status})>"
