# 批次信息数据模型

import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class UploadBatch(SQLModel, table=True):
    """
    批次信息表 - 用于记录批量上传分析的批次信息

    前端将历史修改为一次上传分析作为一次历史记录后，
    此表用于跟踪每次批量上传的整体信息。
    """
    __tablename__ = "upload_batches"

    # --- 核心字段 ---
    id: int = Field(default=None, primary_key=True, index=True)
    batch_id: uuid.UUID = Field(default_factory=uuid.uuid4, unique=True, index=True, description="批次唯一标识")
    batch_name: Optional[str] = Field(default=None, description="批次名称，通常为用户上传的顶层文件夹名")
    file_count: int = Field(default=0, description="批次中的文件数量")
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False, description="创建时间")

    # --- 关联用户 (外键) ---
    user_id: uuid.UUID = Field(foreign_key="users.id", description="所属用户ID")