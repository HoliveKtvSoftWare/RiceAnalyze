# 分析记录数据模型

import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel
from pydantic import ConfigDict


class Analysis(SQLModel, table=True):
    model_config = ConfigDict(protected_namespaces=())
    __tablename__ = "analyses" # 定义数据库表名

    # --- 核心字段 ---
    analysis_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    status: str = Field(default="pending", index=True) # 任务状态: pending, processing, queued, completed, failed
    # 分析类型：stem=茎秆截面 / leaf=剑叶。历史记录、模型列表、导出都按它分流，
    # 保证两个页面的数据互不串台。
    task_type: str = Field(default="stem", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # --- 文件路径字段 ---
    original_file_path: Optional[str] = Field(default=None)
    annotated_image_path: Optional[str] = Field(default=None) # 推理后带掩膜的图片
    result_json_path: Optional[str] = Field(default=None) # LabelMe JSON 文件
    excel_report_path: Optional[str] = Field(default=None) # Excel 报告文件

    # --- 关联用户 (外键) ---
    # 这个字段将这条分析记录与 users 表中的一个用户关联起来
    user_id: uuid.UUID = Field(foreign_key="users.id")
    
    # --- 批量上传支持 ---
    batch_id: Optional[uuid.UUID] = Field(default=None, index=True) # 批次ID，用于批量上传
    batch_index: Optional[int] = Field(default=None) # 在批次中的序号

    # --- 模型选择 ---
    model_used: Optional[str] = Field(default=None) # 本次分析使用的模型名称

    # --- 失败原因 ---
    error_message: Optional[str] = Field(default=None) # 后台任务失败时的错误描述