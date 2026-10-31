# app/database/session.py
"""
数据库会话管理

性能要点：
1. echo=False —— 原实现 echo=True 会把每条 SQL 及其参数打到控制台，
   在批量上传 / 轮询历史时会产生大量同步 I/O，是隐形的响应延迟来源；
2. 连接池按并发量显式配置，并开启 pool_pre_ping，
   避免拿到的连接已被数据库回收（首次查询要重连，表现为偶发变慢）；
3. 会话依赖中不再重复 close()，由 async with 负责释放。
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from app.core.config import settings
import logging

log = logging.getLogger(__name__)

# 1. 创建异步数据库引擎
# 这个引擎是 SQLAlchemy 与数据库沟通的核心接口
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    future=True,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
)

# 2. 创建异步会话工厂
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# 3. 定义一个函数，用于在应用启动时创建所有数据表
#
# 历史记录接口（前端轮询最频繁的接口）按 user_id 过滤、按 created_at 排序，
# 但模型里没有 user_id 的索引。下面的 DDL 是幂等的（IF NOT EXISTS），
# 只为已存在的表补上查询用的复合索引，数据量增长后能避免全表扫描 + 排序。
#
# 同时负责给老库补列：SQLModel 的 create_all 只建新表，不会给已存在的表加字段，
# 因此剑叶需要的 task_type 用 ALTER TABLE ... IF NOT EXISTS 幂等补齐，
# 并把历史记录统一标成 stem（加列时带 DEFAULT，老行会被自动填充）。
_SCHEMA_DDL = (
    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS task_type VARCHAR DEFAULT 'stem'",
    "ALTER TABLE upload_batches ADD COLUMN IF NOT EXISTS batch_name VARCHAR",
    "UPDATE analyses SET task_type = 'stem' WHERE task_type IS NULL",
    "CREATE INDEX IF NOT EXISTS ix_analyses_task_type ON analyses (task_type)",
    "CREATE INDEX IF NOT EXISTS ix_analyses_user_created_at ON analyses (user_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_analyses_user_status ON analyses (user_id, status)",
)


async def _ensure_schema_upgrades(conn):
    """补齐索引与新增字段（仅对 Postgres 生效，失败不影响启动）。"""
    if "postgresql" not in settings.DATABASE_URL:
        return
    for ddl in _SCHEMA_DDL:
        try:
            # PostgreSQL aborts the transaction after a failed statement.
            # Isolate optional upgrades so subsequent ones can still execute.
            async with conn.begin_nested():
                await conn.execute(text(ddl))
        except Exception as error:  # pragma: no cover - 升级失败不应阻断启动
            log.warning(f"数据库结构升级失败（忽略）: {ddl} -> {error}")


async def create_db_and_tables():
    # 注意：这里我们使用 'async with' 语法
    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all) # 如果需要，可以在启动时清空数据表
        await conn.run_sync(SQLModel.metadata.create_all)
        await _ensure_schema_upgrades(conn)

# 4. 异步数据库会话依赖（新增）
async def get_async_session():
    """
    获取异步数据库会话
    用于 FastAPI 依赖注入
    """
    async with AsyncSessionLocal() as session:
        yield session
