import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field "model_name" has conflict with protected namespace',
    category=UserWarning,
)

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import os
from app.auth.backend import auth_backend

from app.database.session import create_db_and_tables
from app.auth.schemas import UserCreate, UserRead, UserUpdate
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.endpoints import analysis_router as analysis_router
from app.api.endpoints import export_excel_router as excel_router
from app.api.endpoints import export_json_router
from app.auth.core import fastapi_users

log = logging.getLogger(__name__)

# 有 orjson 时用更快的 JSON 序列化器（requirements.txt.txt 已包含），缺失时自动回退
try:
    import orjson  # noqa: F401

    from fastapi.responses import ORJSONResponse as _DefaultResponse
except ImportError:  # pragma: no cover
    from fastapi.responses import JSONResponse as _DefaultResponse


class SelectiveGZipMiddleware(GZipMiddleware):
    """
    只对动态接口做 gzip 压缩。

    /static 下是 JPEG 图片和已下载的 xlsx（本身已压缩），再 gzip 一遍
    只会白白消耗 CPU 并增加延迟，因此在请求进入压缩层前就直接放行。
    """

    def __init__(self, app, minimum_size=1024, exclude_prefixes=("/static",)):
        super().__init__(app, minimum_size=minimum_size)
        self.exclude_prefixes = tuple(exclude_prefixes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path", "").startswith(
            self.exclude_prefixes
        ):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class CachedStaticFiles(StaticFiles):
    """
    为静态文件加上强缓存头。

    分析产物（原图 / 标注图 / JSON）的文件名里带有 UUID + 时间戳，
    内容不会原地变化，因此可以放心让浏览器长期缓存，
    再次查看同一条记录时图片可以秒开。
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers.setdefault(
                "cache-control", "public, max-age=31536000, immutable"
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("应用启动...")
    await create_db_and_tables()

    # 后台预热 YOLO 模型：让第一次分析请求不用再等权重加载
    try:
        from app.services.yolo_inference import preload_models

        preload_models(list(settings.get_models_map().values()))
    except Exception as preload_error:  # pragma: no cover - 预热失败不影响服务
        log.warning(f"模型预热启动失败（忽略）: {preload_error}")

    yield
    print("应用关闭...")


app = FastAPI(lifespan=lifespan, default_response_class=_DefaultResponse)

# 响应压缩：历史记录 / 导出统计等 JSON 体积可减少 70% 以上
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024)

# ORS配置
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# 挂载认证路由
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/api/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/api/auth",
    tags=["auth"],
)

# 挂载用户管理路由
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/api/users",
    tags=["users"],
)

# 挂载分析路由
app.include_router(analysis_router.router, prefix="/api/analysis", tags=["Analysis"])

# 挂载Excel导出路由
app.include_router(excel_router.router, prefix="/api/excel", tags=["Excel"])

# 🆕 挂载JSON导出路由（新增这行）
app.include_router(export_json_router.router, tags=["Export"])

# 配置静态文件
static_dir = settings.STORAGE_PATH
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", CachedStaticFiles(directory=static_dir), name="static")

# 根路径
@app.get("/")
def read_root():
    return {"Hello": "World"}