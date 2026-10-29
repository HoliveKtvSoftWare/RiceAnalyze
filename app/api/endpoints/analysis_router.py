from datetime import datetime
import uuid
import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, Body, Form, Query
from fastapi.concurrency import run_in_threadpool
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from sqlmodel import select
from sqlalchemy import or_
from fastapi import Request
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.core import fastapi_users
from app.models.user import UserTable
from app.core.config import settings
from app.services.analysis_service import run_full_analysis
from app.services.image_utils import thumbnail_path_for
from app.auth.db import get_async_session
from app.models.analysis import Analysis
from app.models.batch import UploadBatch
# 设置日志
log = logging.getLogger(__name__)

router = APIRouter()
current_active_user = fastapi_users.current_user(active=True)

def _validate_model(model_name: Optional[str], task_type: str = "stem") -> None:
    """
    校验前端传的模型名是否在该分析类型的注册表中。

    茎秆用 YOLO_MODELS，剑叶用 LEAF_MODELS（.env 可配）。
    注意：不在此处检查文件是否存在，文件缺失会在后台任务中被捕获，
    并记录为失败的分析历史，error_message 字段会写明失败原因。
    """
    if (task_type or "stem").strip().lower() == "leaf":
        available_models = settings.get_leaf_models_map()
        registry_name = "LEAF_MODELS"
    else:
        available_models = settings.get_models_map()
        registry_name = "YOLO_MODELS"

    if model_name is not None and model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型 '{model_name}'（{registry_name}），可用: {list(available_models.keys())}"
        )

@router.get("/models")
async def get_available_models(group: Optional[str] = None) -> Dict[str, Any]:
    """
    返回后端可用的 YOLO 模型列表，供前端渲染模型选择器。

    ``group=leaf`` 时返回剑叶模型（LEAF_MODELS），其余情况返回茎秆模型。
    """
    if (group or "").strip().lower() == "leaf":
        models = settings.get_leaf_available_models()
    else:
        models = settings.get_available_models()
    return {
        "models": models,
        "default": models[0]["name"] if models else None,
        "group": "leaf" if (group or "").strip().lower() == "leaf" else "stem"
    }

def _normalize_storage_base(storage_path: str) -> str:
    """把 STORAGE_PATH 规范成不含 ./ 前缀、使用正斜杠的相对基础路径。"""
    base = storage_path.replace('\\', '/').strip()
    while base.startswith('./'):
        base = base[2:]
    return base.strip('/')


def _path_to_static_url(path: Optional[str], static_url_prefix: str) -> Optional[str]:
    """
    把数据库里保存的本地文件路径转换为 /static 下的可访问 URL。

    处理要点：
    1. Windows 反斜杠统一成正斜杠；
    2. 同时兼容相对路径（./app_storage/user_data/...）和绝对路径
       （D:/.../app_storage/user_data/...）——历史上两种都出现过；
    3. 对路径做 URL 编码：文件名里的空格、中文、# 等字符如果原样拼进 URL，
       浏览器/代理的处理并不一致，部分环境会直接请求失败，
       表现出来就是"图片加载不出来"。
    """
    if not path:
        return None

    try:
        cleaned = path.replace('\\', '/')
        while '//' in cleaned:
            cleaned = cleaned.replace('//', '/')
        while cleaned.startswith('./'):
            cleaned = cleaned[2:]

        base = _normalize_storage_base(settings.STORAGE_PATH)
        relative: Optional[str] = None

        if base and cleaned.startswith(base + '/'):
            relative = cleaned[len(base) + 1:]
        elif base and f'/{base}/' in cleaned:
            relative = cleaned.split(f'/{base}/', 1)[1]
        elif '/user_data/' in cleaned:
            relative = cleaned.split('/user_data/', 1)[1]

        if relative is None:
            log.warning(f"文件路径 '{cleaned}' 不包含预期的基础路径 '{base}'，将直接拼接到 /static 下。")
            relative = cleaned.lstrip('/')

        return f"{static_url_prefix}{quote(relative, safe='/')}"
    except Exception as e:
        log.error(f"转换路径 '{path}' 为 URL 时出错: {e}")
        return None


def _existing_static_url(path: Optional[str], static_url_prefix: str) -> Optional[str]:
    """
    仅当文件真实存在时才返回可访问 URL。

    有些历史记录只剩数据库行、文件已被清理（磁盘清理、手工删除、目录迁移等）。
    如果照常返回 URL，前端只会拿到一个 404，显示成"无法加载图片"，既误导用户也难排查。
    这里直接返回 None，前端就能明确提示"文件已丢失"。
    """
    if not path or not os.path.exists(path):
        return None
    return _path_to_static_url(path, static_url_prefix)


# TIFF 等浏览器不能直接显示的格式：上传时额外生成一张 JPEG 预览图，
# 只用于前端展示；原图仍保留原始格式供比例尺识别与指标计算。
_PREVIEW_SUFFIX = ".preview.jpg"


def _preview_path_for(original_path: Optional[str]) -> Optional[str]:
    if not original_path:
        return None
    return f"{os.path.splitext(original_path)[0]}{_PREVIEW_SUFFIX}"


def _make_image_preview(image_path: str, max_side: int = 1600) -> Optional[str]:
    """把任意 PIL 可解码的图片转成一张 JPEG 预览图（长边不超过 max_side）。"""
    try:
        from PIL import Image

        target = _preview_path_for(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            ratio = min(1.0, float(max_side) / float(max(width, height)))
            if ratio < 1.0:
                image = image.resize(
                    (max(1, int(width * ratio)), max(1, int(height * ratio))),
                    Image.LANCZOS,
                )
            image.save(target, "JPEG", quality=88)
        log.info(f"已生成图片预览: {target}")
        return target
    except Exception as e:
        log.warning(f"生成图片预览失败（不影响分析）: {image_path} -> {e}")
        return None


@router.get("/history")
async def get_analysis_history(
    request: Request,
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
    limit: Optional[int] = Query(
        None, ge=1, le=2000,
        description="最多返回多少条历史记录，缺省返回全部（向后兼容）"
    ),
    group: Optional[str] = Query(
        None, description="按分析大类过滤：stem=茎秆 / leaf=剑叶；不传返回全部"
    ),
) -> List[Dict[str, Any]]:
    """
    获取当前登录用户的分析历史记录，包含可访问的文件 URL。

    ``group`` 用于把茎秆页与剑叶页的数据完全隔开：
    剑叶页只取 ``task_type='leaf'`` 的记录；茎秆页只取 stem（含历史遗留的 NULL）。

    前端在处理中会持续轮询该接口，因此这里保持单次查询、
    批量取批次名（避免 N+1），并支持可选的 limit 限制返回量。
    """
    group_key = (group or "").strip().lower()
    log.info(f"用户 {user.id} 请求分析历史记录（group={group_key or 'all'}）...")

    conditions = [Analysis.user_id == user.id]
    if group_key == "leaf":
        conditions.append(Analysis.task_type == "leaf")
    elif group_key == "stem":
        conditions.append(or_(Analysis.task_type == "stem", Analysis.task_type.is_(None)))

    statement = select(Analysis).where(*conditions).order_by(Analysis.created_at.desc())
    if limit is not None:
        statement = statement.limit(limit)

    result = await db.execute(statement)
    history_records = result.scalars().all()

    log.info(f"为用户 {user.id} 找到 {len(history_records)} 条历史记录。")

    batch_ids = list({record.batch_id for record in history_records if record.batch_id is not None})
    batch_name_map: Dict[str, Optional[str]] = {}
    if batch_ids:
        batch_stmt = select(UploadBatch).where(UploadBatch.batch_id.in_(batch_ids))
        batch_result = await db.execute(batch_stmt)
        for batch in batch_result.scalars().all():
            batch_name_map[str(batch.batch_id)] = batch.batch_name

    base_url = str(request.base_url)
    static_url_prefix = f"{base_url}static/"

    history_with_urls = []
    for record in history_records:
        original_filename = "未知文件名"
        if record.original_file_path:
             basename = os.path.basename(record.original_file_path)
             parts = basename.split('_', 1)
             if len(parts) == 2 and len(parts[0]) == 36:
                 original_filename = parts[1]
             else:
                 original_filename = basename

        original_url = _existing_static_url(record.original_file_path, static_url_prefix)
        original_preview_url = _existing_static_url(
            _preview_path_for(record.original_file_path), static_url_prefix
        )
        annotated_url = _existing_static_url(record.annotated_image_path, static_url_prefix)
        thumbnail_source = (
            record.annotated_image_path
            or _preview_path_for(record.original_file_path)
            or record.original_file_path
        )
        thumbnail_url = _existing_static_url(
            thumbnail_path_for(thumbnail_source), static_url_prefix
        )
        json_url = _existing_static_url(record.result_json_path, static_url_prefix)
        files_missing = bool(
            (record.original_file_path and not original_url)
            or (record.annotated_image_path and not annotated_url)
        )

        history_with_urls.append({
            "analysisId": record.analysis_id,
            "batchId": str(record.batch_id) if record.batch_id else None,
            "batchName": batch_name_map.get(str(record.batch_id)) if record.batch_id else None,
            "batchIndex": record.batch_index,
            "createdAt": record.created_at,
            "status": record.status,
            "taskType": record.task_type or "stem",
            "modelUsed": record.model_used,
            "errorMessage": record.error_message,
            "originalFilename": original_filename,
            "originalImageUrl": original_url,
            "originalPreviewUrl": original_preview_url,
            "annotatedImageUrl": annotated_url,
            "thumbnailUrl": thumbnail_url,
            "resultJsonUrl": json_url,
            "filesMissing": files_missing,
        })

    return history_with_urls

def _extract_folder_name(files: List[UploadFile]) -> Optional[str]:
    """
    从多个上传文件中提取公共顶层文件夹名。

    当前端使用 webkitdirectory 选择整个文件夹时，浏览器会为每个文件
    携带相对路径（如 "水稻品种A/IMG_001.jpg"）。此函数提取路径的第一个
    部分作为批次名。如果所有文件没有共同的顶层目录，则返回 None。
    """
    top_dirs = set()
    for f in files:
        normalized = f.filename.replace('\\', '/')
        if '/' in normalized:
            top_dirs.add(normalized.split('/')[0])
    if len(top_dirs) == 1:
        return top_dirs.pop()
    return None


# 落盘文件名（uuid_原文件名_时间戳.ext）的长度预算。
# Windows 的默认路径上限是 260 字符，而存储目录本身已经很长
# （项目路径 + app_storage/user_data + 用户 uuid + 批次 uuid ≈ 145 字符），
# 原文件名过长时 open() 会直接报 ENOENT（“未能找到路径的一部分”）。
# 这里把文件名主体限制在 96 字符内，保证输出 JSON / 标注图不会超限。
_MAX_OUTPUT_BASENAME = 96
_MIN_NAME_BUDGET = 12


def _build_unique_filename_base(clean_filename: str, analysis_uuid, upload_time: str) -> tuple:
    """拼出 `{uuid}_{原文件名}_{时间戳}{扩展名}`，并按 Windows 路径长度预算截断原文件名。"""
    name, ext = os.path.splitext(clean_filename)
    prefix = f"{analysis_uuid}_"
    suffix = f"_{upload_time}{ext}"
    name_budget = _MAX_OUTPUT_BASENAME - len(prefix) - len(suffix)
    if len(name) > name_budget:
        # 截断时保留开头（品种/编号一般在前），并去掉末尾可能残留的空格/点
        name = name[:max(_MIN_NAME_BUDGET, name_budget)].rstrip(' .')
    return f"{prefix}{name}{suffix}", ext


async def _save_single_file(file: UploadFile, originals_dir: str, index: int) -> tuple:
    """保存单个文件并返回 (saved_file_path, unique_filename_base)"""
    analysis_uuid = uuid.uuid4()
    clean_filename = os.path.basename(file.filename).replace('/', '_').replace('\\', '_')
    upload_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename_base, _ext = _build_unique_filename_base(clean_filename, analysis_uuid, upload_time)
    saved_file_path = os.path.join(originals_dir, unique_filename_base)

    log.info(f"正在保存文件 {index} 至: {saved_file_path} (原始名: {file.filename})")
    try:
        # 磁盘写入是阻塞 I/O，放到线程池执行，避免卡住事件循环（其它请求会一起变慢）
        def _write():
            with open(saved_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer, length=1024 * 1024)

        await run_in_threadpool(_write)
    except Exception as e:
        log.error(f"保存文件 {file.filename} 失败: {e}")
        raise HTTPException(status_code=500, detail=f"无法保存文件 {file.filename}。")

    # TIFF 等浏览器不能直接显示的格式，额外生成一张 JPEG 预览（失败不影响分析）
    if os.path.splitext(saved_file_path)[1].lower() in (".tif", ".tiff"):
        await run_in_threadpool(_make_image_preview, saved_file_path)

    return saved_file_path, unique_filename_base, analysis_uuid


async def _create_batch(db: AsyncSession, user: UserTable, file_count: int, batch_name: Optional[str] = None) -> UploadBatch:
    """创建一个 UploadBatch 记录"""
    batch_uuid = uuid.uuid4()
    new_batch = UploadBatch(
        batch_id=batch_uuid,
        batch_name=batch_name,
        file_count=file_count,
        user_id=user.id
    )
    db.add(new_batch)
    await db.commit()
    log.info(f"创建批次成功: {new_batch.batch_id}, 文件数量: {file_count}, 批次名: {batch_name}")
    return new_batch


def _add_analysis_task(
    tasks: BackgroundTasks,
    analysis: Analysis,
    saved_file_path: str,
    user: UserTable,
    unique_filename_base: str,
    model_name: Optional[str],
    batch_id: Optional[str] = None,
    task_type: str = "stem"
) -> None:
    """提交后台分析任务"""
    try:
        tasks.add_task(
            run_full_analysis,
            analysis_id=str(analysis.analysis_id),
            original_file_path=saved_file_path,
            user_id=str(user.id),
            original_filename=unique_filename_base,
            model_name=model_name,
            batch_id=batch_id,
            task_type=task_type
        )
    except Exception as e:
        log.error(f"添加后台任务时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法启动后台分析任务。")


@router.post("/upload")
async def upload_adaptive(
    tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    task_type: str = Form("stem"),
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    自适应上传接口：同时支持单文件和多文件上传，根据前端传入的文件数量自动适配处理逻辑。

    Form 字段:
        files: 上传的图片文件列表（单文件时也用同样的字段名即可）
        model_name: 用户选择的模型名称，可选；不传则使用默认模型
        task_type: 分析类型，``stem``（茎秆截面，默认）/ ``leaf``（剑叶）
    """
    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件")

    task_type = (task_type or "stem").strip().lower()
    if task_type not in ("stem", "leaf"):
        raise HTTPException(status_code=400, detail=f"未知分析类型 '{task_type}'，只支持 stem / leaf")

    # 剑叶模型名归一化：前端的下拉值可能是旧 key（如 leaf-default）、
    # 权重文件名（leaf-our.pt）或大小写不一致的写法，统一映射回 LEAF_MODELS 的 key，
    # 避免改过 .env 之后旧页面一上传就 400。
    if task_type == "leaf":
        resolved_model = settings.resolve_leaf_model_key(model_name)
        if model_name and resolved_model != model_name:
            log.info(f"剑叶模型名归一化: '{model_name}' -> '{resolved_model}'")
        model_name = resolved_model

    is_single = len(files) == 1
    log.info(
        f"收到上传请求: task_type={task_type}, model_name={model_name}, 文件数={len(files)} "
        f"({'单文件模式' if is_single else '批量模式'})"
    )

    _validate_model(model_name, task_type)

    batch_name = _extract_folder_name(files)
    log.info(f"提取到的批次名(顶层文件夹): {batch_name}")

    batch = await _create_batch(db, user, len(files), batch_name)

    originals_dir = os.path.join(settings.STORAGE_PATH, "originals", str(user.id))
    os.makedirs(originals_dir, exist_ok=True)

    analyses_info = []
    new_analyses = []

    for index, file in enumerate(files):
        saved_file_path, unique_filename_base, analysis_uuid = await _save_single_file(
            file, originals_dir, index + 1
        )

        new_analyses.append(Analysis(
            analysis_id=analysis_uuid,
            user_id=user.id,
            status="processing",
            task_type=task_type,
            original_file_path=saved_file_path,
            batch_id=batch.batch_id,
            batch_index=index,
            model_used=model_name if model_name else "default"
        ))

        analyses_info.append({
            "analysis_id": analysis_uuid,
            "original_filename": file.filename,
            "saved_file_path": saved_file_path,
            "unique_filename_base": unique_filename_base,
        })

    # 一次性提交：批量上传 N 个文件时，避免 N 次事务往返（每个文件一次 commit + refresh）
    if new_analyses:
        db.add_all(new_analyses)
        await db.commit()

    for analysis, info in zip(new_analyses, analyses_info):
        _add_analysis_task(
            tasks, analysis, info["saved_file_path"], user,
            info["unique_filename_base"], model_name,
            batch_id=str(batch.batch_id),
            task_type=task_type
        )

    default_model = model_name if model_name else "default"

    if is_single:
        return {
            "message": "文件已成功提交后台处理。",
            "analysis_id": analyses_info[0]["analysis_id"],
            "batch_id": batch.batch_id,
            "batch_name": batch.batch_name,
            "task_type": task_type,
            "original_filename": analyses_info[0]["original_filename"],
            "model_used": default_model
        }

    return {
        "message": f"{len(files)} 个文件已成功提交后台处理。",
        "batch_id": batch.batch_id,
        "batch_name": batch.batch_name,
        "total_files": len(files),
        "task_type": task_type,
        "model_used": default_model,
        "analyses": [
            {
                "analysis_id": info["analysis_id"],
                "original_filename": info["original_filename"],
            }
            for info in analyses_info
        ]
    }


@router.post("/upload/batch")
async def upload_images_batch(
    tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    task_type: str = Form("stem"),
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    批量上传的兼容接口，内部委托给自适应上传接口。
    保留此路由是为了向后兼容已有前端代码。
    """
    return await upload_adaptive(tasks, files, model_name, task_type, user, db)

@router.get("/metrics/{analysis_id}")
async def get_analysis_metrics(
    analysis_id: str,
    unit: str = Query("μm", description="长度单位：µm / mm / cm"),
    scale: float = Query(500.0, gt=0, description="未识别到图内比例尺时的回退像素当量（µm/px）"),
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """返回某条已完成记录的指标明细（剑叶/茎秆按记录自身的类型自动选择口径）。

    剑叶页面的“指标卡片”用它渲染：面积、周长、各类维管束计数等。
    几何计算 + 比例尺识别需要解码原图，因此放到线程池执行。
    """
    from app.services.stem_excel import excel_service
    from app.services.leaf_excel import leaf_excel_service

    statement = select(Analysis).where(
        Analysis.analysis_id == analysis_id,
        Analysis.user_id == user.id
    )
    result = await db.execute(statement)
    analysis = result.scalars().first()

    if not analysis:
        raise HTTPException(status_code=404, detail="分析记录不存在或无权查看")
    if analysis.status != "completed":
        raise HTTPException(status_code=400, detail=f"记录尚未完成（当前状态: {analysis.status}）")

    task_type = (analysis.task_type or "stem").strip().lower()
    service = leaf_excel_service if task_type == "leaf" else excel_service

    try:
        metrics = await run_in_threadpool(
            service.load_analysis_data_from_record, analysis, unit, scale
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"计算指标失败 {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"计算指标失败: {str(e)}")

    return {
        "analysisId": str(analysis.analysis_id),
        "taskType": task_type,
        "unit": unit,
        "metrics": metrics,
    }


@router.delete("/delete/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    删除指定的分析记录
    """
    try:
        log.info(f"用户 {user.id} 请求删除分析记录: {analysis_id}")

        # 查找分析记录
        statement = select(Analysis).where(
            Analysis.analysis_id == analysis_id,
            Analysis.user_id == user.id
        )
        result = await db.execute(statement)
        analysis = result.scalars().first()

        if not analysis:
            log.warning(f"分析记录 {analysis_id} 不存在或不属于用户 {user.id}")
            raise HTTPException(status_code=404, detail="分析记录不存在或无权操作")

        files_to_delete = [
            analysis.original_file_path,
            analysis.annotated_image_path,
            thumbnail_path_for(analysis.annotated_image_path),
            analysis.result_json_path
        ]

        for file_path in files_to_delete:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    log.info(f"已删除文件: {file_path}")
                except Exception as e:
                    log.error(f"删除文件 {file_path} 时出错: {e}")

        for file_path in files_to_delete:
            if file_path:
                parent_dir = os.path.dirname(file_path)
                if parent_dir and os.path.isdir(parent_dir):
                    try:
                        os.rmdir(parent_dir)
                        log.info(f"已删除空目录: {parent_dir}")
                    except OSError:
                        pass

        await db.delete(analysis)
        await db.commit()
        log.info(f"分析记录 {analysis_id} 已成功删除")

        return {"message": "分析记录已成功删除"}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"删除分析记录时发生错误: {e}")
        raise HTTPException(status_code=500, detail="删除分析记录时发生错误")


@router.post("/delete/batch")
async def delete_analyses_batch(
    request_data: Dict[str, List[str]] = Body(...),
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    批量删除分析记录
    请求体: { "analysisIds": ["id1", "id2", "id3"] }
    """
    try:
        analysis_ids = request_data.get("analysisIds", [])
        if not analysis_ids:
            raise HTTPException(status_code=400, detail="请传入 analysisIds 列表")

        log.info(f"用户 {user.id} 请求批量删除 {len(analysis_ids)} 条分析记录")

        statement = select(Analysis).where(
            Analysis.analysis_id.in_(analysis_ids),
            Analysis.user_id == user.id
        )
        result = await db.execute(statement)
        analyses = result.scalars().all()

        if not analyses:
            raise HTTPException(status_code=404, detail="没有找到可删除的分析记录")

        deleted_count = 0
        skipped_ids = []

        for analysis in analyses:
            try:
                files_to_delete = [
                    analysis.original_file_path,
                    analysis.annotated_image_path,
                    thumbnail_path_for(analysis.annotated_image_path),
                    analysis.result_json_path
                ]
                for file_path in files_to_delete:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            log.error(f"删除文件 {file_path} 时出错: {e}")

                for file_path in files_to_delete:
                    if file_path:
                        parent_dir = os.path.dirname(file_path)
                        if parent_dir and os.path.isdir(parent_dir):
                            try:
                                os.rmdir(parent_dir)
                                log.info(f"已删除空目录: {parent_dir}")
                            except OSError:
                                pass

                await db.delete(analysis)
                deleted_count += 1
            except Exception as e:
                log.error(f"删除分析记录 {analysis.analysis_id} 时出错: {e}")
                skipped_ids.append(str(analysis.analysis_id))

        await db.commit()
        log.info(f"批量删除完成: 成功 {deleted_count} 条, 跳过 {len(skipped_ids)} 条")

        return {
            "message": f"成功删除 {deleted_count} 条记录",
            "deleted_count": deleted_count,
            "total_requested": len(analysis_ids),
            "skipped_ids": skipped_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"批量删除时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")