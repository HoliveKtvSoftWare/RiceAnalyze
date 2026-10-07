from datetime import datetime
import uuid
import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, Body, Form
import logging
from typing import List, Dict, Any, Optional
from sqlmodel import select
from fastapi import Request
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.core import fastapi_users
from app.models.user import UserTable
from app.core.config import settings
from app.services.analysis_service import run_full_analysis
from app.auth.db import get_async_session
from app.models.analysis import Analysis
from app.models.batch import UploadBatch
# 设置日志
log = logging.getLogger(__name__)

router = APIRouter()
current_active_user = fastapi_users.current_user(active=True)

def _validate_model(model_name: Optional[str]) -> None:
    """
    校验前端传的模型名是否在 .env 注册表中。
    注意：不在此处检查文件是否存在，文件缺失会在后台任务中被捕获，
    并记录为失败的分析历史，error_message 字段会写明失败原因。
    """
    available_models = settings.get_models_map()
    if model_name is not None and model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型 '{model_name}'，可用: {list(available_models.keys())}"
        )

@router.get("/models")
async def get_available_models() -> Dict[str, Any]:
    """
    返回后端可用的 YOLO 模型列表，供前端渲染模型选择器。
    """
    models = settings.get_available_models()
    default_name = settings.get_models_map().pop(list(settings.get_models_map().keys())[0]) if settings.get_models_map() else None
    return {
        "models": models,
        "default": models[0]["name"] if models else None
    }

@router.get("/history")
async def get_analysis_history(
    request: Request,
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
) -> List[Dict[str, Any]]:
    """
    获取当前登录用户的分析历史记录，包含可访问的文件 URL。
    """
    log.info(f"用户 {user.id} 请求分析历史记录...")
    statement = select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc())

    result = await db.execute(statement)
    history_records = result.scalars().all()

    log.info(f"为用户 {user.id} 找到 {len(history_records)} 条历史记录。")

    base_url = str(request.base_url)
    static_url_prefix = f"{base_url}static/"
    storage_base_path = settings.STORAGE_PATH.strip('.').strip('/').strip('\\') + '/'

    history_with_urls = []
    for record in history_records:
        def path_to_url(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            try:
                cleaned_path = path.replace('.\\', '').replace('./', '').replace('\\', '/')
                if cleaned_path.startswith(storage_base_path):
                    relative_path = cleaned_path[len(storage_base_path):]
                else:
                    log.warning(f"文件路径 '{cleaned_path}' 不包含预期的基础路径 '{storage_base_path}'，将尝试直接使用。")
                    relative_path = cleaned_path

                if relative_path.startswith('/'):
                     relative_path = relative_path[1:]

                final_url = f"{static_url_prefix}{relative_path}"
                log.debug(f"路径 '{path}' 转换为 URL: '{final_url}'")
                return final_url
            except Exception as e:
                 log.error(f"转换路径 '{path}' 为 URL 时出错: {e}")
                 return None

        original_filename = "未知文件名"
        if record.original_file_path:
             basename = os.path.basename(record.original_file_path)
             parts = basename.split('_', 1)
             if len(parts) == 2 and len(parts[0]) == 36:
                 original_filename = parts[1]
             else:
                 original_filename = basename

        history_with_urls.append({
            "analysisId": record.analysis_id,
            "batchId": str(record.batch_id) if record.batch_id else None,
            "batchIndex": record.batch_index,
            "createdAt": record.created_at,
            "status": record.status,
            "modelUsed": record.model_used,
            "errorMessage": record.error_message,
            "originalFilename": original_filename,
            "originalImageUrl": path_to_url(record.original_file_path),
            "annotatedImageUrl": path_to_url(record.annotated_image_path),
            "resultJsonUrl": path_to_url(record.result_json_path),
        })

    return history_with_urls

async def _save_single_file(file: UploadFile, originals_dir: str, index: int) -> tuple:
    """保存单个文件并返回 (saved_file_path, unique_filename_base)"""
    analysis_uuid = uuid.uuid4()
    clean_filename = os.path.basename(file.filename).replace('/', '_').replace('\\', '_')
    upload_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(clean_filename)
    unique_filename_base = f"{analysis_uuid}_{name}_{upload_time}{ext}"
    saved_file_path = os.path.join(originals_dir, unique_filename_base)

    log.info(f"正在保存文件 {index} 至: {saved_file_path} (原始名: {file.filename})")
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        log.error(f"保存文件 {file.filename} 失败: {e}")
        raise HTTPException(status_code=500, detail=f"无法保存文件 {file.filename}。")

    return saved_file_path, unique_filename_base, analysis_uuid


async def _create_batch(db: AsyncSession, user: UserTable, file_count: int) -> UploadBatch:
    """创建一个 UploadBatch 记录"""
    batch_uuid = uuid.uuid4()
    new_batch = UploadBatch(
        batch_id=batch_uuid,
        file_count=file_count,
        user_id=user.id
    )
    db.add(new_batch)
    await db.commit()
    await db.refresh(new_batch)
    log.info(f"创建批次成功: {new_batch.batch_id}, 文件数量: {file_count}")
    return new_batch


def _add_analysis_task(
    tasks: BackgroundTasks,
    analysis: Analysis,
    saved_file_path: str,
    user: UserTable,
    unique_filename_base: str,
    model_name: Optional[str]
) -> None:
    """提交后台分析任务"""
    try:
        tasks.add_task(
            run_full_analysis,
            analysis_id=str(analysis.analysis_id),
            original_file_path=saved_file_path,
            user_id=str(user.id),
            original_filename=unique_filename_base,
            model_name=model_name
        )
    except Exception as e:
        log.error(f"添加后台任务时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法启动后台分析任务。")


@router.post("/upload")
async def upload_adaptive(
    tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    自适应上传接口：同时支持单文件和多文件上传，根据前端传入的文件数量自动适配处理逻辑。

    Form 字段:
        files: 上传的图片文件列表（单文件时也用同样的字段名即可）
        model_name: 用户选择的模型名称，可选；不传则使用默认模型
    """
    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件")

    is_single = len(files) == 1
    log.info(
        f"收到上传请求: model_name={model_name}, 文件数={len(files)} "
        f"({'单文件模式' if is_single else '批量模式'})"
    )

    _validate_model(model_name)

    batch = await _create_batch(db, user, len(files))

    originals_dir = os.path.join(settings.STORAGE_PATH, "originals", str(user.id))
    os.makedirs(originals_dir, exist_ok=True)

    analyses_info = []

    for index, file in enumerate(files):
        saved_file_path, unique_filename_base, analysis_uuid = await _save_single_file(
            file, originals_dir, index + 1
        )

        new_analysis = Analysis(
            analysis_id=analysis_uuid,
            user_id=user.id,
            status="processing",
            original_file_path=saved_file_path,
            batch_id=batch.batch_id,
            batch_index=index,
            model_used=model_name if model_name else "default"
        )
        db.add(new_analysis)
        await db.commit()
        await db.refresh(new_analysis)

        _add_analysis_task(
            tasks, new_analysis, saved_file_path, user, unique_filename_base, model_name
        )

        analyses_info.append({
            "analysis_id": new_analysis.analysis_id,
            "original_filename": file.filename
        })

    default_model = model_name if model_name else "default"

    if is_single:
        return {
            "message": "文件已成功提交后台处理。",
            "analysis_id": analyses_info[0]["analysis_id"],
            "batch_id": batch.batch_id,
            "original_filename": analyses_info[0]["original_filename"],
            "model_used": default_model
        }

    return {
        "message": f"{len(files)} 个文件已成功提交后台处理。",
        "batch_id": batch.batch_id,
        "total_files": len(files),
        "model_used": default_model,
        "analyses": analyses_info
    }


@router.post("/upload/batch")
async def upload_images_batch(
    tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    user: UserTable = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    批量上传的兼容接口，内部委托给自适应上传接口。
    保留此路由是为了向后兼容已有前端代码。
    """
    return await upload_adaptive(tasks, files, model_name, user, db)

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