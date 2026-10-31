"""
Excel下载路由
"""

import uuid
import os
import re
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import logging
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.core import fastapi_users
from app.models.user import UserTable
from app.auth.db import get_async_session
from app.models.analysis import Analysis
from app.services.stem_excel import excel_service
from app.services.leaf_excel import leaf_excel_service
from app.core.task_groups import task_group, group_condition

EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _group_key(group: Optional[str]) -> str:
    """规范化分析大类：leaf=剑叶，其余一律按 stem（茎秆）处理。"""
    return task_group(group)


def _service_for(group: Optional[str]):
    """按分析大类返回对应的导出服务（列定义 + 指标口径都在服务里）。"""
    return leaf_excel_service if _group_key(group) == "leaf" else excel_service


def _extract_real_name(original_file_path: str) -> str:
    basename = os.path.basename(original_file_path)
    parts = basename.split('_', 1)
    if len(parts) == 2 and len(parts[0]) == 36:
        name_with_ext = parts[1]
    else:
        name_with_ext = basename
    name_no_ext = os.path.splitext(name_with_ext)[0]
    name_no_ext = re.sub(r'_\d{8}_\d{6}$', '', name_no_ext)
    return name_no_ext

def _build_cd(filename: str) -> str:
    ascii_name = filename.encode('ascii', errors='ignore').decode('ascii') or 'download'
    utf8_name = quote(filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"

def _file_to_streaming_response(file_path: str, filename: str, media_type: str) -> FileResponse:
    """
    用 FileResponse 直接流式发送文件。

    旧实现会把整个 xlsx 读进内存再包一层 BytesIO，既多占一份内存，
    又多一次全量拷贝；FileResponse 由 Starlette 在线程池中分块读取，
    首字节更快，也不受文件大小影响。
    """
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Content-Disposition": _build_cd(filename)},
    )
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter()
current_active_user = fastapi_users.current_user(active=True)

_export_tasks: Dict[str, Dict[str, Any]] = {}
_export_tasks_lock = threading.Lock()


class ExportRequest(BaseModel):
    selectedColumns: List[str]
    unit: str = "μm"
    scale: float = 500.0
    # stem=茎秆（默认） / leaf=剑叶：决定用哪套列定义与指标口径
    group: str = "stem"


class AsyncExportRequest(ExportRequest):
    asyncMode: bool = False


class BatchExportRequest(BaseModel):
    analysisIds: Optional[List[str]] = None
    allCompleted: bool = False
    selectedColumns: List[str]
    unit: str = "μm"
    scale: float = 500.0
    asyncMode: bool = False
    group: str = "stem"


@router.get("/columns")
async def get_export_columns(group: str = Query("stem", description="stem=茎秆 / leaf=剑叶")):
    service = _service_for(group)
    return {
        "available_columns": service.get_available_columns(),
        "group": _group_key(group),
        "message": "可导出的列配置"
    }


@router.post("/summary")
async def export_all_analysis_to_excel(
        request_data: AsyncExportRequest = Body(...),
        user: UserTable = Depends(current_active_user),
        db: AsyncSession = Depends(get_async_session),
        background_tasks: BackgroundTasks = None
):
    try:
        group = _group_key(request_data.group)
        statement = select(Analysis).where(
            Analysis.user_id == user.id,
            Analysis.status == "completed",
            group_condition(Analysis.task_type, group)
        ).order_by(Analysis.created_at.desc())

        result = await db.execute(statement)
        all_records = result.scalars().all()

        if not all_records:
            raise HTTPException(status_code=404, detail="没有找到已完成的分析记录")

        if request_data.asyncMode:
            task_id = _submit_async_export(
                records=all_records,
                selected_columns=request_data.selectedColumns,
                unit=request_data.unit,
                scale=request_data.scale,
                prefix="summary",
                total=len(all_records),
                user_id=str(user.id),
                background_tasks=background_tasks,
                group=group,
            )
            return {
                "success": True,
                "message": f"已提交后台导出任务，共 {len(all_records)} 条记录",
                "taskId": task_id,
                "pollUrl": f"/api/excel/tasks/{task_id}",
                "downloadUrl": f"/api/excel/download/{task_id}",
            }

        # 同步导出（pandas / openpyxl / 图片解码）必须放到线程池，
        # 否则在 async 端点里直接调用会阻塞整个事件循环，所有其它请求一起卡住。
        file_path, filename = await run_in_threadpool(
            _do_export,
            records=all_records,
            selected_columns=request_data.selectedColumns,
            unit=request_data.unit,
            scale=request_data.scale,
            prefix="summary",
            group=group,
        )

        return _file_to_streaming_response(file_path, filename, EXCEL_MEDIA_TYPE)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"导出所有分析记录时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@router.post("/batch")
async def export_batch_to_excel(
        request_data: BatchExportRequest = Body(...),
        user: UserTable = Depends(current_active_user),
        db: AsyncSession = Depends(get_async_session),
        background_tasks: BackgroundTasks = None
):
    try:
        group = _group_key(request_data.group)
        if request_data.allCompleted:
            statement = select(Analysis).where(
                Analysis.user_id == user.id,
                Analysis.status == "completed",
                group_condition(Analysis.task_type, group)
            ).order_by(Analysis.created_at.desc())
        elif request_data.analysisIds:
            statement = select(Analysis).where(
                Analysis.analysis_id.in_(request_data.analysisIds),
                Analysis.user_id == user.id,
                Analysis.status == "completed",
                group_condition(Analysis.task_type, group)
            )
        else:
            raise HTTPException(status_code=400, detail="请传入 analysisIds 或设置 allCompleted=true")

        result = await db.execute(statement)
        records = result.scalars().all()

        if not records:
            raise HTTPException(status_code=404, detail="没有找到有效的已完成分析记录")

        if request_data.asyncMode:
            task_id = _submit_async_export(
                records=records,
                selected_columns=request_data.selectedColumns,
                unit=request_data.unit,
                scale=request_data.scale,
                prefix="batch",
                total=len(records),
                user_id=str(user.id),
                background_tasks=background_tasks,
                group=group,
            )
            return {
                "success": True,
                "message": f"已提交后台导出任务，共 {len(records)} 条记录",
                "taskId": task_id,
                "pollUrl": f"/api/excel/tasks/{task_id}",
                "downloadUrl": f"/api/excel/download/{task_id}",
            }

        file_path, filename = await run_in_threadpool(
            _do_export,
            records=records,
            selected_columns=request_data.selectedColumns,
            unit=request_data.unit,
            scale=request_data.scale,
            prefix="batch",
            group=group,
        )

        return _file_to_streaming_response(file_path, filename, EXCEL_MEDIA_TYPE)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"批量导出Excel时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"批量导出失败: {str(e)}")


@router.post("/{analysis_id}")
async def export_analysis_to_excel(
        analysis_id: str,
        request_data: ExportRequest = Body(...),
        user: UserTable = Depends(current_active_user),
        db: AsyncSession = Depends(get_async_session)
):
    try:
        statement = select(Analysis).where(
            Analysis.analysis_id == analysis_id,
            Analysis.user_id == user.id
        )

        result = await db.execute(statement)
        analysis = result.scalar_one_or_none()

        if not analysis:
            raise HTTPException(status_code=404, detail="分析记录不存在")

        if analysis.status != "completed":
            raise HTTPException(status_code=400, detail="分析记录尚未完成，无法导出")

        # 以记录自身的分析类型为准，避免前端传错 group 导致口径串台
        group = _group_key(analysis.task_type or request_data.group)

        # 图片解码 / 几何计算 / Excel 生成同样是重 CPU 操作，放到线程池执行
        file_path, filename = await run_in_threadpool(
            _build_single_export,
            analysis,
            request_data.unit,
            request_data.scale,
            request_data.selectedColumns,
            group,
        )

        return _file_to_streaming_response(file_path, filename, EXCEL_MEDIA_TYPE)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"导出分析记录时发生错误 {e}")
        raise HTTPException(status_code=500, detail=f"导出分析记录失败: {str(e)}")


@router.get("/tasks/{task_id}")
async def get_export_task_status(task_id: str):
    with _export_tasks_lock:
        task = _export_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="导出任务不存在")

    response = {
        "taskId": task_id,
        "status": task["status"],
        "total": task["total"],
        "progress": task["progress"],
        "createdAt": task["created_at"],
    }
    if task["status"] == "completed":
        response["downloadUrl"] = f"/api/excel/download/{task_id}"
        response["filename"] = task.get("filename")
    elif task["status"] == "failed":
        response["error"] = task.get("error", "未知错误")
        response["message"] = task.get("message") or task.get("error")
    elif task["status"] == "cancelled":
        response["message"] = task.get("message", "任务已被用户取消")
    elif task["status"] == "cancelling":
        response["message"] = task.get("message", "任务正在停止...")
    return response


@router.post("/tasks/{task_id}/cancel")
async def cancel_export_task(
        task_id: str,
        user: UserTable = Depends(current_active_user),
):
    with _export_tasks_lock:
        task = _export_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.get("user_id") is not None and task["user_id"] != str(user.id):
            raise HTTPException(status_code=403, detail="Cannot cancel a task belonging to another user")

        if task["status"] in ("completed", "failed"):
            raise HTTPException(status_code=409, detail="Task is already completed or failed, cannot cancel")

        if task["status"] in ("cancelling", "cancelled"):
            return {
                "status": task["status"],
                "message": "任务已标记为取消，正在停止..." if task["status"] == "cancelling" else "任务已被取消",
            }

        task["status"] = "cancelling"
        task["message"] = "任务正在停止..."

    return {
        "status": "cancelling",
        "message": "任务已标记为取消，正在停止...",
    }


@router.get("/download/{task_id}")
async def download_export_task(task_id: str):
    with _export_tasks_lock:
        task = _export_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="导出任务不存在")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"导出尚未完成，当前状态: {task['status']}")

    file_path = task["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件已丢失或被清理")

    dl_name = task.get("filename") or os.path.basename(file_path)
    return _file_to_streaming_response(file_path, dl_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _submit_async_export(
        records: list,
        selected_columns: List[str],
        unit: str,
        scale: float,
        prefix: str,
        total: int,
        user_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
        group: str = "stem",
) -> str:
    task_id = str(uuid.uuid4())
    group = _group_key(group)

    record_info_list = [
        {
            "analysis_id": str(r.analysis_id),
            "original_file_path": r.original_file_path,
            "result_json_path": r.result_json_path,
        }
        for r in records
    ]

    with _export_tasks_lock:
        _export_tasks[task_id] = {
            "status": "processing",
            "total": total,
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "file_path": None,
            "filename": None,
            "error": None,
            "message": None,
            "user_id": user_id,
        }

    kwargs = dict(
        task_id=task_id,
        record_info_list=record_info_list,
        selected_columns=selected_columns,
        unit=unit,
        scale=scale,
        prefix=prefix,
        group=group,
    )

    if background_tasks is not None:
        background_tasks.add_task(_run_export_worker, **kwargs)
    else:
        thread = threading.Thread(target=_run_export_worker, kwargs=kwargs, daemon=True)
        thread.start()

    return task_id


def _is_task_cancelled(task_id: str) -> bool:
    with _export_tasks_lock:
        task = _export_tasks.get(task_id)
        if not task:
            return False
        return task["status"] in ("cancelling", "cancelled")


def _run_export_worker(
        task_id: str,
        record_info_list: list,
        selected_columns: List[str],
        unit: str,
        scale: float,
        prefix: str,
        group: str = "stem",
):
    try:
        from app.models.analysis import Analysis as _Analysis

        service = _service_for(group)
        group = _group_key(group)
        label = "剑叶" if group == "leaf" else "茎秆"

        valid_data = []
        for idx, info in enumerate(record_info_list):
            if _is_task_cancelled(task_id):
                log.info(f"异步导出任务 {task_id} 已被取消，停止处理")
                with _export_tasks_lock:
                    if task_id in _export_tasks:
                        _export_tasks[task_id]["status"] = "cancelled"
                        _export_tasks[task_id]["message"] = "任务已被用户取消"
                return

            record = _Analysis(
                analysis_id=uuid.UUID(info["analysis_id"]),
                original_file_path=info["original_file_path"],
                result_json_path=info["result_json_path"],
            )
            try:
                data = service.load_analysis_data_from_record(record, unit, scale)
                valid_data.append(data)
            except Exception as e:
                log.warning(f"异步导出跳过记录 {info['analysis_id']}: {e}")

            with _export_tasks_lock:
                if task_id in _export_tasks:
                    _export_tasks[task_id]["progress"] = idx + 1

        if _is_task_cancelled(task_id):
            log.info(f"异步导出任务 {task_id} 已被取消，停止处理")
            with _export_tasks_lock:
                if task_id in _export_tasks:
                    _export_tasks[task_id]["status"] = "cancelled"
                    _export_tasks[task_id]["message"] = "任务已被用户取消"
            return

        if not valid_data:
            with _export_tasks_lock:
                _export_tasks[task_id]["status"] = "failed"
                _export_tasks[task_id]["error"] = "没有成功加载任何分析数据"
            return

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{len(valid_data)}个文件_{now_str}_{unit}.xlsx"
        file_path = service.export_all_to_excel(valid_data, selected_columns, filename, unit)

        with _export_tasks_lock:
            if task_id in _export_tasks:
                if _export_tasks[task_id]["status"] in ("cancelling", "cancelled"):
                    _export_tasks[task_id]["status"] = "cancelled"
                    _export_tasks[task_id]["message"] = "任务已被用户取消"
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass
                    return
                _export_tasks[task_id]["status"] = "completed"
                _export_tasks[task_id]["file_path"] = file_path
                _export_tasks[task_id]["filename"] = filename
                _export_tasks[task_id]["progress"] = len(record_info_list)

        log.info(f"异步导出任务 {task_id} 完成: {file_path}（成功 {len(valid_data)}/{len(record_info_list)}）")

    except Exception as e:
        log.error(f"异步导出任务 {task_id} 失败: {e}")
        with _export_tasks_lock:
            if task_id in _export_tasks:
                if _export_tasks[task_id]["status"] in ("cancelling", "cancelled"):
                    _export_tasks[task_id]["status"] = "cancelled"
                    _export_tasks[task_id]["message"] = "任务已被用户取消"
                else:
                    _export_tasks[task_id]["status"] = "failed"
                    _export_tasks[task_id]["error"] = str(e)


def _build_single_export(analysis: Analysis, unit: str, scale: float, selected_columns: List[str],
                         group: str = "stem") -> tuple:
    """单条记录的 Excel 生成（同步函数，由线程池调用）。"""
    service = _service_for(group)
    label = "剑叶" if _group_key(group) == "leaf" else "茎秆"

    analysis_data = service.load_analysis_data_from_record(analysis, unit, scale)
    df = service.generate_excel_data(analysis_data, selected_columns, unit)

    real_name = _extract_real_name(analysis.original_file_path)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{real_name}_{now_str}_{unit}.xlsx"

    file_path = service.create_excel_file(df, filename)
    return file_path, filename


def _do_export(records: list, selected_columns: List[str], unit: str, scale: float,
               prefix: str = "export", group: str = "stem") -> tuple:
    service = _service_for(group)
    label = "剑叶" if _group_key(group) == "leaf" else "茎秆"

    valid_data = []
    for record in records:
        try:
            data = service.load_analysis_data_from_record(record, unit, scale)
            valid_data.append(data)
        except Exception as e:
            log.warning(f"导出中跳过记录 {record.analysis_id}: {e}")
            continue

    if not valid_data:
        raise HTTPException(status_code=404, detail="没有成功加载任何分析数据")

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{len(valid_data)}个文件_{now_str}_{unit}.xlsx"
    file_path = service.export_all_to_excel(valid_data, selected_columns, filename, unit)
    log.info(f"Excel 导出完成: {file_path}（成功 {len(valid_data)}/{len(records)}）")
    return file_path, filename