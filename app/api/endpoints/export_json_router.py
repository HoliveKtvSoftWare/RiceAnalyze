"""
导出API端点 - 异步版本
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, FileResponse
from sqlmodel import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
import os
import re
import json
import logging
import tempfile
import zipfile
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from urllib.parse import quote

from app.database.session import get_async_session
from app.models.analysis import Analysis
from app.auth.core import fastapi_users

log = logging.getLogger(__name__)

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

router = APIRouter(prefix="/api/export", tags=["Export"])


class BatchJsonExportRequest(BaseModel):
    analysisIds: Optional[List[str]] = None


class DataPackageItem(BaseModel):
    """一条被勾选的数据项：同一条记录会拆成 image / json 两项。"""

    analysisId: str
    kind: str


class DataPackageExportRequest(BaseModel):
    """数据中心「下载选中」请求体：只打包用户勾选的图片或 JSON。"""

    items: Optional[List[DataPackageItem]] = None
    group: Optional[str] = None


# ==================== 辅助服务函数 ====================

def build_content_disposition(filename):
    ascii_name = filename.encode('ascii', errors='ignore').decode('ascii') or 'download'
    utf8_name = quote(filename)
    return "attachment; filename={}; filename*=UTF-8''{}".format(ascii_name, utf8_name)


def get_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_json_response(json_path, download_name=None):
    """
    直接以文件流的方式返回 JSON 文件。

    旧实现会先 json.load 再 json.dumps(indent=2)：分析结果里带有
    几 MB 的 base64 图像数据，这一步既耗时又要占用多份内存，
    而且发生在事件循环线程里，会把并发请求一起拖慢。
    这里直接发送原始文件，内容和 LabmeMe 实际使用的完全一致。
    """
    if download_name is None:
        download_name = "export.json"

    return FileResponse(
        json_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": build_content_disposition(download_name)},
    )


def get_json_statistics(json_path, data=None):
    """获取JSON文件的统计信息（可传入已解析的数据，避免重复读取大文件）"""
    try:
        if data is None:
            data = get_json_data(json_path)
        file_stat = os.stat(json_path)

        label_counts = {}
        for shape in data.get('shapes', []):
            label = shape.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            'filename': os.path.basename(json_path),
            'file_size': file_stat.st_size,
            'file_size_human': _format_size(file_stat.st_size),
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'shapes_count': len(data.get('shapes', [])),
            'labels': label_counts,
            'image_width': data.get('imageWidth', 0),
            'image_height': data.get('imageHeight', 0),
        }
    except Exception as e:
        return {'error': str(e)}


def _load_json_and_stats(json_path):
    """读取一次 JSON，同时返回统计信息（同步函数，供线程池调用）。"""
    data = get_json_data(json_path)
    return data, get_json_statistics(json_path, data=data)


def _build_zip_file(json_files, zip_filename) -> str:
    """
    把多个 JSON 打包成 zip，写入临时文件（同步函数，供线程池调用）。

    写临时文件而不是内存 BytesIO：大批量导出时不会把整个压缩包留在内存里，
    同时 FileResponse 可以分块发送，浏览器能立即开始下载。
    """
    fd, temp_path = tempfile.mkstemp(suffix=".zip", prefix="rice_export_")
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for file_info in json_files:
                zipf.write(file_info['path'], file_info['name'])
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return temp_path


def _format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return "{} B".format(size_bytes)
    elif size_bytes < 1024 * 1024:
        return "{:.1f} KB".format(size_bytes / 1024)
    else:
        return "{:.1f} MB".format(size_bytes / (1024 * 1024))


_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _safe_sample_name(original_file_path: Optional[str]) -> str:
    """从原图路径得到安全的样本名，作为 ZIP 内的文件夹名。"""
    raw = _extract_real_name(original_file_path) if original_file_path else "sample"
    cleaned = _INVALID_FILENAME_CHARS.sub("_", raw).strip(" .")
    return (cleaned or "sample")[:80]


def _unique_folder_name(base_name: str, used_names: set) -> str:
    """同一批下载里样本名可能重复（不同目录下的同名图片），加序号避免覆盖。"""
    candidate = base_name
    index = 1
    while candidate.lower() in used_names:
        index += 1
        candidate = "{} ({})".format(base_name, index)
    used_names.add(candidate.lower())
    return candidate


def _build_data_package_zip(selection) -> tuple:
    """
    只把用户勾选的数据项打包：图片与 JSON 分开处理，不再自动成对加入。

    selection 是 [{'analysis': Analysis, 'kind': 'image'|'json'}, ...]。
    图片优先取带掩膜的标注图，缺失时才用原图。
    同一条记录的图片和 JSON 如果都被勾选，会落在同一个样本文件夹里；
    只勾 JSON 时，压缩包中就只有该 JSON。

    返回 (zip 临时文件路径, 打包成功的文件数)。同步函数，由线程池调用。
    """
    fd, temp_path = tempfile.mkstemp(suffix=".zip", prefix="rice_data_")
    os.close(fd)

    used_names = set()
    folder_names = {}
    entries = []

    try:
        for entry in selection:
            analysis = entry['analysis']
            kind = entry['kind']
            analysis_key = str(analysis.analysis_id)

            # 文件夹名按记录只生成一次：图片和 JSON 同时勾选时共用同一个目录
            if analysis_key not in folder_names:
                base_name = _safe_sample_name(
                    analysis.original_file_path or analysis.annotated_image_path
                )
                folder_names[analysis_key] = {
                    'base': base_name,
                    'folder': _unique_folder_name(base_name, used_names),
                }
            names = folder_names[analysis_key]
            base_name = names['base']
            folder_name = names['folder']

            if kind == 'json':
                json_path = analysis.result_json_path
                if json_path and os.path.exists(json_path):
                    entries.append((json_path, "{}/{}.json".format(folder_name, base_name)))
                continue

            if kind == 'image':
                # 数据中心展示 / 下载的都是带掩膜的分割图，旧记录缺失时才退回原图。
                image_path = analysis.annotated_image_path
                if not image_path or not os.path.exists(image_path):
                    image_path = analysis.original_file_path
                if not image_path or not os.path.exists(image_path):
                    continue
                image_ext = (os.path.splitext(image_path)[1] or ".jpg").lower()
                entries.append(
                    (image_path, "{}/{}{}".format(folder_name, base_name, image_ext))
                )

        written = set()
        packed = 0
        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for file_path, entry_name in entries:
                if entry_name in written:
                    continue
                zipf.write(file_path, entry_name)
                written.add(entry_name)
                packed += 1
        return temp_path, packed
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


# ==================== API 端点 ====================

@router.get("/json/{analysis_id}")
async def export_analysis_json(
    analysis_id,
    user=Depends(fastapi_users.current_user(active=True)),
    session: AsyncSession = Depends(get_async_session)
):
    """
    导出分析结果的JSON文件
    """
    try:
        log.info("用户 {} 请求导出JSON: {}".format(user.id, analysis_id))

        statement = select(Analysis).where(
            Analysis.analysis_id == analysis_id,
            Analysis.user_id == user.id
        )
        result = await session.execute(statement)
        analysis = result.scalar_one_or_none()

        if not analysis:
            raise HTTPException(status_code=404, detail="分析记录不存在")

        if analysis.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="分析尚未完成，当前状态: {}".format(analysis.status)
            )

        if not analysis.result_json_path or not os.path.exists(analysis.result_json_path):
            raise HTTPException(status_code=404, detail="JSON文件不存在")

        real_name = _extract_real_name(analysis.original_file_path)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"{real_name}_{now_str}.json"

        return export_json_response(
            json_path=analysis.result_json_path,
            download_name=download_name
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("导出JSON失败: {}".format(e), exc_info=True)
        raise HTTPException(status_code=500, detail="导出失败: {}".format(str(e)))


@router.get("/json/preview/{analysis_id}")
async def preview_analysis_json(
    analysis_id,
    user=Depends(fastapi_users.current_user(active=True)),
    session: AsyncSession = Depends(get_async_session)
):
    """
    预览JSON内容（不下载）
    """
    try:
        log.info("用户 {} 请求预览JSON: {}".format(user.id, analysis_id))

        statement = select(Analysis).where(
            Analysis.analysis_id == analysis_id,
            Analysis.user_id == user.id
        )
        result = await session.execute(statement)
        analysis = result.scalar_one_or_none()

        if not analysis:
            raise HTTPException(status_code=404, detail="分析记录不存在")

        if not analysis.result_json_path or not os.path.exists(analysis.result_json_path):
            raise HTTPException(status_code=404, detail="JSON文件不存在")

        # 大 JSON（含 base64 图像数据）的解析放到线程池，避免阻塞事件循环
        data, stats = await run_in_threadpool(
            _load_json_and_stats, analysis.result_json_path
        )

        return JSONResponse({
            'success': True,
            'data': data,
            'statistics': stats,
            'analysis_info': {
                'analysis_id': str(analysis.analysis_id),
                'original_filename': os.path.basename(analysis.original_file_path),
                'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
                'status': analysis.status
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error("预览JSON失败: {}".format(e), exc_info=True)
        raise HTTPException(status_code=500, detail="预览失败: {}".format(str(e)))


@router.get("/list")
async def list_exportable_analyses(
    user=Depends(fastapi_users.current_user(active=True)),
    session: AsyncSession = Depends(get_async_session),
    limit=50,
    offset=0,
    status=None
):
    """
    获取可导出的分析列表
    """
    try:
        log.info("用户 {} 请求分析列表, status={}".format(user.id, status))

        statement = select(Analysis).where(Analysis.user_id == user.id)

        if status:
            statement = statement.where(Analysis.status == status)
        else:
            statement = statement.where(Analysis.status.in_(['completed', 'failed']))

        statement = statement.order_by(desc(Analysis.created_at))
        statement = statement.offset(offset).limit(limit)

        result = await session.execute(statement)
        analyses = result.scalars().all()

        result_list = []
        for analysis in analyses:
            json_exists = False
            json_size = None
            if analysis.result_json_path and os.path.exists(analysis.result_json_path):
                json_exists = True
                json_size = os.path.getsize(analysis.result_json_path)

            result_list.append({
                'analysis_id': analysis.analysis_id,
                # ✅ 修复：使用 basename 提取文件名
                'original_filename': os.path.basename(analysis.original_file_path),
                'original_file_path': analysis.original_file_path,
                'annotated_image_path': analysis.annotated_image_path,
                'result_json_path': analysis.result_json_path,
                'status': analysis.status,
                'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
                'updated_at': analysis.updated_at.isoformat() if analysis.updated_at else None,
                'json_exists': json_exists,
                'json_size': json_size,
                'json_size_human': _format_size(json_size) if json_size else None
            })

        return {
            'success': True,
            'data': result_list,
            'total': len(result_list),
            'limit': limit,
            'offset': offset
        }

    except Exception as e:
        log.error("获取列表失败: {}".format(e), exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@router.post("/json/batch")
async def export_batch_json(
    analysis_ids_query: Optional[List[str]] = Query(None, alias="analysis_ids"),
    request_data: Optional[BatchJsonExportRequest] = Body(default=None),
    user=Depends(fastapi_users.current_user(active=True)),
    session: AsyncSession = Depends(get_async_session)
):
    """
    批量导出多个分析结果的JSON文件（返回ZIP压缩包）
    支持两种调用方式：
    1. Query 参数: POST /json/batch?analysis_ids=id1&analysis_ids=id2
    2. Body JSON:  { "analysisIds": ["id1", "id2"] }
    """
    import zipfile

    if request_data and request_data.analysisIds:
        analysis_ids = request_data.analysisIds
    elif analysis_ids_query:
        analysis_ids = analysis_ids_query
    else:
        raise HTTPException(status_code=422, detail="请通过 query(analysis_ids) 或 body(analysisIds) 传入要导出的ID列表")

    try:
        log.info("用户 {} 请求批量导出: {} 个文件".format(user.id, len(analysis_ids)))

        statement = select(Analysis).where(
            Analysis.analysis_id.in_(analysis_ids),
            Analysis.user_id == user.id,
            Analysis.status == "completed"
        )
        result = await session.execute(statement)
        analyses = result.scalars().all()

        if not analyses:
            raise HTTPException(
                status_code=404,
                detail="未找到有效的分析记录"
            )

        json_files = []
        for analysis in analyses:
            if analysis.result_json_path and os.path.exists(analysis.result_json_path):
                json_files.append({
                    'path': analysis.result_json_path,
                    'name': os.path.basename(analysis.result_json_path)
                })

        if not json_files:
            raise HTTPException(
                status_code=404,
                detail="没有可导出的JSON文件"
            )

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{len(json_files)}个文件_{now_str}.zip"

        zip_path = await run_in_threadpool(_build_zip_file, json_files, zip_filename)

        from starlette.background import BackgroundTask
        return FileResponse(
            zip_path,
            media_type="application/zip",
            headers={"Content-Disposition": build_content_disposition(zip_filename)},
            background=BackgroundTask(lambda: os.path.exists(zip_path) and os.remove(zip_path)),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("批量导出失败: {}".format(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="批量导出失败: {}".format(str(e))
        )


@router.post("/packages")
async def export_data_packages(
    request_data: DataPackageExportRequest = Body(...),
    user=Depends(fastapi_users.current_user(active=True)),
    session: AsyncSession = Depends(get_async_session)
):
    """
    数据中心「下载选中」：只打包用户勾选的图片或 JSON，不再自动成对加入。

    ZIP 结构示例（只勾 JSON 时就不会出现图片）：
        {样本名}/
            {样本名}{图片扩展名}  # 勾选了「图片」时（带掩膜的分割图）
            {样本名}.json         # 勾选了「JSON」时
    """
    request_items = request_data.items or []
    if not request_items:
        raise HTTPException(status_code=422, detail="请传入要下载的数据项列表")

    analysis_ids = list({item.analysisId for item in request_items})

    try:
        log.info(
            "用户 {} 请求打包下载 {} 个数据项（{} 条记录）".format(
                user.id, len(request_items), len(analysis_ids)
            )
        )

        statement = (
            select(Analysis)
            .where(
                Analysis.analysis_id.in_(analysis_ids),
                Analysis.user_id == user.id,
                Analysis.status == "completed",
            )
            .order_by(desc(Analysis.created_at))
        )
        result = await session.execute(statement)
        analyses = result.scalars().all()

        if not analyses:
            raise HTTPException(status_code=404, detail="未找到可下载的已完成数据")

        analysis_map = {str(analysis.analysis_id): analysis for analysis in analyses}
        selection = []
        for item in request_items:
            kind = (item.kind or "").strip().lower()
            if kind not in ("image", "json"):
                continue
            analysis = analysis_map.get(str(item.analysisId))
            if not analysis:
                continue
            selection.append({"analysis": analysis, "kind": kind})

        if not selection:
            raise HTTPException(status_code=404, detail="选中的数据项没有可下载的文件")

        zip_path, packed = await run_in_threadpool(_build_data_package_zip, selection)
        if packed == 0:
            try:
                os.remove(zip_path)
            except OSError:
                pass
            raise HTTPException(
                status_code=404,
                detail="选中的数据没有找到对应文件，无法打包下载",
            )

        label = {"leaf": "剑叶", "stem": "茎秆"}.get(
            (request_data.group or "").strip().lower(), "分析"
        )
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = "{}数据_{}项_{}.zip".format(label, packed, now_str)

        from starlette.background import BackgroundTask

        log.info("数据打包完成: {} ({} 个文件)".format(zip_path, packed))
        return FileResponse(
            zip_path,
            media_type="application/zip",
            headers={"Content-Disposition": build_content_disposition(zip_filename)},
            background=BackgroundTask(lambda: os.path.exists(zip_path) and os.remove(zip_path)),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("打包下载失败: {}".format(e), exc_info=True)
        raise HTTPException(status_code=500, detail="打包下载失败: {}".format(str(e)))