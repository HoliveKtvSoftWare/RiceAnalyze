# 分析服务

import os
import logging
from datetime import datetime
from typing import Optional

# 导入我们重构后的、健壮的服务模块
from app.services.yolo_inference import run_system

# 导入数据库相关的工具
from sqlmodel import create_engine, Session, select
from app.core.config import settings
from app.models.analysis import Analysis

# 设置日志
log = logging.getLogger(__name__)

# 创建一个同步的数据库引擎（后台任务在独立线程中执行，使用同步驱动更简单）
# pool_pre_ping: 避免复用被数据库服务端回收的连接，导致任务状态更新偶发变慢
sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
sync_engine = create_engine(
    sync_db_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# 确认函数定义包含了 original_filename: str
def run_full_analysis(analysis_id: str, original_file_path: str, user_id: str, original_filename: str, model_name:
Optional[str] = None, batch_id: Optional[str] = None, task_type: str = "stem"):
    """
    后台任务主函数，使用原始文件名进行输出。

    Args:
        analysis_id: 分析记录的 UUID
        original_file_path: 原图存储路径
        user_id: 用户 UUID
        original_filename: 原始文件名（含时间戳）
        model_name: 用户选择的模型名称（茎秆对应 YOLO_MODELS，剑叶对应 LEAF_MODELS 的 key）
        batch_id: 批次 UUID，用于组织输出目录结构
        task_type: 分析类型，``stem``（茎秆截面，默认）或 ``leaf``（剑叶）；
            决定用哪套模型注册表与渲染配色
    """
    task_type = (task_type or "stem").strip().lower()
    if task_type not in ("stem", "leaf"):
        task_type = "stem"
    try:
        log.info(
            f"--- [任务 {analysis_id}] 开始处理 (类型: {task_type}, "
            f"原始文件名: {original_filename}, 模型: {model_name}) ---"
        )

        # 步骤 1: 根据分析类型 + model_name 解析出实际模型路径
        if task_type == "leaf":
            model_path = settings.get_leaf_model_path(model_name)
        else:
            model_path = settings.get_model_path(model_name)
        log.info(f"解析模型: task_type={task_type}, model_name={model_name} -> model_path={model_path}")

        # 步骤 2: 准备本次任务专属的输出文件夹
        original_filename_without_ext = os.path.splitext(original_filename)[0]
        if batch_id:
            output_dir = os.path.join(settings.STORAGE_PATH, user_id, batch_id)
        else:
            output_dir = os.path.join(settings.STORAGE_PATH, user_id, original_filename_without_ext)
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"输出目录已创建: {output_dir}")

        # 步骤 3: 调用 YOLO 推理服务
        # 茎秆沿用原来的多边形平滑与四类配色；剑叶用 11 类自定义配色、不平滑
        # （平滑会改变周长口径，剑叶周长直接按多边形弦长求和）。
        # 剑叶额外对齐 V2 的推理/后处理参数：
        #   retina_masks=True  -> 高分辨率掩膜，轮廓更细
        #   iou=0.45           -> NMS 更严，减少同一结构被重复检测
        #   filter_small_components + min_mask_area_px=64 -> 去掉掩膜内噪点和微小的误检实例
        log.info(f"开始执行 YOLO 推理... 图片: {original_file_path}")
        inference_kwargs = {}
        if task_type == "leaf":
            from app.services.leaf_analysis import LEAF_COLORS

            inference_kwargs = {
                "smooth": False,
                "colors": LEAF_COLORS,
                "conf": 0.25,
                "iou": 0.45,
                "retina_masks": True,
                "filter_small_components": True,
                "min_mask_area_px": 64,
            }
        annotated_image_path, json_output_path = run_system(
            model_path=model_path,
            image_path=original_file_path,
            output_path=output_dir,
            output_basename=original_filename_without_ext,
            **inference_kwargs
        )
        log.info(f"YOLO 推理完成。JSON 已保存至: {json_output_path}")

        # 步骤 4: 更新数据库（含 model_used）
        log.info(f"开始更新数据库状态为 'completed'...")
        with Session(sync_engine) as session:
            statement = select(Analysis).where(Analysis.analysis_id == analysis_id)
            analysis_record = session.exec(statement).one()

            analysis_record.status = "completed"
            analysis_record.task_type = task_type
            analysis_record.annotated_image_path = annotated_image_path
            analysis_record.result_json_path = json_output_path
            analysis_record.model_used = model_name if model_name else "default"
            # 重跑/重试成功时清掉历史失败原因，避免“已完成但还挂着旧报错”
            analysis_record.error_message = None
            analysis_record.updated_at = datetime.utcnow()

            session.add(analysis_record)
            session.commit()

        log.info(f"--- [任务 {analysis_id}] 已成功处理 ---")

    except Exception as e:
        log.error(f"--- [任务 {analysis_id}] 发生严重错误: {e} ---", exc_info=True)
        try:
            with Session(sync_engine) as session:
                statement = select(Analysis).where(Analysis.analysis_id == analysis_id)
                analysis_record = session.exec(statement).one_or_none()
                if analysis_record:
                    analysis_record.status = "failed"
                    analysis_record.error_message = _summarize_error(e, model_name)
                    analysis_record.updated_at = datetime.utcnow()
                    session.add(analysis_record)
                    session.commit()
            log.info(f"数据库状态已更新为 'failed'，error_message={_summarize_error(e, model_name)}")
        except Exception as db_error:
            log.error(f"在更新任务状态为 'failed' 时再次发生错误: {db_error}")

        # 重新抛出异常，让 FastAPI BackgroundTasks 知道出错了
        raise e


def _summarize_error(exc: Exception, model_name: Optional[str]) -> str:
    """
    将底层异常转化为前端能直接看懂的中文提示，用于 Analysis.error_message。
    """
    if isinstance(exc, FileNotFoundError) and "模型文件不存在" in str(exc):
        return f"未部署该模型: {model_name if model_name else 'default'}"
    msg = str(exc) if str(exc) else exc.__class__.__name__
    return msg[:500]