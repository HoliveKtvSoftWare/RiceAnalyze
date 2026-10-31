"""剑叶（水稻剑叶）指标计算。

口径取自 RiceAnylaze_V2.0：

* 区域标签（各应出现一次）：``body1`` 主脉、``side1`` 侧脉1、``side2`` 侧脉2、
  ``body2`` 空腔；
* 维管束标签：``body_big`` / ``body1_small`` / ``body2_small`` /
  ``side1_big`` / ``side1_small`` / ``side2_big`` / ``side2_small``。

同一区域出现多个实例时保留**面积最大**者；带 ``mask_topology`` 标记的轮廓按
"外轮廓面积减洞面积、周长求和"处理，普通多边形直接按鞋带公式 / 弦长求和。

像素当量 k（µm/px）优先由图片右下象限的 500µm 比例尺自动识别，
未识别到有效比例尺时物理量留空，计数保留。
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# 图内比例尺线条代表的物理长度（µm），与茎秆口径一致
SCALE_BAR_UM = 500.0

# 长度单位 -> 1 个该单位等于多少 µm
UNIT_CONVERSION: Dict[str, float] = {
    "um": 1.0,
    "μm": 1.0,
    "mm": 1000.0,
    "cm": 10000.0,
}

LEAF_REGION_LABELS = ("body1", "side1", "side2", "body2")
LEAF_BUNDLE_LABELS = (
    "body_big",
    "body1_small",
    "body2_small",
    "side1_big",
    "side1_small",
    "side2_big",
    "side2_small",
)

# 导出列定义。key 必须与 compute_leaf_metrics 的返回值一致。
LEAF_COLUMNS: Dict[str, str] = {
    "filename": "样本名称",
    # 比例尺按当前导出单位换算，列名在导出时动态生成（比例尺(mm/px) 等）
    "scalePerPx": "比例尺",
    "sectionArea": "截面面积(含腔)",
    "tissueArea": "组织净面积",
    "body1Area": "主脉面积",
    "body2Area": "空腔面积",
    "side1Area": "侧脉1面积",
    "side2Area": "侧脉2面积",
    "body1Perimeter": "主脉周长",
    "body2Perimeter": "空腔周长",
    "side1Perimeter": "侧脉1周长",
    "side2Perimeter": "侧脉2周长",
    "side1Length": "侧脉1长度",
    "side2Length": "侧脉2长度",
    "bodyBigCount": "主脉大维管束数目",
    "bodyBigTotalArea": "主脉大维管束总面积",
    "bodyBigAvgArea": "主脉大维管束平均面积",
    "bodySmallCount": "主脉小维管束数目",
    "bodySmallTotalArea": "主脉小维管束总面积",
    "bodySmallAvgArea": "主脉小维管束平均面积",
    "cavitySmallCount": "空腔小维管束数目",
    "cavitySmallTotalArea": "空腔小维管束总面积",
    "cavitySmallAvgArea": "空腔小维管束平均面积",
    "side1BigCount": "侧脉1大维管束数目",
    "side1BigTotalArea": "侧脉1大维管束总面积",
    "side1BigAvgArea": "侧脉1大维管束平均面积",
    "side1SmallCount": "侧脉1小维管束数目",
    "side1SmallTotalArea": "侧脉1小维管束总面积",
    "side1SmallAvgArea": "侧脉1小维管束平均面积",
    "side2BigCount": "侧脉2大维管束数目",
    "side2BigTotalArea": "侧脉2大维管束总面积",
    "side2BigAvgArea": "侧脉2大维管束平均面积",
    "side2SmallCount": "侧脉2小维管束数目",
    "side2SmallTotalArea": "侧脉2小维管束总面积",
    "side2SmallAvgArea": "侧脉2小维管束平均面积",
}

# 叠加渲染配色（key = 模型输出的标签名，RGBA 0~255）
LEAF_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "body1": (0, 110, 255, 235),
    "side1": (0, 210, 90, 235),
    "side2": (0, 190, 220, 235),
    "body2": (185, 0, 235, 235),
    "body_big": (255, 0, 0, 150),
    "body1_small": (255, 140, 0, 145),
    "body2_small": (255, 0, 255, 145),
    "side1_big": (200, 0, 60, 160),
    "side1_small": (255, 215, 0, 150),
    "side2_big": (0, 120, 0, 160),
    "side2_small": (150, 255, 110, 150),
}

# 指标结果缓存：同一批记录反复导出（单条 + 总表）时不必重算几何量、重读图片。
_LEAF_CACHE: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_LEAF_CACHE_LOCK = threading.Lock()
_LEAF_CACHE_MAX = 512


def _cache_get(key):
    with _LEAF_CACHE_LOCK:
        value = _LEAF_CACHE.get(key)
        if value is None:
            return None
        _LEAF_CACHE.move_to_end(key)
        return dict(value)


def _cache_put(key, value):
    with _LEAF_CACHE_LOCK:
        _LEAF_CACHE[key] = dict(value)
        _LEAF_CACHE.move_to_end(key)
        while len(_LEAF_CACHE) > _LEAF_CACHE_MAX:
            _LEAF_CACHE.popitem(last=False)


def _file_stamp(path: str):
    """返回能代表文件当前版本的信息（修改时间 + 大小）。"""
    try:
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


# Compatibility names use the same pure geometry as the measurements.
from app.domain.analysis.metrics_common import conversion_factor, polygon_area, polygon_perimeter
from app.domain.geometry.mask_geometry import topology_geometry


def compute_leaf_metrics(analysis_data: Dict[str, Any], unit: str, k: Optional[float]) -> Dict[str, Any]:
    """Use the same pure geometry as the standalone tool, with automatic ends."""
    from app.domain.analysis.families.leaf import compute
    from app.domain.geometry.side_length import measure_leaf_sides

    lengths = measure_leaf_sides(analysis_data)
    result = compute(analysis_data, unit, k, lengths=lengths)
    # Preserve the stable µm/px base; exports display the current unit.
    if result.get("scaleUmPerPx") is not None:
        result["scaleUmPerPx"] = round(k, 6)
        result["scalePerPx"] = round(k / conversion_factor(unit), 6)
    else:
        result["scalePerPx"] = None
    result["regionOverlapRatio"] = _selected_overlap(analysis_data)
    return result


def _selected_overlap(data):
    """Diagnostic only: respect holes and exclude expected body/cavity overlap."""
    import cv2
    import numpy as np
    from app.domain.geometry.side_length import select_region

    width, height = int(data.get("imageWidth") or 0), int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        return 0.0
    masks = {}
    for label in LEAF_REGION_LABELS:
        rings = select_region(data.get("shapes", []), label)
        canvas = np.zeros((max(1, height//4), max(1, width//4)), np.uint8)
        for ring in sorted(rings, key=lambda s: (s.get("flags") or {}).get("contour_role") == "hole"):
            points = np.rint(np.asarray(ring["points"], float)/4).astype(np.int32)
            cv2.fillPoly(canvas, [points], 0 if (ring.get("flags") or {}).get("contour_role") == "hole" else 1)
        if canvas.any():
            masks[label] = canvas.astype(bool)
    worst = 0.0
    labels = list(masks)
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            if {a, b} == {"body1", "body2"}:
                continue
            worst = max(worst, float((masks[a] & masks[b]).sum()) / min(masks[a].sum(), masks[b].sum()))
    return round(worst, 4)


def detect_scale_um_per_px(image_path: Optional[str], scale_bar_um: float = SCALE_BAR_UM) -> Optional[float]:
    """从图片右下象限的红/蓝比例尺推算像素当量（µm/px）。

    做法与茎秆口径一致：在右下象限找纯红 (>=240,<=20,<=20) / 纯蓝
    (<=20,<=20,>=240) 像素，取"单行红+蓝像素数"的最大值 s 作为比例尺线条长度，
    则 k = 500 / s。返回 None 表示未识别到比例尺。
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as im:
            arr = np.array(im.convert("RGB"))
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        height, width = arr.shape[:2]
        quadrant = arr[height // 2:, width // 2:, :3].astype(np.int16)
        red = (quadrant[:, :, 0] >= 240) & (quadrant[:, :, 1] <= 20) & (quadrant[:, :, 2] <= 20)
        blue = (quadrant[:, :, 0] <= 20) & (quadrant[:, :, 1] <= 20) & (quadrant[:, :, 2] >= 240)
        rows = red.sum(axis=1) + blue.sum(axis=1)
        if rows.size == 0:
            return None
        scaler = int(rows.max())
        if scaler <= 0:
            return None
        return scale_bar_um / scaler
    except Exception as error:  # pragma: no cover - 识别失败只影响量纲
        log.warning("剑叶比例尺自动识别失败（物理量留空）: %s", error)
        return None


def compute_from_paths(original_file_path: str, json_file_path: str, unit: str = "μm",
                       scale: float = 500.0) -> Dict[str, Any]:
    """根据原图与 LabelMe JSON 计算一行剑叶指标（带进程内缓存）。

    :param original_file_path: 原图路径，用于自动识别比例尺（未识别到比例尺时物理量留空）。
    :param json_file_path: LabelMe 结果 JSON 路径。
    :param unit: 目标单位。
    :param scale: 图中比例尺代表的物理长度（µm），不是像素当量。
    :return: 含 ``filename``、``scaleUmPerPx`` 及全部指标 key 的字典。
    """
    from app.core.paths import backend_path
    original_file_path = backend_path(original_file_path) if original_file_path else None
    json_file_path = backend_path(json_file_path)
    cache_key = (
        os.path.abspath(original_file_path) if original_file_path else None,
        os.path.abspath(json_file_path) if json_file_path else None,
        _file_stamp(original_file_path) if original_file_path else None,
        _file_stamp(json_file_path) if json_file_path else None,
        unit,
        float(scale),
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    with open(json_file_path, "r", encoding="utf-8") as file:
        analysis_data = json.load(file)

    k = detect_scale_um_per_px(original_file_path, float(scale))
    if k and k > 0:
        log.info("[leaf] 自动识别比例尺: %.4f µm/px", k)
    else:
        k = None
        log.warning("[leaf] 未识别到有效图内比例尺，物理量留空")

    from app.services.stem_excel import _normalize_sample_name

    row: Dict[str, Any] = {
        "filename": _normalize_sample_name(
            os.path.splitext(os.path.basename(json_file_path))[0]
        )
    }
    row.update(compute_leaf_metrics(analysis_data, unit, k))

    _cache_put(cache_key, row)
    return row


__all__ = [
    "SCALE_BAR_UM", "UNIT_CONVERSION", "LEAF_COLUMNS", "LEAF_COLORS",
    "LEAF_REGION_LABELS", "LEAF_BUNDLE_LABELS",
    "conversion_factor", "polygon_area", "polygon_perimeter", "topology_geometry",
    "compute_leaf_metrics", "detect_scale_um_per_px", "compute_from_paths",
]
