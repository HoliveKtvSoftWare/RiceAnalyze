"""剑叶（水稻剑叶）指标计算。

口径取自 RiceAnylaze_V2.0：

* 区域标签（各应出现一次）：``body1`` 主脉、``side1`` 侧脉1、``side2`` 侧脉2、
  ``body2`` 空腔；
* 维管束标签：``body_big`` / ``body1_small`` / ``body2_small`` /
  ``side1_big`` / ``side1_small`` / ``side2_big`` / ``side2_small``。

同一区域出现多个实例时保留**面积最大**者；带 ``mask_topology`` 标记的轮廓按
"外轮廓面积减洞面积、周长求和"处理，普通多边形直接按鞋带公式 / 弦长求和。

像素当量 k（µm/px）优先由图片右下象限的 500µm 比例尺自动识别，
失败时回退到调用方传入的 scale 参数。
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
    "side1HalfPerimeter": "侧脉1周长÷2",
    "side2HalfPerimeter": "侧脉2周长÷2",
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


def conversion_factor(unit: str) -> float:
    """长度单位换算系数（相对 µm）；未知单位按 µm 处理。"""
    return UNIT_CONVERSION.get(unit, 1.0)


def polygon_area(points: List[List[float]]) -> float:
    """鞋带（格林）公式计算多边形面积，与 cv2.contourArea 等价。"""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_perimeter(points: List[List[float]]) -> float:
    """多边形周长：逐段弦长求和（含闭合段）。"""
    n = len(points)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def topology_geometry(shapes: List[Dict[str, Any]]) -> Tuple[float, float]:
    """带洞轮廓的总面积与总周长：hole 面积做减法，周长直接累加。"""
    area = 0.0
    perimeter = 0.0
    for shape in shapes:
        points = shape.get("points", [])
        role = (shape.get("flags") or {}).get("contour_role", "outer")
        value = polygon_area(points)
        area += -value if role == "hole" else value
        perimeter += polygon_perimeter(points)
    return max(0.0, area), perimeter


def region_overlap_ratio(region_polygons: Dict[str, List[List[List[float]]]],
                         image_size: Tuple[int, int]) -> float:
    """估算不同区域标签之间的重叠程度（0~1，取最严重的一对）。

    剑叶的四个区域（主脉/侧脉1/侧脉2/空腔）在解剖上互不重叠。
    一旦两个标签的掩膜大面积压在一起，说明模型把同一块组织判给了两个类别，
    此时面积类指标会重复计入、截面面积虚高。

    为了提高速度，这里把多边形缩到 1/4 分辨率再栅格化统计，
    只用于质量告警，不参与指标计算。
    """
    width, height = int(image_size[0] or 0), int(image_size[1] or 0)
    if not width or not height or len(region_polygons) < 2:
        return 0.0
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - cv2 是项目依赖
        return 0.0

    scale = 0.25
    scaled_w = max(1, int(width * scale))
    scaled_h = max(1, int(height * scale))
    masks = {}
    for label, rings in region_polygons.items():
        if not rings:
            continue
        canvas = np.zeros((scaled_h, scaled_w), np.uint8)
        for ring in rings:
            points = np.asarray(ring, dtype=np.float32) * scale
            if len(points) >= 3:
                cv2.fillPoly(canvas, [points.astype(np.int32)], 1)
        masks[label] = canvas.astype(bool)

    worst = 0.0
    labels = list(masks)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            first, second = masks[labels[i]], masks[labels[j]]
            intersection = int((first & second).sum())
            smaller = min(int(first.sum()), int(second.sum()))
            if intersection and smaller:
                ratio = intersection / smaller
                if ratio > worst:
                    worst = ratio
                    worst_pair = (labels[i], labels[j])
    if worst > 0.2:
        log.warning("剑叶区域重叠异常: %s 重叠 %.1f%%，面积类指标可能重复计入",
                    worst_pair, worst * 100)
    return worst


def compute_leaf_metrics(analysis_data: Dict[str, Any], unit: str, k: float) -> Dict[str, Any]:
    """按剑叶口径计算指标。

    :param analysis_data: LabelMe 风格的标注数据（含 ``shapes``）。
    :param unit: 目标单位，支持 µm / um / mm / cm。
    :param k: 像素当量（µm/px）；面积按 k²、长度按 k 折算。
    :return: {列 key: 数值}，与 ``LEAF_COLUMNS`` 一致。
    """
    shapes = analysis_data.get("shapes", [])

    region_metrics: Dict[str, Tuple[float, float]] = {}
    region_polygons: Dict[str, List[List[List[float]]]] = {}
    legacy_regions: Dict[str, List[dict]] = {}
    topology_regions: Dict[str, Dict[Any, List[dict]]] = {}
    bundle_areas: Dict[str, List[float]] = {}

    for shape in shapes:
        label = shape.get("label", "unknown")
        points = shape.get("points", [])

        if label in LEAF_REGION_LABELS:
            flags = shape.get("flags") or {}
            if flags.get("mask_topology"):
                group_id = shape.get("group_id")
                group_key = group_id if group_id is not None else "topology"
                topology_regions.setdefault(label, {}).setdefault(group_key, []).append(shape)
            else:
                legacy_regions.setdefault(label, []).append(shape)
        elif label in LEAF_BUNDLE_LABELS:
            bundle_areas.setdefault(label, []).append(polygon_area(points))
        else:
            log.debug("剑叶结果中忽略未知标签: %s", label)

    for label in LEAF_REGION_LABELS:
        candidates = []
        for group_shapes in topology_regions.get(label, {}).values():
            area, perimeter = topology_geometry(group_shapes)
            candidates.append((area, perimeter, [s.get("points", []) for s in group_shapes]))
        for shape in legacy_regions.get(label, []):
            points = shape.get("points", [])
            candidates.append((polygon_area(points), polygon_perimeter(points), [points]))

        if candidates:
            best = max(candidates, key=lambda item: item[0])
            region_metrics[label] = (best[0], best[1])
            region_polygons[label] = best[2]
            if len(candidates) > 1:
                log.warning("剑叶结果中区域标签 %s 出现 %d 个实例，保留面积最大者", label, len(candidates))
        else:
            log.warning("剑叶结果中缺少区域标签 %s，相关指标按 0 处理", label)

    try:
        k = float(k)
    except (TypeError, ValueError):
        k = 0.0
    if not k or k <= 0:
        k = 1.0
        log.warning("像素当量无效，按 1 µm/px 处理")

    factor = conversion_factor(unit)
    area_factor = (k ** 2) / (factor ** 2)
    length_factor = k / factor

    def region_area(label: str) -> float:
        return region_metrics.get(label, (0.0, 0.0))[0]

    def region_perimeter(label: str) -> float:
        return region_metrics.get(label, (0.0, 0.0))[1]

    def bundle_stats(label: str) -> Tuple[int, float, float]:
        areas = bundle_areas.get(label, [])
        count = len(areas)
        total_px = sum(areas)
        total = total_px * area_factor
        avg = (total_px / count) * area_factor if count else 0.0
        return count, total, avg

    body1_area = region_area("body1")
    body2_area = region_area("body2")
    side1_area = region_area("side1")
    side2_area = region_area("side2")

    side1_perimeter = region_perimeter("side1") * length_factor
    side2_perimeter = region_perimeter("side2") * length_factor

    body_big_count, body_big_total, body_big_avg = bundle_stats("body_big")
    body_small_count, body_small_total, body_small_avg = bundle_stats("body1_small")
    cavity_small_count, cavity_small_total, cavity_small_avg = bundle_stats("body2_small")
    side1_big_count, side1_big_total, side1_big_avg = bundle_stats("side1_big")
    side1_small_count, side1_small_total, side1_small_avg = bundle_stats("side1_small")
    side2_big_count, side2_big_total, side2_big_avg = bundle_stats("side2_big")
    side2_small_count, side2_small_total, side2_small_avg = bundle_stats("side2_small")

    # 区域重叠诊断：四个区域解剖上应互不重叠，重叠过大说明模型把同一块组织
    # 判给了多个类别，面积类指标会重复计入（只做告警与展示，不参与计算）。
    overlap_ratio = region_overlap_ratio(
        region_polygons,
        (analysis_data.get("imageWidth", 0), analysis_data.get("imageHeight", 0)),
    )

    return {
        # scaleUmPerPx 恒为 µm/px（前端做稳定换算用）；scalePerPx 按当前单位换算
        "scaleUmPerPx": round(k, 6),
        "scalePerPx": round(length_factor, 6),
        # 0~1，值越大说明区域标签冲突越严重（理想接近 0）
        "regionOverlapRatio": round(overlap_ratio, 4),
        # 空腔完全位于主脉内部，截面面积不重复计入
        "sectionArea": (body1_area + side1_area + side2_area) * area_factor,
        "tissueArea": max(0.0, body1_area - body2_area + side1_area + side2_area) * area_factor,
        "body1Area": body1_area * area_factor,
        "body2Area": body2_area * area_factor,
        "side1Area": side1_area * area_factor,
        "side2Area": side2_area * area_factor,
        "body1Perimeter": region_perimeter("body1") * length_factor,
        "body2Perimeter": region_perimeter("body2") * length_factor,
        "side1Perimeter": side1_perimeter,
        "side2Perimeter": side2_perimeter,
        "side1HalfPerimeter": side1_perimeter / 2.0,
        "side2HalfPerimeter": side2_perimeter / 2.0,
        "bodyBigCount": body_big_count,
        "bodyBigTotalArea": body_big_total,
        "bodyBigAvgArea": body_big_avg,
        "bodySmallCount": body_small_count,
        "bodySmallTotalArea": body_small_total,
        "bodySmallAvgArea": body_small_avg,
        "cavitySmallCount": cavity_small_count,
        "cavitySmallTotalArea": cavity_small_total,
        "cavitySmallAvgArea": cavity_small_avg,
        "side1BigCount": side1_big_count,
        "side1BigTotalArea": side1_big_total,
        "side1BigAvgArea": side1_big_avg,
        "side1SmallCount": side1_small_count,
        "side1SmallTotalArea": side1_small_total,
        "side1SmallAvgArea": side1_small_avg,
        "side2BigCount": side2_big_count,
        "side2BigTotalArea": side2_big_total,
        "side2BigAvgArea": side2_big_avg,
        "side2SmallCount": side2_small_count,
        "side2SmallTotalArea": side2_small_total,
        "side2SmallAvgArea": side2_small_avg,
    }


def detect_scale_um_per_px(image_path: Optional[str]) -> Optional[float]:
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
        return SCALE_BAR_UM / scaler
    except Exception as error:  # pragma: no cover - 识别失败只影响量纲
        log.warning("剑叶比例尺自动识别失败（回退请求参数）: %s", error)
        return None


def compute_from_paths(original_file_path: str, json_file_path: str, unit: str = "μm",
                       scale: float = 500.0) -> Dict[str, Any]:
    """根据原图与 LabelMe JSON 计算一行剑叶指标（带进程内缓存）。

    :param original_file_path: 原图路径，用于自动识别比例尺（不存在时回退 scale）。
    :param json_file_path: LabelMe 结果 JSON 路径。
    :param unit: 目标单位。
    :param scale: 未识别到比例尺时的回退像素当量（µm/px）。
    :return: 含 ``filename``、``scaleUmPerPx`` 及全部指标 key 的字典。
    """
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

    k = detect_scale_um_per_px(original_file_path)
    if k and k > 0:
        log.info("[leaf] 自动识别比例尺: %.4f µm/px", k)
    else:
        k = float(scale) if scale and scale > 0 else 1.0
        log.warning("[leaf] 未识别到图内比例尺，回退 scale=%.4f µm/px", k)

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