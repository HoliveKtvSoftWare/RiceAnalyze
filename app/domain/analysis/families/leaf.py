# -*- coding: utf-8 -*-
"""剑叶指标（11 类 notail 标签体系，纯计算）。

用于剑叶分析：主脉 / 侧脉1 / 侧脉2 / 空腔四块区域，
以及分布在其中的大、小维管束。

标签体系
--------
* 区域标签（各应出现一次）：``body1`` 主脉、``side1`` 侧脉1、``side2`` 侧脉2、``body2`` 空腔
* 维管束标签：``body_big`` / ``body1_small`` / ``body2_small`` /
  ``side1_big`` / ``side1_small`` / ``side2_big`` / ``side2_small``

同一区域出现多个实例时保留**面积最大**者并告警；带 ``mask_topology`` 标记的
轮廓走拓扑几何（外轮廓减洞），否则按普通多边形处理。

面积、周长与维管束口径与独立版共用；回归资料保存在仓库外的融合验收目录。
侧脉长度为 side 自身两端之间的完整中心线路径弧长，不依赖主脉或头尾方向。
无法可靠确认两端或路径连通的结果为 None，Excel 保留空单元格。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.domain.analysis.metrics_common import (
    conversion_factor,
    polygon_area,
    polygon_perimeter,
)
from app.domain.geometry.mask_geometry import topology_geometry
from app.domain.geometry.side_length import measure_leaf_sides

log = logging.getLogger(__name__)

# 区域标签：主脉 / 侧脉1 / 侧脉2 / 空腔（每个样本应各出现一次）
LEAF_REGION_LABELS = ("body1", "side1", "side2", "body2")

# 维管束标签：主脉大/小、空腔小、两条侧脉的大/小
LEAF_BUNDLE_LABELS = ("body_big", "body1_small", "body2_small",
                      "side1_big", "side1_small", "side2_big", "side2_small")


def compute(analysis_data: Dict[str, Any], unit: str, k: Optional[float], lengths=None) -> Dict[str, Any]:
    """按剑叶口径计算指标。

    k: 像素当量（µm/px）。由调用方统一解析后传入 —— 优先用图内 500µm 比例尺
       自动识别，也可显式传入人工像素当量。缺失时物理量留空，计数保留。
       面积按 k²、长度按 k 折算。
    """
    shapes = analysis_data.get("shapes", [])

    region_metrics: Dict[str, Tuple[float, float]] = {}   # label -> (面积px², 周长px)
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
            log.debug(f"剑叶结果中忽略未知标签: {label}")

    for label in LEAF_REGION_LABELS:
        candidates = [
            topology_geometry(group_shapes)
            for group_shapes in topology_regions.get(label, {}).values()
        ]
        candidates.extend(
            (polygon_area(shape.get("points", [])),
             polygon_perimeter(shape.get("points", [])))
            for shape in legacy_regions.get(label, [])
        )
        if candidates:
            region_metrics[label] = max(candidates, key=lambda item: item[0])
            if len(candidates) > 1:
                log.warning(f"剑叶结果中区域标签 {label} 出现 {len(candidates)} 个实例，保留面积最大者")
        else:
            log.warning(f"剑叶结果中缺少区域标签 {label}，相关指标按 0 处理")

    has_scale = k is not None and math.isfinite(k) and k > 0
    if not has_scale:
        # Compute geometry and counts as usual; the placeholder factor must
        # never be exposed as a measured physical value below.
        k = 1.0
        log.warning("缺少有效像素当量，剑叶物理量留空，维管束计数保留")

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

    lengths = measure_leaf_sides(analysis_data) if lengths is None else lengths
    for label, measurement in lengths.items():
        if measurement.reason:
            log.warning("%s 长度不可用: %s", label, measurement.reason)

    body_big_count, body_big_total, body_big_avg = bundle_stats("body_big")
    body_small_count, body_small_total, body_small_avg = bundle_stats("body1_small")
    cavity_small_count, cavity_small_total, cavity_small_avg = bundle_stats("body2_small")
    side1_big_count, side1_big_total, side1_big_avg = bundle_stats("side1_big")
    side1_small_count, side1_small_total, side1_small_avg = bundle_stats("side1_small")
    side2_big_count, side2_big_total, side2_big_avg = bundle_stats("side2_big")
    side2_small_count, side2_small_total, side2_small_avg = bundle_stats("side2_small")

    metrics = {
        "scaleUmPerPx": round(k, 4),
        # 空腔完全位于主脉内部，截面面积不再重复计入
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
        "side1Length": (lengths["side1"].length_px * length_factor
                        if lengths["side1"].length_px is not None else None),
        "side2Length": (lengths["side2"].length_px * length_factor
                        if lengths["side2"].length_px is not None else None),
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
    if not has_scale:
        for key in metrics:
            if not key.endswith("Count"):
                metrics[key] = None
    return metrics


__all__ = ["compute", "LEAF_REGION_LABELS", "LEAF_BUNDLE_LABELS"]
