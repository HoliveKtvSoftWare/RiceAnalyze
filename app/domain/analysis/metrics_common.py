# -*- coding: utf-8 -*-
"""茎秆 / 剑叶指标**共用**的几何与单位换算工具（纯计算）。

分工
----
* 族专属的指标算法在 ``app/domain/analysis/families/<族>.py``
* 这里只放两族都要用的部分

改动本文件会**同时影响两个族**，属于共享核心，改前请确认另一族是否需要同步。
本模块不依赖 Web、数据库、文件系统，可脱离整个应用单独测试。
"""

import math
from typing import Dict, List

# 图内比例尺线条所代表的物理长度（µm）。
# 这是数据采集端的约定：切片图右下角自带一根 500µm 的比例尺线。
SCALE_BAR_UM = 500.0

# 长度单位 -> 1 个该单位等于多少 µm
UNIT_CONVERSION: Dict[str, float] = {
    "um": 1.0,
    "μm": 1.0,
    "mm": 1000.0,
    "cm": 10000.0,
}


def conversion_factor(unit: str) -> float:
    """长度单位换算系数（相对 µm）；未知单位按 µm（1.0）处理。"""
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
    """多边形周长：逐段弦长求和（含闭合段），与 cv2.arcLength(..., True) 等价。

    注意：周长完全由顶点决定，对标注/预测轮廓的点密度敏感；
    模型输出的轮廓点很密，与人工标注的稀疏多边形不可直接比较。
    """
    n = len(points)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
    return total


def diameter_from_area(area: float) -> float:
    """把面积折算成等效圆直径（茎秆指标用）。"""
    if area <= 0:
        return 0.0
    return 2 * math.sqrt(area / math.pi)


def perimeter_from_area(area: float) -> float:
    """把面积折算成等效圆周长（茎秆指标用）。"""
    if area <= 0:
        return 0.0
    return 2 * math.pi * math.sqrt(area / math.pi)


__all__ = [
    "SCALE_BAR_UM", "UNIT_CONVERSION", "conversion_factor",
    "polygon_area", "polygon_perimeter",
    "diameter_from_area", "perimeter_from_area",
]
