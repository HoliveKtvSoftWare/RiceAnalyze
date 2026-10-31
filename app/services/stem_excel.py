"""
Excel下载服务

性能要点：
1. 指标计算按 (原图, JSON, 单位, 标尺) 做进程内缓存 —— 同一批记录反复导出
   （例如先导出单条再导出总表）时直接命中缓存，不再重复解码图片、重算几何量；
2. 标尺检测只读取图片右下角区域，不再把整张原图转成 numpy 数组；
3. 多边形面积 / 二值化都改用向量化实现，并把填充画布限制在多边形包围盒内。
"""


import os
import re
import json
import pandas as pd
import logging
import math
import threading
from collections import OrderedDict
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from sqlmodel import create_engine
from openpyxl.utils import get_column_letter
from app.core.config import settings
from app.core.paths import backend_path
from app.models.analysis import Analysis

log = logging.getLogger(__name__)

# 指标结果缓存：key -> 指标字典。上限 512 条，超出后淘汰最久未使用的记录。
_RESULT_CACHE: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()
_RESULT_CACHE_MAX = 512

# 二值化闭运算核大小；包围盒外扩该值的一半以上，保证闭运算结果与原实现一致
_CLOSE_KERNEL = 60
_CLOSE_PADDING = _CLOSE_KERNEL // 2 + 4


def _cache_get(key):
    with _RESULT_CACHE_LOCK:
        value = _RESULT_CACHE.get(key)
        if value is None:
            return None
        _RESULT_CACHE.move_to_end(key)
        return dict(value)


def _cache_put(key, value):
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE[key] = dict(value)
        _RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
            _RESULT_CACHE.popitem(last=False)


def _file_stamp(path):
    """返回能代表文件当前版本的信息（修改时间 + 大小），用于缓存有效性判断。"""
    try:
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


# 上传时为了文件名唯一而加的包装：{analysis_uuid}_{原始文件名}_{上传时间戳}
# 导出样本名称时把 UUID 前缀和上传时间戳后缀都去掉，还原成用户上传时的原始文件名
# （UUID 兼容带/不带连字符以及 - 或 _ 分隔的写法；时间戳格式为 _YYYYMMDD_HHMMSS）
_UUID_PREFIX_RE = re.compile(
    r'^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}[_-]+'
)
_UPLOAD_TS_SUFFIX_RE = re.compile(r'_\d{8}_\d{6}$')


def _normalize_sample_name(name: str) -> str:
    """
    还原样本的原始文件名：去掉 UUID 前缀与上传时间戳后缀。

    例如 ``0f8fad5b-d9cb-469f-a165-70867728950e_叶片A_20241027_095852``
    返回 ``叶片A``。若两者都不匹配则原样返回。

    :param name: 原始样本名（不含扩展名）。
    :return: 去掉 UUID 前缀、时间戳后缀后的样本名。
    """
    if not name:
        return name
    stripped = _UUID_PREFIX_RE.sub('', name, count=1) or name
    stripped = _UPLOAD_TS_SUFFIX_RE.sub('', stripped, count=1) or stripped
    return stripped


class ExcelDownloadService:
    def __init__(self):
        self.available_columns = {
            "filename": "样本名称",
            # 比例尺固定排在样本名称后一列；值按当前导出单位换算，列名在导出时动态生成
            "scalePerPx": "比例尺",
            "largeTailCount": "大维管束数目",
            "smallTailCount": "小维管束数目",
            "totalCount": "总维管束数目",
            "largeTailArea": "大维管束面积",
            "smallTailArea": "小维管束面积",
            "stemDiameter": "茎秆直径",
            "stemPerimeter": "茎秆周长",
            "cavityArea": "空腔面积",
            "stemCavityAreaDiff": "茎秆面积与空腔面积差值",
            "largeSmallAreaRatio": "大维管束面积与小维管束面积比值",
            "largeSmallCountRatio": "大维管束数目与小维管束数目比值",
            "stemArea": "茎秆面积",
            "cavityStemAreaRatio": "空腔面积与茎秆面积比值",
            "smallCountPerimeterRatio": "小维管束数目与茎秆周长比值",
            "largeCountPerimeterCavityRatio": "大维管束数目与茎秆周长与空腔面积差值比值"
        }

        self.unit_conversion = {
            "μm": 1.0,
            "mm": 1000.0,
            "cm": 10000.0
        }

        sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
        self._sync_engine = create_engine(sync_db_url, echo=False, pool_pre_ping=True)

    def get_available_columns(self) -> Dict[str, str]:
        """返回当前服务支持的所有可导出列名及其中文显示名映射。"""
        return self.available_columns

    def _unit_symbol(self, unit: str) -> str:
        """把内部单位名转成显示用的符号（μm → µm）。"""
        return {"μm": "µm", "mm": "mm", "cm": "cm"}.get(unit, unit)

    def column_label(self, col: str, unit: str) -> str:
        """
        返回导出列的中文表头。

        比例尺的列名需要跟随导出单位变化，例如“比例尺(mm/px)”；
        其余列使用固定中文名。
        """
        if col == "scalePerPx":
            return f"比例尺({self._unit_symbol(unit)}/px)"
        return self.available_columns[col]

    def _ordered_columns(self, selected_columns: List[str]) -> List[str]:
        """
        按可用列的定义顺序排序列，保证 Excel 的列顺序固定。

        比例尺（scalePerPx）在列定义中紧跟样本名称，所以无论用户以什么顺序勾选，
        导出后它都在第二列。
        """
        order = {col: index for index, col in enumerate(self.available_columns)}
        return sorted(selected_columns, key=lambda col: order.get(col, len(order)))

    def _rebuild_stem_contour(self, p: np.ndarray, um_per_px: float, smooth_um: float = 20, bins: int = 360):
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        c = np.array([np.sum((x + xn) * cross), np.sum((y + yn) * cross)]) / (3 * cross.sum())
        v = p - c
        theta = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
        radius = np.linalg.norm(v, axis=1)
        idx = np.floor(theta * bins / (2 * np.pi)).astype(int) % bins
        r = np.full(bins, np.nan)
        spreads = []
        for i in range(bins):
            vals = radius[idx == i]
            if len(vals):
                r[i] = np.median(vals)
            if len(vals) > 1:
                spreads.append(float(np.ptp(vals)))
        known = np.where(np.isfinite(r))[0]
        if len(known) < bins * .5:
            raise ValueError('径向覆盖不足，需要人工修复轮廓')
        gaps = np.diff(np.r_[known, known[0] + bins])
        max_gap = int(gaps.max())
        grid = np.arange(bins)
        r = np.interp(grid, np.r_[known - bins, known, known + bins], np.tile(r[known], 3))
        step = 2 * np.pi * np.median(r) / bins
        sigma_bins = smooth_um / um_per_px / step
        frequency = np.fft.rfftfreq(bins)
        gaussian = np.exp(-2 * np.pi ** 2 * sigma_bins ** 2 * frequency ** 2)
        rs = np.fft.irfft(np.fft.rfft(r) * gaussian, n=bins)
        angle = (grid + .5) * 2 * np.pi / bins
        curve = c + np.column_stack([rs * np.cos(angle), rs * np.sin(angle)])
        spread = max(spreads, default=0) / np.median(radius)
        return curve, spread, max_gap

    def _calculate_stem_perimeter(self, out_points, um_per_pixel, smooth_um=20.0):
        p = np.asarray(out_points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("无效的茎秆外轮廓")
        if um_per_pixel <= 0 or smooth_um <= 0:
            raise ValueError("像素标定及平滑尺度必须大于0")
        if np.array_equal(p[0], p[-1]):
            p = p[:-1]
        if self._calculate_shape_area(p) <= 0:
            raise ValueError("茎秆外轮廓面积为0")
        length = lambda pts: float(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1).sum())
        curve, spread, gap = self._rebuild_stem_contour(p, um_per_pixel, smooth_um)
        result = length(curve)
        light = length(self._rebuild_stem_contour(p, um_per_pixel, smooth_um * 0.75)[0])
        strong = length(self._rebuild_stem_contour(p, um_per_pixel, smooth_um * 1.25)[0])
        sensitivity = abs(light - strong) / result
        if spread > 0.05 or gap > 10 or sensitivity > 0.02 or length(p) / result > 1.15:
            log.warning(
                "茎秆周长需人工核验：径向跨度比例=%.4f，最大角度缺口=%d，"
                "平滑敏感性=%.2f%%；周长为径向重建估计",
                spread, gap, sensitivity * 100
            )
        return result

    def _compute_from_paths(self, original_file_path: str, json_file_path: str, unit: str, scale: float) -> Dict[str, Any]:
        """
        根据原始图像文件路径和标注结果 JSON 文件路径，计算一组完整的水稻茎秆分析指标。

        结果会按 (原图, JSON, 单位, 标尺, 文件修改状态) 缓存，
        同一记录重复导出时无需再次解码图片和重算几何量。

        主要处理流程：
        1. 从 JSON 中读取标注数据（大/小维管束、茎秆外轮廓、空腔轮廓）。
        2. 通过 `_label_binarize` 计算茎秆面积；空腔面积直接用鞋带公式。
        3. 读取原图右下角并调用 `_get_scaler` 自动检测标尺像素长度，推导像素到实际单位的缩放比例。
        4. 标定完成后，用径向重建算法计算茎秆周长。
        5. 结合 `unit_conversion` 将像素面积/长度转换为用户指定单位（μm/mm/cm）。
        6. 计算各类派生指标（总数、比值、差值等）并汇总返回。

        :param original_file_path: 原始图像的绝对路径，用于检测标尺像素长度。
        :param json_file_path: LabelMe 风格的标注结果 JSON 路径，包含 shapes 字段。
        :param unit: 目标单位字符串，支持 "μm"、"mm"、"cm"。
        :param scale: 用户指定的标尺实际长度（微米），默认 500。
        :return: 包含所有分析指标的字典，键名为英文字段名（与 available_columns 对应）。
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

        with open(json_file_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        img_h = analysis_data.get("imageHeight", 0)
        img_w = analysis_data.get("imageWidth", 0)

        base_info = {
            # 样本名称：JSON 文件名即 {uuid}_{原始文件名}_{时间戳}.json，导出时还原成原始文件名
            "filename": _normalize_sample_name(os.path.splitext(os.path.basename(json_file_path))[0])
        }

        shapes = analysis_data.get("shapes", [])

        large_tail_shapes = []
        small_tail_shapes = []
        out_points = None
        in_points = None

        for shape in shapes:
            label = shape.get("label", "unknown")
            points = shape.get("points", [])
            area = self._calculate_shape_area(points)

            if label == "big":
                large_tail_shapes.append({"area": area})
            elif label == "small":
                small_tail_shapes.append({"area": area})
            elif label == "out":
                out_points = points
            elif label == "in":
                in_points = points

        large_tail_count = len(large_tail_shapes)
        small_tail_count = len(small_tail_shapes)
        total_count = large_tail_count + small_tail_count

        large_tail_area_pixel = sum(shape["area"] for shape in large_tail_shapes)
        small_tail_area_pixel = sum(shape["area"] for shape in small_tail_shapes)

        stem_area_pixel = 0.0
        stem_perimeter_pixel = 0.0
        stem_diameter_pixel = 0.0

        if out_points and img_h > 0 and img_w > 0:
            stem_area_pixel, _ = self._label_binarize(img_h, img_w, out_points)
            try:
                out_pts = np.array(out_points, dtype=np.int32)
                (_, _), radius = cv2.minEnclosingCircle(out_pts)
                stem_diameter_pixel = radius * 2.0
            except Exception:
                stem_diameter_pixel = self._calculate_diameter_from_area(stem_area_pixel)
        elif out_points:
            stem_area_pixel = self._calculate_shape_area(out_points)
            stem_diameter_pixel = self._calculate_diameter_from_area(stem_area_pixel)

        cavity_area_pixel = self._calculate_shape_area(in_points) if in_points else 0.0

        conversion_factor = self.unit_conversion.get(unit, 1.0)

        scaling_ratio = 1.0
        try:
            if os.path.exists(original_file_path):
                scaler = self._detect_scaler_from_file(original_file_path)
                if scaler > 0:
                    scaling_ratio = scale / scaler
            else:
                scaling_ratio = scale / 500.0
        except Exception as e:
            log.warning(f"读取原图检测标尺失败，使用默认比例: {e}")
            scaling_ratio = scale / 500.0

        if out_points:
            try:
                stem_perimeter_pixel = self._calculate_stem_perimeter(out_points, scaling_ratio)
            except Exception as e:
                log.warning(f"径向重建周长失败，使用等效圆估算: {e}")
                stem_perimeter_pixel = self._calculate_perimeter_from_area(stem_area_pixel)

        stem_diameter = stem_diameter_pixel * scaling_ratio / conversion_factor
        stem_perimeter = stem_perimeter_pixel * scaling_ratio / conversion_factor

        area_scale_factor = scaling_ratio ** 2
        large_tail_area = large_tail_area_pixel * area_scale_factor / (conversion_factor ** 2)
        small_tail_area = small_tail_area_pixel * area_scale_factor / (conversion_factor ** 2)
        stem_area = stem_area_pixel * area_scale_factor / (conversion_factor ** 2)
        cavity_area = cavity_area_pixel * area_scale_factor / (conversion_factor ** 2)

        stem_cavity_area_diff = stem_area - cavity_area
        large_small_area_ratio = large_tail_area / small_tail_area if small_tail_area > 0 else 0
        large_small_count_ratio = large_tail_count / small_tail_count if small_tail_count > 0 else 0
        cavity_stem_area_ratio = cavity_area / stem_area if stem_area > 0 else 0
        small_count_perimeter_ratio = small_tail_count / stem_perimeter if stem_perimeter > 0 else 0
        large_count_perimeter_cavity_ratio = large_tail_count / stem_cavity_area_diff if stem_cavity_area_diff > 0 else 0

        base_info.update({
            "largeTailCount": large_tail_count,
            "smallTailCount": small_tail_count,
            "totalCount": total_count,
            "largeTailArea": large_tail_area,
            "smallTailArea": small_tail_area,
            "stemDiameter": stem_diameter,
            "stemPerimeter": stem_perimeter,
            "cavityArea": cavity_area,
            "stemCavityAreaDiff": stem_cavity_area_diff,
            "largeSmallAreaRatio": large_small_area_ratio,
            "largeSmallCountRatio": large_small_count_ratio,
            "stemArea": stem_area,
            "cavityStemAreaRatio": cavity_stem_area_ratio,
            "smallCountPerimeterRatio": small_count_perimeter_ratio,
            "largeCountPerimeterCavityRatio": large_count_perimeter_cavity_ratio,
            # 与剑叶页一致：把检测到的比例尺一并返回，前端“结构指标”面板显示。
            # scaleUmPerPx 恒为 µm/px；scalePerPx 按当前单位换算，前端与 Excel 都用它。
            "scaleUmPerPx": round(scaling_ratio, 6),
            "scalePerPx": round(scaling_ratio / conversion_factor, 6)
        })

        _cache_put(cache_key, base_info)
        return base_info

    def load_analysis_data_from_record(self, record: Analysis, unit: str = "μm", scale: float = 500.0) -> Dict[str, Any]:
        """
        直接基于已从数据库查出的 Analysis 记录计算分析指标。

        该方法跳过二次查库和 os.walk 遍历，直接从 record 中取出
        original_file_path 与 result_json_path 后委托给 _compute_from_paths，
        适用于上层已完成持久化查询的场景，减少 I/O 开销。

        :param record: 已查询到的 Analysis ORM 实体，需包含原始文件和 JSON 文件路径。
        :param unit: 目标单位，默认 "μm"。
        :param scale: 标尺实际长度（微米），默认 500。
        :return: 完整的分析指标字典。
        :raises HTTPException 404: 当原始文件或 JSON 文件路径无效或文件不存在时。
        """
        original_file_path = backend_path(record.original_file_path) if record.original_file_path else None
        json_file_path = backend_path(record.result_json_path) if record.result_json_path else None

        if not original_file_path or not os.path.exists(original_file_path):
            raise HTTPException(status_code=404, detail=f"原始文件不存在: {original_file_path}")
        if not json_file_path or not os.path.exists(json_file_path):
            raise HTTPException(status_code=404, detail=f"分析结果文件不存在: {json_file_path}")

        return self._compute_from_paths(original_file_path, json_file_path, unit, scale)

    def _detect_scaler_from_file(self, image_path: str) -> float:
        """
        直接从原图文件中检测标尺像素长度。

        与旧实现的区别：只在内存中保留图片右下角的四分之一区域做检测，
        避免把整张高分辨率原图（例如 4000x3000）整体转成 numpy 数组，
        单张图片可节省几十 MB 的内存拷贝和相应的耗时。
        """
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return 500.0
            region = img.crop((width // 2, height // 2, width, height))
            if region.mode != "RGB":
                region = region.convert("RGB")
            img_arr = np.asarray(region)

        return self._scaler_from_region(img_arr)

    def _get_scaler(self, img_arr: np.ndarray, img_height: int, img_width: int) -> float:
        """
        通过检测图像右下角区域的红蓝色像素条，自动估算标尺在像素下的长度。

        原理：标尺通常由红色长条和蓝色长条组成，分别在 HSV/RGB 空间中筛选红色与蓝色，
        统计两者的像素和的最大值作为标尺像素长度；若检测失败则回退为 500.0。

        :param img_arr: 原图的 numpy 数组（H x W x 3，RGB 顺序）。
        :param img_height: 图像高度。
        :param img_width: 图像宽度。
        :return: 检测到的标尺像素长度，失败时返回 500.0。
        """
        return self._scaler_from_region(img_arr[img_height // 2:, img_width // 2:, :])

    def _scaler_from_region(self, region: np.ndarray) -> float:
        """
        在给定区域（通常是原图右下角）上统计红/蓝标尺像素，返回标尺像素长度。

        :param region: 区域像素数组（H x W x 3，RGB 顺序）。
        :return: 标尺像素长度；检测失败返回 500.0。
        """
        try:
            red_mask = (region[..., 0] >= 240) & (region[..., 1] <= 20) & (region[..., 2] <= 20)
            blue_mask = (region[..., 0] <= 20) & (region[..., 1] <= 20) & (region[..., 2] >= 240)
            red_pixels = np.sum(red_mask, axis=1)
            blue_pixels = np.sum(blue_mask, axis=1)
            scaler = float(np.max(red_pixels + blue_pixels))
            return scaler if scaler > 0 else 500.0
        except Exception as e:
            log.warning(f"检测标尺像素长度失败: {e}")
            return 500.0

    def _label_binarize(self, img_h: int, img_w: int, points: List[List[float]]) -> tuple:
        """
        将标注点多边形填充为二值图像，并进行形态学闭运算平滑边界，
        最终返回该多边形在像素空间中的面积和周长。

        流程：
        1. 创建全黑二值图，用 cv2.fillPoly 将 points 填充为白色。
        2. 使用 60x60 卷积核做 MORPH_CLOSE，填补小缺口。
        3. 统计非零像素数得到面积；findContours 后对外部轮廓计算 arcLength。

        优化：画布不再固定为整幅原图，而是刚好覆盖多边形包围盒（外扩半个卷积核），
        面积与周长结果完全一致，但在大图上可以少分配、少扫描几十倍的内存。

        :param img_h: 画布高度。
        :param img_w: 画布宽度。
        :param points: 多边形顶点列表，每项为 [x, y]。
        :return: (area_pixel, perimeter_pixel)，若点数不足或异常则返回 (0.0, 0.0)。
        """
        if len(points) < 3:
            return 0.0, 0.0
        try:
            pts_raw = np.asarray(points, dtype=np.float64)
            if pts_raw.ndim != 2 or pts_raw.shape[1] < 2:
                return 0.0, 0.0
            pts_raw = pts_raw[:, :2]
            pts_raw = pts_raw[np.isfinite(pts_raw).all(axis=1)]
            if len(pts_raw) < 3:
                return 0.0, 0.0

            # 与原实现完全一致：先截断成 int32，再做整数平移，
            # 保证光栅化结果与原「整幅画布」实现逐像素相同。
            pts = pts_raw.astype(np.int32)

            min_x = int(pts[:, 0].min()) - _CLOSE_PADDING
            min_y = int(pts[:, 1].min()) - _CLOSE_PADDING
            max_x = int(pts[:, 0].max()) + _CLOSE_PADDING
            max_y = int(pts[:, 1].max()) + _CLOSE_PADDING

            # 裁剪到图像范围内
            min_x = max(0, min(min_x, img_w - 1))
            min_y = max(0, min(min_y, img_h - 1))
            max_x = max(min_x, min(max_x, img_w - 1))
            max_y = max(min_y, min(max_y, img_h - 1))

            canvas_h = max_y - min_y + 1
            canvas_w = max_x - min_x + 1
            if canvas_h <= 0 or canvas_w <= 0:
                return 0.0, 0.0

            binary_image = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            shifted = pts - np.array([min_x, min_y], dtype=np.int32)
            cv2.fillPoly(binary_image, [shifted], 255)
            kernel = np.ones((_CLOSE_KERNEL, _CLOSE_KERNEL), np.uint8)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            area_pixel = float(np.count_nonzero(binary_image))
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and len(contours) > 0:
                perimeter_pixel = float(cv2.arcLength(contours[0], True))
            else:
                perimeter_pixel = 0.0
            return area_pixel, perimeter_pixel
        except Exception as e:
            log.warning(f"闭运算二值化失败: {e}")
            return 0.0, 0.0

    def _calculate_shape_area(self, points: List[List[float]]) -> float:
        """
        使用鞋带公式（Shoelace formula）计算任意简单多边形的面积（像素单位）。

        :param points: 多边形顶点列表，每项为 [x, y]，按顺时针或逆时针顺序排列。
        :return: 多边形面积；当点数少于 3 时返回 0.0。
        """
        if points is None or len(points) < 3:
            return 0.0

        # 说明：这里保留纯 Python 循环。实测 LabelMe 点集是 list[list]，
        # 转成 ndarray 的开销比循环本身还大（16~512 点范围内慢 1.5~8 倍），
        # 只有输入本来就是 ndarray 时才走向量化路径。
        if isinstance(points, np.ndarray):
            pts = points[:, :2]
            x = pts[:, 0]
            y = pts[:, 1]
            return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)

        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    def _calculate_diameter_from_area(self, area: float) -> float:
        """
        假设茎秆截面近似圆形，根据面积反推等效直径 d = 2 * sqrt(area / π)。

        :param area: 截面面积（像素或实际单位）。
        :return: 等效直径；area <= 0 时返回 0.0。
        """
        if area <= 0:
            return 0.0
        return 2 * math.sqrt(area / math.pi)

    def _calculate_perimeter_from_area(self, area: float) -> float:
        """
        假设茎秆截面近似圆形，根据面积反推等效周长 C = 2π * sqrt(area / π)。

        :param area: 截面面积（像素或实际单位）。
        :return: 等效周长；area <= 0 时返回 0.0。
        """
        if area <= 0:
            return 0.0
        return 2 * math.pi * math.sqrt(area / math.pi)

    def generate_excel_data(self, analysis_data: Dict[str, Any], selected_columns: List[str],
                            unit: str = "μm") -> pd.DataFrame:
        """
        根据用户选择的列从单条分析数据中抽取字段，构造单行 DataFrame 并将列名映射为中文。

        - 对浮点值保留 6 位小数；
        - 缺失字段以 "N/A" 填充；
        - 若 selected_columns 中包含未定义列，则抛出 400 错误。

        :param analysis_data: 单条样本的分析指标字典。
        :param selected_columns: 用户选中的英文字段名列表，需全部存在于 available_columns。
        :param unit: 导出单位，比例尺列名据此动态显示（比例尺(µm/px) 等）。
        :return: 列名为中文显示名的单行 DataFrame。
        """
        invalid_columns = [col for col in selected_columns if col not in self.available_columns]
        if invalid_columns:
            raise HTTPException(status_code=400, detail=f"无效的列选择: {invalid_columns}")

        selected_columns = self._ordered_columns(selected_columns)

        base_row = {}
        for col in selected_columns:
            if col in analysis_data:
                value = analysis_data[col]
                if isinstance(value, float):
                    base_row[col] = round(value, 6)
                else:
                    base_row[col] = value
            else:
                base_row[col] = "N/A"

        df = pd.DataFrame([base_row])

        column_mapping = {col: self.column_label(col, unit) for col in selected_columns}
        df = df.rename(columns=column_mapping)

        return df

    def create_excel_file(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        将给定 DataFrame 写入 Excel 文件并保存到存储目录下的 _exports 子目录。

        自动调整每列宽度（最大不超过 50 字符），工作表固定命名为 "分析结果"。
        若未指定文件名则以当前时间戳生成。

        :param df: 待导出的 pandas DataFrame。
        :param filename: 可选的输出文件名（含 .xlsx 后缀）。
        :return: 保存后的 Excel 文件绝对路径。
        """
        if filename is None:
            filename = f"{datetime.now().strftime('%Y.%m.%d_%H:%M')}.xlsx"

        export_dir = os.path.join(settings.STORAGE_PATH, "excel_export")
        os.makedirs(export_dir, exist_ok=True)

        file_path = os.path.join(export_dir, filename)

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='分析结果', index=False)

            worksheet = writer.sheets['分析结果']
            # 用 pandas 的向量化长度统计代替「遍历每个单元格」的双层循环
            for index, column in enumerate(df.columns, start=1):
                try:
                    max_length = int(df[column].astype(str).str.len().max() or 0)
                except Exception:
                    max_length = 0
                max_length = max(max_length, len(str(column)))
                worksheet.column_dimensions[get_column_letter(index)].width = min(max_length + 2, 50)

        log.info(f"Excel 文件已保存至: {file_path}")
        return file_path

    def export_all_to_excel(self, analysis_data_list: List[Dict[str, Any]], selected_columns: List[str],
                            filename: str = None, unit: str = "μm") -> str:
        """
        将多条样本的分析指标批量导出到一个 Excel 文件。

        行为与 generate_excel_data 类似，但处理多条记录，最后委托 create_excel_file 落盘。
        默认文件名格式：export_{样本数}_samples_{时间戳}.xlsx。

        :param analysis_data_list: 多条样本的分析指标字典列表。
        :param selected_columns: 用户选中的英文字段名列表，需全部存在于 available_columns。
        :param filename: 可选的输出文件名。
        :param unit: 导出单位，比例尺列名据此动态显示。
        :return: 保存后的 Excel 文件绝对路径。
        """
        invalid_columns = [col for col in selected_columns if col not in self.available_columns]
        if invalid_columns:
            raise HTTPException(status_code=400, detail=f"无效的列选择: {invalid_columns}")

        selected_columns = self._ordered_columns(selected_columns)

        rows = []
        for analysis_data in analysis_data_list:
            row = {}
            for col in selected_columns:
                if col in analysis_data:
                    value = analysis_data[col]
                    if isinstance(value, float):
                        row[col] = round(value, 6)
                    else:
                        row[col] = value
                else:
                    row[col] = "N/A"
            rows.append(row)

        df = pd.DataFrame(rows)

        column_mapping = {col: self.column_label(col, unit) for col in selected_columns}
        df = df.rename(columns=column_mapping)

        if filename is None:
            filename = f"export_{len(analysis_data_list)}_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return self.create_excel_file(df, filename)


excel_service = ExcelDownloadService()
