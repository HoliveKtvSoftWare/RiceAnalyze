"""
Excel下载服务
"""


import os
import json
import pandas as pd
import logging
import math
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from sqlmodel import create_engine, Session
from app.core.config import settings
from app.models.analysis import Analysis

log = logging.getLogger(__name__)


class ExcelDownloadService:
    def __init__(self):
        self.available_columns = {
            "filename": "样本名称",
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
            "um": 1.0,
            "μm": 1.0,
            "mm": 1000.0,
            "cm": 10000.0
        }

        sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
        self._sync_engine = create_engine(sync_db_url, echo=False)

    def get_available_columns(self) -> Dict[str, str]:
        """返回当前服务支持的所有可导出列名及其中文显示名映射。"""
        return self.available_columns

    def _compute_from_paths(self, original_file_path: str, json_file_path: str, unit: str, scale: float) -> Dict[str, Any]:
        """
        根据原始图像文件路径和标注结果 JSON 文件路径，计算一组完整的水稻茎秆分析指标。

        主要处理流程：
        1. 从 JSON 中读取标注数据（大/小维管束、茎秆外轮廓、空腔轮廓）。
        2. 通过 `_label_binarize` 或多边形面积公式计算茎秆面积、周长、空腔面积。
        3. 读取原图并调用 `_get_scaler` 自动检测标尺像素长度，推导像素到实际单位的缩放比例。
        4. 结合 `unit_conversion` 将像素面积/长度转换为用户指定单位（um/mm/cm）。
        5. 计算各类派生指标（总数、比值、差值等）并汇总返回。

        :param original_file_path: 原始图像的绝对路径，用于检测标尺像素长度。
        :param json_file_path: LabelMe 风格的标注结果 JSON 路径，包含 shapes 字段。
        :param unit: 目标单位字符串，支持 "um"、"μm"、"mm"、"cm"。
        :param scale: 用户指定的标尺实际长度（微米），默认 500。
        :return: 包含所有分析指标的字典，键名为英文字段名（与 available_columns 对应）。
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        img_h = analysis_data.get("imageHeight", 0)
        img_w = analysis_data.get("imageWidth", 0)

        base_info = {
            "filename": os.path.basename(json_file_path).replace('.json', '')
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
        cavity_area_pixel = 0.0

        if out_points and img_h > 0 and img_w > 0:
            stem_area_pixel, stem_perimeter_pixel = self._label_binarize(img_h, img_w, out_points)
            try:
                out_pts = np.array(out_points, dtype=np.int32)
                (_, _), radius = cv2.minEnclosingCircle(out_pts)
                stem_diameter_pixel = radius * 2.0
            except Exception:
                stem_diameter_pixel = self._calculate_diameter_from_area(stem_area_pixel)
        elif out_points:
            stem_area_pixel = self._calculate_shape_area(out_points)
            stem_perimeter_pixel = self._calculate_perimeter_from_area(stem_area_pixel)
            stem_diameter_pixel = self._calculate_diameter_from_area(stem_area_pixel)

        if in_points and img_h > 0 and img_w > 0:
            cavity_area_pixel, _ = self._label_binarize(img_h, img_w, in_points)
        elif in_points:
            cavity_area_pixel = self._calculate_shape_area(in_points)

        conversion_factor = self.unit_conversion.get(unit, 1.0)

        scaling_ratio = 1.0
        try:
            if os.path.exists(original_file_path):
                from PIL import Image
                img = Image.open(original_file_path)
                img_arr = np.array(img)
                scaler = self._get_scaler(img_arr, img.height, img.width)
                if scaler > 0:
                    scaling_ratio = scale / scaler
            else:
                scaling_ratio = scale / 500.0
        except Exception as e:
            log.warning(f"读取原图检测标尺失败，使用默认比例: {e}")
            scaling_ratio = scale / 500.0

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
            "largeCountPerimeterCavityRatio": large_count_perimeter_cavity_ratio
        })

        return base_info

    def load_analysis_data_from_record(self, record: Analysis, unit: str = "um", scale: float = 500.0) -> Dict[str, Any]:
        """
        直接基于已从数据库查出的 Analysis 记录计算分析指标。

        该方法跳过二次查库和 os.walk 遍历，直接从 record 中取出
        original_file_path 与 result_json_path 后委托给 _compute_from_paths，
        适用于上层已完成持久化查询的场景，减少 I/O 开销。

        :param record: 已查询到的 Analysis ORM 实体，需包含原始文件和 JSON 文件路径。
        :param unit: 目标单位，默认 "um"。
        :param scale: 标尺实际长度（微米），默认 500。
        :return: 完整的分析指标字典。
        :raises HTTPException 404: 当原始文件或 JSON 文件路径无效或文件不存在时。
        """
        original_file_path = record.original_file_path
        json_file_path = record.result_json_path

        if not original_file_path or not os.path.exists(original_file_path):
            raise HTTPException(status_code=404, detail=f"原始文件不存在: {original_file_path}")
        if not json_file_path or not os.path.exists(json_file_path):
            raise HTTPException(status_code=404, detail=f"分析结果文件不存在: {json_file_path}")

        return self._compute_from_paths(original_file_path, json_file_path, unit, scale)

    def load_analysis_data(self, analysis_id: str, user_id: str, unit: str = "um", scale: float = 500.0) -> Dict[str, Any]:
        """
        兼容旧接口：根据 analysis_id + user_id 查库，并在 JSON 路径缺失时通过
        os.walk 用户数据目录回退查找标注文件，最终调用 _compute_from_paths 计算指标。

        :param analysis_id: 分析记录在数据库中的唯一标识。
        :param user_id: 当前用户 ID，用于回退时限定搜索目录范围。
        :param unit: 目标单位，默认 "um"。
        :param scale: 标尺实际长度（微米），默认 500。
        :return: 完整的分析指标字典。
        :raises HTTPException 404: 当记录不存在、原始文件或 JSON 文件缺失时。
        :raises HTTPException 500: 其它未知异常。
        """
        try:
            from sqlmodel import select as _select
            with Session(self._sync_engine) as session:
                statement = _select(Analysis).where(Analysis.analysis_id == analysis_id)
                result = session.exec(statement)
                analysis_record = result.first()

            if not analysis_record:
                raise HTTPException(status_code=404, detail="分析记录不存在")

            original_file_path = analysis_record.original_file_path
            if not original_file_path:
                raise HTTPException(status_code=404, detail="原始文件路径不存在")

            json_file_path = analysis_record.result_json_path
            if not json_file_path or not os.path.exists(json_file_path):
                original_filename = os.path.basename(original_file_path)
                sample_name_with_uuid = os.path.splitext(original_filename)[0]
                if '_' in sample_name_with_uuid and len(sample_name_with_uuid.split('_')[0]) == 36:
                    sample_name = sample_name_with_uuid.split('_', 1)[1]
                else:
                    sample_name = sample_name_with_uuid

                user_data_dir = os.path.join(settings.STORAGE_PATH, str(user_id))
                for root, dirs, files in os.walk(user_data_dir):
                    for file in files:
                        if file.endswith('.json'):
                            json_file_path = os.path.join(root, file)
                            break
                    if json_file_path and os.path.exists(json_file_path):
                        break

            if not json_file_path or not os.path.exists(json_file_path):
                raise HTTPException(status_code=404, detail=f"分析结果文件不存在")

            return self._compute_from_paths(original_file_path, json_file_path, unit, scale)

        except HTTPException:
            raise
        except Exception as e:
            log.error(f"加载分析数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"加载分析数据失败: {str(e)}")

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
        try:
            rd = img_arr[img_height // 2:, img_width // 2:, :]
            red_mask = (rd[..., 0] >= 240) & (rd[..., 1] <= 20) & (rd[..., 2] <= 20)
            blue_mask = (rd[..., 0] <= 20) & (rd[..., 1] <= 20) & (rd[..., 2] >= 240)
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

        :param img_h: 画布高度。
        :param img_w: 画布宽度。
        :param points: 多边形顶点列表，每项为 [x, y]。
        :return: (area_pixel, perimeter_pixel)，若点数不足或异常则返回 (0.0, 0.0)。
        """
        if len(points) < 3:
            return 0.0, 0.0
        try:
            binary_image = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(binary_image, [pts], 255)
            kernel = np.ones((60, 60), np.uint8)
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
        if len(points) < 3:
            return 0.0

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

    def generate_excel_data(self, analysis_data: Dict[str, Any], selected_columns: List[str]) -> pd.DataFrame:
        """
        根据用户选择的列从单条分析数据中抽取字段，构造单行 DataFrame 并将列名映射为中文。

        - 对浮点值保留 6 位小数；
        - 缺失字段以 "N/A" 填充；
        - 若 selected_columns 中包含未定义列，则抛出 400 错误。

        :param analysis_data: 单条样本的分析指标字典。
        :param selected_columns: 用户选中的英文字段名列表，需全部存在于 available_columns。
        :return: 列名为中文显示名的单行 DataFrame。
        """
        invalid_columns = [col for col in selected_columns if col not in self.available_columns]
        if invalid_columns:
            raise HTTPException(status_code=400, detail=f"无效的列选择: {invalid_columns}")

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

        column_mapping = {col: self.available_columns[col] for col in selected_columns}
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

        export_dir = os.path.join(settings.STORAGE_PATH, "_exports")
        os.makedirs(export_dir, exist_ok=True)

        file_path = os.path.join(export_dir, filename)

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='分析结果', index=False)

            worksheet = writer.sheets['分析结果']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        log.info(f"Excel 文件已保存至: {file_path}")
        return file_path

    def export_all_to_excel(self, analysis_data_list: List[Dict[str, Any]], selected_columns: List[str], filename: str = None) -> str:
        """
        将多条样本的分析指标批量导出到一个 Excel 文件。

        行为与 generate_excel_data 类似，但处理多条记录，最后委托 create_excel_file 落盘。
        默认文件名格式：export_{样本数}_samples_{时间戳}.xlsx。

        :param analysis_data_list: 多条样本的分析指标字典列表。
        :param selected_columns: 用户选中的英文字段名列表，需全部存在于 available_columns。
        :param filename: 可选的输出文件名。
        :return: 保存后的 Excel 文件绝对路径。
        """
        invalid_columns = [col for col in selected_columns if col not in self.available_columns]
        if invalid_columns:
            raise HTTPException(status_code=400, detail=f"无效的列选择: {invalid_columns}")

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

        column_mapping = {col: self.available_columns[col] for col in selected_columns}
        df = df.rename(columns=column_mapping)

        if filename is None:
            filename = f"export_{len(analysis_data_list)}_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return self.create_excel_file(df, filename)


excel_service = ExcelDownloadService()