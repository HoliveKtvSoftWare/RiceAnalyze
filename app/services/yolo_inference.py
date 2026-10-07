"""
YOLO 模型推理服务
"""

import os
import json
import base64
import logging
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch

log = logging.getLogger(__name__)


def image_to_base64(image_path):
    """
    将指定路径的图片文件读取并转换为 base64 编码字符串。

    Args:
        image_path: 图片文件的绝对路径

    Returns:
        base64 编码后的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def smooth_polygon_chaikin(points, iterations=2):
    """
    Chaikin 角点切割算法：对每条边插入两个新点，
    将一条边拆分为两条边，使多边形边缘更加平滑。

    Args:
        points: 多边形顶点列表，格式 [[x1, y1], [x2, y2], ...]
        iterations: 迭代次数，默认 2 次

    Returns:
        平滑后的多边形顶点列表
    """
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return points

    for _ in range(iterations):
        new_pts = np.zeros((n * 2, 2), dtype=np.float64)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            new_pts[2 * i] = (3 * p1 + p2) / 4
            new_pts[2 * i + 1] = (p1 + 3 * p2) / 4
        pts = new_pts
        n = len(pts)

    return [(float(p[0]), float(p[1])) for p in pts.tolist()]


def smooth_polygon_midpoint(points, subdivisions=1):
    """
    中点细分算法：对每条边插入一个中点，将一条边拆分为两条等长的边。

    Args:
        points: 多边形顶点列表
        subdivisions: 细分级数，每级点数翻倍

    Returns:
        平滑后的多边形顶点列表
    """
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return points

    for _ in range(subdivisions):
        new_pts = np.zeros((n * 2, 2), dtype=np.float64)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            new_pts[2 * i] = p1
            new_pts[2 * i + 1] = (p1 + p2) / 2
        pts = new_pts
        n = len(pts)

    return [(float(p[0]), float(p[1])) for p in pts.tolist()]


def smooth_polygon_spline(points, num_points=None, smooth_factor=0.1):
    """
    B 样条插值平滑：使用 scipy 的 splprep 对多边形进行参数化样条插值，
    生成一条穿过原始点的平滑曲线，再从曲线上均匀采样得到更多点。

    Args:
        points: 多边形顶点列表
        num_points: 输出点数，默认自动计算
        smooth_factor: 平滑因子，值越大越平滑

    Returns:
        平滑后的多边形顶点列表
    """
    from scipy.interpolate import splprep, splev

    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return points

    if num_points is None:
        num_points = max(n * 3, 32)

    closed_pts = np.vstack([pts, pts[0:1]])

    tck, _ = splprep([closed_pts[:, 0], closed_pts[:, 1]], s=smooth_factor, per=True, k=3)

    u_new = np.linspace(0, 1, num_points, endpoint=False)
    x_new, y_new = splev(u_new, tck)

    return [(float(x), float(y)) for x, y in zip(x_new.tolist(), y_new.tolist())]


def smooth_polygon(points, method='chaikin', iterations=2):
    """
    多边形平滑主入口，支持三种方法：
    1. 'chaikin'  - Chaikin 角点切割
    2. 'midpoint' - 中点细分（最简单，边缘保持直线但更密集）
    3. 'spline'   - B 样条插值（最平滑，曲线最自然）

    Args:
        points: 原始多边形顶点列表 [[x, y], ...]
        method: 平滑方法选择，可选 'chaikin'、'midpoint'、'spline'
        iterations: 迭代次数或细分级数

    Returns:
        平滑后的多边形顶点列表
    """
    if len(points) < 3:
        return points

    if method == 'chaikin':
        return smooth_polygon_chaikin(points, iterations=iterations)
    elif method == 'midpoint':
        return smooth_polygon_midpoint(points, subdivisions=iterations)
    elif method == 'spline':
        return smooth_polygon_spline(points)
    else:
        return smooth_polygon_chaikin(points, iterations=iterations)


def reorder_polygon_by_angle(points):
    """
    将多边形顶点按质心极角排序，修复 YOLO 输出乱序导致的自相交/乱穿线问题。

    Args:
        points: 原始多边形顶点列表 [[x, y], ...]

    Returns:
        按极角排序后的顶点列表
    """
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 3:
        return points
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in order]


def simplify_polygon(points, tolerance=2.0):
    """
    使用 Douglas-Peucker 算法简化多边形，在保留形状的同时大幅减少点数。

    Args:
        points: 原始多边形顶点列表 [[x, y], ...]
        tolerance: 简化容差（像素），值越大点越少

    Returns:
        简化后的顶点列表
    """
    import cv2
    pts = np.array(points, dtype=np.float32)
    simplified = cv2.approxPolyDP(pts, tolerance, closed=True)
    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


def convert_to_labelme_format(image_path, image_size, masks_list, labels_list, names,
                              smooth=True, smooth_method='chaikin', smooth_iterations=2):
    """
    将 YOLO 输出的掩码/标签数据转换为 LabelMe JSON 格式。

    Args:
        image_path: 原图路径，用于读取图片数据和提取文件名
        image_size: 原图尺寸
        masks_list: 掩码坐标列表，每个元素为一组多边形点
        labels_list: 标签索引列表，每个元素对应 masks_list 中的类别索引
        names: 类别名称字典，键为索引，值为类别名称
        smooth: 是否对多边形边缘进行平滑处理，默认 True
        smooth_method: 平滑方法，可选 'chaikin'、'midpoint'、'spline'
        smooth_iterations: 平滑迭代次数，越大边缘越平滑

    Returns:
        LabelMe 格式的字典，包含 version、flags、shapes、imagePath、imageData 等字段
    """
    image_data = image_to_base64(image_path)
    labelme_format = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": image_size[1],
        "imageWidth": image_size[0]
    }
    for i, (masks_xy, boxes_cls) in enumerate(zip(masks_list, labels_list)):
        for j, polygon in enumerate(masks_xy):
            cls = int(boxes_cls[j])
            label_name = names[int(cls)]
            points = [(float(point[0]), float(point[1])) for point in polygon]

            points = reorder_polygon_by_angle(points)
            points = simplify_polygon(points, tolerance=2.0)

            if smooth:
                points = smooth_polygon(points, method=smooth_method, iterations=smooth_iterations)
            shape = {
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_format["shapes"].append(shape)
    return labelme_format


def draw_polygons_on_image(image, shapes):
    """
    在原图上根据 LabelMe shapes 数据绘制半透明多边形掩膜，区分四类标签颜色：
    - out: 红色
    - in: 绿色
    - small: 蓝色
    - big: 黄色

    Args:
        image: PIL Image 对象，原始图片
        shapes: LabelMe shapes 列表，每个 shape 包含 label、points 等字段

    Returns:
        绘制完掩膜后的 PIL Image 对象（RGB 模式）
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "out": (255, 0, 0, 64),
        "in": (0, 255, 0, 64),
        "small": (0, 0, 255, 64),
        "big": (255, 255, 0, 64),
    }
    img_w, img_h = image.size

    def _normalize_points(points):
        """
        将点坐标统一转换为 PIL 可接受的格式：[(int x, int y), ...]，
        并裁剪到图像边界内，过滤无效坐标。

        Args:
            points: 原始坐标点列表，格式 [[x, y], ...]

        Returns:
            规范化后的整数坐标点列表，格式 [(x, y), ...]
        """
        normalized = []
        for p in points:
            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                normalized.append((x, y))
            except (TypeError, ValueError, IndexError):
                continue
        return normalized

    for shape in shapes:
        if shape['label'] == 'out':
            points = _normalize_points(shape['points'])
            color = colors.get('out', (255, 255, 0, 64))
            if len(points) >= 3:
                draw.polygon(points, fill=color)
    for shape in shapes:
        if shape['label'] != 'out':
            points = _normalize_points(shape['points'])
            color = colors.get(shape['label'], (255, 255, 0, 64))
            if len(points) >= 3:
                draw.polygon(points, fill=color)
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    return image.convert("RGB")


def run_system(model_path, image_path, output_path, output_basename,
               smooth=True, smooth_method='chaikin', smooth_iterations=2):
    """
    完整 YOLO 推理流程：加载模型、执行推理、生成 LabelMe JSON 标注文件和带掩膜的可视化图片。

    Args:
        model_path: YOLO 模型权重文件路径
        image_path: 输入图片的路径
        output_path: 输出目录路径
        output_basename: 输出文件名，JSON 和 JPG 文件均使用此名称
        smooth: 是否对多边形边缘进行平滑处理，默认 True
        smooth_method: 平滑方法，可选 'chaikin'、'midpoint'、'spline'
        smooth_iterations: 平滑迭代次数，默认 2

    Returns:
        元组 (output_image_path, json_output_path)，分别为输出图片和 JSON 文件的完整路径

    Raises:
        Exception: 推理过程中发生的任何异常
    """
    try:
        log.info(f"开始推理任务... 模型: {model_path}, 图片: {image_path}")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请检查 .env 中 YOLO_MODELS 的配置，或确认该模型文件已放置到正确路径。"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"使用的设备: {device}")
        model = YOLO(model_path).to(device)

        log.info(f"模型加载成功，开始对 {image_path} 进行推理...")
        results = model(image_path, show=False, save=False, visualize=False)

        image = Image.open(image_path)

        masks_list = []
        labels_list = []

        for result in results:
            boxes = result.boxes
            masks = result.masks
            if masks is None:
                log.warning(f"图片 {image_path} 未检测到任何掩码 (masks)。")
                masks_xy = []
            else:
                masks_xy = masks.xy
            boxes_cls = boxes.cls
            masks_list.append(masks_xy)
            labels_list.append(boxes_cls)

        names = result.names
        image_size = image.size

        log.info("推理完成，开始生成 LabelMe JSON...")
        if smooth:
            log.info(f"使用 {smooth_method} 方法平滑多边形边缘，迭代 {smooth_iterations} 次")
        labelme_data = convert_to_labelme_format(
            image_path, image_size, masks_list, labels_list, names,
            smooth=smooth, smooth_method=smooth_method, smooth_iterations=smooth_iterations
        )

        json_output_path = os.path.join(output_path, f"{output_basename}.json")
        with open(json_output_path, "w") as json_file:
            json.dump(labelme_data, json_file, indent=4)
        log.info(f"JSON 文件已保存到: {json_output_path}")

        log.info("开始绘制带掩膜的图片...")
        output_image = draw_polygons_on_image(image, labelme_data["shapes"])
        output_image_path = os.path.join(output_path, f"{output_basename}.jpg")
        output_image.save(output_image_path)
        log.info(f"带掩膜的图片已保存到: {output_image_path}")

        return output_image_path, json_output_path

    except Exception as e:
        log.error(f"在 run_system 中发生严重错误: {e}", exc_info=True)
        raise e