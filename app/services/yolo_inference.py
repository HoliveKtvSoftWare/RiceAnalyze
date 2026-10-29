"""
YOLO 模型推理服务

性能要点：
1. 模型只加载一次（进程级缓存），避免每张图都重复读取权重 / 初始化 CUDA；
2. 推理在 torch.inference_mode() 下执行，关闭自动求导，减少显存与耗时；
3. 多边形平滑算法全部向量化（numpy），点数越多收益越明显；
4. LabelMe JSON 使用紧凑分隔符 + 可选的 orjson 写入，减少序列化开销。
"""

import os
import json
import base64
import logging
import sys
import sysconfig
import threading

import numpy as np
from PIL import Image, ImageDraw

# 项目根目录里还留着一份旧版 ultralytics（8.2.57）副本，运行时会被优先导入，
# 导致依赖新模块（如 YOLO11 的 C3k2）的权重加载失败：
#     AttributeError: Can't get attribute 'C3k2' ...
# requirements.txt.txt 固定的是 ultralytics==8.3.27，因此这里临时把 site-packages
# 挪到 sys.path 最前面完成导入，随后恢复原顺序（子模块走包的 __path__，不受影响）。
_SITE_PACKAGES = sysconfig.get_paths().get("purelib")
_ORIGINAL_SYS_PATH = None
if _SITE_PACKAGES:
    _normalized_site_packages = os.path.abspath(_SITE_PACKAGES)
    _rest = [
        entry for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != _normalized_site_packages
    ]
    if _rest != sys.path:
        _ORIGINAL_SYS_PATH = list(sys.path)
        sys.path = [_SITE_PACKAGES] + _rest
try:
    import ultralytics  # noqa: F401
    from ultralytics import YOLO
finally:
    if _ORIGINAL_SYS_PATH is not None:
        sys.path[:] = _ORIGINAL_SYS_PATH

import torch

# 剑叶 OUR 头适配：必须在上面把 site-packages 的 ultralytics 提到 sys.path 首部之后再导入
from app.services.leaf_head_adapter import OurSegmentationPredictor, adapt_our_model

try:  # orjson 速度约为标准库的 3~10 倍，缺失时自动回退
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    orjson = None

log = logging.getLogger(__name__)
log.info("ultralytics %s (%s)", getattr(ultralytics, "__version__", "unknown"),
         getattr(ultralytics, "__file__", "unknown"))

# --- 模型缓存 ---------------------------------------------------------------
# key: (模型绝对路径, 设备字符串)；value: 已加载并 .to(device) 的 YOLO 实例
_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.RLock()
_DEVICE = None


def _get_device():
    """只探测一次可用设备（torch.cuda.is_available() 每次调用都有开销）。"""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def get_model(model_path, device=None, warmup=True):
    """
    获取（并缓存）已加载到指定设备的 YOLO 模型。

    同一个模型文件在进程内只会被加载一次，后续推理直接复用，
    这对逐张图片的分析任务可以把单张耗时从“秒级”降到“模型前向”本身。

    Args:
        model_path: 模型权重文件路径
        device: 目标设备，None 时自动选择 CUDA / CPU
        warmup: 首次加载后是否做一次空跑预热（触发权重实际初始化与算子选取）

    Returns:
        ultralytics.YOLO 实例
    """
    device = device or _get_device()
    cache_key = (os.path.abspath(model_path), str(device))

    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    with _MODEL_CACHE_LOCK:
        # 双重检查：避免并发首次请求重复加载
        model = _MODEL_CACHE.get(cache_key)
        if model is not None:
            return model

        log.info(f"加载模型 {model_path} 到设备 {device}（仅首次需要）...")
        model = YOLO(model_path).to(device)

        # 剑叶 OUR 权重带 mask_refine，官方 Segment.forward 不执行它，会得到噪点掩膜。
        # 只替换该模型实例的 head 类；茎秆模型这里返回 False，行为不变。
        is_our = adapt_our_model(model)
        if is_our:
            log.info(f"模型 {os.path.basename(model_path)} 启用 OUR mask_refine 适配。")

        if warmup:
            try:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                with torch.inference_mode():
                    # predict() 只在首次调用时接受 predictor，预热必须一起传，
                    # 否则默认 SegmentationPredictor 会被缓存下来，后面无法替换。
                    model.predict(dummy, verbose=False, show=False, save=False,
                                  **({"predictor": OurSegmentationPredictor} if is_our else {}))
                log.info(f"模型 {os.path.basename(model_path)} 预热完成。")
            except Exception as warmup_error:  # 预热失败不应影响正式推理
                log.warning(f"模型预热失败（忽略）: {warmup_error}")

        _MODEL_CACHE[cache_key] = model
        return model


def preload_models(model_paths):
    """在后台线程中预加载模型，缩短用户第一次分析的等待时间。"""
    def _load():
        for path in model_paths:
            try:
                get_model(path)
            except Exception as error:  # pragma: no cover
                log.warning(f"预加载模型失败 {path}: {error}")

    thread = threading.Thread(target=_load, name="yolo-model-preload", daemon=True)
    thread.start()
    return thread


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
        # 向量化：一次性算出每条边的两个切割点，避免 Python 逐点循环
        nxt = np.roll(pts, -1, axis=0)
        new_pts = np.empty((n * 2, 2), dtype=np.float64)
        new_pts[0::2] = 0.75 * pts + 0.25 * nxt
        new_pts[1::2] = 0.25 * pts + 0.75 * nxt
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
        nxt = np.roll(pts, -1, axis=0)
        new_pts = np.empty((n * 2, 2), dtype=np.float64)
        new_pts[0::2] = pts
        new_pts[1::2] = (pts + nxt) / 2
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


def _segments_intersect(a1, a2, b1, b2):
    """
    判断两条线段 (a1-a2) 和 (b1-b2) 是否相交（不含端点重合）。
    返回 True 表示相交（自相交多边形的标志）。
    """
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def polygon_is_simple(points):
    """
    检测多边形是否为简单多边形（无自相交）。
    YOLO masks.xy 输出的轮廓点天然是按顺序排列的，
    但如果中途被重新排序（如极角排序）就会产生自相交。

    注意：本函数是 O(n²) 的。调用方（convert_to_labelme_format）会先用
    simplify_polygon 把上千个轮廓点降到几十个，再调用它，
    这里的点数上限只是兜底保护，避免异常输入拖慢整个请求。
    """
    n = len(points)
    if n < 4:
        return True
    if n > 1000:  # 点数过多时放弃自相交检测（YOLO 原始轮廓不会自相交）
        return True
    for i in range(n):
        a1 = points[i]
        a2 = points[(i + 1) % n]
        for j in range(i + 2, n):
            if j % n == i:
                continue
            b1 = points[j]
            b2 = points[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


def fix_polygon_order(points):
    """
    轻量级的多边形修复：仅当检测到自相交时才尝试修复。
    优先保留原始顺序（YOLO 输出本身就是正确的轮廓顺序），
    只有在极端情况下（点序真的乱了）才回退到凸包。

    注意：极角排序只对凸多边形安全，对非凸多边形一定会产生
    自相交（即用户截图中的"放射性"问题），因此不再作为修复手段。

    Args:
        points: 原始多边形顶点列表 [[x, y], ...]

    Returns:
        修复后的顶点列表
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return points

    if polygon_is_simple(points):
        return points

    import cv2
    hull = cv2.convexHull(pts, clockwise=False)
    return [(float(p[0][0]), float(p[0][1])) for p in hull]


def simplify_polygon(points, tolerance=2.0):
    """
    使用 Douglas-Peucker 算法简化多边形，在保留形状的同时大幅减少点数。

    Args:
        points: 原始多边形顶点列表 [[x, y], ...]
        tolerance: 简化容差（像素），值越大点越少

    Returns:
        简化后的顶点列表；点数不足 3 时返回空列表，由调用方决定回退或跳过。
    """
    import cv2
    if points is None or len(points) < 3:
        # 模型偶尔会输出退化掩膜（0~2 个点）。空数组直接丢给 cv2.approxPolyDP
        # 会触发断言 “npoints >= 0 && (depth == CV_32S || depth == CV_32F)”，
        # 之前表现为整条分析失败，这里提前返回。
        return []
    pts = np.array(points, dtype=np.float32)
    simplified = cv2.approxPolyDP(pts, tolerance, closed=True)
    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


def _components_polygons(mask_bitmap, min_component_area=64, min_component_area_ratio=0.001,
                         keep_largest_only=True):
    """把一张实例掩膜里的连通域清理干净，返回外轮廓多边形列表（原图坐标系）。

    默认只保留最大的连通域：一个检测实例只应产出一个多边形，
    否则同一颗维管束被拆成几块会抬高计数。保留多块的模式留给特殊场景。
    """
    import cv2

    count, component_map, stats, _ = cv2.connectedComponentsWithStats(mask_bitmap, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        if keep_largest_only:
            kept = (component_map == largest).astype(np.uint8)
        else:
            threshold = max(int(min_component_area),
                            int(np.ceil(float(areas.max()) * min_component_area_ratio)))
            keep = [index for index in range(1, count)
                    if int(stats[index, cv2.CC_STAT_AREA]) >= threshold]
            if largest not in keep:
                keep.append(largest)
            kept = np.zeros_like(mask_bitmap)
            for index in keep:
                kept[component_map == index] = 1
        mask_bitmap = kept

    contours = cv2.findContours(mask_bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygon_list = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon_list.append((points.astype(float), float(cv2.contourArea(contour))))
    return polygon_list


def _mask_polygons(masks, image_size, classes, filter_components=True, min_area=0):
    """把 ultralytics 的掩膜位图转成 (多边形, 类别索引) 列表。

    相比直接用 ``masks.xy``，这里基于位图做连通域清理，能顺带去掉
    几十像素的噪点，并且在 retina_masks=True 时得到更细的轮廓。
    """
    import cv2

    data = masks.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    data = np.asarray(data)
    width, height = int(image_size[0]), int(image_size[1])

    items = []
    for index, class_index in enumerate(classes):
        bitmap = (np.asarray(data[index]).squeeze() > 0).astype(np.uint8)
        if bitmap.ndim != 2 or not bitmap.any():
            continue
        if bitmap.shape != (height, width):
            bitmap = cv2.resize(bitmap, (width, height), interpolation=cv2.INTER_NEAREST)
            bitmap = (bitmap > 0).astype(np.uint8)

        if filter_components:
            polygons = _components_polygons(bitmap)
        else:
            contours = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            polygons = [(c.reshape(-1, 2).astype(float), float(cv2.contourArea(c)))
                        for c in contours if len(c.reshape(-1, 2)) >= 3]

        for points, area in polygons:
            if min_area and area < float(min_area):
                continue
            items.append((points, int(class_index)))
    return items


def convert_to_labelme_format(image_path, image_size, polygons, names,
                              smooth=True, smooth_method='chaikin', smooth_iterations=2):
    """
    将 YOLO 输出的掩膜多边形转换为 LabelMe JSON 格式。

    Args:
        image_path: 原图路径，用于读取图片数据和提取文件名
        image_size: 原图尺寸 (width, height)
        polygons: [(多边形顶点数组, 类别索引), ...]；一个实例可能贡献多个多边形
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
    for polygon, class_index in polygons:
        label_name = names[int(class_index)]
        raw_points = [(float(point[0]), float(point[1])) for point in polygon]

        # 退化掩膜（点数为 0~2）无法构成多边形，直接跳过：
        # 剑叶模型在大图上偶尔会给出这种空轮廓，保留它只会让下游几何计算出错。
        if len(raw_points) < 3:
            log.debug("跳过退化掩膜: label=%s, 点数=%d", label_name, len(raw_points))
            continue

        # 关键优化：先用 Douglas-Peucker 把上千个轮廓点降到几十个，
        # 再做自相交检测 / 平滑。fix_polygon_order 是 O(n²)，
        # 先简化可以让这一段的计算量下降几个数量级。
        points = simplify_polygon(raw_points, tolerance=2.0)
        if len(points) < 3:  # 简化过度时回退到原始点，避免产生非法多边形
            points = raw_points

        points = fix_polygon_order(points)

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


def dump_labelme_json(labelme_data, json_output_path):
    """
    高速写出 LabelMe JSON。

    - 有 orjson 时走 orjson（字节写入，速度约为标准库的 3~10 倍）；
    - 没有时回退到 json.dump，并使用紧凑分隔符：LabelMe 里 imageData
      是几 MB 的 base64 字符串，indent=4 只会白白放大文件体积。

    Args:
        labelme_data: LabelMe 格式字典
        json_output_path: 输出文件路径
    """
    if orjson is not None:
        with open(json_output_path, "wb") as json_file:
            json_file.write(orjson.dumps(labelme_data))
        return

    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(labelme_data, json_file, ensure_ascii=False, separators=(",", ":"))


def draw_polygons_on_image(image, shapes, colors=None):
    """
    在原图上根据 LabelMe shapes 数据绘制半透明多边形掩膜。

    默认（茎秆）按四类标签配色并先画外轮廓 out：
    - out: 红色
    - in: 绿色
    - small: 蓝色
    - big: 黄色

    传入 ``colors`` 时使用自定义配色（剑叶的 11 类标签），按 shapes 顺序直接绘制。

    Args:
        image: PIL Image 对象，原始图片
        shapes: LabelMe shapes 列表，每个 shape 包含 label、points 等字段
        colors: 可选，{标签名: RGBA 元组}；为 None 时使用茎秆默认配色

    Returns:
        绘制完掩膜后的 PIL Image 对象（RGB 模式）
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    default_colors = {
        "out": (255, 0, 0, 64),
        "in": (0, 255, 0, 64),
        "small": (0, 0, 255, 64),
        "big": (255, 255, 0, 64),
    }
    palette = colors or default_colors
    fallback_color = (255, 255, 0, 64)
    img_w, img_h = image.size

    def _normalize_points(points):
        """
        将点坐标统一转换为 PIL 可接受的格式：[(int x, int y), ...]，
        并裁剪到图像边界内，过滤无效坐标。

        使用 numpy 一次性完成取整 / 越界裁剪 / 非法值过滤，
        比逐点 Python 循环快一个数量级以上（平滑后的点通常有数百个）。

        Args:
            points: 原始坐标点列表，格式 [[x, y], ...]

        Returns:
            规范化后的整数坐标点列表，格式 [(x, y), ...]
        """
        if points is None or len(points) == 0:
            return []
        try:
            arr = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError):
            return []
        if arr.ndim != 2 or arr.shape[1] < 2:
            return []
        arr = arr[:, :2]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
            return []
        coords = np.rint(arr).astype(np.int64)
        coords[:, 0] = np.clip(coords[:, 0], 0, img_w - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, img_h - 1)
        return [tuple(int(v) for v in row) for row in coords.tolist()]

    if colors:
        # 自定义配色（剑叶）：所有标签按原顺序直接绘制，没有 out/in 的先后要求
        for shape in shapes:
            points = _normalize_points(shape.get("points"))
            color = palette.get(shape.get("label"), fallback_color)
            if len(points) >= 3:
                draw.polygon(points, fill=color)
    else:
        for shape in shapes:
            if shape['label'] == 'out':
                points = _normalize_points(shape['points'])
                color = palette.get('out', fallback_color)
                if len(points) >= 3:
                    draw.polygon(points, fill=color)
        for shape in shapes:
            if shape['label'] != 'out':
                points = _normalize_points(shape['points'])
                color = palette.get(shape['label'], fallback_color)
                if len(points) >= 3:
                    draw.polygon(points, fill=color)
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    return image.convert("RGB")


def run_system(model_path, image_path, output_path, output_basename,
               smooth=True, smooth_method='chaikin', smooth_iterations=2, colors=None,
               conf=None, iou=None, retina_masks=False,
               filter_small_components=False, min_mask_area_px=0):
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
        colors: 可选，{标签名: RGBA}；剑叶用自定义 11 类配色，None 时用茎秆默认配色
        conf: 置信度阈值，None 时用 ultralytics 默认值
        iou: NMS 的 IoU 阈值，None 时用 ultralytics 默认值（剑叶用 0.45，减少重叠实例）
        retina_masks: 是否输出高分辨率掩膜（轮廓更细，剑叶开启）
        filter_small_components: 是否清理掩膜内部的细小连通域（去噪点）
        min_mask_area_px: 丢弃面积小于该值（像素²）的实例，0 表示不过滤

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
                f"请检查 .env 中的模型路径配置（YOLO_MODELS / LEAF_MODELS），"
                f"或确认该模型文件已放置到正确路径。"
            )

        # 复用进程内已加载的模型，避免每张图都重新读取权重（最大的耗时来源）
        device = _get_device()
        model = get_model(model_path, device=device)

        log.info(f"开始对 {image_path} 进行推理...")
        predict_kwargs = dict(show=False, save=False, visualize=False, verbose=False)
        if getattr(model, "_rice_our", False):
            predict_kwargs["predictor"] = OurSegmentationPredictor
        if conf is not None:
            predict_kwargs["conf"] = float(conf)
        if iou is not None:
            predict_kwargs["iou"] = float(iou)
        if retina_masks:
            predict_kwargs["retina_masks"] = True

        # inference_mode 关闭 autograd，比默认模式更快、更省内存
        with torch.inference_mode():
            results = model.predict(image_path, **predict_kwargs)

        image = Image.open(image_path)
        image_size = image.size

        polygons = []
        for result in results:
            masks = result.masks
            if masks is None or getattr(masks, "xy", None) is None:
                log.warning(f"图片 {image_path} 未检测到任何掩码 (masks)。")
                continue
            classes = result.boxes.cls.tolist()
            if filter_small_components or min_mask_area_px:
                polygons.extend(_mask_polygons(
                    masks, image_size, classes,
                    filter_components=filter_small_components,
                    min_area=min_mask_area_px,
                ))
            else:
                for class_index, polygon in zip(classes, masks.xy):
                    polygons.append((np.asarray(polygon), int(class_index)))

        names = result.names

        log.info("推理完成，开始生成 LabelMe JSON...")
        if smooth:
            log.info(f"使用 {smooth_method} 方法平滑多边形边缘，迭代 {smooth_iterations} 次")
        labelme_data = convert_to_labelme_format(
            image_path, image_size, polygons, names,
            smooth=smooth, smooth_method=smooth_method, smooth_iterations=smooth_iterations
        )

        json_output_path = os.path.join(output_path, f"{output_basename}.json")
        dump_labelme_json(labelme_data, json_output_path)
        log.info(f"JSON 文件已保存到: {json_output_path}")

        log.info("开始绘制带掩膜的图片...")
        output_image = draw_polygons_on_image(image, labelme_data["shapes"], colors=colors)
        output_image_path = os.path.join(output_path, f"{output_basename}.jpg")
        output_image.save(output_image_path, quality=90)
        log.info(f"带掩膜的图片已保存到: {output_image_path}")

        return output_image_path, json_output_path

    except Exception as e:
        log.error(f"在 run_system 中发生严重错误: {e}", exc_info=True)
        raise e