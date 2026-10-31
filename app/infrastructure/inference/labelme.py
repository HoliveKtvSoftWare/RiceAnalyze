"""LabelMe JSON conversion and polygon smoothing."""

import base64
import os

import cv2
import numpy as np

from app.domain.geometry.mask_geometry import extract_topology_contours


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def smooth_polygon_chaikin(points, iterations=2):
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return points
    for _ in range(iterations):
        new_pts = np.zeros((n * 2, 2), dtype=np.float64)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            new_pts[2 * i] = (3 * p1 + p2) / 4
            new_pts[2 * i + 1] = (p1 + 3 * p2) / 4
        pts = new_pts
        n = len(pts)
    return [(float(p[0]), float(p[1])) for p in pts.tolist()]


def smooth_polygon_midpoint(points, subdivisions=1):
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return points
    for _ in range(subdivisions):
        new_pts = np.zeros((n * 2, 2), dtype=np.float64)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            new_pts[2 * i] = p1
            new_pts[2 * i + 1] = (p1 + p2) / 2
        pts = new_pts
        n = len(pts)
    return [(float(p[0]), float(p[1])) for p in pts.tolist()]


def smooth_polygon_spline(points, num_points=None, smooth_factor=0.1):
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


def smooth_polygon(points, method="chaikin", iterations=2):
    if len(points) < 3:
        return points
    if method == "chaikin":
        return smooth_polygon_chaikin(points, iterations=iterations)
    if method == "midpoint":
        return smooth_polygon_midpoint(points, subdivisions=iterations)
    if method == "spline":
        return smooth_polygon_spline(points)
    return smooth_polygon_chaikin(points, iterations=iterations)


def prepare_preview_points(points, method="chaikin", iterations=2):
    if len(points) < 3:
        return list(points)
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = min(6.0, max(1.25, perimeter * 0.0005))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    prepared = simplified.tolist() if len(simplified) >= 3 else list(points)
    return smooth_polygon(prepared, method=method, iterations=iterations)


def convert_to_labelme_format(image_path, image_size, masks_list, labels_list, names,
                              smooth=True, smooth_method="chaikin", smooth_iterations=2,
                              embed_image=True, smooth_exclude=("out",),
                              raw_masks_list=None, topology_labels=(), shape_flags_list=None):
    data = {
        "version": "5.0.1", "flags": {}, "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageHeight": image_size[1], "imageWidth": image_size[0],
    }
    if embed_image:
        data["imageData"] = image_to_base64(image_path)
    topology_labels = tuple(topology_labels or ())
    raw_masks_list = raw_masks_list or [{} for _ in masks_list]
    shape_flags_list = shape_flags_list or [[] for _ in masks_list]
    for i, (masks_xy, boxes_cls) in enumerate(zip(masks_list, labels_list)):
        raw_masks = raw_masks_list[i] if i < len(raw_masks_list) else {}
        detection_flags = shape_flags_list[i] if i < len(shape_flags_list) else []
        for j, polygon in enumerate(masks_xy):
            label_value = boxes_cls[j]
            label_name = label_value if isinstance(label_value, str) else names[int(label_value)]
            base_flags = dict(detection_flags[j]) if j < len(detection_flags) else {}
            if label_name in topology_labels and j in raw_masks:
                rings = extract_topology_contours(raw_masks[j])
                if rings:
                    for ring in rings:
                        flags = dict(base_flags)
                        flags.update({"mask_topology": True, "contour_role": ring["role"],
                                      "component_id": ring["component_id"]})
                        data["shapes"].append({"label": label_name, "points": ring["points"],
                                                "group_id": j, "shape_type": "polygon", "flags": flags})
                    continue
            points = [(float(point[0]), float(point[1])) for point in polygon]
            if smooth and label_name not in smooth_exclude:
                points = smooth_polygon(points, method=smooth_method, iterations=smooth_iterations)
            data["shapes"].append({"label": label_name, "points": points, "group_id": None,
                                   "shape_type": "polygon", "flags": base_flags})
    return data
