"""Topology-preserving helpers for segmentation-mask post-processing."""

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


SIDE_BUNDLE_SWAP = {
    "side1_big": "side2_big",
    "side1_small": "side2_small",
    "side2_big": "side1_big",
    "side2_small": "side1_small",
}


def resize_binary_mask(mask, image_size=None) -> np.ndarray:
    """Return a uint8 binary mask, optionally resized to ``(width, height)``."""
    if hasattr(mask, "detach"):
        mask = mask.detach().byte().cpu().numpy()
    binary = (np.asarray(mask).squeeze() > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {binary.shape}")
    if image_size:
        width, height = (int(image_size[0]), int(image_size[1]))
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
            binary = (binary > 0).astype(np.uint8)
    return binary


def filter_small_components(
    mask,
    min_component_area: int = 16,
    min_component_area_ratio: float = 0.001,
) -> np.ndarray:
    """Remove tiny islands while always preserving the largest component."""
    binary = resize_binary_mask(mask)
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(areas)) + 1
    threshold = max(int(min_component_area), int(math.ceil(float(areas.max()) * min_component_area_ratio)))
    keep = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= threshold]
    if largest_index not in keep:
        keep.append(largest_index)

    cleaned = np.zeros_like(binary)
    for index in keep:
        cleaned[component_map == index] = 1
    return cleaned


def extract_topology_contours(
    mask,
    min_component_area: int = 16,
    min_component_area_ratio: float = 0.001,
) -> List[dict]:
    """Extract separate outer and hole rings without connecting components."""
    cleaned = filter_small_components(mask, min_component_area, min_component_area_ratio)
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []

    hierarchy = hierarchy[0]

    def depth_and_root(index: int) -> Tuple[int, int]:
        depth = 0
        root = index
        parent = int(hierarchy[index][3])
        while parent >= 0:
            depth += 1
            root = parent
            parent = int(hierarchy[parent][3])
        return depth, root

    candidates = []
    for index, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if len(points) < 3 or cv2.contourArea(contour) <= 0:
            continue
        depth, root = depth_and_root(index)
        candidates.append({
            "points": points.astype(float).tolist(),
            "role": "outer" if depth % 2 == 0 else "hole",
            "root": root,
            "area": float(cv2.contourArea(contour)),
        })

    roots = sorted(
        {item["root"] for item in candidates},
        key=lambda root: cv2.contourArea(contours[root]),
        reverse=True,
    )
    component_ids = {root: index for index, root in enumerate(roots)}
    candidates.sort(key=lambda item: (component_ids[item["root"]], item["role"] != "outer", -item["area"]))
    return [
        {
            "points": item["points"],
            "role": item["role"],
            "component_id": component_ids[item["root"]],
        }
        for item in candidates
    ]


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    shifted = np.roll(array, -1, axis=0)
    return abs(float(np.sum(array[:, 0] * shifted[:, 1] - shifted[:, 0] * array[:, 1]))) / 2.0


def _polygon_perimeter(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.roll(array, -1, axis=0) - array, axis=1).sum())


def topology_geometry(shapes: Iterable[Mapping]) -> Tuple[float, float]:
    """Calculate net area and total boundary length for topology-aware rings."""
    area = 0.0
    perimeter = 0.0
    for shape in shapes:
        points = shape.get("points", [])
        role = shape.get("flags", {}).get("contour_role", "outer")
        value = _polygon_area(points)
        area += -value if role == "hole" else value
        perimeter += _polygon_perimeter(points)
    return max(0.0, area), perimeter


def reassign_side_bundle_labels(
    labels: Sequence[str],
    boxes_xyxy,
    region_masks: Dict[str, np.ndarray],
) -> Tuple[List[str], Dict[int, str]]:
    """Conservatively reassign side bundles whose centers lie only in the other side."""
    corrected = list(labels)
    corrections: Dict[int, str] = {}
    side_support = {}
    side_contours = {}
    for side_label in ("side1", "side2"):
        if side_label not in region_masks or region_masks[side_label] is None:
            continue
        cleaned = filter_small_components(region_masks[side_label])
        contours = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            continue
        support = np.zeros_like(cleaned)
        cv2.drawContours(support, contours, -1, 1, thickness=cv2.FILLED)
        side_support[side_label] = support
        side_contours[side_label] = contours

    if len(side_support) != 2:
        return corrected, corrections

    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    for index, label in enumerate(labels):
        replacement = SIDE_BUNDLE_SWAP.get(label)
        if replacement is None or index >= len(boxes):
            continue
        own = "side1" if label.startswith("side1") else "side2"
        other = "side2" if own == "side1" else "side1"
        height, width = side_support[own].shape
        x1, y1, x2, y2 = boxes[index]
        x = max(0, min(width - 1, int(round((x1 + x2) / 2.0))))
        y = max(0, min(height - 1, int(round((y1 + y2) / 2.0))))
        inside_own = bool(side_support[own][y, x])
        inside_other = bool(side_support[other][y, x])
        if not inside_own and inside_other:
            corrections[index] = label
            corrected[index] = replacement
        elif not inside_own and not inside_other:
            point = (float(x), float(y))
            own_distance = max(cv2.pointPolygonTest(contour, point, True) for contour in side_contours[own])
            other_distance = max(cv2.pointPolygonTest(contour, point, True) for contour in side_contours[other])
            distance_margin = max(2.0, math.hypot(width, height) * 0.002)
            if other_distance - own_distance > distance_margin:
                corrections[index] = label
                corrected[index] = replacement
    return corrected, corrections
