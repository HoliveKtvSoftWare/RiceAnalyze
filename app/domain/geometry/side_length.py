"""Endpoint-to-endpoint side length, independent of body regions or landmarks.

Lengths are in original-image pixels. Unreliable paths return ``None`` together
with a reason; callers must not substitute zero or half the perimeter.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .mask_geometry import topology_geometry


@dataclass
class LengthMeasurement:
    length_px: Optional[float] = None
    reason: Optional[str] = None
    path: List = field(default_factory=list)
    # Retained field names for review artifacts: arbitrary endpoints A and B,
    # not anatomical root/tip or a match to annotated first/end labels.
    first: Optional[tuple] = None
    end: Optional[tuple] = None
    coverage: float = 0.0
    path_px: Optional[float] = None
    resolution: float = 1.0
    bundle_order: List = field(default_factory=list)
    bundle_centers: List = field(default_factory=list)
    bundle_waypoints: List = field(default_factory=list)
    centering_strength: float = 1.0
    smoothing: dict = field(default_factory=dict)
    navigation_coverage: float = 0.0


def select_region(shapes, label):
    """Match metric selection: largest net-area instance, preserving its rings."""
    candidates, groups = [], {}
    for shape in shapes:
        if shape.get("label") != label or len(shape.get("points", [])) < 3:
            continue
        flags = shape.get("flags") or {}
        if flags.get("mask_topology"):
            key = shape.get("group_id")
            groups.setdefault(key if key is not None else "topology", []).append(shape)
        else:
            candidates.append([shape])
    candidates.extend(groups.values())
    if not candidates:
        return []
    chosen = max(candidates, key=lambda rings: topology_geometry(rings)[0])
    return chosen


def _path_rings(rings):
    """Fill compact bundle voids, but preserve large/long structural gaps."""
    exterior_area = sum(abs(cv2.contourArea(np.asarray(r.get("points", []), dtype=np.float32)))
                        for r in rings if (r.get("flags") or {}).get("contour_role") != "hole")
    exterior_points = np.asarray([p for r in rings
                                  if (r.get("flags") or {}).get("contour_role") != "hole"
                                  for p in r.get("points", [])], dtype=np.float32)
    exterior_span = float(np.linalg.norm(np.ptp(exterior_points, axis=0))) if len(exterior_points) else 0
    selected = []
    for ring in rings:
        points = np.asarray(ring.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            continue
        if (ring.get("flags") or {}).get("contour_role") == "hole":
            # Individual small vascular voids should not divert the midline.
            # A large void between curled corridors is geometry, not a bundle.
            axes = cv2.minAreaRect(points.astype(np.float32))[1]
            major, minor = max(axes), max(1.0, min(axes))
            long_slit = major / minor >= 4 and major > 0.25 * exterior_span
            if abs(cv2.contourArea(points.astype(np.float32))) > exterior_area * 0.10 or long_slit:
                selected.append(ring)
            continue
        selected.append(ring)
    return selected


def _rasterize(rings, origin, scale, size, outline=False):
    result = np.zeros(size, dtype=np.uint8)
    selected = _path_rings(rings)
    for ring in sorted(selected, key=lambda r: (r.get("flags") or {}).get("contour_role") == "hole"):
        points = np.asarray(ring["points"], dtype=float)
        vertices = np.rint((points - origin) * scale).astype(np.int32)
        if outline:
            cv2.polylines(result, [vertices], True, 1, thickness=1)
        else:
            fill = 0 if (ring.get("flags") or {}).get("contour_role") == "hole" else 1
            cv2.fillPoly(result, [vertices], fill)
    return result.astype(bool)


def _extend(point, direction, rings, origin, scale, size):
    """Intersect the local tangent with original boundary segments, not a width cap.

    Using continuous polygon coordinates also preserves a taper narrower than
    the raster resolution. The first positive intersection cannot skip a fold
    wall or structural hole. No intersection is an unavailable measurement.
    """
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return None
    direction = direction / norm
    hits = []
    for ring in _path_rings(rings):
        a = (np.asarray(ring["points"], dtype=float) - origin) * scale
        edge = np.roll(a, -1, axis=0) - a
        delta = a - point
        cross = direction[0] * edge[:, 1] - direction[1] * edge[:, 0]
        valid = np.abs(cross) > 1e-9
        t = np.full(len(a), np.inf)
        u = np.full(len(a), np.inf)
        t[valid] = (delta[valid, 0] * edge[valid, 1] - delta[valid, 1] * edge[valid, 0]) / cross[valid]
        u[valid] = (delta[valid, 0] * direction[1] - delta[valid, 1] * direction[0]) / cross[valid]
        acceptable = valid & (t >= 0) & (u >= -1e-8) & (u <= 1 + 1e-8)
        hits.extend(t[acceptable].tolist())
    if not hits:
        return None
    result = point + direction * min(hits)
    x, y = result
    if not (-0.5 <= y < size[0] + 0.5 and -0.5 <= x < size[1] + 0.5):
        return None
    return result


def _straight_cap(xy, rings, origin, scale, width_distance):
    """Recover a straight end cap even when only one corner spur survives.

    Only a local boundary segment spanning the band width and transverse to
    the interior path qualifies. Rounded/tapered caps keep tangent extension.
    """
    point = xy[0]
    yx = np.rint(xy[:, ::-1]).astype(int)
    widths = width_distance[yx[:, 0], yx[:, 1]]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    near = arc <= max(6.0, 4 * float(widths.max()))
    half_width = float(widths[near].max())
    interior = min(len(xy) - 1, int(np.searchsorted(arc, 3 * half_width)))
    direction = point - xy[interior]
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return None
    direction /= norm
    candidates = []
    for ring in rings:
        if (ring.get("flags") or {}).get("contour_role") == "hole":
            continue
        vertices = (np.asarray(ring["points"], dtype=np.float32) - origin) * scale
        vertices = cv2.approxPolyDP(vertices.astype(np.float32), 0.75, True).reshape(-1, 2)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = b - a
            span = float(np.linalg.norm(edge))
            midpoint = (a + b) / 2
            if not half_width <= span <= 3 * half_width:
                continue
            if abs(float(np.dot(edge / span, direction))) > 0.7:
                continue
            gap = float(np.linalg.norm(midpoint - point))
            if gap <= 2 * half_width:
                candidates.append((gap, midpoint))
    if not candidates:
        return None
    midpoint = min(candidates, key=lambda item: item[0])[1]
    # Discard the corner tail until the medial path has moved a half-width
    # inside the cap. This removes corner detours, not an interior bend.
    offset = midpoint - xy
    gap = np.linalg.norm(offset, axis=1)
    eligible = np.flatnonzero((widths >= half_width * 0.85) &
                             (arc <= 3 * half_width))
    if not len(eligible):
        return None
    index = int(eligible[np.argmin(gap[eligible])])
    return midpoint.astype(float), index


def measure_side(side_rings, width, height, resolution=0.5, bundles=()):
    """Boundary-constrained full-interior traversal, without a skeleton."""
    from .bundle_path import measure_band
    return measure_band(side_rings, width, height, resolution, bundles)


def measure_leaf_sides(data, resolution=0.5):
    """Pure geometry, using model regions only; landmark labels are ignored."""
    shapes = data.get("shapes", [])
    return {label: measure_side(select_region(shapes, label),
                                data.get("imageWidth"), data.get("imageHeight"), resolution,
                                [s for s in shapes if s.get("label") in (label + "_small", label + "_big")])
            for label in ("side1", "side2")}
