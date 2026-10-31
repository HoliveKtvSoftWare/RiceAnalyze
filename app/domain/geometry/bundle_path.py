"""Boundary-constrained, vascular-guided band traversal without skeletons.

The full band interior is the navigation domain. Original contour polylines
remain walls even where two filled portions touch. Vascular centers provide
ordered navigation checkpoints. Whole-curve elastic centering treats those
points as soft guides, so the measured line need not visit each centroid.
"""
import math
from numbers import Real

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from .side_length import LengthMeasurement, _rasterize, _path_rings, _extend, _straight_cap


def _walk(cost, start, end=None, solver=None):
    solver = solver if solver is not None else MCP_Geometric(cost, fully_connected=True)
    distances, _ = solver.find_costs([tuple(start)], None if end is None else [tuple(end)])
    if end is None:
        return distances
    if not np.isfinite(distances[tuple(end)]):
        return None
    return np.asarray(solver.traceback(tuple(end)), dtype=float)[:, ::-1]


def _centers(shapes):
    centers = []
    for index, shape in enumerate(shapes):
        if (shape.get("flags") or {}).get("contour_role") == "hole":
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            continue
        moments = cv2.moments(points)
        center = (np.array([moments["m10"], moments["m01"]]) / moments["m00"]
                  if moments["m00"] else points.mean(axis=0))
        centers.append((index, center))
    return centers


def _nearest(points, mask):
    yy, xx = np.nonzero(mask)
    nodes = np.column_stack([xx, yy])
    tree = cKDTree(nodes)
    distance, index = tree.query(points)
    return nodes[index], distance


def _intersections(a, b, c, d, include_vertices=False):
    edge, other = b-a, d-c
    delta = c[None, :, :] - a[:, None, :]
    determinant = edge[:, None, 0]*other[None, :, 1] - edge[:, None, 1]*other[None, :, 0]
    valid = np.abs(determinant) > 1e-9
    safe = np.where(valid, determinant, 1)
    t = (delta[:, :, 0]*other[None, :, 1]-delta[:, :, 1]*other[None, :, 0])/safe
    u = (delta[:, :, 0]*edge[:, None, 1]-delta[:, :, 1]*edge[:, None, 0])/safe
    tolerance = 0.005/np.maximum(np.linalg.norm(edge, axis=1), 1e-9)
    if include_vertices:
        hits = valid & (t>=-tolerance[:, None]) & (t<=1+tolerance[:, None]) & (u>=-1e-8) & (u<=1+1e-8)
        # Collinear overlapping segments also constitute a revisit/contact.
        length2 = np.sum(edge*edge,axis=1)
        projection = np.sum(delta*edge[:,None,:],axis=2)/np.maximum(length2[:,None],1e-12)
        finish = projection + np.sum(other[None,:,:]*edge[:,None,:],axis=2)/np.maximum(length2[:,None],1e-12)
        line_gap = np.abs(delta[:,:,0]*edge[:,None,1]-delta[:,:,1]*edge[:,None,0])
        begin,end = np.maximum(0,np.minimum(projection,finish)),np.minimum(1,np.maximum(projection,finish))
        overlap = (~valid) & (line_gap<1e-8) & ((end-begin)>tolerance[:,None])
        t[overlap] = begin[overlap]
        hits |= overlap
    else:
        hits = valid & (t>tolerance[:, None]) & (t<1-tolerance[:, None]) & (u>=0) & (u<=1)
    return hits, t


def _walls(rings, origin, scale):
    points = [(np.asarray(r["points"], dtype=float)-origin)*scale for r in _path_rings(rings)]
    return np.concatenate(points), np.concatenate([np.roll(p,-1,axis=0) for p in points])


def _crosses_wall(xy, walls):
    a,b = walls
    for i in range(0,len(xy)-1,128):
        starts,ends = xy[:-1][i:i+128],xy[1:][i:i+128]
        low=np.minimum(starts.min(axis=0),ends.min(axis=0))-.01
        high=np.maximum(starts.max(axis=0),ends.max(axis=0))+.01
        nearby=np.all(np.maximum(a,b)>=low,axis=1)&np.all(np.minimum(a,b)<=high,axis=1)
        hits,t = _intersections(starts,ends,a[nearby],b[nearby],include_vertices=True)
        tolerance = 0.005/np.maximum(np.linalg.norm(ends-starts,axis=1),1e-9)
        if i==0:
            hits[0] &= t[0]>tolerance[0]
        if i+len(starts)==len(xy)-1:
            hits[-1] &= t[-1]<1-tolerance[-1]
        if hits.any():
            return True
    return False


def _outside_region(xy, rings, origin, scale):
    """Check vertices and segment interiors against continuous original rings."""
    points = np.vstack([xy,(xy[:-1]+xy[1:])/2])
    selected = _path_rings(rings)
    outers = [((np.asarray(r['points'],dtype=np.float32)-origin)*scale).astype(np.float32)
              for r in selected if (r.get('flags') or {}).get('contour_role')!='hole']
    holes = [((np.asarray(r['points'],dtype=np.float32)-origin)*scale).astype(np.float32)
             for r in selected if (r.get('flags') or {}).get('contour_role')=='hole']
    for point in points:
        p=tuple(float(v) for v in point)
        if not any(cv2.pointPolygonTest(r,p,True)>=-0.005 for r in outers):
            return True
        if any(cv2.pointPolygonTest(r,p,True)>0.005 for r in holes):
            return True
    return False


def _erase_local_loops(xy, half_width):
    """Erase short waypoint detours; refuse a large ambiguous revisited leg."""
    for _ in range(20):
        hits,t = _intersections(xy[:-1],xy[1:],xy[:-1],xy[1:],include_vertices=True)
        i,j = np.indices(hits.shape)
        pairs = np.argwhere(hits & (j>i+1))
        if not len(pairs):
            return xy
        left,right = pairs[0]
        crossing = xy[left] + t[left,right]*(xy[left+1]-xy[left])
        loop = np.vstack([crossing,xy[left+1:right+1],crossing])
        loop_length = np.linalg.norm(np.diff(loop,axis=0),axis=1).sum()
        if loop_length > max(12,8*half_width):
            return None
        xy = np.vstack([xy[:left+1],crossing,xy[right+1:]])
        xy = xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-6]]
        if len(xy)<2:
            return None
    return None


def _returns_without_wall(xy, rings, origin, scale):
    """Detect a separated reverse leg in the same open corridor.

    Actual U folds have an original contour wall between adjacent legs.
    A transverse correction inside a broad bulge must not manufacture such
    a leg. Use a geometric width estimate, independent of raster resolution.
    """
    boundaries = [(np.asarray(r['points'],dtype=float)-origin)*scale for r in _path_rings(rings)]
    area = sum((-1 if (r.get('flags') or {}).get('contour_role')=='hole' else 1)
               * abs(cv2.contourArea(p.astype(np.float32)))
               for r,p in zip(_path_rings(rings),boundaries))
    perimeter = sum(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum() for p in boundaries)
    width = max(3*scale,area/max(perimeter,1e-9))
    arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    positions = np.linspace(0,arc[-1],max(3,int(arc[-1]/(4*scale))+1))
    points = np.column_stack([np.interp(positions,arc,xy[:,axis]) for axis in (0,1)])
    tangent = np.gradient(points,axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    walls = _walls(rings,origin,scale)
    for left,right in cKDTree(points).query_pairs(width):
        if positions[right]-positions[left]<=4*width or np.dot(tangent[left],tangent[right])>-0.5:
            continue
        if not _crosses_wall(points[[left,right]],walls):
            return True
    return False


def _recenter(points, tangent, rings, origin, scale, radius, strength=1.0):
    """Center each vascular waypoint across its local transverse section.

    Intersect the original contour, never a neighboring strip's far edge.
    This does not depend on a precomputed shortest route or a skeleton.
    """
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
    edges = [(np.asarray(r["points"], dtype=float) - origin) * scale for r in _path_rings(rings)]
    a = np.concatenate(edges)
    edge = np.concatenate([np.roll(p, -1, axis=0) - p for p in edges])
    moved = points.copy().astype(float)
    for offset in range(0, len(points), 128):
        p, direction = points[offset:offset+128], normal[offset:offset+128]
        delta = a[None, :, :] - p[:, None, :]
        cross = direction[:, None, 0] * edge[None, :, 1] - direction[:, None, 1] * edge[None, :, 0]
        valid = np.abs(cross) > 1e-9
        safe = np.where(valid, cross, 1)
        t = (delta[:, :, 0] * edge[None, :, 1] - delta[:, :, 1] * edge[None, :, 0]) / safe
        u = (delta[:, :, 0] * direction[:, None, 1] - delta[:, :, 1] * direction[:, None, 0]) / safe
        on = valid & (u >= -1e-8) & (u <= 1+1e-8)
        forward = np.min(np.where(on & (t > 1e-6), t, np.inf), axis=1)
        backward = np.min(np.where(on & (t < -1e-6), -t, np.inf), axis=1)
        finite = np.isfinite(forward) & np.isfinite(backward)
        shift = np.zeros(len(p))
        shift[finite] = (forward[finite] - backward[finite]) / 2
        local_radius = radius[offset:offset+128]
        # A broad asymmetric bulge is not a reliable transverse section.
        # Keep its vascular center; elsewhere bound the correction so nearby
        # points cannot jump to opposite sides of a large bundle.
        bounded = finite & (forward + backward <= 4 * local_radius)
        shift = np.clip(shift, -local_radius, local_radius)
        moved[offset:offset+len(p)][bounded] += strength * direction[bounded] * shift[bounded, None]
    return moved


def _adjustment_is_ordered(reference, candidate, radius):
    """A local correction stays near navigation and advances along its steps."""
    if np.any(np.linalg.norm(candidate-reference,axis=1)>0.5*radius+1e-6):
        return False
    previous=np.diff(reference,axis=0)
    changed=np.diff(candidate,axis=0)
    active=np.linalg.norm(previous,axis=1)>1e-6
    if np.any(np.sum(previous[active]*changed[active],axis=1)<-1e-6):
        return False
    length=np.linalg.norm(previous,axis=1).sum()
    # Short sections may legitimately lengthen when moving from an inside
    # corner towards the band center. Bound this by the local corridor width,
    # rather than rejecting every change above a percentage of a short leg.
    allowance=max(0.02*length,float(np.median(radius)))
    return np.linalg.norm(changed,axis=1).sum()<=length+allowance+1e-6


def _center_route(xy, rings, origin, scale, half_width, wall_distance, checkpoints=(), strength=1.0):
    """Adjust adjoining vascular sections while preserving their anchors."""
    original = xy.copy()
    walls = _walls(rings, origin, scale)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    positions = np.linspace(0, arc[-1], max(3, int(arc[-1] / 2) + 1))
    xy = np.column_stack([np.interp(positions, arc, xy[:, axis]) for axis in (0, 1)])
    anchors = [0,len(xy)-1]
    if len(checkpoints):
        _, indices = cKDTree(xy).query(np.asarray(checkpoints))
        anchors = sorted(set(anchors + indices.tolist()))
    else:
        # Unmarked bands still use local sections, so one tapered end does
        # not discard the safe centering of an entire long band.
        anchors = sorted(set(anchors + np.searchsorted(
            positions,np.arange(0,arc[-1],max(10,4*half_width))).tolist()))
    for _ in range(2):
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        window = max(3, 1.5 * half_width)
        behind = np.column_stack([np.interp(np.maximum(0, arc-window), arc, xy[:, axis]) for axis in (0, 1)])
        ahead = np.column_stack([np.interp(np.minimum(arc[-1], arc+window), arc, xy[:, axis]) for axis in (0, 1)])
        vertices = np.rint(xy[:, ::-1]).astype(int)
        radius = np.maximum(wall_distance[vertices[:, 0], vertices[:, 1]], half_width / 2)
        centered = _recenter(xy, ahead-behind, rings, origin, scale, radius, strength)
        backward, forward = xy-behind, ahead-xy
        alignment = np.sum(backward * forward, axis=1) / np.maximum(
            np.linalg.norm(backward, axis=1) * np.linalg.norm(forward, axis=1), 1e-9)
        centered[alignment < 0.8] = xy[alignment < 0.8]
        # Endpoint corners are handled by the cap routine below.
        centered[0], centered[-1] = xy[0], xy[-1]
        smoothed = gaussian_filter1d(centered, 1.5, axis=0, mode="nearest")
        smoothed[0], smoothed[-1] = xy[0], xy[-1]
        # Shared anchors can move a bounded amount. If either adjoining
        # section is unsafe, revert its anchors and recheck its neighbors.
        pending = list(zip(anchors[:-1],anchors[1:]))
        while pending:
            rejected=[]
            for left,right in pending:
                candidate = smoothed[left:right+1]
                if (not _adjustment_is_ordered(xy[left:right+1],candidate,2*radius[left:right+1])
                        or _crosses_wall(candidate,walls)
                        or _outside_region(candidate,rings,origin,scale)):
                    if np.any(candidate!=xy[left:right+1]):
                        rejected.append((left,right))
            if not rejected:
                break
            for left,right in rejected:
                smoothed[left:right+1]=xy[left:right+1]
            pending=list(zip(anchors[:-1],anchors[1:]))
        xy = smoothed
    simplified = cv2.approxPolyDP(xy.astype(np.float32), 0.25, False).reshape(-1,2).astype(float)
    simplified = _erase_local_loops(simplified, half_width)
    if (simplified is None or _crosses_wall(simplified,walls)
            or _outside_region(simplified,rings,origin,scale)
            or _returns_without_wall(simplified,rings,origin,scale)):
        return original
    xy = simplified
    return xy


def _revisited(xy, size, half_width):
    """Reject repeated navigation, allowing local overlap at adjoining legs."""
    vertices = np.rint(xy).astype(int)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    first_visit = np.full(size, np.inf)
    repeat = 0.0
    for i, (x, y) in enumerate(vertices):
        if arc[i] - first_visit[y, x] > max(6, 4 * half_width):
            repeat += arc[i] - arc[max(0, i-1)]
        first_visit[y, x] = min(first_visit[y, x], arc[i])
    return repeat > max(6, arc[-1] * 0.02)


def _band_coverage(xy,mask,interior,wall_distance,scale):
    route_mask=np.zeros(mask.shape,dtype=np.uint8)
    cv2.polylines(route_mask,[np.rint(xy).astype(np.int32)],False,1,1)
    gap,nearest=distance_transform_edt(~route_mask.astype(bool),return_indices=True)
    width=wall_distance[nearest[0],nearest[1]]
    uncovered=interior & (gap>np.maximum(1.5*width,6*scale))
    return 1-float(uncovered.sum())/int(mask.sum())


def _finish(xy, rings, origin, scale, size, wall_distance, mask):
    """Use actual transverse caps or symmetric tangent-boundary intersections."""
    start_cap = _straight_cap(xy, rings, origin, scale, wall_distance)
    end_cap = _straight_cap(xy[::-1], rings, origin, scale, wall_distance)
    walls = _walls(rings,origin,scale)
    if start_cap and _crosses_wall(np.asarray([start_cap[0],xy[start_cap[1]]]),walls):
        start_cap = None
    if end_cap and _crosses_wall(np.asarray([xy[-1-end_cap[1]],end_cap[0]]),walls):
        end_cap = None
    left = start_cap[1] if start_cap else 0
    right = len(xy) - end_cap[1] if end_cap else len(xy)
    xy = xy[left:right]
    if len(xy) < 3:
        return None
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    vertices = np.rint(xy[:, ::-1]).astype(int)
    window = max(6 * scale, 2 * float(np.median(wall_distance[vertices[:, 0], vertices[:, 1]])))
    a = max(2, min(len(xy) // 3, int(np.searchsorted(arc, window))))
    back = arc[-1] - arc[::-1]
    b = max(2, min(len(xy) // 3, int(np.searchsorted(back, window))))
    first = start_cap[0] if start_cap else _extend(xy[0], xy[0] - xy[a], rings, origin, scale, size)
    end = end_cap[0] if end_cap else _extend(xy[-1], xy[-1] - xy[-1-b], rings, origin, scale, size)
    if first is None or end is None:
        return None
    extended = np.vstack([first, xy, end])
    return None if _crosses_wall(extended,walls) else extended


def measure_band(rings, width, height, resolution=0.5, bundles=()):
    result = _measure_band(rings,width,height,resolution,bundles)
    if result.reason == 'path_revisited' and len(bundles):
        # Reduce only the centering correction. The retry must independently
        # pass every coverage, wall, endpoint and revisit check.
        conservative = _measure_band(rings,width,height,resolution,bundles,0.5)
        if conservative.length_px is not None:
            return conservative
    return result


def _measure_band(rings, width, height, resolution=0.5, bundles=(), strength=1.0):
    result = LengthMeasurement()
    result.centering_strength = strength
    if not rings:
        result.reason = "side_missing"
        return result
    if width is None or height is None:
        result.reason = "image_dimensions_missing"
        return result
    if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
           for v in (width, height)):
        result.reason = "image_dimensions_invalid"
        return result
    points = np.asarray([p for r in rings for p in r.get("points", [])], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        result.reason = "invalid_region"
        return result
    if not isinstance(resolution, Real) or not math.isfinite(resolution) or not 0 < resolution <= 1:
        raise ValueError("resolution must be in (0, 1]")
    origin = np.maximum(0, np.floor(points.min(axis=0)) - 40)
    maximum = np.minimum([width - 1, height - 1], np.ceil(points.max(axis=0)) + 40)
    extent = maximum - origin
    if np.any(extent <= 0):
        result.reason = "empty_region"
        return result
    scale = min(float(resolution), math.sqrt(1500000 / max(1, np.prod(extent))))
    result.resolution = scale
    size = (int(extent[1] * scale) + 1, int(extent[0] * scale) + 1)
    mask = _rasterize(rings, origin, scale, size)
    if not mask.any():
        result.reason = "empty_region"
        return result
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    chosen = int(np.argmax(areas)) + 1
    if (areas.sum() - areas.max()) / areas.sum() > 0.03:
        result.reason = "side_fragmented"
        return result
    mask &= labels == chosen
    outline = _rasterize(rings, origin, scale, size, outline=True)
    wall_distance = distance_transform_edt(~outline)
    interior = mask & (wall_distance >= max(1.5, 3 * scale))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(interior.astype(np.uint8), 4)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        result.reason = "interior_empty"
        return result
    chosen = int(np.argmax(areas)) + 1
    interior &= labels == chosen
    if (areas.sum() - areas.max()) / areas.sum() > 0.03:
        result.reason = "interior_fragmented"
        return result
    half_width = float(np.median(wall_distance[interior]))
    # Routing prefers the band interior but uses every interior pixel. No
    # thinning, medial axis, skeleton graph or skeleton endpoint detection.
    cost = np.where(interior, 1 + (half_width / np.maximum(wall_distance, 1)) ** 2, np.inf)
    coordinate_cost = np.where(interior, 1.0, np.inf)
    centers = _centers(bundles)
    vascular_radii={}
    for index,center in centers:
        points=np.asarray(bundles[index]['points'],dtype=np.float32)
        radius=math.sqrt(abs(cv2.contourArea(points))/math.pi)*scale
        vascular_radii[index]=radius
    checkpoint_xy, checkpoint_ids = [], []
    if centers:
        local = np.asarray([c for _, c in centers])
        snapped, gaps = _nearest((local - origin) * scale, interior)
        for (index, center), point, gap in zip(centers, snapped, gaps):
            if gap <= max(3 * scale, half_width):
                checkpoint_xy.append(point)
                checkpoint_ids.append(index)
    checkpoint_xy = np.asarray(checkpoint_xy, dtype=int).reshape(-1, 2)
    result.bundle_centers = [tuple((p / scale + origin).tolist()) for p in checkpoint_xy]
    # Deterministic seed ignores input order and vascular big/small type.
    if len(checkpoint_xy):
        seed = checkpoint_xy[np.lexsort((checkpoint_xy[:, 1], checkpoint_xy[:, 0]))[0]][::-1]
    else:
        seed = np.asarray(np.unravel_index(np.argmax(wall_distance * interior), size))
    # Two full-interior geodesic sweeps locate the two ends. Candidate pixels
    # exclude the wall so cap corners cannot create an artificial branch.
    distances = _walk(coordinate_cost, seed)
    first = np.asarray(np.unravel_index(np.argmax(np.where(interior, distances, -1)), size))
    distances = _walk(coordinate_cost, first)
    end = np.asarray(np.unravel_index(np.argmax(np.where(interior, distances, -1)), size))
    # Geodesic distance sorts the detections within the original boundary
    # walls. Connecting nearby centers across a fold is never permitted.
    if len(checkpoint_xy):
        values = distances[checkpoint_xy[:, 1], checkpoint_xy[:, 0]]
        order = np.lexsort((checkpoint_xy[:, 1], checkpoint_xy[:, 0], values))
        result.bundle_order = [checkpoint_ids[i] for i in order]
        ordered = checkpoint_xy[order].astype(float)
        if len(ordered) >= 2:
            tangent = np.empty_like(ordered)
            tangent[0], tangent[-1] = ordered[1]-ordered[0], ordered[-1]-ordered[-2]
            if len(ordered) > 2:
                tangent[1:-1] = ordered[2:]-ordered[:-2]
            radius = np.full(len(ordered), max(half_width, 3))
            ordered = _recenter(ordered, tangent, rings, origin, scale, radius, strength)
        ordered, _ = _nearest(ordered, interior)
        # Transverse corrections retain geodesic order before the separate,
        # boundary-checked local guide projection around large bundles.
        for _ in range(len(ordered)):
            progress=distances[ordered[:,1],ordered[:,0]]
            conflicts=np.flatnonzero(np.diff(progress)<-1e-6)
            if not len(conflicts):
                break
            affected=np.unique(np.r_[conflicts,conflicts+1])
            ordered[affected]=checkpoint_xy[order][affected]
        # A small detection beside a large vascular bulge is a guide, not a
        # reason to leave the local route through the larger bundle center.
        # Project it only onto that adjoining section, with a clear original
        # boundary connection and a displacement bounded by the bulge size.
        large=np.array([vascular_radii[checkpoint_ids[k]]>=half_width for k in order])
        walls=_walls(rings,origin,scale)
        for i in range(1,len(ordered)-1):
            if large[i]:
                continue
            if large[i-1] and i+1<len(ordered):
                start,finish=ordered[i-1],ordered[i+1]
                big_radius=vascular_radii[checkpoint_ids[order[i-1]]]
            elif large[i+1] and i>0:
                start,finish=ordered[i-1],ordered[i+1]
                big_radius=vascular_radii[checkpoint_ids[order[i+1]]]
            else:
                continue
            if min(np.linalg.norm(ordered[i]-start),np.linalg.norm(ordered[i]-finish))>2*big_radius:
                continue
            bounds=np.asarray([start,ordered[i],finish])
            margin=max(2*half_width,big_radius)
            low=np.maximum(0,np.floor(bounds.min(axis=0)-margin).astype(int))
            high=np.minimum([size[1]-1,size[0]-1],np.ceil(bounds.max(axis=0)+margin).astype(int))
            local_cost=cost[low[1]:high[1]+1,low[0]:high[0]+1]
            route=_walk(local_cost,(start-low)[::-1].astype(int),(finish-low)[::-1].astype(int))
            if route is None:
                continue
            route+=low
            gap=np.linalg.norm(route-ordered[i],axis=1)
            candidate=route[np.argmin(gap)]
            if (gap.min()<=2*big_radius and not _crosses_wall(np.asarray([ordered[i],candidate]),walls)
                    and not _crosses_wall(route,walls)):
                ordered[i]=candidate.astype(int)
        checkpoints = np.vstack([first[::-1], ordered, end[::-1]])
        result.bundle_waypoints = (ordered / scale + origin).tolist()
        pieces = []
        # find_costs resets its search state. Reuse this per-measurement solver
        # instead of rebuilding full-image routing buffers for every bundle.
        navigator = MCP_Geometric(cost, fully_connected=True)
        for a, b in zip(checkpoints[:-1], checkpoints[1:]):
            if np.array_equal(a, b):
                continue
            piece = _walk(cost, a[::-1], b[::-1], solver=navigator)
            if piece is None:
                result.reason = "no_path"
                return result
            pieces.extend(piece[:-1].tolist())
        xy = np.asarray(pieces + [end[::-1].tolist()])
    else:
        xy = _walk(cost, first, end)
    if xy is None or len(xy) < 3:
        result.reason = "no_path"
        return result
    # Match the reference repair policy: local checkpoint retraces may be
    # removed by centering and soft smoothing. Only the repaired route is
    # judged for repeated traversal; final wall and no-return checks remain.
    from .soft_curve import center_soft_route
    xy = _center_route(xy,rings,origin,scale,half_width,wall_distance,
                       ordered if len(checkpoint_xy) else (),strength)
    result.navigation_coverage = _band_coverage(xy,mask,interior,wall_distance,scale)
    if result.navigation_coverage < .95:
        result.reason = 'path_incomplete'
        return result
    if len(checkpoint_xy):
        # Stabilize true automatic caps before fixing elastic endpoints;
        # raster extrema can otherwise pin an entire tapered corner.
        xy = _finish(xy,rings,origin,scale,size,wall_distance,mask)
        if xy is None:
            result.reason = 'extension_unresolved'
            return result
        xy, projected, result.smoothing = center_soft_route(
            xy, rings, origin, scale, half_width, wall_distance, ordered, strength)
        result.bundle_waypoints = (projected / scale + origin).tolist()
    if _revisited(xy, size, half_width):
        result.reason = "path_revisited"
        return result
    # Validate the whole band, including regions without vascular detections.
    result.coverage = _band_coverage(xy,mask,interior,wall_distance,scale)
    # Area reach is diagnostic after soft centering: a legitimate through-
    # curve need not enter every vascular bulge. Complete navigation plus
    # bounded, wall-safe curve displacement preserves the traversed corridor.
    if result.navigation_coverage < 0.95:
        result.reason = "path_incomplete"
        return result
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if length ** 2 / max(1, mask.sum()) < 2:
        result.reason = "region_not_band"
        return result
    if not len(checkpoint_xy):
        xy = _finish(xy, rings, origin, scale, size, wall_distance, mask)
    if xy is None:
        result.reason = "extension_unresolved"
        return result
    # Remove grid steps without altering fold topology or drawing across walls.
    walls = _walls(rings,origin,scale)
    accepted = False
    for tolerance in (0.15,0.05,0):
        # Zero tolerance keeps the actual float64 curve. Even an "unsimplified"
        # float32 copy can nudge a near-wall point across an original edge.
        simplified = (cv2.approxPolyDP(xy.astype(np.float32), tolerance, False).reshape(-1,2).astype(float)
                      if tolerance else xy.copy())
        if not _crosses_wall(simplified,walls) and not _outside_region(simplified,rings,origin,scale):
            accepted = True
            break
    if not accepted:
        result.reason = "boundary_crossed"
        return result
    xy = _erase_local_loops(simplified,half_width)
    if xy is None or _returns_without_wall(xy,rings,origin,scale):
        result.reason = "path_revisited"
        return result
    result.length_px = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) / scale
    result.path_px = result.length_px
    xy = xy / scale + origin
    result.path = xy.tolist()
    result.first, result.end = tuple(xy[0]), tuple(xy[-1])
    return result
