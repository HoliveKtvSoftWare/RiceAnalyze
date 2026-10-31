"""Conservative, image-supported refinement of side regions, in source pixels.

No skeleton or watershed. Smoothing preserves topology; bridge removal is a
separate, audited operation permitted only when foreground stays connected.
"""
from dataclasses import dataclass
import copy

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from app.domain.geometry.mask_geometry import extract_topology_contours, filter_small_components

VERSION = 'side-conservative-1.0'


@dataclass
class RefinedSide:
    mask: np.ndarray
    rings: list
    audit: dict


def topology(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return (0, 0)
    depths = []
    for i in range(len(contours)):
        depth, parent = 0, int(hierarchy[0,i,3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0,parent,3])
        depths.append(depth)
    return (sum(d%2 == 0 for d in depths), sum(d%2 == 1 for d in depths))


def rasterize(rings, shape):
    mask = np.zeros(shape, np.uint8)
    # Each component is drawn separately so a hole cannot erase an island.
    for component in sorted({r['component_id'] for r in rings}):
        layer = np.zeros(shape, np.uint8)
        contours = [np.rint(r['points']).astype(np.int32).reshape(-1,1,2)
                    for r in rings if r['component_id']==component]
        # OpenCV's even-odd contour fill preserves the hole's foreground border.
        cv2.drawContours(layer,contours,-1,1,cv2.FILLED)
        mask |= layer
    return mask


def same_components(before,after,connectivity=8):
    """Every original component overlaps exactly one distinct final component."""
    old_count,old = cv2.connectedComponents(before.astype(np.uint8),connectivity=connectivity)
    new_count,new = cv2.connectedComponents(after.astype(np.uint8),connectivity=connectivity)
    if old_count!=new_count:
        return False
    both = (old>0)&(new>0)
    pairs = np.unique(old[both].astype(np.int64)*new_count+new[both])
    return (len(pairs)==old_count-1 and
            len(np.unique(pairs//new_count))==old_count-1 and
            len(np.unique(pairs%new_count))==new_count-1)


def same_topology(before,after):
    # Foreground8/background4 is the digital topology used by RETR_TREE rings.
    return (topology(before)==topology(after) and same_components(before,after) and
            same_components(1-before,1-after,connectivity=4))


def ring_area(rings):
    area = 0.0
    for ring in rings:
        points = np.asarray(ring['points'],np.float64)
        other = np.roll(points,-1,axis=0)
        value = abs(float(np.sum(points[:,0]*other[:,1]-other[:,0]*points[:,1])))/2
        area += -value if ring['role']=='hole' else value
    return area


def _color_evidence(rgb, mask, inside, outside):
    """Learn a saturation separator locally; abstain on low-contrast staining."""
    saturation = cv2.cvtColor(np.asarray(rgb,np.uint8), cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32)
    saturation = gaussian_filter(saturation, 1.0)
    core = saturation[(mask > 0) & (inside >= 8)]
    background = saturation[(mask == 0) & (outside >= 8) & (outside <= 28)]
    if len(core) < 100 or len(background) < 100:
        return None, {'color_evidence': 'insufficient_samples'}
    tissue_q = float(np.percentile(core, 25))
    # Upper tails contain neighboring folded tissue; use the background bulk.
    background_q = float(np.percentile(background, 70))
    gap = tissue_q-background_q
    audit = dict(color_evidence='usable' if gap >= 24 else 'low_contrast',
                 tissue_saturation_q25=tissue_q, background_saturation_q70=background_q)
    if gap < 24:
        return None, audit
    # Negative means clear background, positive means tissue-like.
    evidence = np.clip((saturation-(background_q+gap*.30))/max(gap*.35,8), -1, 1)
    return evidence, audit


def _remove_background_bridges(mask, probability, evidence, inside):
    current = mask.copy()
    audit = dict(bridge_candidates=0, bridges_removed=0, bridge_removed_pixels=0,
                 bridge_rejections={})
    if probability is None or evidence is None:
        audit['bridge_skip'] = 'missing_probability' if probability is None else 'insufficient_image_evidence'
        return current, audit
    # Only thin low-score background corridors, never an entire erosion.
    candidate = ((mask > 0) & (inside <= 12) & (evidence < -.35) &
                 (probability < .90)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    max_removal = int(mask.sum()*.02)
    old_components, old_holes = topology(current)
    # Largest candidates first, deterministic. Small boundary specks need no cut.
    for index in sorted(range(1,count), key=lambda i: (-int(stats[i,cv2.CC_STAT_AREA]),i))[:128]:
        area = int(stats[index,cv2.CC_STAT_AREA])
        if area < 6:
            continue
        audit['bridge_candidates'] += 1
        if area+audit['bridge_removed_pixels'] > max_removal:
            reason = 'area_limit'
        else:
            proposed = current.copy()
            proposed[labels == index] = 0
            components, holes = topology(proposed)
            # Pure removal cannot merge foreground components; equal counts
            # therefore also preserve their identities at this stage.
            if components != old_components:
                reason = 'would_disconnect_tissue'
            elif holes >= old_holes:
                reason = 'no_false_loop_opened'
            else:
                # A cellular pinhole is not sufficient evidence of a folded gap.
                _,old_background,old_stats,_ = cv2.connectedComponentsWithStats(1-current,connectivity=4)
                _,new_background = cv2.connectedComponents(1-proposed,connectivity=4)
                exterior = new_background[0,0]
                opened_labels = np.unique(old_background[(old_background>0)&(new_background==exterior)])
                exterior_old = old_background[0,0]
                opened_areas = [int(old_stats[i,cv2.CC_STAT_AREA]) for i in opened_labels if i!=exterior_old]
                if not opened_areas or max(opened_areas)<max(256,int(mask.sum()*.002)):
                    reason = 'only_small_holes_opened'
                    audit['bridge_rejections'][reason] = audit['bridge_rejections'].get(reason,0)+1
                    continue
                current = proposed
                old_holes = holes
                audit['bridges_removed'] += 1
                audit['bridge_removed_pixels'] += area
                continue
        audit['bridge_rejections'][reason] = audit['bridge_rejections'].get(reason,0)+1
    return current, audit


def _smooth_ring(points, shape):
    points = np.asarray(points,np.float64)
    closed = np.vstack((points,points[0]))
    distances = np.linalg.norm(np.diff(closed,axis=0),axis=1)
    cumulative = np.r_[0,np.cumsum(distances)]
    if cumulative[-1] < 32:
        return points.tolist()
    parameter = np.linspace(0,cumulative[-1],max(8,int(np.ceil(cumulative[-1]/1.5))),endpoint=False)
    sampled = np.column_stack([np.interp(parameter,cumulative,closed[:,axis]) for axis in (0,1)])
    smoothed = gaussian_filter1d(sampled, 2.0, axis=0, mode='wrap')
    smoothed[:,0] = np.clip(smoothed[:,0],0,shape[1]-1)
    smoothed[:,1] = np.clip(smoothed[:,1],0,shape[0]-1)
    # Mild simplification removes redundant dense points without big corner cuts.
    return cv2.approxPolyDP(smoothed.astype(np.float32).reshape(-1,1,2), .20, True).reshape(-1,2).astype(float).tolist()


def has_crossings(rings):
    """Reject proper segment crossings, using a grid to bound comparisons."""
    grid,edges = {},[]
    def cross(a,b):
        return a[0]*b[1]-a[1]*b[0]
    for ring_index,ring in enumerate(rings):
        points = np.asarray(ring['points'],np.float64)
        for index,a in enumerate(points):
            b = points[(index+1)%len(points)]
            low = np.floor(np.minimum(a,b)/16).astype(int)
            high = np.floor(np.maximum(a,b)/16).astype(int)
            cells = [(x,y) for x in range(low[0],high[0]+1) for y in range(low[1],high[1]+1)]
            seen = {j for cell in cells for j in grid.get(cell,())}
            for j in seen:
                ri,ei,n,c,d = edges[j]
                if ri==ring_index and (abs(index-ei)<=1 or abs(index-ei)==len(points)-1):
                    continue
                ab,cd = b-a,d-c
                u,v = cross(ab,c-a),cross(ab,d-a)
                w,z = cross(cd,a-c),cross(cd,b-c)
                if u*v < -1e-9 and w*z < -1e-9:
                    return True
            current = len(edges)
            edges.append((ring_index,index,len(points),a,b))
            for cell in cells:
                grid.setdefault(cell,[]).append(current)
    return False


def refine_side(mask, probability, rgb):
    raw = filter_small_components(mask)
    audit = dict(version=VERSION, bridges_removed=0, input_pixels=int(raw.sum()))
    if not raw.any():
        return RefinedSide(raw, [], dict(audit,status='empty'))
    if np.asarray(rgb).shape[:2] != raw.shape:
        raise ValueError('side后处理的原图与掩膜尺寸不一致')
    if probability is not None:
        probability = np.asarray(probability,np.float32)
        if probability.shape != raw.shape or not np.isfinite(probability).all():
            raise ValueError('side后处理掩膜分数无效')
    # Work on a padded ROI; includes real surrounding background at image edges.
    ys,xs = np.nonzero(raw)
    x0,x1 = max(0,int(xs.min())-32), min(raw.shape[1],int(xs.max())+33)
    y0,y1 = max(0,int(ys.min())-32), min(raw.shape[0],int(ys.max())+33)
    local = np.pad(raw[y0:y1,x0:x1],2)
    scores = np.pad(probability[y0:y1,x0:x1],2) if probability is not None else None
    image = np.pad(np.asarray(rgb)[y0:y1,x0:x1],((2,2),(2,2),(0,0)),mode='edge')
    inside = cv2.distanceTransform(local,cv2.DIST_L2,5)
    outside = cv2.distanceTransform(1-local,cv2.DIST_L2,5)
    evidence,color_audit = _color_evidence(image,local,inside,outside)
    audit.update(color_audit)
    cut,bridge_audit = _remove_background_bridges(local,scores,evidence,inside)
    audit.update(bridge_audit)
    target_topology = topology(cut)
    signed = cv2.distanceTransform(cut,cv2.DIST_L2,5)-cv2.distanceTransform(1-cut,cv2.DIST_L2,5)
    field = gaussian_filter(signed,1.6)
    if evidence is not None and scores is not None:
        # Image-supported displacement <=1.5 px, confined to uncertain boundary.
        uncertainty = np.clip(1-2*np.abs(scores-.5),0,1)
        field += 1.5*evidence*uncertainty*(np.abs(signed)<=6)
    proposed = (field > 0).astype(np.uint8)
    # Never grow through the gap that was explicitly removed.
    proposed[(local > 0) & (cut == 0)] = 0
    baseline_area = int(cut.sum())
    if not same_topology(cut,proposed) or abs(int(proposed.sum())/baseline_area-1) > .015:
        proposed = cut
        audit['smoothing_fallback'] = 'topology_or_area_limit'
    rings = extract_topology_contours(proposed,min_component_area=1,min_component_area_ratio=0)
    cut_rings = extract_topology_contours(cut,min_component_area=1,min_component_area_ratio=0)
    cut_float_area = ring_area(cut_rings)
    # Keep hole geometry exactly; smoothing the large exterior must not silently
    # fill small vascular voids. Outer rings carry the visible boundary noise.
    smooth_rings = [dict(r,points=_smooth_ring(r['points'],proposed.shape)
                         if r['role']=='outer' else r['points']) for r in rings]
    smooth_mask = rasterize(smooth_rings,proposed.shape)
    edge = cv2.morphologyEx(cut,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    edge_distance = cv2.distanceTransform(1-edge,cv2.DIST_L2,5)
    vertices = np.vstack([r['points'] for r in smooth_rings]) if smooth_rings else np.empty((0,2))
    index = np.rint(vertices).astype(int)
    movement = float(edge_distance[index[:,1],index[:,0]].max()) if len(index) else 0
    final_edge = cv2.morphologyEx(smooth_mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    reverse_distance = cv2.distanceTransform(1-final_edge,cv2.DIST_L2,5)
    movement = max(movement,float(reverse_distance[edge>0].max()))
    reverted = False
    if (not same_topology(cut,smooth_mask) or
            abs(int(smooth_mask.sum())/baseline_area-1) > .02 or movement > 6 or
            (cut_float_area>0 and abs(ring_area(smooth_rings)/cut_float_area-1)>.02)):
        smooth_rings = cut_rings
        smooth_mask = cut
        audit['smoothing_fallback'] = 'final_geometry_guard'
        reverted = True
    elif has_crossings(smooth_rings):
        smooth_rings = cut_rings
        smooth_mask = cut
        audit['smoothing_fallback'] = 'polygon_crossing_guard'
        reverted = True
    # Restore global coordinates. Preserve every retained original component.
    for ring in smooth_rings:
        ring['points'] = [[float(np.clip(x+x0-2,0,raw.shape[1]-1)),
                           float(np.clip(y+y0-2,0,raw.shape[0]-1))] for x,y in ring['points']]
    output = rasterize(smooth_rings,raw.shape)
    audit.update(status='processed', output_pixels=int(output.sum()),
                 pixel_area_change_ratio=float(int(output.sum())/int(raw.sum())-1),
                 changed_pixels=int(np.count_nonzero(output != raw)),
                 topology_before=list(topology(raw)), topology_after=list(topology(output)),
                 foreground_components_preserved=same_components(raw,output),
                 smoothing_topology_preserved=same_topology(cut,smooth_mask),
                 smoothing_float_area_change_ratio=(ring_area(smooth_rings)/cut_float_area-1) if cut_float_area else 0.0,
                 candidate_boundary_movement_px=movement,
                 max_boundary_movement_px=0.0 if reverted else movement)
    audit['review_reasons'] = ([audit['smoothing_fallback']] if 'smoothing_fallback' in audit else [])
    if audit.get('bridge_rejections',{}).get('would_disconnect_tissue'):
        audit['review_reasons'].append('background_candidates_retained_to_protect_connectivity')
    return RefinedSide(output,smooth_rings,audit)


def refine_annotation(data, masks, scores, rgb):
    """Replace only side rings; all other annotations remain byte-for-byte data."""
    result = copy.deepcopy(data)
    shapes, audit, handled = [], [], set()
    for shape in data['shapes']:
        label, index = shape['label'], shape.get('group_id')
        if label not in ('side1','side2') or index not in masks:
            shapes.append(copy.deepcopy(shape))
            continue
        if (label,index) in handled:
            continue
        handled.add((label,index))
        refined = refine_side(masks[index],scores.get(index),rgb)
        audit.append(dict(refined.audit,label=label,instance=index))
        for ring in refined.rings:
            flags = dict(shape.get('flags',{}),mask_topology=True,
                         contour_role=ring['role'],component_id=ring['component_id'],
                         side_postprocess=VERSION)
            shapes.append(dict(shape,points=ring['points'],flags=flags))
    result['shapes'] = shapes
    result['side_postprocess'] = dict(version=VERSION,regions=audit)
    return result
