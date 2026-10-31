"""Local, image-supported body contour refinement in original image pixels.

Only existing outer rings move. Hole polygons remain verbatim. Local normal
search uses color transitions and continuous model scores, with displacement,
ownership, topology and area guards. This module does not perform a graph cut.
"""
import copy
from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, map_coordinates

from app.domain.geometry.mask_geometry import extract_topology_contours
from .side_postprocess import rasterize, ring_area, same_topology, topology, has_crossings

VERSION = 'body-local-contour-1.0'
PARAMETERS = dict(sample_step_px=2.0, max_movement_px=8.0, min_color_step=8.0,
                  smoothing_sigma_samples=1.25, max_area_change=.05,
                  max_changed_fraction=.15, local_rollback_halo_samples=4)


@dataclass
class RefinedBody:
    mask: np.ndarray
    rings: list
    audit: dict
    seeds: np.ndarray


def _sample(field, points):
    coordinates=np.array([points[...,1],points[...,0]],dtype=np.float64)
    # cv2.remap uses short integer dimensions (<32767), including the number
    # of requested boundary points. SciPy supports long and wide source data.
    if field.ndim==2:
        return map_coordinates(field,coordinates,order=1,mode='nearest',prefilter=False)
    return np.stack([map_coordinates(field[:,:,i],coordinates,order=1,mode='nearest',prefilter=False)
                     for i in range(field.shape[2])],axis=-1)


def _resample(points, step):
    points=np.asarray(points,np.float64)
    # Keep every original corner as an anchor. Uniform global resampling
    # skips short corners: reverting such samples still changes protected
    # pixels. Subdivide each original edge instead, so a reverted arc is exact.
    following=np.roll(points,-1,axis=0)
    counts=np.maximum(1,np.ceil(np.linalg.norm(following-points,axis=1)/step).astype(int))
    return np.concatenate([a+(b-a)*np.arange(count)[:,None]/count
                           for a,b,count in zip(points,following,counts)],axis=0)


def _polygon_points(points):
    # Remove exactly collinear subdivision samples BEFORE integer rasterizing.
    # Otherwise rounding a point on a diagonal edge can change a protected
    # boundary by one pixel even though its arc has already been reverted.
    before=points-np.roll(points,1,axis=0)
    after=np.roll(points,-1,axis=0)-points
    cross=before[:,0]*after[:,1]-before[:,1]*after[:,0]
    keep=(np.abs(cross)>1e-7)|(np.sum(before*after,axis=1)<0)
    return points[keep].tolist() if np.count_nonzero(keep)>=3 else points.tolist()


def _propose_ring(points, rgb, score, mode, size):
    original=_resample(points,PARAMETERS['sample_step_px'])
    smoothed=gaussian_filter1d(original,PARAMETERS['smoothing_sigma_samples'],axis=0,mode='wrap')
    # Smoothing has its own small movement budget; it cannot round off a tip.
    change=smoothed-original
    norm=np.linalg.norm(change,axis=1)
    smoothed=original+change*np.minimum(1,2.0/np.maximum(norm,1e-8))[:,None]
    tangent=np.roll(smoothed,-2,axis=0)-np.roll(smoothed,2,axis=0)
    normals=np.column_stack((tangent[:,1],-tangent[:,0]))
    signed=np.sum(original[:,0]*np.roll(original[:,1],-1)-np.roll(original[:,0],-1)*original[:,1])
    normals*=1 if signed>0 else -1
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    radius=PARAMETERS['max_movement_px']
    offsets=np.arange(-radius,radius+.25,.5)
    center=smoothed[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    inward=center-normals[:,None,:]*2
    outward=center+normals[:,None,:]*2
    # Small ROI avoids converting or blurring a full-resolution image per ring.
    low=np.maximum(0,np.floor(original.min(axis=0)-radius-8).astype(int))
    high=np.minimum(np.array([size[1],size[0]]),np.ceil(original.max(axis=0)+radius+9).astype(int))
    x0,y0=low; x1,y1=high
    image=np.ascontiguousarray(rgb[y0:y1,x0:x1])
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2LAB).astype(np.float32),(0,0),1.0)
    saturation=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32),(0,0),1.0)
    shift=np.array([x0,y0])
    inner_color=_sample(lab,inward-shift);outer_color=_sample(lab,outward-shift)
    contrast=np.linalg.norm(inner_color-outer_color,axis=2)
    polarity=_sample(saturation,inward-shift)-_sample(saturation,outward-shift)
    if mode=='cavity':polarity=-polarity
    inner_score=_sample(score,inward);outer_score=_sample(score,outward)
    # Model boundaries supply identity and a soft prior. Local color edges
    # supply position; dark vascular features alone cannot pull the curve far.
    quality=np.minimum(contrast/35,3)+.25*np.clip(polarity/40,-1,1)
    quality+=.15*np.clip(inner_score-outer_score,0,1)
    quality-=.045*offsets[None,:]**2
    quality[polarity < -20]-=1.0
    valid=(center[:,:,0]>=0)&(center[:,:,0]<size[1])&(center[:,:,1]>=0)&(center[:,:,1]<size[0])
    quality[~valid]=-np.inf
    best=np.argmax(quality,axis=1)
    local_contrast=contrast[np.arange(len(original)),best]
    confidence=np.clip((local_contrast-PARAMETERS['min_color_step'])/20,0,1)
    movement=offsets[best]*confidence
    movement=gaussian_filter1d(movement,1.5,mode='wrap')
    proposed=smoothed+normals*movement[:,None]
    displacement=proposed-original
    length=np.linalg.norm(displacement,axis=1)
    proposed=original+displacement*np.minimum(1,radius/np.maximum(length,1e-8))[:,None]
    # Do not pull a cropped/touching body away from the frame.
    frame=(original[:,0]<=.5)|(original[:,0]>=size[1]-1.5)|(original[:,1]<=.5)|(original[:,1]>=size[0]-1.5)
    proposed[frame]=original[frame]
    proposed[:,0]=np.clip(proposed[:,0],0,size[1]-1)
    proposed[:,1]=np.clip(proposed[:,1],0,size[0]-1)
    return original,proposed,int((confidence>.25).sum())


def refine_body(mask, probability, rgb, bundle_mask=None, blocked_mask=None, *, color_mode='tissue', rings=None):
    started=time.monotonic()
    if color_mode not in ('tissue','cavity'):
        raise ValueError('未知body颜色模式')
    raw=(np.asarray(mask)>0).astype(np.uint8)
    rgb=np.asarray(rgb,np.uint8)
    if raw.ndim!=2 or rgb.shape!=raw.shape+(3,):
        raise ValueError('body轮廓修正原图与掩膜尺寸不一致')
    if probability is None:
        raise ValueError('body轮廓修正需要连续掩膜分数')
    probability=np.asarray(probability,np.float32)
    if probability.shape!=raw.shape or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('body轮廓修正连续分数无效')
    for value in (bundle_mask,blocked_mask):
        if value is not None and np.asarray(value).shape!=raw.shape:
            raise ValueError('body轮廓修正约束尺寸不一致')
    original=copy.deepcopy(rings) if rings is not None else extract_topology_contours(raw,0,0)
    baseline=rasterize(original,raw.shape)
    output=copy.deepcopy(original)
    blocked=np.zeros_like(raw,dtype=bool) if blocked_mask is None else np.asarray(blocked_mask)>0
    blocked &= baseline==0
    protected=np.zeros_like(raw,dtype=bool) if bundle_mask is None else (np.asarray(bundle_mask)>0)&(baseline>0)
    area=max(1,int(baseline.sum()));float_area=ring_area(original)
    audit=dict(version=VERSION,method='constrained_contour',parameters=PARAMETERS.copy(),
        color_mode=color_mode,input_pixels=int(baseline.sum()),topology_before=list(topology(baseline)),
        hole_policy='original_polygons_unchanged',review_reasons=[],rings_refined=0,
        rings_reverted=0,local_vertices_reverted=0,image_supported_vertices=0,ring_guard_rejections={})

    def conflicts(candidate):
        return ((candidate>0)&blocked)|((candidate==0)&protected)

    def geometry_valid(candidate, candidate_rings):
        checks=dict(topology=same_topology(baseline,candidate),
            pixel_area=abs(int(candidate.sum())/area-1)<=PARAMETERS['max_area_change'],
            changed_fraction=np.count_nonzero(candidate!=baseline)/area<=PARAMETERS['max_changed_fraction'],
            polygon_area=(float_area<=0 or abs(ring_area(candidate_rings)/float_area-1)<=PARAMETERS['max_area_change']),
            ownership_and_seeds=not conflicts(candidate).any(),
            polygon_crossing=not has_crossings(candidate_rings))
        for key,ok in checks.items():
            if not ok:
                counts=audit['ring_guard_rejections'];counts[key]=counts.get(key,0)+1
        return all(checks.values())

    for index,ring in enumerate(original):
        # Preserve every hole polygon, even tiny ones. No ring is created.
        if ring['role']!='outer' or len(ring['points'])<3:
            continue
        sampled,proposed,supported=_propose_ring(ring['points'],rgb,probability,color_mode,raw.shape)
        audit['image_supported_vertices']+=supported
        if len(sampled)<8:
            continue
        current=proposed.copy()
        # Remove only offending arcs before applying the whole-ring guards.
        for _ in range(3):
            trial=copy.deepcopy(output);trial[index]['points']=_polygon_points(current)
            candidate=rasterize(trial,raw.shape)
            bad=conflicts(candidate)
            if not bad.any():break
            distance=cv2.distanceTransform((~bad).astype(np.uint8),cv2.DIST_L2,5)
            nearby=_sample(distance,sampled[:,None,:]).reshape(-1)<=PARAMETERS['max_movement_px']+2
            nearby=maximum_filter1d(nearby.astype(np.uint8),2*PARAMETERS['local_rollback_halo_samples']+1,mode='wrap')>0
            audit['local_vertices_reverted']+=int(nearby.sum())
            current[nearby]=sampled[nearby]
        accepted=False
        # Back off just this ring if its thin neck, area or topology is at risk.
        for fraction in (1.,.5,.25):
            trial=copy.deepcopy(output)
            points=sampled+(current-sampled)*fraction
            trial[index]['points']=_polygon_points(points)
            candidate=rasterize(trial,raw.shape)
            if geometry_valid(candidate,trial):
                output=trial;accepted=True;audit['rings_refined']+=1
                break
        if not accepted:
            audit['rings_reverted']+=1
    final=rasterize(output,raw.shape)
    old_edge=cv2.morphologyEx(baseline,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    new_edge=cv2.morphologyEx(final,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    movement=0.
    if old_edge.any() and new_edge.any():
        movement=max(float(cv2.distanceTransform(1-old_edge,cv2.DIST_L2,5)[new_edge>0].max()),
                     float(cv2.distanceTransform(1-new_edge,cv2.DIST_L2,5)[old_edge>0].max()))
    if movement>PARAMETERS['max_movement_px']+1:
        output=original;final=baseline;movement=0.
        audit['review_reasons'].append('final_displacement_guard')
        audit['rings_reverted']+=audit['rings_refined'];audit['rings_refined']=0
    if audit['rings_reverted']:
        audit['review_reasons'].append('local_geometry_retained')
    changed=int(np.count_nonzero(final!=baseline))
    audit.update(status='processed' if audit['rings_refined'] else 'fallback',
        output_pixels=int(final.sum()),topology_after=list(topology(final)),changed_pixels=changed,
        added_pixels=int(np.count_nonzero((final>0)&(baseline==0))),
        removed_pixels=int(np.count_nonzero((final==0)&(baseline>0))),
        pixel_area_change_ratio=float(int(final.sum())/area-1),
        polygon_area_change_ratio=float(ring_area(output)/float_area-1) if float_area>0 else 0.,
        max_boundary_movement_px=movement,polygon_smoothing_applied=bool(audit['rings_refined']),
        seconds=round(time.monotonic()-started,3))
    return RefinedBody(final,output,audit,np.full(raw.shape,cv2.GC_PR_BGD,np.uint8))
