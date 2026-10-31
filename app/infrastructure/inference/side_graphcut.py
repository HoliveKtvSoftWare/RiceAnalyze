"""Side graph cut and constrained body contours at original resolution.

GMM color models are fitted on a bounded sample. Graph edges and cuts are
evaluated on overlapping ORIGINAL image pixels, with a frozen shared GMM.
No manual annotations/endpoints, skeleton, new weights or training are used.
"""
from dataclasses import dataclass
import copy
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from app.domain.geometry.mask_geometry import extract_topology_contours,filter_small_components
from .side_postprocess import rasterize,topology,same_topology,ring_area,has_crossings,_smooth_ring

VERSION='leaf-hybrid-region-refinement-3.0'
REGION_LABELS=('body1','body2','side1','side2')
PARAMETERS=dict(band_px=32,core_px=5,tile_px=384,halo_px=48,
                model_sample_long_edge=640,max_area_change=.15,max_changed_fraction=.35,
                min_color_contrast=12,rng_seed=20261027)


@dataclass
class GraphCutSide:
    mask: np.ndarray
    rings: list
    audit: dict
    seeds: np.ndarray


def refine_side(mask,probability,rgb,bundle_mask=None,blocked_mask=None,*,color_mode='tissue'):
    if color_mode not in ('tissue','cavity'):
        raise ValueError('未知区域颜色模式')
    started=time.monotonic()
    raw=filter_small_components(mask)
    rgb=np.asarray(rgb,np.uint8)
    if rgb.shape!=raw.shape+(3,):
        raise ValueError('方案B原图与掩膜尺寸不一致')
    if probability is None:
        raise ValueError('方案B需要连续掩膜分数，请使用analyze图片分析')
    probability=np.asarray(probability,np.float32)
    if probability.shape!=raw.shape or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('方案B连续掩膜分数无效')
    for candidate in (bundle_mask,blocked_mask):
        if candidate is not None and np.asarray(candidate).shape!=raw.shape:
            raise ValueError('方案B种子区域尺寸不一致')
    audit=dict(version=VERSION,parameters=PARAMETERS.copy(),color_mode=color_mode,input_pixels=int(raw.sum()),
               topology_before=list(topology(raw)),review_reasons=[])
    empty_seeds=np.full(raw.shape,cv2.GC_PR_BGD,np.uint8)

    def finish(output,seeds,status,reason=None):
        rings=extract_topology_contours(output,min_component_area=1,min_component_area_ratio=0)
        actual=rasterize(rings,raw.shape)
        if reason:
            audit['review_reasons'].append(reason)
        audit.update(status=status,output_pixels=int(actual.sum()),topology_after=list(topology(actual)),
                     changed_pixels=int(np.count_nonzero(actual!=raw)),
                     added_pixels=int(np.count_nonzero((actual>0)&(raw==0))),
                     removed_pixels=int(np.count_nonzero((actual==0)&(raw>0))),
                     pixel_area_change_ratio=float(int(actual.sum())/max(1,int(raw.sum()))-1),
                     seconds=round(time.monotonic()-started,3))
        return GraphCutSide(actual,rings,audit,seeds)

    if not raw.any():
        return finish(raw,empty_seeds,'empty')
    # Original-resolution crop; padding is supplied by the actual image.
    yy,xx=np.nonzero(raw)
    pad=PARAMETERS['band_px']+8
    x0,x1=max(0,int(xx.min())-pad),min(raw.shape[1],int(xx.max())+pad+1)
    y0,y1=max(0,int(yy.min())-pad),min(raw.shape[0],int(yy.max())+pad+1)
    local=raw[y0:y1,x0:x1]
    image=np.ascontiguousarray(rgb[y0:y1,x0:x1])
    score=probability[y0:y1,x0:x1]
    inside=cv2.distanceTransform(local,cv2.DIST_L2,5)
    outside=cv2.distanceTransform(1-local,cv2.DIST_L2,5)
    envelope=np.zeros_like(local)
    contours=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(envelope,contours,-1,1,cv2.FILLED)
    bundles=np.zeros_like(local) if bundle_mask is None else (np.asarray(bundle_mask)[y0:y1,x0:x1]>0).astype(np.uint8)
    bundles &= envelope
    blocked=np.zeros_like(local,dtype=bool) if blocked_mask is None else np.asarray(blocked_mask)[y0:y1,x0:x1]>0
    # Never claim previously unowned tissue from another predicted side/body.
    blocked &= local==0
    bundles[blocked]=0
    allowed=((outside<=PARAMETERS['band_px'])|(bundles>0))&~blocked
    saturation=gaussian_filter(cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32),1)
    core=(inside>=PARAMETERS['core_px'])&(score>=.7)
    surrounding=(local==0)&(outside>=12)&(outside<=32)&~blocked
    if core.sum()<30 or surrounding.sum()<30:
        return finish(raw,empty_seeds,'fallback','insufficient_color_seeds')
    # The cavity is white/low-saturation; its foreground polarity is
    # opposite to stained body1/side tissue. GMM fitting still uses RGB.
    evidence=saturation if color_mode=='tissue' else -saturation
    foreground_q=float(np.percentile(evidence[core],60))
    background_q=float(np.percentile(evidence[surrounding],65))
    contrast=foreground_q-background_q
    audit.update(color_contrast=contrast,foreground_color_q60=foreground_q,
                 background_color_q65=background_q,bundle_seed_pixels=int(bundles.sum()))
    if contrast<PARAMETERS['min_color_contrast']:
        return finish(raw,empty_seeds,'fallback','low_color_contrast')
    seeds=np.where(local,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
    seeds[~allowed]=cv2.GC_BGD
    trusted=core&(evidence>background_q+max(8,contrast*.20))
    trusted |= bundles>0
    seeds[trusted&allowed]=cv2.GC_FGD
    # Clear low-score background can be inside a real hole or folded gap.
    clear_bg=(evidence<background_q+contrast*.05)&(score<.10)&(bundles==0)&(local==0)
    seeds[clear_bg]=cv2.GC_BGD
    if np.count_nonzero(seeds==cv2.GC_FGD)<30 or np.count_nonzero(seeds==cv2.GC_BGD)<30:
        return finish(raw,empty_seeds,'fallback','insufficient_hard_seeds')
    h,w=local.shape
    ratio=min(1,PARAMETERS['model_sample_long_edge']/max(h,w))
    sample_size=(max(8,round(w*ratio)),max(8,round(h*ratio)))
    sample_image=cv2.resize(image,sample_size,interpolation=cv2.INTER_AREA)
    sample_seeds=cv2.resize(seeds,sample_size,interpolation=cv2.INTER_NEAREST)
    if np.count_nonzero((sample_seeds&1)==1)<5 or np.count_nonzero((sample_seeds&1)==0)<5:
        return finish(raw,empty_seeds,'fallback','insufficient_sampled_classes')
    bg,fg=np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
    cv2.setRNGSeed(PARAMETERS['rng_seed'])
    # iterCount=0 fits the two GMMs without solving a low-resolution mask.
    cv2.grabCut(sample_image,sample_seeds,None,bg,fg,0,cv2.GC_INIT_WITH_MASK)
    proposed=(seeds&1).astype(np.uint8)
    tile,halo=PARAMETERS['tile_px'],PARAMETERS['halo_px']
    tiles=0
    for top in range(0,h,tile):
        bottom=min(h,top+tile)
        for left in range(0,w,tile):
            right=min(w,left+tile)
            center=seeds[top:bottom,left:right]
            if not np.any(center>=cv2.GC_PR_BGD):
                continue
            a,b=max(0,top-halo),min(h,bottom+halo)
            c,d=max(0,left-halo),min(w,right+halo)
            tile_seeds=seeds[a:b,c:d].copy()
            cv2.grabCut(np.ascontiguousarray(image[a:b,c:d]),tile_seeds,None,bg.copy(),fg.copy(),
                        1,cv2.GC_EVAL_FREEZE_MODEL)
            proposed[top:bottom,left:right]=tile_seeds[top-a:bottom-a,left-c:right-c]&1
            tiles+=1
    # New islands without any reliable foreground seed carry no identity evidence.
    n,components=cv2.connectedComponents(proposed,connectivity=8)
    anchored=np.unique(components[(seeds==cv2.GC_FGD)&(components>0)])
    proposed=(np.isin(components,anchored)&allowed).astype(np.uint8)
    base_area=int(local.sum())
    change=abs(int(proposed.sum())/base_area-1)
    changed=int(np.count_nonzero(proposed!=local))/base_area
    audit.update(tiles=tiles,foreground_seed_pixels=int((seeds==cv2.GC_FGD).sum()),
                 background_seed_pixels=int((seeds==cv2.GC_BGD).sum()),
                 candidate_area_change_ratio=float(int(proposed.sum())/base_area-1),
                 candidate_changed_fraction=changed)
    global_seeds=empty_seeds.copy()
    global_seeds[y0:y1,x0:x1]=seeds
    if change>PARAMETERS['max_area_change'] or changed>PARAMETERS['max_changed_fraction']:
        return finish(raw,global_seeds,'fallback','large_change_requires_review')
    if np.any((proposed==0)&(seeds==cv2.GC_FGD)):
        return finish(raw,global_seeds,'fallback','foreground_seed_loss')
    output=np.zeros_like(raw)
    output[y0:y1,x0:x1]=proposed
    # Smooth only after the semantic correction. Preserve the corrected topology.
    rings=extract_topology_contours(output,min_component_area=1,min_component_area_ratio=0)
    smooth=[dict(r,points=_smooth_ring(r['points'],raw.shape)) for r in rings]
    smooth_mask=rasterize(smooth,raw.shape)
    area=ring_area(rings)
    original_float_area=ring_area(extract_topology_contours(raw,min_component_area=1,min_component_area_ratio=0))

    def final_geometry_valid(candidate,candidate_rings):
        return (abs(int(candidate.sum())/int(raw.sum())-1)<=PARAMETERS['max_area_change'] and
                np.count_nonzero(candidate!=raw)/int(raw.sum())<=PARAMETERS['max_changed_fraction'] and
                (not original_float_area or abs(ring_area(candidate_rings)/original_float_area-1)<=PARAMETERS['max_area_change']) and
                not np.any((candidate==0)&(global_seeds==cv2.GC_FGD)) and
                not np.any((candidate>0)&(global_seeds==cv2.GC_BGD)) and
                not np.any(candidate[y0:y1,x0:x1][~allowed]))

    smooth_ok=(same_topology(output,smooth_mask) and not has_crossings(smooth) and
               (not area or abs(ring_area(smooth)/area-1)<.02) and
               final_geometry_valid(smooth_mask,smooth))
    if not smooth_ok and not final_geometry_valid(rasterize(rings,raw.shape),rings):
        return finish(raw,global_seeds,'fallback','final_geometry_limit')
    result=finish(smooth_mask if smooth_ok else output,global_seeds,'processed')
    if smooth_ok:
        # Keep the floating polygons used in both the preview and workbook.
        result.rings=smooth
    audit['polygon_smoothing_applied']=bool(smooth_ok)
    return result


def _nearest_instance(masks, indices, size):
    """Assign previously unowned pixels to one mutually exclusive instance."""
    nearest=np.full(size,-1,np.int32)
    distance=np.full(size,np.inf,np.float32)
    for index in sorted(indices):
        candidate=cv2.distanceTransform((masks[index]==0).astype(np.uint8),cv2.DIST_L2,5)
        closer=candidate<distance-1e-6
        tied=np.abs(candidate-distance)<=1e-6
        nearest[closer]=index
        nearest[tied]=-1
        distance=np.minimum(distance,candidate)
    return nearest


def _outer_envelope(mask):
    """Containment support only; never written back as region geometry."""
    envelope=np.zeros_like(mask,dtype=np.uint8)
    contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(envelope,contours,-1,1,cv2.FILLED)
    return envelope


def _annotation_mask(shapes,label,size,index=None):
    rings=[]
    components={}
    for shape in shapes:
        if shape['label']!=label or (index is not None and shape.get('group_id')!=index):
            continue
        flags=shape.get('flags',{})
        key=(shape.get('group_id'),flags.get('component_id',0))
        component=components.setdefault(key,len(components))
        rings.append(dict(points=shape['points'],role=flags.get('contour_role','outer'),
                          component_id=component))
    return rasterize(rings,size)


def refine_annotation(data,masks,scores,rgb,seed_output_dir=None):
    """Refine sides with graph cut and bodies with constrained local contours.

    Body2 is a nested cavity with separate ownership from body1. Existing
    body/cavity overlap stays valid; no extra hole-fill operation is performed.
    """
    result=copy.deepcopy(data)
    shapes,audits,handled=[],[],set()
    size=(data['imageHeight'],data['imageWidth'])
    ownership={label:np.zeros(size,np.uint8) for label in REGION_LABELS}
    indices={label:set() for label in REGION_LABELS}
    for shape in data['shapes']:
        label,index=shape['label'],shape.get('group_id')
        if label in ownership and index in masks:
            ownership[label] |= (masks[index]>0).astype(np.uint8)
            indices[label].add(index)
    # Body1 and sides share tissue ownership. Body2 can be nested in body1,
    # so it has a separate instance competition within the cavity family.
    tissue_indices=set().union(*(indices[label] for label in ('body1','side1','side2')))
    tissue_nearest=_nearest_instance(masks,tissue_indices,size)
    cavity_nearest=_nearest_instance(masks,indices['body2'],size)
    body_envelope=_outer_envelope(ownership['body1'])
    bundle_labels={
        'body1':('body_big','body1_small'), 'body2':('body2_small',),
        'side1':('side1_small','side1_big'), 'side2':('side2_small','side2_big'),
    }
    for shape in data['shapes']:
        label,index=shape['label'],shape.get('group_id')
        if label not in ownership or index not in masks:
            shapes.append(copy.deepcopy(shape))
            continue
        pair=(label,index)
        if scores.get(index) is None:
            # Old cached predictions contain only side scores. Keep body rings
            # verbatim and explicitly abstain; never invent binary probabilities.
            shapes.append(copy.deepcopy(shape))
            if pair not in handled:
                audits.append(dict(label=label,instance=index,status='fallback',
                    review_reasons=['missing_continuous_scores']))
                handled.add(pair)
            continue
        if pair in handled:
            continue
        handled.add(pair)
        raw=(masks[index]>0)
        bundles=np.zeros(size,np.uint8)
        for annotation in data['shapes']:
            if annotation['label'] in bundle_labels[label] and len(annotation['points'])>=3:
                cv2.fillPoly(bundles,[np.rint(annotation['points']).astype(np.int32)],1)
        if label=='body2':
            blocked=(cavity_nearest!=index)&~raw
            blocked |= ((ownership['side1']|ownership['side2'])>0)&~raw
            if body_envelope.any():
                # Keep cavity growth within the organ's outer silhouette.
                blocked |= (body_envelope==0)&~raw
        else:
            blocked=(tissue_nearest!=index)&~raw
            blocked |= (ownership['body2']>0)&~raw
            if label=='body1':
                # Preserve already annotated containment, without filling any
                # previously unowned cavity or hole pixels into body1.
                bundles |= ((ownership['body2']>0)&raw).astype(np.uint8)
        if label in ('body1','body2'):
            from .body_contour import refine_body
            original_rings=[dict(points=s['points'],role=s.get('flags',{}).get('contour_role','outer'),
                                 component_id=s.get('flags',{}).get('component_id',0))
                            for s in data['shapes'] if s['label']==label and s.get('group_id')==index]
            refined=refine_body(masks[index],scores[index],rgb,bundles,blocked,
                                color_mode='cavity' if label=='body2' else 'tissue',rings=original_rings)
        else:
            refined=refine_side(masks[index],scores[index],rgb,bundles,blocked)
        audits.append(dict(refined.audit,label=label,instance=index,
                           method='constrained_contour' if label.startswith('body') else 'seeded_graphcut'))
        for ring in refined.rings:
            flags=dict(shape.get('flags',{}),mask_topology=True,contour_role=ring['role'],
                       component_id=ring['component_id'],region_postprocess=VERSION)
            shapes.append(dict(shape,points=ring['points'],flags=flags))
        if seed_output_dir:
            from PIL import Image
            destination=Path(seed_output_dir)
            destination.mkdir(parents=True,exist_ok=True)
            picture=np.asarray(rgb).copy()
            seeds=refined.seeds
            near=cv2.distanceTransform((masks[index]==0).astype(np.uint8),cv2.DIST_L2,5)<40
            picture[(seeds==cv2.GC_BGD)&near]=(245,70,90)
            picture[seeds==cv2.GC_FGD]=(0,210,220)
            image=Image.fromarray(picture)
            image.thumbnail((1600,1600))
            image.save(destination/(label+'_'+str(index)+'.jpg'),quality=90)
    # Cuts are independent proposals. Check their combined final geometry as
    # well: a shrinking body1 and growing body2 must not create a new escape
    # from the organ silhouette. Use outer silhouettes so a real body1 hole
    # can contain a cavity. On conflict abstain on the body pair, without
    # adding pixels or filling either region. Original annotation order does
    # not affect this check or its result.
    if indices['body1'] and indices['body2']:
        original_envelope=_outer_envelope(_annotation_mask(data['shapes'],'body1',size))
        final_envelope=_outer_envelope(_annotation_mask(shapes,'body1',size))
        conflict=False
        for index in indices['body2']:
            original_cavity=_annotation_mask(data['shapes'],'body2',size,index)
            final_cavity=_annotation_mask(shapes,'body2',size,index)
            existing_escape=(original_cavity>0)&(original_envelope==0)
            if np.any((final_cavity>0)&(final_envelope==0)&~existing_escape):
                conflict=True
                break
        if conflict:
            body_pairs={(label,index) for label in ('body1','body2') for index in indices[label]}
            kept=[s for s in shapes if (s['label'],s.get('group_id')) not in body_pairs]
            originals=[copy.deepcopy(s) for s in data['shapes']
                       if (s['label'],s.get('group_id')) in body_pairs]
            shapes=kept+originals
            for audit in audits:
                if (audit['label'],audit['instance']) not in body_pairs:
                    continue
                label,index=audit['label'],audit['instance']
                restored=_annotation_mask(originals,label,size,index)
                raw=filter_small_components(masks[index])
                audit.update(status='fallback',output_pixels=int(restored.sum()),
                    topology_after=list(topology(restored)),
                    changed_pixels=int(np.count_nonzero(restored!=raw)),
                    added_pixels=int(np.count_nonzero((restored>0)&(raw==0))),
                    removed_pixels=int(np.count_nonzero((restored==0)&(raw>0))),
                    pixel_area_change_ratio=float(int(restored.sum())/max(1,int(raw.sum()))-1),
                    polygon_smoothing_applied=False)
                audit['review_reasons'].append('final_body_containment')
    result['shapes']=shapes
    metadata=dict(version=VERSION,method='hybrid_region_refinement',region_labels=list(REGION_LABELS),
        region_methods={'body1':'constrained_contour','body2':'constrained_contour',
                        'side1':'seeded_graphcut','side2':'seeded_graphcut'},
        hole_policy='no_extra_fill',side_holes='preserved',
        bundle_annotations='independent_overlays',regions=audits)
    result['region_postprocess']=metadata
    # Existing consumers can continue reading this key during the transition.
    result['side_postprocess']=copy.deepcopy(metadata)
    result['side_postprocess']['method']='seeded_graphcut'
    return result
