"""Whole-band elastic centering with soft guides and continuous wall constraints.

Navigation supplies the corridor and guide order, not mandatory centroids.
The smoothing scale is in band-width units, independent of vertex density.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def _resample(xy, step):
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-8]
    xy,arc=xy[keep],arc[keep]
    positions=np.linspace(0,arc[-1],max(3,int(np.ceil(arc[-1]/step))+1))
    points=np.column_stack([np.interp(positions,arc,xy[:,axis]) for axis in (0,1)])
    return points,positions


def _hit_segments(a,b,walls,allow_caps=False):
    from .bundle_path import _intersections
    hits=np.zeros(len(a),dtype=bool)
    for offset in range(0,len(a),128):
        start,finish=a[offset:offset+128],b[offset:offset+128]
        # Exact bounding-box rejection avoids testing every remote contour
        # edge on dense model masks. The margin includes endpoint tolerance.
        low=np.minimum(start.min(axis=0),finish.min(axis=0))-.01
        high=np.maximum(start.max(axis=0),finish.max(axis=0))+.01
        wall_a,wall_b=walls
        nearby=np.all(np.maximum(wall_a,wall_b)>=low,axis=1)&np.all(np.minimum(wall_a,wall_b)<=high,axis=1)
        cross,t=_intersections(start,finish,wall_a[nearby],wall_b[nearby],include_vertices=True)
        if allow_caps:
            if offset==0:
                cross[0]&=t[0]>0.005/max(np.linalg.norm(b[0]-a[0]),1e-9)
            if offset+len(cross)==len(a):
                cross[-1]&=t[-1]<1-0.005/max(np.linalg.norm(b[-1]-a[-1]),1e-9)
        hits[offset:offset+len(cross)]=cross.any(axis=1)
    return hits


def _safe_update(current,proposal,reference,limit,walls):
    """Local line search checks both swept vertices and final curve segments.

    A blocked location cannot cancel smoothing elsewhere along the band.
    Safe movement segments prevent hopping between adjacent fold corridors.
    """
    delta=proposal-reference
    amount=np.linalg.norm(delta,axis=1)
    proposal=reference+delta*np.minimum(1,limit/np.maximum(amount,1e-9))[:,None]
    proposal[0],proposal[-1]=current[0],current[-1]
    fraction=np.ones(len(current))
    for _ in range(14):
        candidate=current+fraction[:,None]*(proposal-current)
        moved=np.linalg.norm(candidate-current,axis=1)>1e-7
        blocked=_hit_segments(current,candidate,walls)&moved
        edge_hits=_hit_segments(candidate[:-1],candidate[1:],walls,allow_caps=True)
        blocked[:-1]|=edge_hits
        blocked[1:]|=edge_hits
        # Navigation endpoints may touch a boundary; they never move.
        blocked[0]=blocked[-1]=False
        if not blocked.any():
            return candidate
        fraction[blocked]*=.5
    # Restore only unresolved points and their incident segments.
    fraction[blocked]=0
    for _ in range(len(current)):
        candidate=current+fraction[:,None]*(proposal-current)
        edge_hits=_hit_segments(candidate[:-1],candidate[1:],walls,allow_caps=True)
        if not edge_hits.any():
            return candidate
        indices=np.unique(np.r_[np.flatnonzero(edge_hits),np.flatnonzero(edge_hits)+1])
        active=indices[fraction[indices]>0]
        if not len(active):
            return current.copy()
        fraction[active]=0
    return current.copy()


def _project_guides(xy,guides,reference,positions):
    """Project in navigation order; constrain each guide to its own arc window."""
    if not len(guides):
        return np.empty((0,2)),[]
    old=np.asarray(guides,dtype=float)
    # The original route explicitly visited guides; nearest reference samples
    # give their arc locations without confusing spatially adjacent U legs.
    indices=np.argmin(np.linalg.norm(reference[None,:,:]-old[:,None,:],axis=2),axis=1)
    indices=np.maximum.accumulate(indices)
    low=np.r_[0,(indices[:-1]+indices[1:])/2]
    high=np.r_[(indices[:-1]+indices[1:])/2,len(xy)-1]
    edge=np.diff(xy,axis=0)
    lengths=np.linalg.norm(edge,axis=1)
    arc=np.r_[0,np.cumsum(lengths)]
    projected=[]
    progress=[]
    for p,left,right in zip(old,low,high):
        eligible=np.arange(max(0,int(np.floor(left))),min(len(edge),int(np.ceil(right))))
        if not len(eligible):
            eligible=np.array([min(len(edge)-1,int(left))])
        t=np.clip(np.sum((p-xy[eligible])*edge[eligible],axis=1)/np.maximum(lengths[eligible]**2,1e-12),0,1)
        t=np.maximum(t,np.clip(left-eligible,0,1))
        t=np.minimum(t,np.clip(right-eligible,0,1))
        candidates=xy[eligible]+t[:,None]*edge[eligible]
        k=np.argmin(np.linalg.norm(candidates-p,axis=1))
        projected.append(candidates[k])
        progress.append(float(arc[eligible[k]]+t[k]*lengths[eligible[k]]))
    return np.asarray(projected),progress


def center_soft_route(xy,rings,origin,scale,half_width,wall_distance,guides=(),strength=1):
    from .bundle_path import _walls,_recenter,_crosses_wall,_outside_region,_returns_without_wall
    reference,positions=_resample(np.asarray(xy,dtype=float),max(1,half_width/2))
    current=reference.copy()
    step=positions[1]-positions[0]
    walls=_walls(rings,origin,scale)
    n=len(current)
    second=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    # Fourth-power scaling is required for a second-difference bending energy.
    penalty=(3*half_width/max(step,1e-6))**4
    bending=penalty*(second.T@second)
    initial_width=map_coordinates(wall_distance,reference[:,::-1].T,order=1,mode='nearest')
    limit=np.maximum(3*half_width,1.5*initial_width)
    for iteration in range(6):
        # Broad tangents ignore vascular-scale zigzags; true large bends remain.
        broad=gaussian_filter1d(current,max(1,2*half_width/step),axis=0,mode='nearest')
        tangent=np.gradient(broad,axis=0)
        width=map_coordinates(wall_distance,current[:,::-1].T,order=1,mode='nearest')
        targets=_recenter(current,tangent,rings,origin,scale,
                          np.full(n,max(2*half_width,3)),strength)
        # Wide bulges have weak centering evidence; adjoining narrow sections
        # and bending energy determine the through-direction instead.
        weight=np.clip((half_width/np.maximum(width,half_width))**2,.08,1)
        matrix=bending+diags(weight,format='csc')
        rhs=weight[:,None]*targets
        # Fixed endpoints are eliminated rather than held with huge weights.
        fixed=np.array([0,n-1])
        rhs=rhs[1:-1]-matrix[1:-1][:,fixed]@reference[fixed]
        proposal=current.copy()
        proposal[1:-1]=spsolve(matrix[1:-1,1:-1],rhs)
        next_xy=_safe_update(current,proposal,reference,limit,walls)
        change=np.max(np.linalg.norm(next_xy-current,axis=1))
        current=next_xy
        if change<.03:
            break
    # Fully independent final continuous checks retain conservative fallback.
    fallback=(_crosses_wall(current,walls) or _outside_region(current,rings,origin,scale)
              or _returns_without_wall(current,rings,origin,scale))
    if fallback:
        current=np.asarray(xy,dtype=float)
        reference=current.copy()
        positions=np.r_[0,np.cumsum(np.linalg.norm(np.diff(current,axis=0),axis=1))]
    projected,progress=_project_guides(current,guides,reference,positions)
    info=dict(method='soft_guides_v1',iterations=iteration+1,fallback=bool(fallback),
              smoothing_scale_px=3*half_width/scale,
              max_shift_px=float(np.max(np.linalg.norm(current-reference,axis=1)))/scale,
              guide_progress_px=[p/scale for p in progress])
    return current,projected,info
