"""Frozen ANODv3 graph with the validated model-NMS corrections."""

import numpy as np

from . import anod3_inference as base
from . import detector_preprocess
from .face_postprocess import fma32,postprocess_faces,validate_meta

F=np.float32
READY=True  # Known97/169/204/NMS/285 nominal representation corrections applied.
PROFILE_ID='anod3-bilinear-owned97-outer107-inner129-fresh-winner-nms-vdsp-order-v2'


def anchor_cwh(xyxy):
    xyxy=np.asarray(xyxy,dtype=F)
    delta=xyxy[:,2:]-xyxy[:,:2]
    return np.concatenate([xyxy[:,:2]+delta*F(.5),delta+F(1)],axis=1)


def proposal_outer_corners(cwh):
    """Native97 inset then clip, represented OUTER for existing169/204 code."""
    centers,sizes=np.asarray(cwh,dtype=F)[:,:2],np.asarray(cwh,dtype=F)[:,2:]
    limit=np.array([base.INPUT_W-1,base.INPUT_H-1],dtype=F)
    left=np.clip(centers-sizes*F(.5)+F(.5),F(0),limit)-F(.5)
    right=np.clip(centers+sizes*F(.5)-F(.5),F(0),limit)+F(.5)
    return np.concatenate([left,right],axis=1)


def geometry(boxes, extent_padding):
    """Reconstruct mode3 center/size from explicitly declared host corners."""
    boxes=np.asarray(boxes,dtype=F)
    delta=boxes[:,2:]-boxes[:,:2]
    center=boxes[:,:2]+delta*F(.5)
    size=np.maximum(delta+F(extent_padding),F(0))
    half=size*F(.5)
    return np.concatenate([size,center-half,center+half],axis=1)


def fma32_vector(a,b,c):
    """Vector f32 FMA: exact f64 product, TwoSum residue, midpoint correction."""
    product=np.float64(F(a))*np.float64(F(b))
    c32=np.asarray(c,dtype=F)
    if np.float64(F(product))==product:
        return F(product)+c32  # exact product makes ordinary f32 addition an exact FMA
    addend=c32.astype(np.float64)
    total=product+addend
    z=total-product
    error=(product-(total-z))+(addend-z)
    rounded=total.astype(F)
    if not np.any(error):
        return rounded
    high=np.nextafter(rounded,F(np.inf))
    low=np.nextafter(rounded,F(-np.inf))
    # Only an f64 sum exactly on an f32 midpoint can double-round differently.
    up=(error>0)&(total==(rounded.astype(np.float64)+high.astype(np.float64))*.5)
    down=(error<0)&(total==(low.astype(np.float64)+rounded.astype(np.float64))*.5)
    return np.where(up,high,np.where(down,low,rounded))


def overlaps(sender, others):
    # CPU0x1940a1718: f32 edge difference; f64 product; f32 intersection.
    difference=np.minimum(sender[4:6],others[:,4:6])-np.maximum(sender[2:4],others[:,2:4])
    positive=np.maximum(difference.astype(np.float64),0.)
    intersection=(positive[:,0]*positive[:,1]).astype(F)
    other_area=others[:,0]*others[:,1]
    area_sum=fma32_vector(sender[0],sender[1],other_area)
    union=area_sum-intersection
    return (intersection.astype(np.float64)/(union.astype(np.float64)+1e-7)).astype(F)


def vdsp_descending_indices(values):
    """0x1d22ab040..0x1d22ab300 finite-float descending branch, no stability promise."""
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('finite 1D scores required')
    order=list(range(len(values)))
    def sift(hole,last,saved):
        # Assembly picks LEFT when child values tie, and moves equal children down.
        while 2*hole+1<=last:
            child=2*hole+1
            if child+1<=last and values[order[child]]>values[order[child+1]]:child+=1
            if values[order[child]]>values[saved]:break
            order[hole]=order[child];hole=child
        order[hole]=saved
    for root in range(len(order)//2-1,-1,-1):sift(root,len(order)-1,order[root])
    for end in range(len(order)-1,0,-1):
        saved=order[end];order[end]=order[0];sift(0,end-1,saved)
    return np.asarray(order,dtype=np.int64)


def run_nms(boxes,scores,iou_threshold,score_threshold,num_pre_nms,max_boxes,per_class,*,extent_padding):
    boxes=np.asarray(boxes,dtype=F)
    scores=np.asarray(scores,dtype=F)
    if boxes.ndim!=2 or boxes.shape[1]!=4 or scores.ndim!=2 or len(scores)!=len(boxes) or scores.shape[1]<1:
        raise ValueError('expected N x 4 boxes and N x C scores')
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()) or np.any(scores<0) or np.any(scores>1) or max_boxes<1 or extent_padding not in (0,1):
        raise ValueError('invalid model NMS inputs or extent policy')
    if 0<num_pre_nms<max_boxes:
        raise ValueError('num_boxes_pre_nms must be 0 or at least max_boxes')
    if not len(boxes):
        return np.empty(0,dtype=np.int64),scores.copy()
    winners=np.argmax(scores,axis=1)
    maxima=scores[np.arange(len(scores)),winners]
    eligible=np.flatnonzero(maxima>F(score_threshold))
    if not len(eligible):
        # Fresh zero-seed CPU-context profile. Reused native seeds can differ.
        selected=np.array([int(np.argmax(maxima))],dtype=np.int64)
        return selected,scores[selected]
    order=eligible[vdsp_descending_indices(maxima[eligible])]
    if num_pre_nms>0:
        order=order[:num_pre_nms]
    g=geometry(boxes[order],extent_padding)
    suppressed=np.zeros(len(order),dtype=bool)
    kept=[]
    for position,index in enumerate(order):
        if suppressed[position]:
            continue
        kept.append(int(index))
        if len(kept)==max_boxes:
            break
        later=np.flatnonzero(~suppressed[position+1:])+position+1
        if per_class:
            later=later[winners[order[later]]==winners[index]]
        if len(later):
            suppressed[later[overlaps(g[position],g[later])>F(iou_threshold)]]=True
    selected=np.asarray(kept,dtype=np.int64)
    return selected,scores[selected]


class ANOD3ModelNMS(base.ANOD3Inference):
    """Owned RPN representation and model NMS corrections; other kernels retained."""

    def __init__(self,net,weights,anchors,profile_sha256):
        super().__init__(str(net),str(weights))
        self._anchors_path=str(anchors)
        self.profile_sha256=profile_sha256

    def _get_anchors_cwh(self):
        if self._anchor_cache is None:
            self._anchor_cache=anchor_cwh(base.load_baked_rpn_anchors(self._anchors_path))
        return self._anchor_cache

    def infer_faces(self,tensor,meta):
        """One controlled full detector graph + shared face postprocess per image."""
        tensor=np.asarray(tensor)
        if tensor.shape!=(1,3,448,608) or tensor.dtype!=F or not np.isfinite(tensor).all():
            raise ValueError('expected finite float32 RGB [1,3,448,608] image tensor')
        validate_meta(meta)
        if meta.get('detector_preprocessing')!=detector_preprocess.PROFILE_ID:
            raise ValueError('input tensor preprocessing profile is missing or incompatible')
        if not getattr(self,'profile_sha256',None):
            raise RuntimeError('construct the validated producer with load_engine()')
        self.nms_diagnostics={}
        outputs=self.run(tensor,input_name='data',verbose=False,
                         output_layers=['rcnn_output_pre','rcnn_output_scores_pre'])
        result=postprocess_faces(outputs['rcnn_output_pre'],outputs['rcnn_output_scores_pre'],meta)
        result.update(detector_profile=PROFILE_ID,detector_profile_sha256=self.profile_sha256,
                      nms_diagnostics=self.nms_diagnostics,
                      nms_state_policy='fresh_zero_seed_per_image; actual device kernel reuse unmeasured')
        return result

    def _loopkit_97_rpn_proposal(self,tensors,layer,bottom,tops):
        names=[name.strip() for name in bottom.split(',')]
        if any(tensors.get(name) is None for name in names):
            for name in tops:
                tensors[name]=None
            return
        delta=np.asarray(tensors[names[0]],dtype=F)
        cls=np.asarray(tensors[names[1]],dtype=F)
        bg,fg=cls[0,:30],cls[0,30:60]
        delta=delta[0].reshape(base.ANCHORS_PER_POSITION,4,base.FEAT_H,base.FEAT_W)
        delta=delta.transpose(2,3,0,1).reshape(-1,4)
        anchors=self._get_anchors_cwh()
        # Retain host exp/F32 arithmetic; native polynomial/FMA parity unclaimed.
        center=anchors[:,:2]+delta[:,:2]*anchors[:,2:]
        size=anchors[:,2:]*np.exp(delta[:,2:])
        outer=proposal_outer_corners(np.concatenate([center,size],axis=1))
        n=base.RPN_ANCHOR_COUNT
        tensors[tops[0]]=base._saturate_fp16(outer.reshape(1,1,n,4))
        tensors[tops[1]]=base._saturate_fp16(fg.transpose(1,2,0).reshape(1,1,n,1))
        tensors[tops[2]]=base._saturate_fp16(bg.reshape(1,30,28,38))

    def _exec_nms(self,tensors,layer,bottom,top):
        names=[n.strip() for n in bottom.split(',')]
        tops=[n.strip() for n in top.split(',')]
        if any(tensors.get(name) is None for name in names):
            for name in tops:
                tensors[name]=None
            return
        extent={'nms_boxes_nms':0,'rcnn_output_nms':1}.get(layer['name'])
        if extent is None:
            raise ValueError('unexpected NMS layer '+layer['name'])
        boxes=np.asarray(tensors[names[0]],dtype=F).reshape(-1,4)
        raw_scores=np.asarray(tensors[names[1]],dtype=F)
        scores=raw_scores.reshape(len(boxes),raw_scores.shape[-1])
        maximum=int(layer['max_boxes'])
        maxima=scores.max(axis=1) if len(scores) else np.empty(0,dtype=F)
        threshold=F(layer.get('score_threshold',0.))
        eligible=int(np.count_nonzero(maxima>threshold))
        self.nms_diagnostics[layer['name']]=dict(input_rows=len(boxes),eligible_count=eligible,
            fallback=bool(len(boxes) and not eligible),row0max=float(maxima[0]) if len(maxima) else None,
            peak_score=float(maxima.max()) if len(maxima) else None,
            native_reuse_unknown=True,state_policy='fresh_zero_seed_per_image')
        selected,selected_scores=run_nms(boxes,scores,
            iou_threshold=float(layer.get('iou_threshold',.5)),
            score_threshold=float(layer.get('score_threshold',0.)),
            num_pre_nms=int(layer.get('num_boxes_pre_nms',0)),max_boxes=maximum,
            per_class=bool(layer.get('per_class_suppression',False)),extent_padding=extent)
        out_boxes=np.zeros((maximum,4),dtype=F)
        out_scores=np.zeros((maximum,scores.shape[1]),dtype=F)
        indices=np.zeros(maximum,dtype=F)
        n=len(selected)
        out_boxes[:n]=boxes[selected];out_scores[:n]=selected_scores;indices[:n]=selected
        values=[out_boxes.reshape(1,1,maximum,4),out_scores.reshape(1,1,maximum,-1),
                indices.reshape(1,1,1,maximum),np.array([[[[n]]]],dtype=F)]
        for name,value in zip(tops,values):
            tensors[name]=base._saturate_fp16(value)
