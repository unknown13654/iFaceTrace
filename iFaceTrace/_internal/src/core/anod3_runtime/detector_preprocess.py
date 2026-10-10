"""Detector-only PIL BILINEAR widened profile; host pixel-calibrated approximation."""

import numpy as np
from PIL import Image

PROFILE_ID='anod3_rgb_pil_bilinear_widened_v1'
SCALE=np.float32(0.01742919348180294)
BIAS=np.array([-2.1398117542266846,-2.0208539962768555,
               -1.7948601245880127],dtype=np.float32)


def frame_size(w,h):
    if not (w>0 and h>0 and w>=h):
        raise ValueError('positive, landscape dimensions required')
    w,h=np.float32(w),np.float32(h)
    if np.float32(w/h)<np.float32(608/448):
        return int(np.float32(np.float32(w/h)*np.float32(448))),448
    return 608,int(np.float32(np.float32(h/w)*np.float32(608)))


def affine_rgb(rgb):
    return ((rgb.astype(np.float32)*SCALE)+BIAS).transpose(2,0,1)[None]


def preprocess_rgb(rgb):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('expected nonempty uint8 RGB image')
    h,w=rgb.shape[:2]
    portrait=h>w
    rotated=np.rot90(rgb,1) if portrait else rgb
    sw,sh=frame_size(rotated.shape[1],rotated.shape[0])
    canvas=np.zeros((448,608,3),dtype=np.uint8)
    canvas[:sh,:sw]=np.asarray(Image.fromarray(rotated).resize((sw,sh),Image.Resampling.BILINEAR))
    meta=dict(orig_w=w,orig_h=h,scale_w=sw,scale_h=sh,portrait=portrait,
              detector_preprocessing=PROFILE_ID,kernel='PIL_BILINEAR_WIDENED',
              kernel_status='HOST_APPROX_PIXEL_CALIBRATED',spatial_offset='none')
    return affine_rgb(canvas),meta,canvas
