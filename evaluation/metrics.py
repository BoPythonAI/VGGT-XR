"""GT is used only for evaluation/alignment, never for inference or optimization."""
import numpy as np
from scipy.spatial.transform import Rotation


def sim3(source,target):
    source,target=np.asarray(source),np.asarray(target)
    sx,tx=source.mean(0),target.mean(0)
    x,y=source-sx,target-tx
    u,s,vt=np.linalg.svd(y.T@x/len(x))
    signs=np.ones(3);signs[-1]=np.linalg.det(u@vt)
    r=u@np.diag(signs)@vt
    scale=float((s*signs).sum()/max((x*x).sum()/len(x),1e-12))
    if scale<=1e-8:raise ValueError('Degenerate camera trajectory: cannot fit Sim(3)')
    t=tx-scale*r@sx
    return scale,r,t


def camera_centers(extrinsics):
    return -np.einsum('nji,nj->ni',extrinsics[:,:3,:3],extrinsics[:,:3,3])


def align_cameras(extrinsics,gt_c2w):
    s,r,t=sim3(camera_centers(extrinsics),gt_c2w[:,:3,3])
    aligned=s*camera_centers(extrinsics)@r.T+t
    learned_r=extrinsics[:,:3,:3].transpose(0,2,1)
    err=gt_c2w[:,:3,:3].transpose(0,2,1)@(r[None]@learned_r)
    rotation_error=np.rad2deg(Rotation.from_matrix(err).magnitude())
    translation=aligned-gt_c2w[:,:3,3]
    # Ordered selected viewpoints, not a claim of dense trajectory evaluation.
    rpe=np.diff(aligned,axis=0)-np.diff(gt_c2w[:,:3,3],axis=0)
    return (s,r,t),dict(ate_sim3_rmse=float(np.sqrt((translation**2).sum(-1).mean())),relative_translation_rmse=float(np.sqrt((rpe**2).sum(-1).mean())),rotation_error_deg_mean=float(rotation_error.mean()),alignment='Sim(3) fitted to training camera centers',num_cameras=len(extrinsics))


def test_camera_in_reconstruction(gt_c2w,k,alignment):
    s,r,t=alignment
    centers=(gt_c2w[:,:3,3]-t)@r/s
    orient=r.T[None]@gt_c2w[:,:3,:3]
    out=np.tile(np.eye(4,dtype=np.float32),(len(gt_c2w),1,1))
    out[:,:3,:3]=orient.transpose(0,2,1)
    out[:,:3,3]=-np.einsum('nij,nj->ni',out[:,:3,:3],centers)
    return out.astype(np.float32),k.astype(np.float32)


def depth_metrics(pred,gt,mask,scale=None):
    valid=mask&np.isfinite(gt)&np.isfinite(pred)&(gt>1e-5)&(pred>1e-5)
    if not valid.any():return dict(valid_pixels=0)
    if scale is None:scale=float(np.median(gt[valid]/pred[valid]))
    p,g=pred[valid]*scale,gt[valid]
    return dict(scale=scale,rmse=float(np.sqrt(np.mean((p-g)**2))),absrel=float(np.mean(np.abs(p-g)/g)),valid_pixels=int(valid.sum()),coverage=float(valid.mean()),alignment='global median depth scale on observed training views')
