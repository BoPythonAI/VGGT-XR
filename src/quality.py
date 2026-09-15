"""Quality trainer: real SH, bounded adaptive density and optional rig pose refinement.

Uses gsplat 1.5.3 APIs. All custom constraints/budget decisions live here; the
upstream package is not modified. Legacy experiment results remain untouched.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, split, reset_opa

from src.gaussian import render
from src.export import export_gaussians
from src.quality_constraints import constrain_scales, progressive_sh_degree
from src.storage import check_storage, write_json


class BudgetStrategy(DefaultStrategy):
    """Select strongest gradients within a hard point cap; repair pinned reset bug."""
    def __init__(self, cap, **kwargs):
        super().__init__(**kwargs)
        self.cap=cap
        self.reset_events=[]

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        count=len(params['means']);slots=self.cap-count
        if slots<=0:return 0,0
        gradients=state['grad2d']/state['count'].clamp_min(1)
        small=params['scales'].exp().amax(-1)<=self.grow_scale3d*state['scene_scale']
        eligible=gradients>self.grow_grad2d
        if step<self.refine_scale2d_stop_iter:
            eligible|=state['radii']>self.grow_scale2d
        candidates=torch.where(eligible)[0]
        if len(candidates)>slots:
            candidates=candidates[torch.topk(gradients[candidates],slots).indices]
        selected=torch.zeros(count,device=gradients.device,dtype=torch.bool)
        selected[candidates]=True
        duplicates=selected&small
        splits=selected&~small
        nd,ns=int(duplicates.sum()),int(splits.sum())
        if nd:duplicate(params=params,optimizers=optimizers,state=state,mask=duplicates)
        if ns:
            splits=torch.cat([splits,torch.zeros(nd,dtype=torch.bool,device=splits.device)])
            split(params=params,optimizers=optimizers,state=state,mask=splits,revised_opacity=self.revised_opacity)
        assert len(params['means'])<=self.cap
        return nd,ns

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        super().step_post_backward(params,optimizers,state,step,info,packed)
        # v1.5.3 uses `== 0 & step > 0`, a chained comparison that never fires.
        # Keep reset inside the refinement window, then allow opacity recovery.
        if 0<step<self.refine_stop_iter and step%self.reset_every==0:
            reset_opa(params=params,optimizers=optimizers,state=state,value=self.prune_opa*2)
            self.reset_events.append(step)


def pose_matrices(deltas, groups, base_c2w, scene_scale):
    """One world-space rigid correction per panorama center, with group 0 fixed."""
    d=torch.tanh(deltas)
    d=d*torch.as_tensor([.035,.035,.035,.02*scene_scale,.02*scene_scale,.02*scene_scale],device=d.device)
    d=torch.cat([torch.zeros_like(d[:1]),d[1:]],0)
    x,y,z=d[:,:3].unbind(-1)
    skew=torch.stack([torch.zeros_like(x),-z,y,z,torch.zeros_like(x),-x,-y,x,torch.zeros_like(x)],-1).reshape(-1,3,3)
    generators=torch.zeros((len(d),4,4),device=d.device)
    generators[:,:3,:3]=skew;generators[:,:3,3]=d[:,3:]
    correction=torch.matrix_exp(generators)
    return torch.linalg.inv(correction[groups]@base_c2w)


def image_metrics(params, extrinsics, intrinsics, images, outdir, masks=None,
                  antialiased=False, background=None, perceptual=False):
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    lp=None
    if perceptual:
        import lpips
        lp=lpips.LPIPS(net='squeeze').cuda().eval()
    records=[]
    with torch.no_grad():
        for i,gt_np in enumerate(images):
            h,w=gt_np.shape[:2]
            e=torch.as_tensor(extrinsics[i:i+1],dtype=torch.float32,device='cuda')
            k=torch.as_tensor(intrinsics[i:i+1],dtype=torch.float32,device='cuda')
            predicted=render(params,e,k,w,h,antialiased=antialiased,background=background).clip(0,1)
            gt=torch.as_tensor(gt_np[None],device='cuda',dtype=torch.float32)
            p,g=predicted,gt
            if masks is not None:
                yy,xx=np.where(masks[i]);y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1
                p=p[:,y0:y1,x0:x1];g=g[:,y0:y1,x0:x1]
            mse=(p-g).square().mean()
            p,g=p.permute(0,3,1,2),g.permute(0,3,1,2)
            row=dict(view=i,psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),ssim=float(ssim(p,g,data_range=1.)))
            if lp is not None:row['lpips_squeeze']=float(lp(p*2-1,g*2-1).mean())
            records.append(row)
            pred_np=(predicted[0].cpu().numpy()*255).astype(np.uint8)
            if i<6 or perceptual:
                Image.fromarray(pred_np).save(outdir/f'{i:03d}.png')
                Image.fromarray(np.concatenate([(gt_np*255).astype(np.uint8),pred_np],1)).save(outdir/f'comparison_{i:03d}.png')
    result=dict(count=len(records),psnr_mean=float(np.mean([x['psnr'] for x in records])),ssim_mean=float(np.mean([x['ssim'] for x in records])),per_view=records)
    if perceptual:result['lpips_squeeze_mean']=float(np.mean([x['lpips_squeeze'] for x in records]))
    return result


def train_quality(xyz, rgb, images, extrinsics, intrinsics, output, *, steps=12000,
                  sh_degree=3, densify=True, cap=200000, pose_opt=False, groups=None,
                  masks=None, antialiased=False, test_factory=None, seed=42,
                  depths=None, depth_confidence=None, depth_weight=0.,
                  max_scale_ratio=2., max_anisotropy=1e6, scale_reg_weight=0.,
                  sh_start_fraction=0., sh_lr_multiplier=1.):
    output=Path(output);check_storage(output,growth_gb=.5)
    config=dict(steps=steps,sh_degree=sh_degree,densify=densify,cap=cap,pose_opt=pose_opt,antialiased=antialiased,seed=seed,
                depth_weight=depth_weight,max_scale_ratio=max_scale_ratio,max_anisotropy=max_anisotropy,
                scale_reg_weight=scale_reg_weight,sh_start_fraction=sh_start_fraction,sh_lr_multiplier=sh_lr_multiplier)
    if (output/'metrics.json').exists():
        old=json.loads((output/'quality_config.json').read_text())
        if old!=config:raise ValueError('Quality config changed; use another output')
        return json.loads((output/'metrics.json').read_text())
    if not 0<=sh_degree<=3 or len(xyz)>cap:raise ValueError('Invalid SH degree or initialization exceeds cap')
    if not 0<=sh_start_fraction<1 or depth_weight<0 or max_scale_ratio<=0 or max_anisotropy<1 or sh_lr_multiplier<0:
        raise ValueError('Invalid artifact-control configuration')
    if depth_weight and (depths is None or depth_confidence is None):
        raise ValueError('Depth supervision requires depths and depth confidence')
    write_json(output/'quality_config.json',config)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device='cuda')
    n,h,w,_=images.shape
    e4=np.tile(np.eye(4,dtype=np.float32),(n,1,1));e4[:,:3]=extrinsics[:,:3]
    centers=-np.einsum('nji,nj->ni',e4[:,:3,:3],e4[:,:3,3])
    # Camera radius defines optimization scale; do not let distant outliers set LR.
    scene_scale=max(float(np.max(np.linalg.norm(centers-centers.mean(0),axis=-1))),float(np.linalg.norm(np.percentile(xyz,90,0)-np.percentile(xyz,10,0)))*.05,1e-3)
    distances,_=cKDTree(xyz).query(xyz,k=4)
    maximum_scale=scene_scale*max_scale_ratio
    scales=np.sqrt(np.mean(distances[:,1:]**2,axis=-1)).clip(scene_scale*1e-5,min(scene_scale*.1,maximum_scale))
    q=rng.normal(size=(len(xyz),4)).astype(np.float32);q/=np.linalg.norm(q,axis=-1,keepdims=True)
    sh=np.zeros((len(xyz),(sh_degree+1)**2,3),np.float32);sh[:,0]=(rgb-.5)/.28209479177387814
    params=torch.nn.ParameterDict({k:torch.nn.Parameter(tensor(v)) for k,v in dict(means=xyz,scales=np.log(scales[:,None].repeat(3,1)),quats=q,opacities=np.full(len(xyz),np.log(.1/.9)),sh0=sh[:,:1],shN=sh[:,1:]).items()})
    rates=dict(means=1.6e-4*scene_scale,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.0025/20*sh_lr_multiplier)
    optimizers={k:torch.optim.Adam([dict(params=[v],lr=rates[k])],eps=1e-15) for k,v in params.items()}
    targets,es,ks=tensor(images),tensor(e4),tensor(intrinsics)
    group_ids=np.asarray(groups if groups is not None else np.arange(n),dtype=np.int64)
    group_ids=np.unique(group_ids,return_inverse=True)[1]
    group_tensor=torch.as_tensor(group_ids,device='cuda')
    deltas=torch.nn.Parameter(torch.zeros((int(group_ids.max())+1,6),device='cuda'))
    pose_optimizer=torch.optim.Adam([deltas],lr=.001) if pose_opt else None
    base_c2w=torch.linalg.inv(es)
    mask_tensor=torch.ones((n,h,w),device='cuda') if masks is None else tensor(masks)
    depth_targets=tensor(depths) if depth_weight else None
    depth_weights=None
    if depth_weight:
        depth_weights=np.empty_like(depth_confidence,dtype=np.float32)
        for j,c in enumerate(depth_confidence):
            valid=np.isfinite(c)&np.isfinite(depths[j])&(depths[j]>1e-5)
            lo,hi=np.percentile(c[valid],[20,90])
            depth_weights[j]=(.1+.9*np.clip((c-lo)/max(hi-lo,1e-6),0,1))*valid
        depth_weights=tensor(depth_weights)*mask_tensor
    # Supervise only the image rectangle, not artificial aspect-ratio padding.
    rectangles=[]
    for m in mask_tensor.cpu().numpy():
        yy,xx=np.where(m);rectangles.append((yy.min(),yy.max()+1,xx.min(),xx.max()+1))
    background=[1.,1.,1.] if masks is not None else [0.,0.,0.]
    strategy=BudgetStrategy(cap,refine_start_iter=max(100,int(steps/60)),refine_stop_iter=int(steps*.5),refine_every=100,reset_every=max(500,int(steps*.2)),pause_refine_after_reset=n,absgrad=True,grow_grad2d=.0008,prune_scale3d=.3)
    strategy.check_sanity(params,optimizers);state=strategy.initialize_state(scene_scale)
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[]
    order=rng.permutation(n)
    for step in range(steps):
        if step%n==0:order=rng.permutation(n)
        i=int(order[step%n])
        for optimizer in optimizers.values():optimizer.zero_grad(set_to_none=True)
        if pose_optimizer:pose_optimizer.zero_grad(set_to_none=True)
        active_pose=pose_opt and max(300,steps//20)<=step<int(steps*.8)
        refined=pose_matrices(deltas,group_tensor,base_c2w,scene_scale) if pose_opt else es
        view=refined[i:i+1] if active_pose else refined[i:i+1].detach()
        degree=progressive_sh_degree(step,steps,sh_degree,sh_start_fraction)
        mode='RGB+ED' if depth_weight else 'RGB'
        rendered_all,alpha,info=render(params,view,ks[i:i+1],w,h,return_info=True,sh_degree=degree,absgrad=densify,antialiased=antialiased,background=background,render_mode=mode)
        rendered=rendered_all[...,:3]
        if densify:strategy.step_pre_backward(params,optimizers,state,step,info)
        y0,y1,x0,x1=rectangles[i]
        p=rendered[:,y0:y1,x0:x1];g=targets[i:i+1,y0:y1,x0:x1]
        l1=(p-g).abs().mean();similarity=ssim(p.permute(0,3,1,2),g.permute(0,3,1,2),data_range=1.)
        depth_loss=torch.zeros((),device='cuda')
        if depth_weight:
            predicted_depth=rendered_all[:,y0:y1,x0:x1,3]
            target_depth=depth_targets[i:i+1,y0:y1,x0:x1]
            weights=depth_weights[i:i+1,y0:y1,x0:x1]
            visible=(alpha[:,y0:y1,x0:x1,0].detach()>.05)&(target_depth>1e-5)&torch.isfinite(predicted_depth)
            weights=weights*visible
            residual=F.smooth_l1_loss(torch.log(predicted_depth.clamp_min(1e-4)),torch.log(target_depth.clamp_min(1e-4)),reduction='none',beta=.05)
            depth_loss=(residual*weights).sum()/weights.sum().clamp_min(1)
        log_extent=params['scales'].amax(-1)-params['scales'].amin(-1)
        size_excess=F.relu(params['scales'].amax(-1)-float(np.log(maximum_scale)))
        shape_excess=F.relu(log_extent-float(np.log(max_anisotropy)))
        scale_loss=size_excess.square().mean()+shape_excess.square().mean()
        loss=.8*l1+.2*(1-similarity)+depth_weight*depth_loss+scale_reg_weight*scale_loss
        if active_pose:loss=loss+.001*deltas.square().mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite quality loss: step={step}')
        loss.backward()
        for optimizer in optimizers.values():optimizer.step()
        if active_pose:pose_optimizer.step()
        for group in optimizers['means'].param_groups:group['lr']=rates['means']*.01**((step+1)/steps)
        if densify:strategy.step_post_backward(params,optimizers,state,step,info)
        with torch.no_grad():
            params['scales'].clamp_(min=float(np.log(scene_scale*1e-6)))
            constrain_scales(params['scales'],maximum_scale,max_anisotropy)
            params['opacities'].clamp_(-12,12)
        if len(params['means'])<64:raise RuntimeError('Densification pruned nearly all Gaussians')
        if step%500==0 or step==steps-1:
            row=dict(step=step+1,loss=float(loss.detach()),l1=float(l1.detach()),depth_loss=float(depth_loss.detach()),scale_loss=float(scale_loss.detach()),gaussian_count=len(params['means']),sh_degree=degree)
            history.append(row);print(f'QUALITY {output.parent.name} {row}',flush=True);check_storage(output)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    refined=pose_matrices(deltas,group_tensor,base_c2w,scene_scale) if pose_opt else es
    final_e=refined.detach().cpu().numpy()
    final_scales=params['scales'].detach().exp()
    anisotropy=final_scales.amax(-1)/final_scales.amin(-1).clamp_min(1e-12)
    result=dict(**config,gaussian_count=len(params['means']),scene_scale=scene_scale,maximum_scale=maximum_scale,
                scale_statistics=dict(max=float(final_scales.max()),p99=float(torch.quantile(final_scales.amax(-1),.99)),anisotropy_p99=float(torch.quantile(anisotropy,.99)),anisotropy_max=float(anisotropy.max())),
                train_seconds=seconds,peak_vram_gb=torch.cuda.max_memory_allocated()/2**30,history=history,opacity_reset_events=strategy.reset_events,gsplat='1.5.3',opacity_reset_fix='Corrected Python chained-comparison condition in local wrapper; package unchanged',pose_protocol='Training RGB only; first camera/group fixed; shared world SE(3) correction preserves panorama rig; bounded corrections' if pose_opt else 'Fixed training cameras',background=background)
    result['training']=image_metrics(params,final_e,intrinsics,images,output/'train_renders',masks,antialiased,background)
    result['training']['note']='Training reconstruction scores, not heldout quality'
    if test_factory is not None:
        test_e,test_k,test_images=test_factory(final_e[:,:3])
        result['heldout']=image_metrics(params,test_e,test_k,test_images,output/'test_renders',antialiased=antialiased,background=background,perceptual=True)
        result['heldout']['pose_protocol']='GT cameras transformed by training-only Sim(3), no test RGB in inference/optimization; development set'
    export_gaussians(output/'scene.ply',params)
    torch.save({k:v.detach().cpu() for k,v in params.items()},output/'checkpoint.pt')
    np.savez_compressed(output/'training_cameras.npz',extrinsics=final_e,intrinsics=intrinsics,pose_deltas=deltas.detach().cpu().numpy(),groups=group_ids)
    write_json(output/'camera.json',dict(extrinsic=final_e[0].tolist(),intrinsic=intrinsics[0].tolist(),width=w,height=h,sh_degree=sh_degree,antialiased=antialiased,background=background,convention='OpenCV world-to-camera; convert world y to -y in Unity'))
    cam_r=final_e[0,:3,:3].T;cam_pos=-cam_r@final_e[0,:3,3];flip=np.array([1,-1,1])
    vec=lambda a:dict(x=float(a[0]),y=float(a[1]),z=float(a[2]))
    write_json(output/'unity_camera.json',dict(position=vec(cam_pos*flip),forward=vec(cam_r[:,2]*flip),up=vec(-cam_r[:,1]*flip),verticalFov=float(np.rad2deg(2*np.arctan(h/(2*intrinsics[0,1,1])))),gaussianCount=len(params['means'])))
    with torch.no_grad():
        Image.fromarray((render(params,refined[:1],ks[:1],w,h,antialiased=antialiased,background=background)[0].clip(0,1).cpu().numpy()*255).astype(np.uint8)).save(output/'train_preview.png')
    write_json(output/'metrics.json',result)
    return result
