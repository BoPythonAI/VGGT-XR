"""Budgeted SH0 Gaussian optimizer using gsplat's actual differentiable CUDA rasterizer."""
import json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from src.export import export_gaussians
from src.storage import check_storage,write_json


def render(params, e, k, width, height, *, return_info=False, sh_degree=None,
           absgrad=False, antialiased=False, background=None, render_mode='RGB'):
    """Render legacy RGB checkpoints or standard SH checkpoints with the same API."""
    if 'sh0' in params:
        colors=torch.cat([params['sh0'],params['shN']],dim=1)
        if sh_degree is None:sh_degree=int(round(colors.shape[1]**.5))-1
    else:
        colors=params['colors'].sigmoid();sh_degree=None
    backgrounds=torch.zeros((len(e),3),device=e.device) if background is None else torch.as_tensor(background,device=e.device,dtype=e.dtype).expand(len(e),3)
    result=rasterization(means=params['means'],quats=torch.nn.functional.normalize(params['quats'],dim=-1),scales=params['scales'].exp(),opacities=params['opacities'].sigmoid(),colors=colors,sh_degree=sh_degree,viewmats=e,Ks=k,width=width,height=height,packed=False,near_plane=.01,far_plane=1e4,backgrounds=backgrounds,absgrad=absgrad,rasterize_mode='antialiased' if antialiased else 'classic',render_mode=render_mode)
    return result if return_info else result[0]


def train(xyz,rgb,scales,train_images,extrinsics,intrinsics,output,steps=1500,test=None,seed=42):
    output=Path(output);check_storage(output,growth_gb=.2)
    done=output/'metrics.json'
    if done.exists():
        cached=json.loads(done.read_text())
        if cached['steps']!=steps:raise ValueError('Cached trainer steps differ; use new output directory')
        return cached
    torch.manual_seed(seed);np.random.seed(seed)
    def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda')
    extent=float(np.linalg.norm(np.percentile(xyz,95,axis=0)-np.percentile(xyz,5,axis=0)))
    q=np.zeros((len(xyz),4),np.float32);q[:,0]=1
    scales=np.clip(scales,extent*.0002,extent*.02)
    colors=np.clip(rgb,.001,.999)
    params=torch.nn.ParameterDict(dict(means=torch.nn.Parameter(tensor(xyz)),quats=torch.nn.Parameter(tensor(q)),scales=torch.nn.Parameter(tensor(np.log(scales[:,None].repeat(3,axis=1)))),opacities=torch.nn.Parameter(tensor(np.full(len(xyz),np.log(.3/.7)))),colors=torch.nn.Parameter(tensor(np.log(colors/(1-colors))))))
    rates=dict(means=.00016*extent,quats=.001,scales=.005,opacities=.05,colors=.01)
    optimizer=torch.optim.Adam([dict(params=[v],lr=rates[k],name=k) for k,v in params.items()],eps=1e-15)
    n,h,w,_=train_images.shape
    e4=np.tile(np.eye(4,dtype=np.float32),(n,1,1));e4[:,:3]=extrinsics
    es,ks,targets=tensor(e4),tensor(intrinsics),tensor(train_images)
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    history=[]
    order=np.random.default_rng(seed).permutation(n)
    for step in range(steps):
        if step%n==0:order=np.random.default_rng(seed+step//n).permutation(n)
        i=int(order[step%n])
        optimizer.zero_grad(set_to_none=True)
        rendered=render(params,es[i:i+1],ks[i:i+1],w,h)
        l1=(rendered[0]-targets[i]).abs().mean()
        similarity=ssim(rendered.permute(0,3,1,2),targets[i:i+1].permute(0,3,1,2),data_range=1.)
        loss=.8*l1+.2*(1-similarity)
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at step {step}')
        loss.backward();optimizer.step()
        with torch.no_grad():params['scales'].clamp_(np.log(extent*.0001),np.log(extent*.1));params['opacities'].clamp_(-10,10)
        if step%100==0 or step==steps-1:
            entry=dict(step=step,l1=float(l1.detach()),loss=float(loss.detach()))
            history.append(entry);print(f'GS {output.name}: {step+1}/{steps}, loss={entry["loss"]:.5f}',flush=True)
        if step%250==0:check_storage(output)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    result=dict(steps=steps,gaussian_count=len(xyz),train_seconds=seconds,peak_vram_gb=torch.cuda.max_memory_allocated()/2**30,seed=seed,sh_degree=0,densification=False,protocol='Fixed Gaussian budget; optimize position, covariance, opacity and RGB; identical steps/seed between initialization variants',history=history)
    export_gaussians(output/'scene.ply',params)
    torch.save({k:v.detach().cpu() for k,v in params.items()},output/'checkpoint.pt')
    write_json(output/'camera.json',dict(extrinsic=e4[0].tolist(),intrinsic=intrinsics[0].tolist(),width=w,height=h,convention='OpenCV world-to-camera; convert world y to -y in Unity'))
    cam_r=e4[0,:3,:3].T;cam_pos=-cam_r@e4[0,:3,3]
    flip=np.array([1,-1,1])
    def vec(a):return dict(x=float(a[0]),y=float(a[1]),z=float(a[2]))
    write_json(output/'unity_camera.json',dict(position=vec(cam_pos*flip),forward=vec(cam_r[:,2]*flip),up=vec(-cam_r[:,1]*flip),verticalFov=float(np.rad2deg(2*np.arctan(h/(2*intrinsics[0,1,1])))),gaussianCount=len(xyz)))
    with torch.no_grad():
        preview=render(params,es[:1],ks[:1],w,h)[0].clip(0,1).cpu().numpy()
        Image.fromarray((preview*255).astype(np.uint8)).save(output/'train_preview.png')
    if test is not None:
        test_e,test_k,test_images=test
        outdir=output/'test_renders';outdir.mkdir(exist_ok=True)
        import lpips
        # SqueezeNet keeps downloaded evaluator weights below 5 MB, on data disk.
        perceptual=lpips.LPIPS(net='squeeze').to('cuda').eval()
        records=[]
        for i in range(len(test_images)):
            with torch.no_grad():
                pred=render(params,tensor(test_e[i:i+1]),tensor(test_k[i:i+1]),test_images.shape[2],test_images.shape[1]).clip(0,1)
                gt=tensor(test_images[i:i+1])
                mse=(pred-gt).square().mean()
                p,g=pred.permute(0,3,1,2),gt.permute(0,3,1,2)
                row=dict(view=i,psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),ssim=float(ssim(p,g,data_range=1.)),lpips_squeeze=float(perceptual(p*2-1,g*2-1).mean()))
                records.append(row)
                rgb_pred=(pred[0].cpu().numpy()*255).astype(np.uint8)
                Image.fromarray(rgb_pred).save(outdir/f'{i:03d}.png')
                Image.fromarray(np.concatenate([(test_images[i]*255).astype(np.uint8),rgb_pred],axis=1)).save(outdir/f'comparison_{i:03d}.png')
        result['heldout']=dict(count=len(records),psnr_mean=float(np.mean([x['psnr'] for x in records])),ssim_mean=float(np.mean([x['ssim'] for x in records])),lpips_squeeze_mean=float(np.mean([x['lpips_squeeze'] for x in records])),per_view=records,pose_protocol='GT test cameras transformed using training-only Sim(3); test images never input to VGGT or GS optimizer')
    with torch.no_grad():
        for _ in range(10):render(params,es[:1],ks[:1],w,h)
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(100):render(params,es[:1],ks[:1],w,h)
        torch.cuda.synchronize();result['server_raster_fps']=100/(time.perf_counter()-t)
        result['fps_note']='Headless gsplat CUDA at native image resolution, excludes display/Unity; not Unity FPS'
    write_json(done,result)
    del params,optimizer,targets,es,ks
    torch.cuda.empty_cache()
    return result
