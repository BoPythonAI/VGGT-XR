import time
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.storage import write_json


def infer(image_paths, output, size=336, max_views=24):
    """Inference-only VGGT. No training, no GT poses or depths passed to model."""
    if len(image_paths)>max_views:
        raise ValueError(f'{len(image_paths)} views exceeds limit {max_views}; select fewer panorama centers or raise limit deliberately.')
    if size%14:raise ValueError('VGGT image size must be divisible by patch size 14')
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from huggingface_hub import hf_hub_download
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'predictions.npz'
    sha=hashlib.sha256()
    for path in image_paths:sha.update(Path(path).read_bytes())
    fingerprint=sha.hexdigest()
    if target.exists():
        info=json.loads((output/'input.json').read_text())
        if info['sha256']!=fingerprint:raise ValueError('Input images differ from cached VGGT predictions; use a new output directory.')
        saved=np.load(target)
        if saved['images'].shape[1:3] != (size,size) or len(saved['images'])!=len(image_paths):
            raise ValueError('Cached prediction dimensions differ. Use a new output directory.')
        return {k:saved[k] for k in saved.files}
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for bounded experiment')
    torch.manual_seed(42)
    rgb=np.stack([np.asarray(Image.open(p).convert('RGB').resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32)/255 for p in image_paths])
    images=torch.from_numpy(rgb).permute(0,3,1,2).to('cuda')
    from huggingface_hub.errors import LocalEntryNotFoundError
    import os
    local=Path(os.environ.get('VGGTXR_ROOT','/root/autodl-tmp/vggt-xr'))/'cache/vggt_model.pt'
    if local.exists():weights=str(local)
    else:
        try:
            weights=hf_hub_download('facebook/VGGT-1B','model.pt',local_files_only=True)
        except LocalEntryNotFoundError:
            weights=hf_hub_download('facebook/VGGT-1B','model.pt')
    model=VGGT()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    # Remove unneeded branches only after strict checkpoint load.
    model.point_head=None
    model.track_head=None
    model=model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(images)
        extrinsic,intrinsic=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:])
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    def cpu(t):return t.detach().float().cpu().numpy()
    result=dict(images=rgb,depth=cpu(pred['depth'])[0,...,0],confidence=cpu(pred['depth_conf'])[0],extrinsics=cpu(extrinsic)[0],intrinsics=cpu(intrinsic)[0])
    temporary=output/'predictions.tmp.npz'
    np.savez_compressed(temporary,**result)
    write_json(output/'input.json',dict(sha256=fingerprint,images=[Path(p).name for p in image_paths],size=size))
    temporary.replace(target)
    write_json(output/'runtime.json',dict(views=len(images),size=size,seconds=seconds,peak_vram_gb=torch.cuda.max_memory_allocated()/2**30,torch=torch.__version__,checkpoint='facebook/VGGT-1B',seed=42,depth_unprojection=True))
    del pred,model,images
    torch.cuda.empty_cache()
    return result
