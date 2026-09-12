import json, time
import torch
from gsplat import rasterization

assert torch.cuda.is_available()
start = time.perf_counter()
means = torch.tensor([[0., 0., 3.]], device='cuda', requires_grad=True)
rgb, alpha, info = rasterization(means, torch.tensor([[1., 0., 0., 0.]], device='cuda'), torch.full((1, 3), .1, device='cuda'), torch.tensor([.8], device='cuda'), torch.tensor([[1., .2, .1]], device='cuda'), torch.eye(4, device='cuda')[None], torch.tensor([[[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]]], device='cuda'), 64, 64, packed=False)
rgb.sum().backward()
assert torch.isfinite(means.grad).all() and alpha.max() > .1
result = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), gsplat_forward_backward=True, seconds=time.perf_counter()-start)
print(json.dumps(result, indent=2))
from pathlib import Path
Path('outputs').mkdir(exist_ok=True)
Path('outputs/cuda_smoke.json').write_text(json.dumps(result, indent=2))
