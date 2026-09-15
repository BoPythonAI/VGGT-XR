"""Small, device-agnostic controls used by the Gaussian quality trainer."""
import math

try:
    import torch
except ModuleNotFoundError:  # Lightweight geometry-only CI does not install CUDA/PyTorch.
    torch=None


def progressive_sh_degree(step, steps, maximum, start_fraction):
    """Keep appearance view-independent until geometry and opacity stabilize."""
    if maximum==0:return 0
    if start_fraction<=0:
        return min(maximum,step//max(250,steps//12))
    start=int(steps*start_fraction)
    if step<start:return 0
    interval=max(1,(steps-start)//maximum)
    return min(maximum,1+(step-start)//interval)


def constrain_scales(log_scales, maximum, max_anisotropy):
    """Project covariance axes onto explicit size and anisotropy bounds."""
    if torch is None:raise RuntimeError('Scale projection requires PyTorch')
    with torch.no_grad():
        log_scales.clamp_(max=math.log(maximum))
        lower=log_scales.amin(-1,keepdim=True)
        log_scales.copy_(torch.minimum(log_scales,lower+math.log(max_anisotropy)))
