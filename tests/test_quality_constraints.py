import pytest

from src.quality_constraints import constrain_scales, progressive_sh_degree


def test_sh_stays_zero_until_geometry_stabilizes():
    assert progressive_sh_degree(3599,6000,3,.6)==0
    assert progressive_sh_degree(3600,6000,3,.6)==1
    assert progressive_sh_degree(5999,6000,3,.6)==3


def test_zero_start_preserves_original_schedule():
    assert progressive_sh_degree(499,6000,3,0)==0
    assert progressive_sh_degree(500,6000,3,0)==1
    assert progressive_sh_degree(1500,6000,3,0)==3


def test_scale_projection_limits_size_and_anisotropy():
    torch=pytest.importorskip('torch')
    scales=torch.log(torch.tensor([[.001,.02,2.],[.3,.3,.3]]))
    constrain_scales(scales,.1,10.)
    linear=scales.exp()
    assert float(linear.max())<=.100001
    assert float((linear.max(-1).values/linear.min(-1).values).max())<=10.0001
