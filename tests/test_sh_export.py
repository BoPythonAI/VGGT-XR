import numpy as np
import pytest
from src.export import sh_to_ply_rest


def test_sh_ply_channel_order_and_degree_padding():
    sh=np.arange(24,dtype=np.float32).reshape(1,8,3)
    packed=sh_to_ply_rest(sh).reshape(1,3,15)
    np.testing.assert_array_equal(packed[0,:,:8],sh[0].T)
    np.testing.assert_array_equal(packed[0,:,8:],0)


def test_sh_export_rejects_unrepresentable_degree():
    with pytest.raises(ValueError):sh_to_ply_rest(np.zeros((1,24,3)))
