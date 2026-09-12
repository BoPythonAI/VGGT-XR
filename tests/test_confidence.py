import numpy as np
from src.geometry import filter_points
from src.panorama import perspective_rays
from src.geometry import unproject


def test_confidence_floor_ties_are_removed():
    _,k=perspective_rays(28,90);e=np.eye(4,dtype=np.float32)[:3][None];d=np.ones((1,28,28),np.float32)
    points=unproject(d,e,k[None]);colors=np.ones_like(points)*.5;confidence=np.ones_like(d);confidence[:,:14]=2
    xyz,rgb,conf,scales,mask,stats=filter_points(points,colors,confidence,d,e,k[None],method='confidence',quantile=.35,max_points=1000)
    assert stats['confidence_threshold']==1
    assert mask.sum()==28*14
    assert np.all(conf==2)
