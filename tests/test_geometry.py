import numpy as np
from src.panorama import erp_rays, project, perspective_rays, rotation, constrain_poses
from src.geometry import unproject, consistency_mask
from evaluation.metrics import sim3


def test_erp_projection_rays_and_seam():
    erp=erp_rays(512,1024)
    local,_=perspective_rays(70,100)
    for yaw,pitch in [(0,0),(179,0),(270,30),(0,90),(0,-90)]:
        actual,_,r=project(erp,yaw,pitch,100,70)
        expected=local@r.T;expected/=np.linalg.norm(expected,axis=-1,keepdims=True)
        assert np.max(np.abs(actual-expected))<.002


def test_depth_round_trip_with_translation():
    rays,k=perspective_rays(14,90)
    e=np.concatenate([rotation(30).T, np.array([[.3],[.1],[-.4]])],axis=1)[None]
    depth=np.full((1,14,14),2,dtype=np.float32)
    xyz=unproject(depth,e,k[None])
    cam=xyz[0]@e[0,:3,:3].T+e[0,:3,3]
    projected=cam@k.T
    yy,xx=np.mgrid[:14,:14]
    np.testing.assert_allclose(projected[...,:2]/projected[...,2:],np.stack([xx,yy],-1),atol=2e-5)


def test_known_sim3():
    x=np.random.default_rng(7).normal(size=(20,3));r=rotation(40,20);y=2.3*x@r.T+np.array([1,2,3])
    s,rr,t=sim3(x,y)
    np.testing.assert_allclose(s*x@rr.T+t,y,atol=1e-6)


def test_group_constraint_preserves_consistent_rig():
    rs=np.stack([rotation(0),rotation(90),rotation(180)])
    center=np.array([1,2,3]);e=np.zeros((3,3,4));e[:,:3,:3]=rs.transpose(0,2,1);e[:,:3,3]=-np.einsum('nij,j->ni',e[:,:3,:3],center)
    np.testing.assert_allclose(constrain_poses(e,[0,0,0],rs),e,atol=1e-5)


def test_consistency_rejects_depth_disagreement():
    _,k=perspective_rays(14,90);e=np.tile(np.eye(4)[:3],(2,1,1));d=np.full((2,14,14),2,dtype=np.float32)
    xyz=unproject(d,e,k[None].repeat(2,0));mask=np.ones_like(d,bool)
    good,_=consistency_mask(xyz,d,e,k[None].repeat(2,0),mask)
    assert good.sum()>200
    d[1]*=2
    xyz=unproject(d,e,k[None].repeat(2,0))
    bad,_=consistency_mask(xyz,d,e,k[None].repeat(2,0),mask)
    assert bad.sum()==0
