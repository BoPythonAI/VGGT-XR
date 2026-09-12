# Actual experiment results

run | views | retained | ate | depth_absrel | depth_coverage | psnr | ssim | lpips_squeeze | vggt_vram_gb | raster_fps
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
cubemap | 18 | 30000 | 0.8618 | 0.1218 | 0.6498 | 6.4676 | 0.0081 | 0.8406 | 5.6822 | 3316.0690
erp_direct | 3 | 30000 | N/A | N/A | N/A | N/A | N/A | N/A | 4.7838 | 3574.5545
kitchen | 24 | 60000 | N/A | N/A | N/A | N/A | N/A | N/A | 5.8024 | 3070.0498
nerf_confidence | 23 | 30000 | 0.3001 | N/A | N/A | 12.2979 | 0.6574 | 0.4024 | 5.7756 | 3667.0549
nerf_confidence_masked | 23 | 30000 | 0.3001 | N/A | N/A | 12.1427 | 0.6492 | 0.3176 | 5.7756 | 3412.1653
nerf_consistency | 23 | 30000 | 0.3001 | N/A | N/A | 12.0550 | 0.6436 | 0.3194 | 5.7756 | 3298.8530
nerf_consistency_masked | 23 | 30000 | 0.3001 | N/A | N/A | 12.1575 | 0.6500 | 0.3152 | 5.7756 | 3381.2957
nerf_raw | 23 | 30000 | 0.3001 | N/A | N/A | 12.1177 | 0.6550 | 0.4193 | 5.7756 | 3612.6455
nerf_raw_masked | 23 | 30000 | 0.3001 | N/A | N/A | 11.9799 | 0.6211 | 0.3437 | 5.7756 | 3496.3009
overlap | 24 | 30000 | 0.8610 | 0.1774 | 0.6486 | 6.8346 | 0.0437 | 0.8617 | 5.8024 | 3553.7326
overlap_anchor | 30 | 30000 | 0.2572 | 0.2252 | 0.6489 | 14.6459 | 0.3637 | 0.4733 | 5.9145 | 3701.9407
overlap_consistency | 24 | 30000 | 0.8610 | 0.1700 | 0.5624 | 6.8487 | 0.0411 | 0.8577 | 5.8024 | 3541.3368
overlap_constrained | 24 | 30000 | 0.5544 | 0.1774 | 0.6486 | 10.1214 | 0.2630 | 0.7691 | 5.8024 | 3220.3209
overlap_fullsphere | 30 | 30000 | 0.8553 | 0.2252 | 0.6489 | 7.1393 | 0.0643 | 0.7955 | 5.9145 | 3686.5391

Protocols: NeRF source is the public 100x100 Tiny NeRF dataset enlarged to 336; do not interpret as full-resolution Lego benchmark. Analytical room is a controlled synthetic box scene, not Replica/Structured3D or a real room. Heldout images are excluded from VGGT and GS optimization. Sim(3) alignment uses training cameras only. Depth scale is fixed across filtering ablations; coverage is reported beside error. LPIPS uses SqueezeNet. Gaussian optimizer uses SH0, fixed Gaussian count, no densification. ERP-direct does not admit pinhole GT depth or camera evaluation. Raster FPS is headless server gsplat, not Unity.
