# 实际交付状态

状态日期：2026-09-15

根据用户最终决定，Unity 运行验证已移出 Stop Line。项目以服务器端 VGGT、360° Adapter、置信度过滤、gsplat 优化、冻结定量评估和 GitHub Demo 为验收范围；当前范围已经完成。

| Stop Line 必做项 | 状态 | 证据 |
| --- | --- | --- |
| VGGT baseline | 完成 | 官方冻结权重；camera/depth；COLMAP；输入指纹 |
| 360° Adapter | 完成 | direct/cubemap/overlap/极点/rig anchor；真实 ERP 方向标定 |
| confidence filtering | 完成 | 置信度阈值、跨全景深度一致性、voxel 过滤 |
| Gaussian Splatting | 完成 | CUDA 前向/反向、SH0 基线、SH3 质量路径、PLY 与视频 |
| quantitative evaluation | 完成 | 合成精确真值与真实 ZInD 留出位置；ATE/depth/PSNR/SSIM/LPIPS/时间/显存 |
| GitHub Demo | 完成 | 代码、报告、对比图、指标 JSON 与 v0.3.0 成果包 |

Unity 键鼠工程与导出脚本仍保留在仓库中，但不要求安装 Editor，不进行 Unity 编译、视频或 FPS 验证，也不作为完成条件。

最新自然解析场景 v2 在 30 个未见视图达到 22.7171 dB PSNR、0.8012 SSIM、0.2308 LPIPS。真实 ZInD pano-5 留出位置上，质量路径相对同为 6000 步的 SH0 基线从 18.7464 提升到 19.2660 dB，SSIM 从 0.6700 提升到 0.6881，LPIPS 从 0.5108 降到 0.4148。真实未见位置仍有明显模糊，报告不把它描述为生产级重建。

13 项测试和 GitHub CI 通过。服务器质量验证通过有限 checkpoint、Gaussian 硬上限、非零 SH、有限相机、全景共享中心、冻结测试哈希和三种子完整性检查。

服务器目录为 `/root/autodl-tmp/vggt-xr`。系统盘使用约 829 MiB；数据盘剩余约 157 GiB。环境、权重、数据、缓存和输出均放在数据盘。完整扩展结果见 `PANORAMA_EXTENSION_REPORT.zh-CN.md` 和 `panorama_extension.json`。
