# 实际交付状态

| Stop Line 必做项 | 状态 | 证据 |
| --- | --- | --- |
| VGGT baseline | 完成服务器实跑 | 官方冻结权重；camera/depth；COLMAP 模型；输入指纹 |
| 360 adapter | 完成服务器实跑 | direct/cubemap/overlap/极点/rig anchor 消融 |
| confidence filtering | 完成服务器实跑 | 严格阈值、跨全景组 depth consistency、voxel；回归测试 |
| Gaussian Splatting | 完成服务器实跑 | gsplat CUDA 前向/反向、SH0 优化、真实 PLY/视频 |
| Unity 键鼠交互 | 工程和源码准备完成，运行待验收 | D 盘工程已生成；Editor 安装启动返回 Windows “操作已被用户取消”；没有编译结果、Unity 视频或 Unity FPS |
| quantitative evaluation | 完成开发期消融 | 14 最终配置；ATE/rotation/depth/PSNR/SSIM/LPIPS/时间/显存/数量；不是独立最终泛化测试 |
| GitHub Demo | 已发布代码、报告和 gsplat 预览 | https://github.com/BoPythonAI/VGGT-XR |

6 个几何/过滤测试通过。所有 14 配置的 COLMAP 图像/点数量、有限深度/位姿/Gaussian 参数和输入指纹通过验证，见 `artifact_verification.json`。

实测 30 视图 fullsphere PSNR 为 7.1393 dB，anchor + known intrinsics 为 14.6459 dB，ATE 为 0.8553 → 0.2572（解析房间，训练中心 Sim(3)）。最终深度 AbsRel 仍为 0.2252，重建仍有纹理失真/漂浮点。Tiny NeRF 的 PSNR 改善较小，不声明全面提升。

服务器目录：`/root/autodl-tmp/vggt-xr`。系统盘已用约 0.764 GiB、剩余 29.24 GiB；数据盘剩余约 158.14 GiB。权重、新增环境、缓存、数据、临时文件和输出全部位于数据盘。

本机 Unity 工程：`D:\VGGT-XR-artifacts\UnityProject`，包含真实厨房 Gaussian、RGB 点云、confidence 点云和相机。另一个合成全景场景在 `D:\VGGT-XR-artifacts\demo\overlap_anchor`。官方 Editor 安装包已下载至 D 盘并验证 Authenticode 签名，但安装尚未成功。

完成 Unity 验收的具体入口：安装并激活 Unity 2022.3，再运行 `scripts/desktop_demo.ps1` 传入 Editor 路径以自动生成/打包场景；用 `scripts/open_unity.ps1` 运行交互界面并保持缓存/临时文件在 D 盘。详细验收步骤见 `docs/UNITY_VALIDATION.md`。

真实多中心全景数据尚未提供，目前 360 定量证据来自可控解析合成场景；真实厨房照片只用于定性 Demo。没有宣称实现头显 XR/OpenXR；本次目标按用户最后的限定采用 Unity 键鼠。
