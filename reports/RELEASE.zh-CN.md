VGGT-XR v0.1.0 是实测的重建原型：冻结 VGGT、360 透视投影、confidence/depth-consistency filtering、COLMAP、gsplat 优化和 Unity 键鼠工程。

14 个最终配置完成并通过产物验证，6 项几何/过滤测试通过。合成全景的 30 视图 anchor + known intrinsics 配置开发期留出 PSNR 为 14.65 dB；仍有明显重建误差。原始指标、参数、源头与协议均包含在 reports 中。

附件包含真实厨房和合成全景的标准 Gaussian PLY、点云、相机、报告及实际 gsplat 视频。模型权重与输入数据未打包。视频来自 CUDA gsplat；Unity 的实际运行、视频和 FPS 尚未完成，因为 Windows 取消了 Editor 安装启动。

原始工程代码采用 MIT。示例场景由原始非商用 VGGT-1B 权重推理产生，按学术/非商用用途展示；上游 VGGT、数据、gsplat 和 UnityGaussianSplatting 的各自许可仍适用。
