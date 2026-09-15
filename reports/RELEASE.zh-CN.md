VGGT-XR 是实测的重建原型：冻结 VGGT、360 透视投影、confidence/depth-consistency filtering、COLMAP、gsplat 优化、定量评估和可复现演示。Unity 键鼠工程作为可选导出代码保留，不属于最终验收范围。

14 个最终配置完成并通过产物验证，6 项几何/过滤测试通过。合成全景的 30 视图 anchor + known intrinsics 配置开发期留出 PSNR 为 14.65 dB；仍有明显重建误差。原始指标、参数、源头与协议均包含在 reports 中。

附件包含真实厨房、自然解析全景和真实 ZInD 样例的标准 Gaussian PLY、点云、相机、报告及实际 gsplat 视频。模型权重与输入数据未打包。最终视频来自 CUDA gsplat；本项目不使用 Unity 视频或 Unity FPS 作为实验依据。

原始工程代码采用 MIT。示例场景由原始非商用 VGGT-1B 权重推理产生，按学术/非商用用途展示；上游 VGGT、数据、gsplat 和 UnityGaussianSplatting 的各自许可仍适用。
