# VGGT-XR 执行计划与 Stop Line

## 交付范围

只做冻结 VGGT 推理、360° Adapter、confidence filtering、gsplat Gaussian 优化、定量评估和可复现 GitHub Demo。不训练 VGGT，不增加语义分割，不加入 OpenXR/头显、Quest standalone 或大型数据集训练。根据用户 2026-09-15 的最终决定，Unity 运行验证已从 Stop Line 移除；现有 Unity 导出代码仅作为可选能力保留。

## 资源规则

服务器工作根目录：`/root/autodl-tmp/vggt-xr`。已有 Python/CUDA 等系统工具留在系统盘。新增隔离 Python 环境、模型、pip/Hugging Face/Torch 缓存、CUDA 编译产物、临时文件、数据、日志、输出全部放数据盘。每个阶段检查数据盘剩余空间，至少保留 15 GiB；权重分片下载另预留 10 GiB。第一轮新增占用控制在 20 GiB 左右。已有 `paircheck` 项目与 181GB 数据不动。

本机下载、Unity 项目/Library、模型和打包结果放 `D:\VGGT-XR-artifacts`。本地 GitHub 代码目录只放源码、文档和小型结果图。服务器 5090 有约 32GB 显存；默认 336×336、最多 24 个视图、最多 30,000 个初始 Gaussian。推理和优化顺序执行。

## 数据和有效性

1. 公开 Tiny NeRF Lego：106 张原始 100×100 图像，抽取最多 24 张训练，预留 5 张测试。它适合验证流程与消融，不等同于完整高分辨率 NeRF Synthetic benchmark。
2. 带纹理的解析室内合成场景：三个不同相机中心的 ERP，准确的位姿和逐像素深度，另一个相机中心的 10 张留出图（8 个水平朝向 + 上/下）。它验证 ERP 投影和几何过滤，不冒充 Replica、Structured3D 或真实拍摄。
3. 真实 ZInD 固定协议：同一区域 pano 2/4/6 用于训练，pano 5 完全留出，仅用于最终新位置渲染评估。它验证真实同区域新位置泛化，不代表跨住宅泛化。

同一张全景的多个透视视图共享中心，只有旋转。不能把它们当作有基线的多视角来声称完成可靠三角化。多视角深度一致性投票排除同一全景组。

## 实验与阶段门槛

| 阶段 | 工作 | 验收与动态调整 |
| --- | --- | --- |
| 0 | 存储、CUDA、权重 | gsplat 前向/反向成功，权重 SHA256 校验，5 个几何测试通过 |
| 1 | VGGT baseline | 输出位姿、深度、真实 COLMAP 模型与彩色点云；记录运行时间和峰值显存 |
| 2 | 360 Adapter | ERP direct、6 面 90° cubemap、8 水平面 100° overlap；检查全景接缝、极点和旋转约定 |
| 3 | filtering | raw、confidence 分位数、confidence + 跨全景组 depth consistency；记录错误与覆盖率 |
| 4 | gsplat | 各初始化使用相同 seed、步数和 Gaussian 上限；输出标准 PLY、留出视图和 PSNR/SSIM/LPIPS |
| 5 | 定量与 Demo | 冻结留出视图；报告 PSNR/SSIM/LPIPS、相机/深度误差、时间/显存；输出真实 gsplat 视频和 PLY |
| 6 | GitHub Demo | 说明、单命令入口、真实指标、预览图/视频、PLY、复现命令；Unity 工程只作可选附件 |

第一轮共 8 个配置：Lego raw/confidence/consistency；ERP direct；cubemap；overlap；overlap + pose constraint；overlap + consistency。每个配置默认 Gaussian 1500 步。这是项目级、固定预算验证，不是 SOTA 训练。

8 个水平 overlap 视图只覆盖 360° 水平方位，不能完整覆盖天顶/地面。留出视图包含两极，防止只评估水平视图而隐藏缺口。如观察到两极缺失，在相同 3 个全景中心增加 `--include-poles --max-views 32`（共 30 视图）作为一次有针对性的覆盖率调整，并单独报告增加的计算预算。

## 指标协议

- 相机：训练相机中心拟合 Sim(3)，ATE RMSE、选定视图间相对平移误差、旋转角误差。不是未经对齐的米级绝对定位，也不是完整序列轨迹基准。
- 深度：解析合成场景上，所有像素共同拟合一个全局 median scale。不同过滤方法使用同一 scale，同时报告 AbsRel、RMSE 和 coverage。
- 渲染：留出图从未用于 VGGT 或优化。测试 GT 相机仅通过训练相机拟合的 Sim(3) 转到重建坐标。报告 PSNR、SSIM、LPIPS-SqueezeNet。
- 系统：VGGT 推理时间/显存，Gaussian 优化时间/显存、数量、文件大小；最终运行数据来自服务器端 CUDA gsplat。
- ERP direct 不满足 pinhole 几何模型，GT pinhole 深度与相机指标标记 N/A，不为凑表格创造不可解释的数字。

## 动态调整规则

1. OOM 时先降低输入视图/分辨率，保存失败日志与新配置。不能通过重复单张全景补视差。
2. 过滤后少于 64 个点则失败并停止该配置，不静默回退到 raw。必要时显式调低 confidence 分位数或放宽一致性阈值，并记录修改。
3. shared-center/known-rotation 后处理只作为消融。如果破坏 learned depth 与 pose 的一致性，默认关闭；不预先宣称改善。
4. 如果过滤降低几何错误却损害留出渲染，选择兼顾覆盖率的配置，不把其中一个指标的改善称为全面提升。
5. 如果 1500 步仍显著欠拟合，可将 raw 和最好的 filtered 配置一起增加到 3000 步；不只延长有利配置。
6. 真实数据或 GitHub 发布失败时，把该项标为待完成，并具体记录依赖。其余阶段继续运行。
7. 完成上述范围后停止扩展功能。后续提升优先增加真实全景采集中心和跨 ERP 几何一致性，不新增 Unity/XR 验证工作。
