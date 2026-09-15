# VGGT-XR 重建质量：从输入、几何到 Gaussian 优化的证据与实验路线

## 摘要

现有 Demo 能辨认主体，却存在模糊、重影和漂浮。将这些现象笼统归因于 VGGT，无法指导有效改进。本调查围绕输入信息、几何一致性、Gaussian 表达与可观测性，核验原始论文、作者代码和固定版本 API，并对当前实现进行审计。调查覆盖 17 篇论文，采用三条独立检索视角和两轮以上检索，分别处理几何、优化与全景评价。现实现将几何预测和颜色监督绑定在 336 分辨率，采用固定数量的 SH0 Gaussian，保持相机不变；这些简化同时限制细节恢复和收敛。文献提示容量扩展具有合理性，但稀疏全景研究也表明，不正确的相机和有限的平移观测可能被更强训练器拟合成错误结构。本项目据此采用有数量上限的增密、真实 SH、训练分辨率解耦及保持 panorama rig 的小幅相机优化，并以训练/开发集分离和同预算对照评估。文献结论用于提出可检验路线，实际收益以独立保存的服务器实验记录为准。

## 1. 研究问题与项目边界

现有真实图片 Demo 使用 24 张原始 779×520 图像，却仅以 336×336 填充后的图像训练；全景 Demo 来自三个合成拍摄位置。两者具有不同的证据能力：前者可检验原图细节与视觉稳定性，后者支持已知 pose/depth 的诊断，不能相互替代。

- RQ1：模糊、重影和漂浮分别可能来自哪些实现限制，怎样区分输入、相机、深度与表示容量？
- RQ2：哪些公开方法能够在冻结 VGGT、保留 gsplat、使用 RTX 5090 32 GB 的边界内落地，其失败条件是什么？
- RQ3：什么实测足以支持“质量改善”，怎样避免训练拟合和更换评价协议带来的假提升？

本报告不提出重新训练 VGGT 或生成式补图路线；不把 Unity 功能增加计入重建质量改善。旧 14 组是已用于决策的开发集实验。新的质量结果、代码、视频分别保存，不覆盖历史结果。

## 2. 检索、核验与证据组织

2026-09-15 使用 web 检索、arXiv/CVF、作者项目及官方源码。依据 deep-research 方法，三条独立检索视角并行执行：几何与相机、Gaussian 表达与优化、全景与评价反证。每条至少先宽检索再按发现的术语集中补检。关键词包括 `VGGT bundle adjustment preprocessing 518`、`Gaussian inaccurate camera poses`、`gsplat SH densification opacity reset`、`AbsGS MCMC`、`panorama sparse wide-baseline`、`floaters unseen views`、`COLMAP rig`。主报告在合并核验记录后统一综合，分支记录位于 [几何来源](research/geometry_sources.zh-CN.md)、[优化来源](research/gaussian_sources.zh-CN.md)、[全景评价来源](research/panorama_evaluation_sources.zh-CN.md)。

每篇文献先核对标题与作者，再核对摘要、正文或作者实现是否支持具体主张。正式语料包含 17 篇论文；官方代码作为实施证据另计。WS-PSNR 仅检索到出版摘要，未将其具体算法纳入实施依据。检索初期 arXiv 摘要写代码将发布；补检官方仓库确认作者已在 2026 年 4 月发布 PanoVGGT 高、低分辨率权重。它因此成为可执行的后续原生全景对照，但会替换本项目的冻结 VGGT+adapter 几何前端，不能混入本轮训练器消融。博客和第三方解读不作为方法证据；issue 仅用于发现问题，版本错误的结论由固定源码和实际测试确认。

## 3. 质量误差的分解

下表按导致残差的对象组织证据，避免把所有问题归为“模型不够强”。跨类别的方法在后续讨论说明其同时改变的变量。

| 类别 | 当前可验证限制 | 候选干预 | 可以证明什么 | 不能据此证明什么 |
|---|---|---|---|---|
| 输入与几何 | 336 监督、padding、固定 E/K、深度不一致 | 高清监督、精确 K 变换、真实 tracks BA、受约束 pose refinement | 已观察纹理与相机一致性的改善 | 修复全部未观测几何 |
| Gaussian 表达与优化 | SH0、固定点数、常数位置 LR | SH3、局部尺度、位置 LR 衰减、受预算增密 | 同输入下的表达与收敛改善 | 新增有效拍摄位置 |
| 稀疏全景与可观测性 | 三个位置、投影畸变、组间位姿误差 | 固定 rig、可靠深度软约束、更多真实平移位置 | 满足已知投影物理约束 | 三个位置等价三十个独立中心 |
| 评价 | 已查看的 dev 图、无真实全景 GT | 固定测试清单、同轨迹、pose/depth/RGB 分报 | 在明确协议下的有界收益 | 真实全景泛化或 SOTA |

## 4. 输入信息和几何一致性

VGGT 提供前馈相机、深度与点图初始化，COLMAP 则利用经过验证的多视图观测进行三角化和 BA；前者减少初始化成本，后者需要实际 tracks，二者都没有保证任意低纹理或退化数据可重建。[1][2] 当前项目的无 tracks COLMAP 格式导出不能视为已经运行 BA。[VGGT](https://arxiv.org/abs/2503.11651)、[COLMAP](https://demuc.de/papers/schoenberger2016sfm.pdf)

锁定版本官方 VGGT COLMAP demo 使用 518 几何推理和更高分辨率跟踪；其保比例预处理与当前 kitchen 的保比例 padding 属于兼容路线，区别在于当前颜色监督也被压到 336。这里不能把 resize 函数误读成已修正 kitchen 仍被拉伸，也不能把导出 1080p 视频当作获得高清重建。工程推论是解耦推理与颜色监督，按实际内容 rectangle 的缩放/偏移计算 `K_new = A K_old`，屏蔽人工 padding。TinyNeRF 原始 100×100，任何上采样都不能恢复未存在的信息。[固定官方 demo](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/demo_colmap.py)、[预处理](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/utils/load_fn.py)

BARF 研究相机与 NeRF 的联合配准，而 BAD-Gaussians 和 Robust Gaussian Splatting 在 GS 中分别处理相机/曝光轨迹或多类输入退化；它们共同提示相机是独立质量变量，却也说明模糊具有多种机制，不能仅凭外观确诊。[3][4][5] 本项目的小幅光度 refinement 只是一种可落地工程候选，未复现它们的完整方法或承诺论文增益。[BARF](https://arxiv.org/abs/2104.06405)、[BAD-Gaussians](https://arxiv.org/abs/2403.11831)、[Robust Gaussian Splatting](https://arxiv.org/abs/2404.04211)

| 几何路线 | 观测约束 | 可用条件 | 必须记录的失败边界 |
|---|---|---|---|
| 前馈 VGGT | 学习的多视图先验 | 与训练分布兼容、有足够关联视角 | 大旋转、弱纹理、全景域差异 |
| tracks + 几何 BA | 多帧真实像素重投影 | track 连接充分、三角化非退化 | 少 tracks、错误对应、尺度/gauge 漂移 |
| 小幅光度 pose refinement | 渲染 RGB 与训练 RGB | 初始位姿接近、固定 gauge、渐进开放 | 几何与相机共适应、训练拟合增加 |

本项目的 BA 探索用真实 SIFT tracks、固定预测 K 与 robust BA；这不是官方 VGGT 跟踪器复现。BA 改变 E/K 后，必须重新反投影，检查深度与 camera gauge 的一致性。用训练相机中心 Sim(3) 回到原坐标只处理整体相似变换，不等于纠正逐帧深度系统偏差。相机微调冻结首相机/rig，并限制旋转和平移，所有测试 RGB 都不参与更新。

## 5. Gaussian 表达、分配与收敛

原始 3DGS 同时优化各向异性 Gaussian、SH 颜色与密度分配，AbsGS 则分析梯度抵消导致的大 Gaussian 欠分裂；两者都说明只有 CUDA rasterizer 并不等于完整质量训练。[6][7] 当前固定 SH0 的限制在镜头变化与细纹理处尤其值得检查，但 SH 不提供新的几何观测。[3DGS](https://arxiv.org/abs/2308.04079)、[AbsGS](https://arxiv.org/abs/2404.10484)

Default/AbsGrad 按投影梯度复制和分裂，MCMC 改为重定位与随机探索；二者改善密度配置的路径不同，但都不能把错误相机自动变成正确相机。[7][8] 当前选择项目内受数量上限约束的 Default/AbsGrad，以最大投影梯度分配剩余点数，沿用 gsplat ops 同步更新 Parameter、Adam 和 strategy 状态；MCMC 作为备选，不同时引入以免混淆变量。[AbsGS](https://arxiv.org/abs/2404.10484)、[MCMC](https://papers.nips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)

固定 gsplat 1.5.3 官方示例支持 SH3、分阶激活、局部三个近邻均方距离尺度、opacity 0.1、位置 LR 指数衰减。这些默认值用于建立合理参考，而非被证明适合当前数据的最优超参数。PLY 导出必须保存真正高阶系数，按 channel-major 顺序打包，不能继续只填零。[官方训练器](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py)

| 优化变量 | 当前原型 | 质量候选 | 控制条件 |
|---|---|---|---|
| 颜色 | sigmoid RGB / SH0 | SH3，逐阶激活 | 相同输入、步数、测试 |
| 密度 | 固定 30k/60k | 梯度增密与剪枝 | 硬 cap；状态同步；记录计数 |
| 初始尺度 | 平均近邻距离及全局 extent 截断 | RMS 三近邻、camera-scale 范围 | 不由远处离群点主导 LR |
| 位置更新 | 常数 LR | 指数衰减 | 后半段固定密度精修 |
| opacity | 0.3，无有效重置 | 0.1，受控重置 | 留出恢复阶段，避免末步重置 |

源码发现 v1.5.3 的 `step % reset_every == 0 & step > 0` 被解析为链式比较而恒假。小跑直接核验了该条件和修复后的实际重置事件。项目 wrapper 使用明确的布尔条件，只在增密窗口重置；不修改系统安装包。这个错误是可验证版本问题，但不能据此断言所有旧结果质量差都由它造成。[固定 strategy 源码](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/default.py)

Mip-Splatting 针对采样率变化加入 3D smoothing 与二维 Mip filter，而 AbsGS 针对空间细节分配，两者处理不同的残差来源。[9][7] gsplat 的二维 antialias 仅作可选独立对照；不能称为完整 Mip-Splatting，Unity 中也必须核验一致性。[Mip-Splatting](https://arxiv.org/abs/2311.16493)、[AbsGS](https://arxiv.org/abs/2404.10484)

## 6. 稀疏全景的几何边界

SparseGS、FSGS 与 depth-regularized optimization 分别用深度、未见视角/剪枝或尺度对齐改善稀疏视角退化；共同证据支持几何约束与容量同时考虑，不能将组合方法的增益全部归于 densification。[10][11][12] 本项目可复用可靠 VGGT 深度作软约束，但其错误也会被强监督固化，需单独消融；本轮不把 GT 深度输入优化器。[SparseGS](https://arxiv.org/abs/2312.00206)、[FSGS](https://arxiv.org/abs/2312.00451)、[Depth-Regularized Optimization](https://arxiv.org/abs/2311.13398)

360-GS 使用室内布局和球面投影，OmniGS 提供直接全景 rasterizer；两者比 perspective adapter 更深入改变成像或结构假设，不能把它们的指标直接归于切片数量。[13][14] Splatter-360 通过球面 cost volume 处理 wide-baseline 几何，而 PanoVGGT 改变球面编码与训练；这些说明冻结 perspective VGGT 的域差异仍存在。[15][16] PanoVGGT 官方权重现已发布，可作为另列的全景原生前端对照；它不是 adapter 的小改动，也不能把论文或新权重的结果写成我们的结果。[360-GS](https://arxiv.org/abs/2402.00763)、[OmniGS](https://github.com/liquorleaf/OmniGS)、[Splatter-360](https://arxiv.org/abs/2412.06250)、[PanoVGGT 官方仓库](https://github.com/YijingGuo-June/PanoVGGT)

| 全景路径 | 改变对象 | 对本轮的作用 | 当前不采用原因 |
|---|---|---|---|
| perspective rig | 输入与相机参数化 | 保留已知 K、相对旋转、共同中心 | 已采用；需改善组间位姿 |
| 布局/深度正则 | 场景几何先验 | 诊断漂浮和稀疏覆盖 | 布局假设不能外推任意物体；深度需可靠 |
| 原生 ERP rasterizer | 成像模型 | 减少切片近似 | 增加 CUDA 和 Unity 一致性工作 |
| 球面学习模型 | 网络与训练域 | 可用已发布 PanoVGGT 权重作独立对照 | 会替换核心几何前端；依赖和协议需另行锁定 |

同一 ERP 的虚拟视角共享中心、固定相对旋转。30 张切片来自三个位置，只有三个平移观测中心。当前相机优化以每中心一个世界刚体增量复合所有切片，保持组内约束；独立移动各切片虽可能降低训练损失，却不满足中心全景模型。官方 COLMAP 当前有 panorama rig 示例，但服务器 pycolmap 4.2.0 实测无 panorama API，因此本轮不拼接未经版本核验的 main 模块。[COLMAP rig 文档](https://colmap.github.io/rigs.html)

## 7. 实验路线和评价

3DGS 与 sparse-view 工作都以未见视角质量作为关键指标，LPIPS 则补充 PSNR/SSIM 对感知差异的不足；它不能作为几何正确性的替代。[6][10][17] 本轮保留 PSNR、SSIM、LPIPS-SqueezeNet、pose 指标和逐图对照，不混用 LPIPS backbone。[3DGS](https://arxiv.org/abs/2308.04079)、[SparseGS](https://arxiv.org/abs/2312.00206)、[LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)

第一阶段同一数据/初始相机采用 6000 steps，对比旧训练器、SH3+固定密度、SH3+增密、SH3+增密+受约束 pose。旧训练器延长步数的结果用于识别“只是多训练”的收益。质量候选包含局部尺度和 LR 的共同升级，不能从该对照单独声称 SH 的因果贡献。真实图像单独升 518 推理、672 监督，保留无 GT 的定性边界。参数、预测 hash、训练 seed、点数、runtime、显存与磁盘均保存。

开发集用于选择配置；预先冻结的新合成位置只用于选择结束后的最终诊断。所有测试图像不进入 VGGT 或训练器，GT pose 只经训练相机 Sim(3) 转换后用于渲染评价。新测试仍在相同合成场景，不能叫跨场景泛化。真实图像无可靠 GT test pose，本轮报告真实图视觉对照与明确标记的 training reconstruction scores，不能把它们称 heldout 质量。

## 8. 跨分支判断、开放问题与结论

输入升级增加已记录的纹理信息，密度和 SH 增加表达，相机/深度提高观测之间的一致性；三者不能互相替代。相机优化和更强 Gaussian 可能共同降低训练损失却破坏新位置几何，因此最终路线取决于对照结果。局部姿态调整只覆盖小误差，不解决组间的大幅旋转错误；错误深度正则也可能阻止几何修复。真实高质量多位置全景采集仍是未完成证据，新增交互不会改变这个事实。

RQ1：实现审计确立了分辨率、表示和优化简化；重影根因仍需消融区分，不从画面单独确诊。RQ2：合理落地项是保比例高清监督/精确 K、真实 SH、camera-scale 初始化、LR 衰减、预算增密与保 rig 的小幅 pose；真实 tracks BA 可探索但受连接和 gauge 条件约束。RQ3：支持质量改善的证据须来自同预算开发对照、冻结最终新位置测试和相同轨迹视频，并明确真实数据与 GT 使用边界。本调查的贡献是将质量问题拆成可以直接验证的干预对象，具体实验收益见 [质量实验记录](../reports/QUALITY_REPORT.zh-CN.md)。

## 参考文献

[1] Jianyuan Wang, Minghao Chen, Nikita Karaev, et al., "VGGT: Visual Geometry Grounded Transformer," CVPR, 2025.

[2] Johannes L. Schönberger, Jan-Michael Frahm, "Structure-from-Motion Revisited," CVPR, 2016.

[3] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, et al., "BARF: Bundle-Adjusting Neural Radiance Fields," ICCV, 2021.

[4] Lingzhe Zhao, Peng Wang, Peidong Liu, "BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting," ECCV, 2024.

[5] François Darmon, Lorenzo Porzi, Samuel Rota-Bulò, et al., "Robust Gaussian Splatting," arXiv:2404.04211, 2024.

[6] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM Transactions on Graphics, 2023.

[7] Zongxin Ye, Wenyu Li, Sidun Liu, et al., "AbsGS: Recovering Fine Details for 3D Gaussian Splatting," arXiv:2404.10484, 2024.

[8] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, et al., "3D Gaussian Splatting as Markov Chain Monte Carlo," NeurIPS, 2024.

[9] Zehao Yu, Anpei Chen, Binbin Huang, et al., "Mip-Splatting: Alias-free 3D Gaussian Splatting," CVPR, 2024.

[10] Haolin Xiong, Sairisheek Muttukuru, Hanyuan Xiao, et al., "SparseGS: Sparse View Synthesis using 3D Gaussian Splatting," 3DV, 2025; revised arXiv:2312.00206, 2026.

[11] Zehao Zhu, Zhiwen Fan, Yifan Jiang, et al., "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting," ECCV, 2024.

[12] Jaeyoung Chung, Jeongtaek Oh, Kyoung Mu Lee, "Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images," arXiv:2311.13398, 2024.

[13] Jiayang Bai, Letian Huang, Jie Guo, et al., "360-GS: Layout-guided Panoramic Gaussian Splatting For Indoor Roaming," arXiv:2402.00763, 2024.

[14] Longwei Li, Huajian Huang, Sai-Kit Yeung, et al., "OmniGS: Fast Radiance Field Reconstruction using Omnidirectional Gaussian Splatting," WACV, 2025.

[15] Zheng Chen, Chenming Wu, Zhelun Shen, et al., "Splatter-360: Generalizable 360° Gaussian Splatting for Wide-baseline Panoramic Images," CVPR, 2025.

[16] Yijing Guo, Mengjun Chao, Luo Wang, et al., "PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery," arXiv:2603.17571, 2026.

[17] Richard Zhang, Phillip Isola, Alexei A. Efros, et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR, 2018.
