# 全景稀疏重建与评价证据审查

检索与核验日期：2026-09-15。范围：冻结 VGGT，保留现有 perspective adapter 与 gsplat，改善实际几何和新视角质量；不展开模型重新训练。本文件是 deep-research 的 panorama / critics / methodology 分支，未执行新的 GPU 实验。

## 摘要与研究问题

RQ1：将同位置 panorama 拆成更多视角，能够解决哪些问题，仍缺哪些几何信息？RQ2：哪些已有证据支持在当前实现中加入几何约束，哪些改进需更换模型或 rasterizer？RQ3：怎样区分训练图拟合、开发集调参收益和真实新位置重建收益？

## 检索方法

第一轮使用 OmniGS、360 Gaussian sparse panorama、SparseGS floaters 等宽检索；第二轮追索 360-GS、Splatter-360、FSGS、COLMAP rig 与训练/测试协议；补检 PanoVGGT、BARF 和 LPIPS。候选工作先核对论文题名/作者/年份，再核对摘要、方法原文或作者代码。只采用论文、CVF/arXiv 和官方仓库；未采用搜索聚合站的解释。CVF HTML 部分请求返回 403，已用可检索的 CVF 摘要及 arXiv/官方仓库交叉核验。

## 一、投影与可观测性

[360-GS](https://arxiv.org/abs/2402.00763) 将问题分成 spherical projection 与 sparse/textureless geometry，并用房间布局初始化和约束墙面/地面；[OmniGS](https://openaccess.thecvf.com/content/WACV2025/html/Li_OmniGS_Fast_Radiance_Field_Reconstruction_using_Omnidirectional_Gaussian_Splatting_WACV_2025_paper.html) 则实现直接 ERP 可微 rasterizer。两者都超出“多生成几张 pinhole 图”的作用范围。现有 adapter 解决 VGGT 的输入投影兼容和球面覆盖，但三个输入中心仍然只是三个具有平移信息的位置。

[Splatter-360](https://arxiv.org/abs/2412.06250) 通过 spherical cost volume 加强 wide-baseline 深度估计，而 [PanoVGGT](https://arxiv.org/abs/2603.17571) 改变球面位置编码、旋转增广与训练锚定。它们提供了几何建模的方向，但不能把这些需模型训练的机制直接当作冻结 VGGT 的 adapter 模块。arXiv 摘要仍写代码将发布；补检[官方仓库](https://github.com/YijingGuo-June/PanoVGGT)确认作者已在 2026 年 4 月发布高、低分辨率权重，可作为替换几何前端的独立对照。

官方 [COLMAP rig 文档](https://github.com/colmap/colmap/blob/main/doc/rigs.rst) 已明确给出 panorama → virtual pinhole rig → reconstruction 的路径，固定已知 sensor 相对位姿和内参；对应 [panorama_sfm.py](https://github.com/colmap/colmap/blob/main/python/examples/panorama_sfm.py) 默认采用 overlapping perspective rendering。推论：当前 anchor 固定组内旋转与中心后，下一项真正影响几何的变量是组间 panorama 位姿。相机微调必须共享每组中心与刚体旋转，不能让各 crop 独立平移而破坏 panorama 成像模型。此 API 属于当前 COLMAP main，部署前必须核对已安装 pycolmap 版本，不能假定现有 wheel 已包含。

## 二、稀疏视角中的错误几何

[SparseGS](https://arxiv.org/abs/2312.00206) 用深度先验、未见视角正则和 floater pruning 降低背景塌陷；[FSGS](https://arxiv.org/abs/2312.00451) 联合 Gaussian unpooling、单目深度与增广视角；[Depth-Regularized Optimization](https://arxiv.org/abs/2311.13398) 用 COLMAP 稀疏点对齐单目深度的尺度与偏移。它们共同支持“增加可用几何约束”，并不支持“增加 Gaussian 数量就能保证新视角正确”。

在本项目中可以复用 VGGT 高置信深度作为训练时的软约束，不需要再训练 VGGT，也不必新增大型 depth 模型。建议使用相对深度或鲁棒 inverse-depth loss、低置信和边界 mask、逐步衰减权重；先确认 rendered depth 是 camera-z 还是 ray distance。VGGT 深度存在系统偏差时，强约束会固化错误，因此应同时报告有/无深度项的独立视角差异。以上具体损失设计为本项目的工程推论，尚未由当前实验验证。

容量扩展可改善已观测纹理，但 SH 是视角相关颜色表达，不提供新的几何观测；densification 可填细节，也可把错误相机下的 RGB 残差转成更多浮点。应保留训练相机优化、depth prior、scale/opacity 约束的消融，而不是只比较训练 loss。[BARF](https://arxiv.org/abs/2104.06405) 对 pose 与 scene 的联合优化说明准确相机是独立变量；该工作使用 NeRF，不能直接声称其效果数字适用于 gsplat。

## 三、评价与可信改进

[LPIPS 原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) 支持加入感知质量指标，因为 PSNR/SSIM 无法覆盖所有感知差异；LPIPS 本身也不是几何正确性的检验。[WS-PSNR](https://doi.org/10.1109/LSP.2017.2720693) 的出版摘要说明按球面面积加权以补偿投影不均匀。ERP 最终测试应同时记录等面积球面指标和固定 pinhole viewport；若目前只评估 pinhole 图，应明确命名为 viewport PSNR，不能称作整球 ERP 指标。

固定 [gsplat v1.5.3 simple_trainer](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py) 可核验 test_every=8、SH3、30k steps 与分阶段评估，且提供 pose/depth 配置。这是训练器参考条件，不能把我们现有 SH0 / 1500–3000 steps 的结果冒充其默认基准结果。

本项目建议冻结以下评价协议：

1. 按 panorama 的原始中心分割 train/dev/test，同一 panorama 的 crops 全部属于同一 split；同中心另一朝向不是新位置测试。
2. 新测试集在确定参数前写入清单和 hash；旧 14 组已检查的 heldout 保留为 development ablation。生成新的 synthetic test center 可作为独立诊断，但不能替代真实 panorama 测试。
3. 同一数据、相机、初始化与固定预算比较训练策略，另做升级完整配置对比；至少固定三个 seeds，报告均值、标准差、每图差异。
4. 分开报告 pose ATE/rotation、depth AbsRel/RMSE/coverage 和 RGB PSNR/SSIM/LPIPS；Sim3 和尺度仅从训练相机/深度拟合。GT camera oracle 可诊断容量上限，需显式标为 oracle，不能并入无 GT pipeline 的主结果。
5. kitchen 无 GT camera 时，用提前冻结的真实 heldout 图片评估仍需可靠 test pose；若利用 heldout RGB 做 localization 或 photometric pose fitting，应披露其使用方式，不把它宣称为完全无测试图参与的重建。
6. 新旧 flythrough 使用相同世界坐标、路径、分辨率、裁剪与曝光；只在相机周围极短移动容易隐藏错误，需包含有平移的固定路径。

## 核验表

| 工作 | 元数据确认 | 内容核验来源 | 判定及使用边界 |
|---|---|---|---|
| 360-GS，Jiayang Bai 等，arXiv 2024；作者仓库标 3DV 2025 | arXiv 与 [官方代码](https://github.com/LeoDarcy/360GS) | arXiv HTML 方法 3.1–3.5 | VERIFIED；布局含室内平面假设，不适用于任意物体 |
| OmniGS，Longwei Li 等，WACV 2025 | CVF 与 [官方代码](https://github.com/liquorleaf/OmniGS) | CVF 摘要、README camera_type=3 | VERIFIED；需独立 spherical CUDA 实现；仓库提到 perspective 功能存在已知问题 |
| Splatter-360，Zheng Chen 等，CVPR 2025 | CVF、arXiv 与 [官方代码](https://github.com/thucz/splatter360) | arXiv HTML、代码数据与训练说明 | VERIFIED；学习式 spherical matching，非简单 adapter；HM3D 训练数据声明约 2–3 TB，不应全量下载 |
| PanoVGGT，Yijing Guo 等，2026 | arXiv；页面标 Accepted by CVPR 2026；官方仓库补检 | 摘要用于机制；README 说明 2026 年 4 月已发布高、低分辨率权重 | VERIFIED；可作替换几何前端的独立对照，不混入冻结 VGGT 消融 |
| SparseGS，Haolin Xiong 等 | arXiv 修订题名、[作者仓库](https://github.com/ForMyCat/SparseGS) | OpenReview PDF 方法与 arXiv 摘要 | VERIFIED；早期题名含 Real-Time 360°，修订版不含；此 360°不等于 ERP 输入 |
| FSGS，Zehao Zhu 等，ECCV 2024 | arXiv 与 [官方代码](https://github.com/VITA-Group/FSGS) | arXiv HTML 与摘要 | VERIFIED；unpooling 与 depth 同时存在，不能归因到容量单因素 |
| Depth-Regularized Optimization，Jaeyoung Chung 等，2023/2024 | arXiv v3 | 摘要 | VERIFIED；只采纳 depth 对齐及减少过拟合的摘要结论 |
| BARF，Chen-Hsuan Lin 等，ICCV 2021 | arXiv 与 [官方代码](https://github.com/chenhsuanlin/bundle-adjusting-NeRF) | 摘要、CVF PDF | VERIFIED；跨表示机制证据，不移植其定量效果 |
| LPIPS，Richard Zhang 等，CVPR 2018 | CVF 与 arXiv | CVF 摘要 | VERIFIED；不同 backbone 的分数不可直接混合 |
| Weighted-to-Spherically-Uniform Quality Evaluation，IEEE SPL 2017 | DOI 10.1109/LSP.2017.2720693 出版摘要 | 检索返回的出版摘要；全文未读 | PAYWALL；仅采用球面面积加权动机，不引用正文细节 |
| COLMAP panorama/rig；gsplat v1.5.3 | 官方仓库 | 官方代码与文档 | VERIFIED；需版本锁定，不视为新论文贡献 |

## 综合判断与开放问题

RQ1：增加朝向补覆盖和投影兼容，新增平移中心才增加 depth triangulation 信息。RQ2：在冻结 VGGT 的边界内，group-rigid camera refinement、可靠深度软约束和有预算的容量优化最可落地；PanoVGGT 现有权重属于替换几何前端的独立路线。RQ3：独立中心、冻结 test 清单、GT 使用披露、pose/depth/RGB 联合评价和相同演示路径，是辨别实际改进与训练过拟合的必要条件。

自我反驳：现有低容量确实可能是主要瓶颈，不能先验否定 SH/densification；因此应先跑固定输入的容量对照。另一方面，新训练 loss 更低不保证 geometry 更好，depth 先验也可能错误；任何模块须按同一独立 test 验收。最终真实 XR 场景的判断仍取决于真实多位置 panorama 数据，synthetic box 的收益不外推为真实重建能力。
