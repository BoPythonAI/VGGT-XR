# 重建质量调查：几何、相机与输入分辨率

调查日期：2026-09-15。本文是质量改进的文献与源码证据，不是新实验结果。范围限定在既有 VGGT → 360° Adapter → confidence filtering → gsplat 流水线；不改 VGGT 权重，不以生成式补图替代几何验证。

## 摘要与研究问题

RQ1：当前重影和模糊是否足以判断为相机误差，哪些证据能区分相机、深度与表示容量问题？RQ2：官方 VGGT 的高质量推理与 BA 路径和现实现有哪些差异？RQ3：360° 虚拟视角如何保留物理约束，在单张 RTX 5090 上怎样进行有边界的改进？

文献支持相机准确性是可微重建的重要条件，但画面模糊本身不能定位根因。源码显示当前项目将 GS 监督图像绑定为 VGGT 的 336×336 输入，且 GS 阶段保持相机不变。这使低频输入、相机误差和固定高斯预算同时成为候选原因。建议先解耦监督分辨率与推理分辨率，再用固定测试协议判断是否需要几何 BA 或小幅光度 pose refinement。

## 方法与检索记录

第一轮广检索使用 `VGGT Visual Geometry Grounded Transformer bundle adjustment demo_colmap`、`Structure from Motion Revisited`、`BARF Bundle Adjusting Neural Radiance Fields`，覆盖前馈几何、经典 SfM 与光度相机优化。第二轮集中搜索 `Gaussian splatting inaccurate camera poses`、`VGGT preprocessing 518`、`COLMAP panoramic pure rotation`，核验论文正文、官方代码、最新 rig 文档。补检索直接读取 `pycolmap/panorama.py`，确认 360° 支持并非仅文档设想。

所有使用的论文经标题及作者核验，方法主张经摘要或正文复核；博客、论坛与第三方代码讲解未作为技术证据。未找到本项目数据上 BA 必然改善的证据，这一点保留为待实验判断。

## 一、输入信息与坐标变换

VGGT 论文采用保比例缩放，长边 518，短边裁剪且为 14 的倍数；官方 COLMAP demo 则先保比例填充至正方形，在 518 做几何预测，在 1024 做跟踪。这两种官方路径都不能等同于把任意原图拉伸至正方形。[论文正文](https://arxiv.org/html/2503.11651v1#S3.SS4)、[项目锁定版本 demo](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/demo_colmap.py#L114)、[预处理源码](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/load_fn.py#L88)

本地 `src/inference.py` 审计：`infer(..., size=336)` 保存的 `images` 同时用于 GS 监督；`src/gaussian.py` 根据这些监督图的尺寸渲染。因此仅把视频导出到高清不能恢复没有训练过的细节，单纯升 VGGT 分辨率也不等于使用原始高清监督。kitchen 已在上游准备阶段保比例填充，不能把这里的一般 resize 行为误称为当前 kitchen 一定被拉伸。

实施建议属于本调查的工程推论：保存原图或高质量虚拟投影图，使用独立的 GS 训练尺寸；从真实缩放、padding、crop 的仿射映射计算 `K_new = A K_old`，同步有效像素 mask。不要让 padding 白边参与几何点初始化与全部光度损失。分辨率对照需要在相同评估尺寸比较，避免把指标变化与画面采样变化混在一起。

## 二、稀疏重投影几何与光度相机优化

VGGT 论文同时评估前馈与 VGGT+BA；这说明 BA 是允许的后处理路线，而不是模型使用的必要前提。COLMAP 则通过几何验证、三角化、离群点剔除和 BA 联合提高结构与相机的一致性：前者提供强初始化，后者依赖可靠观测关系。两者都没有保证任意场景优化后会更好。[VGGT 论文附录 C](https://arxiv.org/html/2503.11651v1#A3)、[Schönberger 与 Frahm 论文](https://demuc.de/papers/schoenberger2016sfm.pdf)

官方 `demo_colmap.py` 的 BA 分支首先生成多帧 tracks，按可见性和重投影条件构建带观测关系的 reconstruction，再调用 `pycolmap.bundle_adjustment`。导出一个格式正确但缺少多视图 tracks 的点云不具备同样的几何优化条件。[官方 BA 实现](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/demo_colmap.py#L127)

BARF 证明在 NeRF 上联合相机与场景优化可处理不准确初始位姿，但也指出直接高频编码损害配准；因此不能从“有梯度”推断“任意大位姿误差都能解决”。BAD-Gaussians 与 Robust Gaussian Splatting 将这一问题推进到 3DGS，并同时建模输入模糊或色彩差异：二者支持相机误差是重要候选原因，却不支持将所有 GS 模糊归因于相机。[BARF](https://arxiv.org/abs/2104.06405)、[BAD-Gaussians](https://lingzhezhao.github.io/BAD-Gaussians/)、[Robust Gaussian Splatting](https://arxiv.org/abs/2404.04211)

| 路线 | 可验证对象 | 应用条件 | 当前优先级 |
|---|---|---|---|
| VGGT + tracks + 几何 BA | sparse reprojection error、track coverage、注册相机数 | 多位置、有纹理、非退化匹配；增加跟踪依赖 | 中：先做小数据可行性与诊断 |
| 训练帧 SE(3) 光度 refinement | 固定相机与可调相机的独立测试渲染差异 | 小误差初始化；固定 gauge；正则化；渲染器支持 viewmat 梯度 | 中高：先确认梯度及做消融 |
| 曝光轨迹/去模糊建模 | 原始输入是否存在曝光模糊 | 有真实输入模糊证据 | 低：现阶段不盲目加模块 |

建议先报告训练相机之间的重投影分布、深度一致性分布和新视角错误位置；小幅 pose refinement 冻结首帧或 rig gauge，先建立粗几何再开放低学习率增量，设置旋转/平移正则及偏移上限。优化失败或测试指标下降时保留固定相机结果，而不是不断扩大 pose 自由度。

特别注意：几何 BA 改变 E/K 后，旧深度反投影点与新相机不再自动一致。若继续以 VGGT depth 初始化 GS，应明确记录采用 BA 点云，还是用新的 E/K 重新反投影，并检查 depth 的尺度一致性。统一 Sim(3) 不能纠正每帧各自的位姿误差；对 depth 做 GT 中位数对齐只是一种评估协议，不应偷渡为训练修复。

## 三、360° 观测约束

COLMAP 官方文档现在直接支持 360° 图像转 virtual pinhole rig；`pycolmap.panorama.py` 创建固定相对旋转、零相对平移的 rig，官方示例可用于独立几何对照。但文档明确要求脚本匹配安装版本，当前 HEAD 不代表服务器已有 pycolmap 一定支持该 API。[Rig 文档](https://colmap.github.io/rigs.html#reconstruction-from-360-spherical-images)、[panorama_sfm.py](https://github.com/colmap/colmap/blob/main/python/examples/panorama_sfm.py)、[panorama.py](https://github.com/colmap/colmap/blob/main/python/pycolmap/panorama.py#L180)

同一 ERP 的多个投影视角只增加方向覆盖，并未增加相机中心或平移视差。因此 30 个虚拟视角来自 3 个全景位置时，不能解释为 30 个独立观测中心。经典 SfM 论文避免从纯旋转对进行三角化，官方采集建议也要求换位置而非原地转动。[COLMAP 论文 §4.1–4.3](https://demuc.de/papers/schoenberger2016sfm.pdf)、[官方采集建议](https://colmap.github.io/tutorial.html#structure-from-motion)

本项目若增加 pose refinement，应只优化每个 ERP 的一个中心位姿，虚拟视角由已知相对旋转复合产生；保留已知投影 K。独立优化每张切片可能通过不物理的相机移动降低训练损失，却损坏跨方向几何。真实全景还要检查拼接错位、非中心成像和曝光差异；理想中心投影约束对存在严重拼接视差的输入仅是近似模型。

补检说明（2026-09-15）：PanoVGGT 官方仓库显示高、低分辨率预训练权重已在 2026 年 4 月发布。它可作为替换几何前端的后续对照，但不属于冻结 VGGT 的相机后处理，本分支结论不依赖它。

## 四、RTX 5090 可落地的有限实验

这些预算为本项目的实验设计，不是论文实测资源或性能承诺。

1. **输入阶段**：先跑 kitchen 与既有全景的 518 推理，维持既有视图数；30 视图在显存压力下减少全景位置/方向或分阶段处理，禁止偷偷切断所有全局多视图关系并宣称等价。模型、临时文件、cache 均继续在 `/root/autodl-tmp/vggt-xr`。
2. **监督阶段**：固定几何与 GS 设置，比较低分辨率与原图保比例监督；通过有效 mask 限制背景和 padding，报告同尺寸测试结果。
3. **几何阶段**：在同一监督和训练预算下比较固定 E/K 与小幅 pose refinement；全景以 rig 位姿为优化变量。若使用官方 BA，先在少量真实图上核验 tracks、观测连接和前后重投影统计，再决定是否扩展。
4. **评价阶段**：camera pose refinement 只用训练 RGB；GT 测试相机使用训练相机估计的坐标变换。若使用测试 RGB 单独优化测试相机，必须标记为 test-time fitting，不能继续写“完全 heldout”。kitchen 无独立 GT 位姿时，报告固定已知相机的重渲染与视觉对照，避免伪造自由新视角指标。

输入上采样无法恢复 TinyNeRF 100×100 原始数据的高频信息；要检验清晰度改善，必须把 kitchen 的高分辨率原图或新真实采集作为主要视觉对象。固定 SH0 和无 densification 的容量瓶颈由另一研究分支负责，本分支不能用几何优化承诺解决这一瓶颈。

## 引用核验表

| 论文 | 作者 | 年份/发表 | 存在核验 | 支持主张核验 | 裁决 |
|---|---|---|---|---|---|
| VGGT: Visual Geometry Grounded Transformer | Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny | CVPR 2025 | [arXiv 元数据](https://arxiv.org/abs/2503.11651)与作者正文一致 | 正文 §3.4 与 Appendix C 支持保比例518预处理及 VGGT+BA；不外推5090性能 | VERIFIED |
| Structure-from-Motion Revisited | Johannes L. Schönberger, Jan-Michael Frahm | CVPR 2016 | 作者托管 PDF 与 [COLMAP BibTeX](https://colmap.github.io/)一致 | PDF §2、§4 支持 BA、离群点与纯旋转退化条件 | VERIFIED |
| BARF: Bundle-Adjusting Neural Radiance Fields | Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, Simon Lucey | ICCV 2021 | [arXiv](https://arxiv.org/abs/2104.06405)与[官方代码](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)一致 | 摘要支持联合配准与 coarse-to-fine；适用 NeRF，不直接声称本GS复现其结果 | VERIFIED |
| BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting | Lingzhe Zhao, Peng Wang, Peidong Liu | ECCV 2024 | [arXiv](https://arxiv.org/abs/2403.11831)与[作者项目页](https://lingzhezhao.github.io/BAD-Gaussians/)一致 | 项目摘要支持运动曝光轨迹与 GS 联合优化；未用于承诺本项目PSNR | VERIFIED |
| Robust Gaussian Splatting | François Darmon, Lorenzo Porzi, Samuel Rota-Bulò, Peter Kontschieder | arXiv 2024 | [原始元数据](https://arxiv.org/abs/2404.04211)匹配标题作者 | 摘要支持相机/运动模糊/失焦/色差区分；当前仅按预印本引用 | VERIFIED |

## 结论与边界

RQ1 的答案是“候选因素明确，根因尚未由独立消融确证”；相机、输入高频信息和表示容量必须分别检验。RQ2 的答案是“当前实现已有相当简化，官方518与多视图tracks+BA是合理对照，格式导出并非 BA”。RQ3 的答案是“先修监督分辨率与坐标一致性，再以 rig 小幅优化，避免独立切片位姿漂移”。这些建议仍需要服务器实测，不能以论文结果替代本项目效果验证。
