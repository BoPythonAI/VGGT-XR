# Gaussian 表达与优化：质量改善的来源核验与实施判断

检索日期：2026-09-15。范围：现有 `src/gaussian.py` 的固定 SH0、固定 Gaussian 数量、纯图像损失训练。本文是系统主报告的优化视角分报告，未修改实现或服务器。

## 摘要与研究问题

本调查回答：RQ1，哪些现有表达与训练限制会造成模糊、漂浮和细节丢失？RQ2，官方 gsplat 1.5.3 可直接支持哪些改善，并有哪些版本陷阱？RQ3，如何在有 GPU、存储上限的条件下区分优化收益与相机、数据收益？文献表明，表达容量、密度分配、采样过滤分别处理不同误差；它们不能替代正确相机与覆盖充分的输入。建议先建立相机可靠的诊断对照，再升级真实 SH、局部尺度、学习率调度和有预算的密度控制。

## 检索方法

第一轮检索原始 3DGS、gsplat 官方 strategy 文档、Mip-Splatting；第二轮用已发现的 MCMC、AbsGS、opacity reset、初始化和 learning-rate schedule 进一步检索论文正文与固定 `v1.5.3` 官方代码。另行检查官方 issue 作为反例线索，并用源码判断是否足以支持结论。仅采用作者论文、作者项目和官方代码/文档。API 实施判断以 `v1.5.3` 固定源码为准，避免将当前 main 新增参数误用于已安装版本。

## 文献核验表

| 工作与作者 | 题录存在核验 | 方法/结论核验 | 判定与适用边界 |
|---|---|---|---|
| **3D Gaussian Splatting for Real-Time Radiance Field Rendering**；Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis；SIGGRAPH / TOG 2023 | [作者项目](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 与 [arXiv](https://arxiv.org/abs/2308.04079) 标题、作者一致 | [作者代码](https://github.com/graphdeco-inria/gaussian-splatting) 证实 SH3、densification、位置 LR 调度和 30k 默认训练 | VERIFIED。当前仅调用 CUDA rasterizer 不等于已复现原论文完整训练 |
| **3D Gaussian Splatting as Markov Chain Monte Carlo**；Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi；NeurIPS 2024 | [会议页](https://papers.nips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html) 与 [作者代码](https://github.com/KWU-CVRL/3dgs-mcmc) 一致；早期 arXiv 将 Tseng 记作 Jeff Tseng，采用正式会议署名 | [论文正文](https://arxiv.org/html/2404.09591v3) 确认 SGLD、relocalization、L1 regularizer；附录 C 明确 aliasing 与反射仍受限 | VERIFIED。初始化鲁棒性不代表坏姿态可被自动修复 |
| **AbsGS: Recovering Fine Details for 3D Gaussian Splatting**；Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou；arXiv 2404.10484，2024 | [arXiv](https://arxiv.org/abs/2404.10484) 与 [作者项目](https://ty424.github.io/AbsGS.github.io/) 一致；正式会议信息有标题介词 `in`/`for` 差异，此处明确使用预印本标题 | [论文正文](https://arxiv.org/html/2404.10484v1) 确认 gradient collision 会阻止过大 Gaussian 分裂，提出 homodirectional gradient | VERIFIED。只有有 densification 的训练才会利用此信号；单独打开 absgrad 不能改善固定点训练 |
| **Mip-Splatting: Alias-free 3D Gaussian Splatting**；Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger；CVPR 2024 | [CVF 正文](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf) 与 [作者代码](https://github.com/autonomousvision/mip-splatting) 一致 | [arXiv 摘要](https://arxiv.org/abs/2311.16493) 与正文共同证实 3D smoothing 和 2D Mip filter，关注采样率变化 | VERIFIED。gsplat 的二维 opacity compensation 不等于完整 Mip-Splatting |

## 证据分类与交叉比较

### 1. 表达容量和优化收敛

原始 3DGS 同时使用 anisotropic covariance、view-dependent SH 和交错密度控制，因此不能将当前 SH0、30k/60k 固定初始化点、1500/3000 步原型的质量上限归因于 VGGT 本身。当前颜色是 sigmoid RGB，视角变化时没有方向条件；PLY 导出的高阶 SH 为零，也不会凭导出格式获得 SH3 能力。[原始工作](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

已核验的官方示例用 3 个非自身近邻的均方距离设置初始尺度；将 SH 系数分为 `sh0` 与 `shN`，高阶 LR 为零阶的 1/20，并逐步激活阶数；means 学习率指数衰减至初值的 1%。这些是可复现的默认参考，而不是针对当前小数据已证明最优的超参数。建议用真实 SH2/3 对照和局部尺度替代当前全局 extent 截断主导的初始化。[gsplat v1.5.3 示例](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/examples/simple_trainer.py)

### 2. 密度分配：Default、AbsGS 与 MCMC

DefaultStrategy 根据 screen-space gradient 对小 Gaussian 复制、对大 Gaussian 分裂并剪除低 opacity 点；AbsGS 针对梯度相互抵消造成的欠分裂，改变 densification 信号。与二者相比，MCMC 通过重定位和随机探索减少对初始化及手工分裂规则的依赖，并可控制数量。三者改变的是分配与优化，并未消除相机误差、遮挡缺口、曝光变化或不可观测表面。[Default 固定源码](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/strategy/default.py)、[AbsGS](https://arxiv.org/abs/2404.10484)、[MCMC 正文](https://arxiv.org/html/2404.09591v3)

当前项目应优先测试 **MCMC + 明确 cap_max**，因为数据盘和显存是硬约束，官方 DefaultStrategy 没有数量上限。若测试 Default/AbsGS，需自行实现受控增长上限，不能只在更新后裁剪裸 Parameter：strategy 的状态数组及 Adam 状态必须同步。AbsGS 的 `absgrad=True` 必须同时传给 rasterizer 和 strategy，参考 `grow_grad2d=0.0008`；这个阈值应视为起点，不是通用保证。[gsplat strategy 文档](https://docs.gsplat.studio/main/apis/strategy.html)

### 3. 采样率稳定性

Mip-Splatting 处理改变焦距/距离导致的 dilation 与 aliasing；这一维度与增加点数、SH 或 MCMC 的改善不同。gsplat `rasterize_mode='antialiased'` 用二维投影 covariance 的 determinant 比值补偿 opacity。它可作为镜头移动稳定性的对照，但缺少完整论文所需的由训练视角采样频率确定的 3D smoothing，不能标注为完整 Mip-Splatting 复现。[Mip-Splatting](https://arxiv.org/abs/2311.16493)、[固定 rasterization 源码](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/rendering.py)

## 可实施 API 与必须检查的接口

| 改善 | 已核验 API / 调用约束 | 当前实现需要调整 |
|---|---|---|
| 真实 SH | `rasterization(colors=SH_coefficients, sh_degree=active_degree)`；系数形状 N×(degree+1)^2×3 | 禁止对 SH 系数做 RGB sigmoid；初始化 RGB→SH0，高阶初值零；checkpoint 和 PLY 同步保存实际高阶系数 |
| DefaultStrategy | `check_sanity`、`initialize_state(scene_scale)`、`step_pre_backward`、`step_post_backward(..., packed=False)` | params 和 optimizers 同名，**每个 optimizer 仅一个 Parameter**；当前单一 Adam 多组不满足接口；保留 rasterizer 的 info |
| MCMCStrategy | `cap_max`、`noise_lr`、`refine_*`；`step_post_backward(..., lr=current_means_lr)` | 同步 Parameter/optimizer 结构；实际传入衰减后的 means LR；加入 opacity/scale 正则，记录变化后的 Gaussian 数量 |
| Depth diagnostic | `render_mode='RGB+ED'`、alpha 输出 | ED 为 opacity-normalized projection depth；仅对有效高 confidence 深度计算，不能将错误 VGGT 深度当真值 |
| Antialias | `rasterize_mode='antialiased'` | 训练和评估明确一致；Unity 渲染器需单独核验二维 compensation 支持，防止 server/Unity 行为不同 |
| VRAM | `packed=True` 可减少投影中间量；`sparse_grad=True` 通常需 SparseAdam | 初次接策略保持已验证 packed=False，先确认效果，再优化内存；不把两项同时引入掩盖原因 |

接口来源：[固定 rasterizer](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/rendering.py)、[固定 Default](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/strategy/default.py)、[固定 MCMC](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/strategy/mcmc.py)。

### v1.5.3 的 opacity reset 源码陷阱

`default.py` 的条件实际为 `if step % self.reset_every == 0 & step > 0:`。Python 将其解析为链式比较，右侧 `0 & step` 恒为零，第二个比较 `0 > 0` 为假，因此这个条件无法触发 reset。这里的结论来自源码与语言语义；官方 [issue #797](https://github.com/nerfstudio-project/gsplat/issues/797) 本身只报告疑似症状，不足以证明全部训练问题由此导致。

若项目选择 Default，必须明确本地受控修复正确条件 `(step > 0) and (step % reset_every == 0)` 并核验 optimizer 状态；不能静默修改系统包。选择 MCMC 可避免这个 Default-specific 条件，但仍要验证重定位、增长及显存预算。该版本的 MCMC 在 refine_stop 后仍会注入按 LR 缩放的噪声，所以若要求最后阶段固定几何精修，要显式停止注噪并在报告中标明它是本地修改。

## 建议的受控实验顺序

以下预算是针对现有 RTX 5090 32 GB、稀疏小场景的工程建议，不是论文推荐值或已取得成绩。

1. 固定训练/测试划分、相机与初始化点，保留当前 SH0 原型作为 A0。增加至少训练 PSNR、heldout 指标、固定轨迹视频与 alpha/depth 可视化；训练好而测试差指向相机/覆盖或过拟合，两者都差才更支持表达/优化限制。
2. A1：真实 SH2/3 + 局部 3-NN scales + opacity 0.1 + means LR 指数衰减，6000 步。和 A0 做同 6000 步额外对照，不能把步数收益当成改进收益。
3. A2：在 A1 上增 MCMC，初始30k/60k，`cap_max=150000` 起，最多300000；opacity/scale L1 权重从官方示例0.01作为对照起点；refine 开始约200、结束约4000、每100步，最后2000步明确固定/停止噪声精修。具体调度应按收敛与风险调整。
4. 如 A2 仍因大 Gaussian 过度覆盖而模糊，A3 用受预算限制的 Default/AbsGS；opacity reset 修复做独立记录。诊断成功后再在原始较高分辨率图像上训练，VGGT 输入分辨率与 GS 监督分辨率可以分开，但必须正确缩放 K、处理 padding 和有效像素 mask。
5. Antialiased 单独对照，并测试多个距离和焦距。质量 gate 使用同轨迹主体近景、边界和未见视角；不能仅靠训练图或平均指标挑展示。

每500步记录 Gaussian 数量、显存 allocated/reserved、磁盘剩余量、loss 与有效 alpha 覆盖。触及预算就停止增长；非有限参数/loss、显存增长异常、测试大幅退化触发暂停。新输出独立目录保存，维持原型可复现性，不覆盖旧结果。

## 讨论、未解问题与结论

RQ1：当前表达与训练明显低于完整 3DGS 常规配置，但视频重影不能仅凭外观归因到点数或 SH；错误相机与深度会使更强优化器拟合错误的多层结构。RQ2：固定版本已具备 SH、Default、AbsGrad、MCMC 与二维 antialias 基础 API，同时存在 reset 条件与噪声调度的具体陷阱。RQ3：实施应固定相机、数据、步数和测试轨迹做分层消融，并用硬数量和内存预算控制成本。优先恢复合理完整训练能力，再用真实数据和正确相机核验收益；MCMC 与 antialias 的论文优势不能直接当作当前场景的已证实改善。
