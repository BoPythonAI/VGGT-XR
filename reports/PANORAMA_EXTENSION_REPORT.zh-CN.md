# 自然与真实 360° 重建扩展报告

日期：2026-09-15

服务器：AutoDL RTX 5090，项目与全部大文件位于 `/root/autodl-tmp/vggt-xr`

结果数据：[panorama_extension.json](panorama_extension.json)

## 目标与结论

原分析型 360° 房间使用高饱和彩色坐标纹理，虽然便于诊断对应关系，但容易被误认为重建产生的方格光影。本扩展保留原基准用于回归测试，同时新增两个互补实验：

1. `natural_room`：视觉更自然的可控合成房间，仍提供解析 RGB、深度与相机真值；
2. `zind_room09_protocol`：来自 ZInD 官方仓库固定提交的真实多位置 360° 样例，以未参与重建的位置检验泛化。

自然场景 v2 相比首次自然场景版本有明显提升。真实场景上，SH3、受限自适应密度和训练位姿优化在 PSNR、SSIM、LPIPS 三项指标上均优于 6000 步 SH0 固定预算基线。真实未见位置仍存在显著拖影和模糊，因此本结果证明改进方向有效，但还未达到生产级自由视点质量。

## 自然解析场景 v2

场景由 `scripts/prepare_data.py::trace_natural` 生成。它包含暖色低对比墙面、木地板、踢脚线、窗户、挂画和家具，并为每条光线计算精确交点；不存在把估计深度当真值的问题。训练输入为 3 个 ERP 中心，测试为另外 3 个相机中心，每个中心包含 8 个水平朝向和 2 个极区朝向，共 30 张未见视图。

| 版本 | Heldout PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 自然场景 v1 | 19.2205 dB | 0.7756 | 0.3435 |
| 自然场景 v2 | **22.7171 dB** | **0.8012** | **0.2308** |
| v2 − v1 | **+3.4966 dB** | **+0.0256** | **−0.1127** |

VGGT 几何前端在 30 个训练投影视图上的相机指标为：训练中心拟合 Sim(3) 后 ATE 0.1721 m、相对平移 RMSE 0.0906 m、平均旋转误差 4.3359°。全部深度像素的尺度对齐 RMSE 为 0.6037 m、AbsRel 为 0.2408；置信度保留率为 64.98%。

![自然解析场景的未见视图](../assets/quality/natural360_v2_heldout_comparison.png)

## 真实 ZInD 未见位置实验

数据来自 [ZInD 官方仓库](https://github.com/zillow/zind) 的 `sample_tour/000`，固定到提交 `06cfdbf295fa21500a857efe81f80f2243b8fb40`。ZInD 收录真实室内 360° 全景及楼层平面标注，数据集方法发表于 [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cruz_Zillow_Indoor_Dataset_Annotated_Floor_Plans_With_360deg_Panoramas_and_CVPR_2021_paper.html)；数据使用遵循 ZInD Terms of Use。本仓库只保存下载脚本、协议和衍生评测结果，不重新分发原始全景。

冻结协议如下：

- 同一 `partial_room_09` 内，以 pano 2、4、6 的三张 2048×1024 ERP 作为训练输入；
- pano 5 完全留出，仅在训练完成后投影为 8 个水平视角与 2 个极区视角；
- pano 5 的 RGB 不进入 VGGT 推理、点云过滤、Gaussian 初始化或优化；
- 相机水平位置和 yaw 来自官方 `floor_plan_transformation`；评估前的 Sim(3) 只使用训练相机中心拟合；
- 下载脚本固定上游提交并校验官方 MD5，同时在生成协议后写入所有文件的 SHA-256。

| 方法（相同 seed=42、6000 步） | Gaussians | SH | Heldout PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| 固定预算基线 | 60,000 | 0 | 18.7464 dB | 0.6700 | 0.5108 |
| 质量路径 | 119,986 | 3 | **19.2660 dB** | **0.6881** | **0.4148** |
| 改变量 | — | — | **+0.5196 dB** | **+0.0181** | **−0.0960** |

VGGT 前端耗时 0.7610 s，峰值显存 5.9145 GiB。固定预算基线训练耗时 50.33 s；质量路径训练耗时 63.07 s、训练峰值显存 0.3290 GiB。训练相机指标为 Sim(3) 对齐后 ATE 0.1416 m、相对平移 RMSE 0.0736 m、平均旋转误差 6.7108°。

![ZInD pano 5 未见位置对比](../assets/quality/zind_room09_heldout_comparison.png)

从图中可以看到，质量路径减少了部分颜色发散并提高了结构可辨性，但两种方法在离训练中心较远的未见位置都出现明显模糊。三张稀疏 ERP 的视差、VGGT 深度/位姿误差和表面覆盖不足是当前主要瓶颈。继续单纯增加 SH 阶数或训练步数不会补回未观测几何；下一轮最有价值的是增加真实采集中心，并加入跨 ERP 的深度一致性与遮挡约束。

## 复现

大文件应继续放在数据盘。以下命令均在 `/root/autodl-tmp/vggt-xr` 执行：

```bash
source scripts/environment.sh

# 自然解析场景
python scripts/prepare_data.py --root /root/autodl-tmp/vggt-xr/data --size 336
python run.py --input data/natural_room --output outputs/natural360_v2_base \
  --panorama --projection overlap --include-poles --max-views 30 \
  --constrain-poses --pose-fusion anchor --known-intrinsics \
  --filter confidence --max-points 600000 --size 336 --geometry-only
python scripts/improve_quality.py --base outputs/natural360_v2_base \
  --output outputs/quality/natural360_v2_quality --steps 6000 \
  --cap 120000 --pose-opt

# 真实 ZInD 样例；需先阅读并接受 ZInD Terms of Use
python scripts/prepare_zind_sample.py \
  --output /root/autodl-tmp/vggt-xr/data/zind_room09_protocol \
  --size 336 --accept-zind-terms
python run.py --input data/zind_room09_protocol --output outputs/zind_room09_base \
  --panorama --projection overlap --include-poles --max-views 30 \
  --constrain-poses --pose-fusion anchor --known-intrinsics \
  --filter confidence --max-points 600000 --size 336 --geometry-only
python scripts/improve_quality.py --base outputs/zind_room09_base \
  --output outputs/quality/zind_room09_quality --steps 6000 \
  --cap 120000 --pose-opt

python scripts/verify_quality.py
```

`verify_quality.py` 已通过有限数值、Gaussian 硬上限、非零 SH、相机有限性、全景共享中心、冻结测试哈希和三种子完整性检查。发布包另含两段 gsplat 飞行视频、两张对比图、两个 PLY 和完整 JSON，不含 ZInD 原始图像。

## 结果边界

ZInD 本次只使用官方仓库中的一个样例住宅、一个相连区域和一个留出位置，因此结论是同区域新位置泛化，不代表跨住宅泛化。自然场景结果仍属于解析合成数据。要形成论文级最终结论，还需在遵守数据条款的前提下扩展到多个住宅，并在冻结测试集上报告均值和方差。
