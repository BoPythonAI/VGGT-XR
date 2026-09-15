# 质量配置冻结选择规则

冻结时间：2026-09-15；写入时尚未得到全景质量消融结果，新的 final test 已生成但未渲染或查看。

每个数据集只使用原有 development heldout 指标选择一次配置。候选为 `sh3_fixed`、`sh3_dense`、`sh3_dense_pose`；旧训练器 6000 steps 是参考线，不进入候选。

1. 合格候选必须相对旧训练器满足：SSIM 不低于 0.005，LPIPS-SqueezeNet 不劣于 0.005。
2. 在合格候选中选 development PSNR 最高者；PSNR 差小于 0.05 dB 时选 LPIPS 更低者；再相同时选更简单的 fixed > dense > dense_pose。
3. 若没有合格候选，则不声称质量配置成功，并保留旧训练器。
4. 配置选定后才对冻结 final test 渲染；seed 42、43、44 使用同一配置，全部报告均值和样本标准差，不丢弃失败 seed。
5. final test 不再用于改参数。旧 dev 与 final 都是同场景诊断，不改称真实跨场景泛化。
