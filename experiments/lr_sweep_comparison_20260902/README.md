# Learning Rate Sweep — 2026-09-02

## 目标

参考 `Hurricane0698/TransformerLM-from-scratch` 的两阶段 TinyStories 学习率扫描：先用较短 probe
淘汰较差学习率，再对候选学习率使用更长预算，并以验证集逐 token loss 选择最佳值。

## 固定配置

- 数据：TinyStories train 前 2,000,000 字符、valid 前 500,000 字符
- Token 数：train 486,956，valid 121,603
- 模型：4 layers, d_model=512, 16 heads, d_ff=1344, context=256, vocab=10,000
- Batch size：16
- 优化器：本项目 AdamW，betas=(0.9, 0.95)，weight decay=0.01
- 调度：线性 warmup + cosine decay，min_lr=max_lr×0.1
- 梯度裁剪：1.0
- 随机种子：42；所有 LR 使用相同初始化与数据采样顺序
- 粗扫：300 steps；完整阶段：1000 steps；每次验证平均 5 个 batch

## 两阶段选择

粗扫学习率：0.0003, 0.001, 0.00125, 0.002, 0.004, 0.008。

根据粗扫 best validation loss 自动选择：0.002, 0.00125, 0.001。

## 结果

| Stage | Max LR | Steps | Best valid loss | Final valid loss | Time | Diverged |
|---|---:|---:|---:|---:|---:|:---:|
| full | 0.001 | 1000 | 2.8427 @ 500 | 3.0840 | 221.0s | no |
| full | 0.00125 | 1000 | 2.8498 @ 500 | 3.1107 | 220.4s | no |
| full | 0.002 | 1000 | 2.8638 @ 500 | 3.1101 | 222.5s | no |
| probe | 0.0003 | 300 | 3.4257 @ 250 | 3.4586 | 67.5s | no |
| probe | 0.001 | 300 | 3.0229 @ 250 | 3.0485 | 67.1s | no |
| probe | 0.00125 | 300 | 2.9685 @ 250 | 2.9980 | 67.3s | no |
| probe | 0.002 | 300 | 2.9214 @ 250 | 2.9500 | 66.8s | no |
| probe | 0.004 | 300 | 3.1470 @ 300 | 3.1470 | 67.3s | no |
| probe | 0.008 | 300 | 3.4542 @ 300 | 3.4542 | 74.7s | no |

## 结论

本次预算和数据子集下，最佳学习率为 **0.001**，最佳验证损失为
**2.8427**（step 500），最终验证损失为
**3.0840**。

粗扫阶段最好的学习率是 `0.002`，但完整阶段由 `0.001` 反超，说明短 probe 适合淘汰明显较差的范围，不能替代完整预算比较。第二名 `0.00125` 的最佳验证损失为 `2.8498`，只比最佳值高 `0.0071`，仍是值得在完整数据上复核的候选。

三个完整运行的最佳验证损失都出现在 step 500；此后训练损失继续下降，而验证损失回升到
`3.0840` 或更高，表明这个 486,956-token 子集已开始过拟合。后续应增大训练数据覆盖并保存 best-validation checkpoint，而不是只保留最后一步。

该结果用于选择后续完整 TinyStories 训练的候选学习率，不应与参考仓库 10,000-step、完整数据结果作绝对数值比较。
建议用最佳值及其相邻值做更长训练后再最终定参。

## 复现

```bash
modal run modal_lr_sweep.py
```

原始逐点评估记录位于 `results.json` 和 `runs/*.json`，汇总见 `summary.csv`，参考风格图见
`lr_sweep_research_summary.svg`。相同文件也持久化在 Modal Volume 的
`/experiments/lr_sweep_comparison_20260902/`。
