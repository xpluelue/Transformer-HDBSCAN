# Transformer + HDBSCAN

独立的雷达脉冲分选实现：先将 ToA 转成 delta-ToA 并按窗口标准化，再用 Transformer 和 triplet loss 学习脉冲 embedding，最后对每个窗口运行 HDBSCAN。

## 使用

```bash
cd transformer_hdbscan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

训练（目录中的 `.h5` 文件需要包含 `data` 和 `labels`）：

```bash
python train.py --train-dir /path/to/train_scan --output model.pt
```

预测：

```bash
python predict.py --data-dir /path/to/test_scan \
  --checkpoint model.pt --output predictions.npz \
  --device cuda --batch-size 256 --hdbscan-jobs 8
```

`data` 的形状应为 `[脉冲数, 特征数]`，第 1 列必须是 ToA；`labels` 的形状为 `[脉冲数]`。测试默认与原基线一致：仅统计完整窗口且窗口中至少有 2 个真实发射机。Transformer 使用指定的 GPU，HDBSCAN 使用 CPU；`--hdbscan-jobs` 控制并行聚类进程数。预测结果保存在 `predictions.npz`，噪声标签为 `-1`。
