# Transformer Metric Learning + HDBSCAN

该实现已并入主项目包：
`src/turing_deinterleaving_challenge/transformer_hdbscan/`。训练、预测和数据工具
通过 `turing_deinterleaving_challenge.transformer_hdbscan` 模块调用。
Transformer 仍采用固定窗口控制显存，但训练按源文件组织窗口，预测时再对
完整源脉冲文件聚类：

```text
raw PDW
→ 边界连续的 delta-ToA + 每个文件统一标准化
→ 50% 重叠滑动窗口通过 Transformer Encoder
→ 同一脉冲的重叠窗口 embedding 求算术平均，不做二次归一化
→ 对完整源脉冲流运行一次 HDBSCAN
→ 每个原始源文件一个自包含 HDF5 结果文件
→ 后续单独读取聚类 HDF5 计算指标
```

同一源文件内的窗口共享文件级 HDBSCAN 簇编号。一个预测簇的唯一标识是：

```text
(file_index, predicted_label)
```

不同源文件的相同数字标签仍然没有关联。完整文件不会一次送入 Transformer；
只有滑动窗口进入模型，因此 Transformer 显存开销仍由 `window_length` 和
`batch_size` 控制。每个原始脉冲最终只保留一个聚合 embedding；完整文件的
embedding 与 HDBSCAN 工作集保存在内存中。

训练batch只包含同一个源文件的窗口。Triplet正样本优先来自该文件的不同
窗口，负样本来自该文件的其他真实发射机；额外的同源紧致损失把一个真实源
在各窗口的embedding拉向同一质心。这样既不把完整脉冲流送入Transformer，
又给完整文件聚类提供跨窗口一致性监督。单发射机文件虽然无法构造Triplet，
仍可通过紧致损失提供训练信号。

当前 Transformer 投影头直接输出原始 embedding，不执行 L2 归一化，向量
模长会保留并参与 HDBSCAN 的欧氏距离。当前编码器结构为：

```text
Linear(5,64) + LayerNorm + GELU
→ 4层 Pre-Norm Transformer Encoder
  → 4头自注意力，RoPE作用于Q、K
  → SwiGLU FFN，hidden_dim=128
  → dropout=0.1
→ Linear(64,8)
```

训练总损失为：

```text
loss = triplet_margin_loss + compactness_weight × compactness_loss
triplet = max(||anchor-positive||₂ - ||anchor-negative||₂ + margin, 0)
compactness = mean(||embedding - 同文件同真实源质心||₂²)
```

默认 `margin=0.2`、`compactness_weight=0.05`。重叠窗口中的同一个原始脉冲
不会被选作自己的 Triplet 正样本；正样本优先来自其他窗口中的同源不同脉冲。

HDBSCAN 返回的 `-1` 与其他整数标签一样，是参与分组和全部评价指标的
普通预测簇，不使用或排除所谓 `noise` 组。逐簇文件使用连续输出序号命名，
原始 HDBSCAN 标签（包括 `-1`）记录在 `_clusters.csv`。

## 安装

在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu]"
```

输入 HDF5 的 `data` 形状必须是 `[脉冲数, 特征数]`，第 0 列是 ToA；
有标签训练和评价文件还需要 `[脉冲数]` 的 `labels`。
数据发布中可能存在 `data.shape[0] == 0` 的合法空配置；训练、验证和预测会
打印 `Skipping ... empty HDF5 source file(s)` 并跳过这些文件。损坏文件、
缺少 `data` 或维度错误仍会直接报错，不会被静默忽略。

## 数据集完整性检查与按需修复

`repair_dataset` 会通过 Hugging Face 镜像读取远端文件清单，检查本地缺失文件，
并逐块读取所有 HDF5 的 `data` 和 `labels`。它会识别空文件、损坏数据块、
NaN/Inf、缺少数据集以及 `data/labels` 长度不一致。默认只审计，不修改文件：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.repair_dataset \
  --dataset-root turing-synthetic-radar-dataset/scan \
  --endpoint https://hf-mirror.com \
  --subdir scan \
  --report dataset_audit_before.json
```

确认报告后增加 `--repair`。程序只重新下载审计失败或本地缺失的文件；下载
副本通过完整校验后才会原子替换，并把旧文件按原目录结构备份：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.repair_dataset \
  --dataset-root turing-synthetic-radar-dataset/scan \
  --endpoint https://hf-mirror.com \
  --subdir scan \
  --repair \
  --report dataset_audit_after.json
```

该数据集需要先在 Hugging Face 页面获得访问权限，并在服务器执行
`hf auth login`；旧版 `huggingface_hub` 使用 `huggingface-cli login`。如果
镜像中的下载副本也为空或损坏，程序不会覆盖本地文件，并在报告的
`unresolved` 中记录。

## 训练

下面的命令会建立新的 RoPE+SwiGLU 实验，不覆盖已有
`transformer_metric_scan_4096`。每轮先计算文件感知的 Triplet 与同源紧致
损失，再恢复每个验证文件的完整 embedding，通过双GPU cuML HDBSCAN聚类，
以逐文件 V-measure 的宏平均选择最佳 checkpoint 和早停：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.train \
  --train-dir turing-synthetic-radar-dataset/scan/scan/train_scan \
  --validation-dir turing-synthetic-radar-dataset/scan/scan/val_scan \
  --output experiments/transformer_metric_scan_rope_vmeasure/model.pt \
  --log-dir experiments/transformer_metric_scan_rope_vmeasure/logs \
  --window-length 4096 \
  --window-stride 2048 \
  --batch-size 512 \
  --num-workers 8 \
  --epochs 50 \
  --shuffle-train-windows \
  --normalization per_file \
  --training-scope file \
  --model-dim 64 \
  --num-layers 4 \
  --num-heads 4 \
  --feedforward-dim 128 \
  --embedding-dim 8 \
  --dropout 0.1 \
  --margin 0.2 \
  --compactness-weight 0.05 \
  --max-anchors-per-emitter 64 \
  --min-cluster-size 5 \
  --min-cluster-fraction 0.0005 \
  --allow-single-cluster \
  --validation-max-files 128 \
  --selection-metric v_measure \
  --validation-hdbscan-devices 0,1 \
  --validation-hdbscan-parallel-files 2 \
  --early-stopping-patience 5 \
  --amp \
  --amp-dtype bf16 \
  --devices cuda:0,cuda:1
```

该实验严格使用独立验证集：

```bash
--validation-dir turing-synthetic-radar-dataset/scan/scan/val_scan
```

因此 `train_scan` 只参与参数更新，`val_scan` 只参与每轮验证、checkpoint选择和
早停，`test_scan` 不参与训练过程。最佳模型训练完成后，使用独立脚本运行一次
测试集完整文件推理、双GPU cuML HDBSCAN和指标计算：

```bash
bash scripts/test_transformer_rope_vmeasure.sh
```

测试结果写到
`experiments/transformer_metric_scan_rope_vmeasure/pdw_studio_clusters_dual_gpu_v1`。
输出严格采用下文的PDW Studio逐簇格式，与
`transformer_metric_scan_4096/pdw_studio_clusters_dual_gpu_v1`一致；测试集汇总指标
保存在`summary.json`。输出目录必须为空，重复测试时请指定新的输出目录。

如果已经生成了native格式的`test_predictions`，无需重新执行Transformer或
HDBSCAN，可以直接转换已保存的预测标签：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.convert_native_to_pdw_studio \
  experiments/transformer_metric_scan_rope_vmeasure/test_predictions \
  experiments/transformer_metric_scan_rope_vmeasure/pdw_studio_clusters_dual_gpu_v1
```

转换会保留原目录，新目录包含`_summary.csv`、`_clusters.csv`、`summary.json`、
`run.log`以及逐源文件的逐簇HDF5。

`--validation-max-files 128` 表示每轮使用固定的128个验证文件计算损失和完整文件
GPU HDBSCAN V-measure；设为 `0` 才会使用全部验证文件。V-measure 采用逐文件
宏平均，因为不同文件的真实标签和预测标签都是文件内局部编号，不能直接跨文件
拼接。`window_stride=2048` 表示每次滑动半个4096窗口。
若省略 `--window-stride`，训练程序自动使用 `window_length // 2`。这里的
`batch-size` 是两张卡合计的窗口数。上述实验沿用已验证的总 batch 512；如果
新注意力内核在服务器环境中OOM，应依次降到256、128。序列长度使注意力计算
近似按平方增长，因此不建议直接把4096翻倍到8192。

训练会原生生成以下文件，不需要再通过 `tee` 截取终端输出：

```text
runs/transformer_hdbscan/logs/
├── train.log       # 初始化信息、每轮摘要、早停和最终checkpoint
├── metrics.csv     # 损失、V-measure、选择值、耗时、best和stale_epochs
└── run_config.json # 完整启动参数和命令
```

训练显示不再使用基于 `\r` 刷新的动态 tqdm 条。文件归一化约每5%输出一行，
训练、验证embedding和GPU HDBSCAN约每30秒输出一条以换行结束的短状态，
因此在 `screen`、重定向日志或IDE终端中都不会发生多条进度互相覆盖。每轮结束
再输出一条完整摘要。推荐进入screen后直接运行上述命令，不要添加 `| tee ...`：

```bash
screen -S thdbscan-train
cd /path/to/turing-deinterleaving-challenge
# 在这里执行上面的 CUDA_VISIBLE_DEVICES=0,1 python ... 命令
```

按 `Ctrl-a d` 脱离，之后用下面命令恢复：

```bash
screen -r thdbscan-train
```

如果只希望看每轮摘要，可以增加 `--no-progress`；此时screen中只打印初始化
信息和每轮摘要，完整数值仍实时写入日志。可在另一个终端查看纯文本日志：

```bash
tail -f experiments/transformer_metric_scan_rope_vmeasure/logs/train.log
```

每轮结束后比较 `validation_v_measure`。V-measure 连续5轮没有提高时触发早停；
如果仍持续改善，则训练到 `--epochs` 指定的上限。最佳 checkpoint 始终是
V-measure 宏平均最高的那一轮。`validation_loss`、Triplet和Compactness仍写入
日志供诊断，但不参与该实验的checkpoint选择。每轮都要对32个完整验证文件
聚类，因此训练耗时会比只看验证损失明显增加。

训练期验证聚类和后续预测使用相同的有效 `min_cluster_size`：

```text
max(min_cluster_size, ceil(完整文件脉冲数 × min_cluster_fraction))
```

每个源文件先使用文件内全部脉冲计算一套均值和标准差，该文件的所有窗口复用
同一套统计量。每个非首窗口的第一个delta-ToA使用原文件前一个脉冲计算，
不再人为重置为0。预测新文件时也会先计算该文件自己的统计量。

预测器会根据 state dict 自动识别旧的正弦位置编码 checkpoint，并继续使用旧
结构加载；但旧权重不会自动转换为 RoPE+SwiGLU。新结构必须使用上述独立实验
重新训练。

## 预测与完整源文件输出

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.predict \
  --data-dir /path/to/scan \
  --split test \
  --checkpoint runs/transformer_hdbscan/model.pt \
  --output-dir runs/transformer_hdbscan/predictions \
  --device cuda \
  --batch-size 64 \
  --num-workers 8 \
  --hdbscan-jobs 8 \
  --amp
```

双卡仅用于并行计算 Transformer 窗口 embedding：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.predict \
  --data-dir /path/to/scan \
  --split test \
  --checkpoint runs/transformer_hdbscan/model.pt \
  --output-dir runs/transformer_hdbscan/predictions \
  --devices cuda:0,cuda:1 \
  --batch-size 64 \
  --num-workers 8 \
  --hdbscan-jobs 8 \
  --amp \
  --amp-dtype bf16
```

### PDW Studio 逐簇文件输出

`--output-format pdw_studio` 会在完整文件 HDBSCAN 后，按照
PDW Studio 所需结构把每个预测簇写成独立 HDF5。服务器上的
`transformer_metric_scan_4096/model.pt` 可直接运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.predict \
  --data-dir turing-synthetic-radar-dataset/scan/scan/test_scan \
  --split test \
  --checkpoint experiments/transformer_metric_scan_4096/model.pt \
  --output-dir experiments/transformer_metric_scan_4096/pdw_studio_clusters \
  --output-format pdw_studio \
  --devices cuda:0,cuda:1 \
  --batch-size 32 \
  --num-workers 8 \
  --hdbscan-backend cuml \
  --hdbscan-devices 0,1 \
  --hdbscan-parallel-files 2 \
  --amp \
  --amp-dtype bf16
```

窗口长度、步长、
归一化方式以及 HDBSCAN 参数默认从 checkpoint 读取，不需要再次手工指定。
输出目录必须不存在或为空。

`requirements_gpu.txt` 已固定服务器使用的 CUDA 13 / cuML 26.8.0；
`pip install -e ".[gpu]"` 会连同项目一起安装。安装后验证：

```bash
python -c "import cuml; from cuml.cluster import HDBSCAN; print(cuml.__version__)"
```

如果 cuML 未安装或版本不兼容，程序会在读取数据前直接报错，不会静默退回
CPU。`--hdbscan-devices` 是应用 `CUDA_VISIBLE_DEVICES` 后的逻辑编号。
`--hdbscan-devices 0,1 --hdbscan-parallel-files 2` 会建立两个独立工作线程：
一张GPU负责一个完整源文件，两张卡同时处理两个不同源文件。单个文件内部仍
使用单GPU cuML HDBSCAN，不切换到近似NN-descent，因此不会因双卡调度改变
该文件的HDBSCAN参数或聚类语义。聚类完成顺序可能与文件名顺序不同，但
每个输出目录、源文件元数据和清单记录仍保持正确对应。

旧参数 `--hdbscan-device 1` 仍受支持，表示仅在逻辑GPU 1上串行聚类；它不能
与 `--hdbscan-devices` 同时使用。`--hdbscan-parallel-files` 默认等于列出的
设备数，不能超过设备数。`--hdbscan-jobs` 仅对 sklearn CPU 后端生效，cuML
模式会忽略它。

需要恢复CPU聚类时显式指定：

```bash
--hdbscan-backend sklearn --hdbscan-jobs 16
```

cuML与sklearn都保留 `-1` 标签，但二者构建近邻图和最小生成树的实现不同，
边界点或等距邻居较多时，预测簇标签可能存在少量差异。因此切换后应先用几个
代表性文件比较运行时间和聚类质量，再执行全部250个文件。

输出结构为：

```text
pdw_studio_clusters/
├── _summary.csv
├── _clusters.csv
├── summary.json
├── run.log
├── config_0/
│   ├── config_0_0.h5
│   ├── config_0_1.h5
│   └── ...
├── config_1/
│   └── ...
└── ...
```

每个 `<源文件名>_<输出簇序号>.h5` 与 `config_590_2.h5` 的结构一致，仅包含：

```text
data        (5, N) float64  # 该预测簇的完整原始PDW，按ToA升序
labels      (N,)   int32    # 对应脉冲的原始真实标签
```

预测簇按脉冲数从大到小重编号为 `_0`、`_1` 等。为严格匹配样例，逐簇 HDF5
不写根属性，也不再重复保存 `true_label`。原始 HDBSCAN 标签、输出簇序号和源
文件路径统一记录在 `_clusters.csv`。原始标签 `-1` 不会被删除或并入最近簇，
而是作为普通预测簇参与排序并输出。使用 `--allow-unlabeled` 处理无标签输入时，
`labels` 全部写成 `-1`。

已有 `pdw_studio_clusters_dual_gpu_v1/config_0/clusterNNN.h5` 无需重新运行
Transformer、embedding 或 HDBSCAN。先执行只读检查：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.convert_pdw_studio_format \
  experiments/transformer_metric_scan_4096/pdw_studio_clusters_dual_gpu_v1
```

确认没有错误后，就地转换：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.convert_pdw_studio_format \
  experiments/transformer_metric_scan_4096/pdw_studio_clusters_dual_gpu_v1 \
  --apply
```

转换程序从旧文件的 `true_label`（若不存在则用 `labels`）生成新 `labels`，将
`data` 转置为 `(5,N)`、转换为 `float64`，把文件改名为如 `config_0_0.h5`，
并生成 `_clusters.csv`。程序逐个打开并关闭 HDF5，不会触发模型推理；为避免
失败时损坏旧结果，会先生成并验证全部新文件，成功后才删除旧 `clusterNNN.h5`，
因此转换期间需要同时容纳新旧文件的临时磁盘空间。

`--devices` 使用 PyTorch 进程可见的逻辑卡号。显式设置
`CUDA_VISIBLE_DEVICES=0,1` 可避免服务器已有环境变量只暴露某一张物理卡。
程序启动行会打印设备列表和 `CUDA_VISIBLE_DEVICES`，便于核对映射。
多卡模式采用 `torch.nn.DataParallel`，不会改变 checkpoint 或预测结果格式。
sklearn后端在CPU上运行；cuML单卡参数 `--hdbscan-device` 会串行处理完整
文件，复数参数 `--hdbscan-devices` 会让不同完整文件在多张GPU上并行处理。
每个工作线程始终绑定一张逻辑GPU，共享清单和HDF5输出只由主线程写入。

运行时会显示两条独立进度：

```text
Embedding windows ... active=... rate=... win/s
Complete files    ... stage=HDBSCAN, active=[cuda:0=..., cuda:1=...]
```

每完成一个源文件还会打印其实际GPU、HDBSCAN、写盘耗时以及动态
`remaining~` 估计。总时间估计会根据当前机器已测得的 embedding 吞吐、
HDBSCAN 耗时、并发GPU数和写盘耗时逐文件校准；由于 HDBSCAN 没有内部进度回调，单个
文件聚类期间只显示当前文件，无法显示该文件内部的完成百分比。HDBSCAN
剩余时间按脉冲数平方近似，因此最初几个文件完成前的 ETA 波动会比较大。

Transformer推理与两个cuML工作线程会共享两张GPU。若并行聚类时发生OOM，
应先降低 `--batch-size`。`--hdbscan-parallel-files 1`只能降低并发显存，不能
把两张卡显存合并，也不能降低单个完整文件本身的工作集。

4096窗口的双卡推理建议从总 `--batch-size 32` 开始；显存仍有余量时尝试
48或64，OOM时退回上一档。训练比推理保存更多中间激活，训练和推理的可用
batch不能直接等同。Transformer注意力计算量随窗口长度平方增长，不建议仅为
提高显存占用而把4096继续翻倍到8192。

输出目录必须为空，以避免新旧结果文件混合。预测采用流式写出，不会把
完整测试集预测保存在内存中；双卡聚类时主要保留两项正在运行的完整文件
embedding以及当前正在生成的窗口，完成后立即逐文件写出。

每个非空滑动窗口都会用于生成 embedding：

- 使用checkpoint保存的窗口长度（当前新模型为4096）；
- 使用checkpoint保存的滑动步长（当前新模型为2048）；
- 不足一个完整窗口的最后一个窗口；
- 只有一个真实发射机的窗口。

同一脉冲若出现在两个窗口，其两个 embedding 只做算术平均，平均结果不再
执行 L2 归一化；非重叠脉冲只有一个贡献。聚合后 embedding 数量严格等于
原文件脉冲数，随后共同参与一次 HDBSCAN。逐窗口 HDF5 输出已经取消。真实
标签不参与预测或筛选，只随聚类结果保存，供后续独立评价使用。

无标签数据可以使用 `--allow-unlabeled`。此时仍输出预测簇，但不创建
`true_clusters`、列联表和评价指标。

预测默认只完成完整文件聚类和保存，不计算任何指标。旧命令中已有的下列参数
仍可保留，但现在可以省略：

```bash
--skip-evaluation
```

默认模式完全跳过 Homogeneity、Completeness、V-measure、ARI、AMI、MCC、F1
和列联表计算。每个源文件仍独立保存为一个 HDF5，其中保留
`pulses/predicted_labels`、`pulses/true_labels`、原始 PDW、预测簇 group 和
真实簇 group，因此后续评价不需要重新生成 embedding 或重新运行 HDBSCAN。
`manifest.csv` 的指标列为空，`summary.json` 的 `metrics` 为空且
`evaluation_enabled` 为 `false`。

## 后续单独评价聚类文件

聚类全部完成后，使用独立命令读取已经保存的标签：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.evaluate_saved \
  runs/transformer_hdbscan/predictions \
  --output-dir runs/transformer_hdbscan/evaluation
```

该命令只读取每个 HDF5 的 `pulses/predicted_labels` 和
`pulses/true_labels`，不会加载模型、重新生成 embedding、重新执行 HDBSCAN，
也不会修改聚类文件。评价结果单独保存为 `evaluation.csv` 和 `summary.json`；
`-1` 与其他标签一样作为普通预测簇参与全部指标。输入也可以是单个聚类 HDF5。

## 输出目录

```text
predictions/
├── manifest.csv
├── summary.json
└── files/
    ├── file_0000__config_0.h5
    ├── file_0001__config_1.h5
    └── ...
```

`manifest.csv` 每个完整源文件一行，记录脉冲数、Transformer 窗口数、
文件级簇数；默认聚类流程的指标列为空。`summary.json` 记录运行配置与文件数。

每个结果文件包含对应原始源文件的全部有效脉冲：

```text
attributes
├── source_file, file_index
├── pulse_count, window_count, window_length, window_stride
└── clustering_scope, has_true_labels

/embedding_windows
├── starts
└── valid_lengths

/pulses
├── raw_pdws
├── source_indices
├── predicted_labels
└── true_labels

/predicted_clusters
├── cluster_-1
│   ├── member_indices
│   ├── source_indices
│   ├── pdws
│   └── counterpart_labels
└── cluster_0, cluster_1, ...

/true_clusters
└── cluster_<真实标签>/...

/evaluation（仅使用预测阶段可选的 `--evaluate` 时存在；推荐独立评价命令）
├── predicted_cluster_ids
├── true_cluster_ids
├── contingency_matrix
└── 完整源文件的 V-measure、ARI、AMI、MCC、F1 等属性
```

每个簇 group 保存该文件级簇的全部原始 PDW、源文件索引及另一种划分的
标签，不再把跨窗口的同一簇拆成多个文件。默认聚类流程不会创建
`/evaluation`。

## 检查和导出单个簇

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.inspect \
  runs/transformer_hdbscan/predictions/files/<source-result>.h5
```

查看并按需导出预测标签 `-1` 的完整脉冲组：

```bash
python -m turing_deinterleaving_challenge.transformer_hdbscan.inspect \
  runs/transformer_hdbscan/predictions/files/<source-result>.h5 \
  --predicted-label -1 \
  --export-cluster /tmp/cluster_minus_1.npz
```

## 测试

```bash
pip install -r requirements_dev.txt
pytest -q tests/transformer_hdbscan
```

测试会核对完整文件只运行一次 HDBSCAN、尾窗参与文件级聚类、每个源文件
只输出一个 HDF5、原始 PDW、`cluster_-1`、列联表、manifest 和 summary；
同时核对全局标准化的窗口边界delta-ToA、单文件batch采样、跨窗口Triplet和
单发射机紧致损失，以及保存后独立评价不会修改聚类文件。
