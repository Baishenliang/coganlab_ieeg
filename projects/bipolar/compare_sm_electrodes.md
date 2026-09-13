# 运行 SM_vWM 电极比较

激活 ieeg 环境后，在仓库根目录运行：

```powershell
python projects/bipolar/compare_sm_electrodes.py
```

也可以在 IDE 中直接运行该 Python 文件。不需要准备标签变量。

脚本用 `utils.group.load_stats` 自动加载 average 的 Auditory_inRep mask，
以及 bipolar 的 Auditory、Delay、Go、Resp 的 mask 和 CORRECT zscore epochs，
取通道交集并保留 Auditory mask 顺序，以复现 bipolar 分组分析的标签顺序。
加载 epochs 需要一定时间和内存，不会重新计算显著性或读取坐标。

默认数据目录：`~/Box/CoganLab/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/stats`。
可用 `--stats-root "目录"` 指定其他位置。

读取的索引文件是 `projects/bipolar/Lex_twin_idxes_hg_bipolar.npy` 和
`projects/GLM/data/Lex_twin_idxes_hg.npy`，使用 `LexDelay_Sensorimotor_in_Delay_sig_idx`。
排除 D24、D26；双极 A1-A2 只与 average A1 比较，仅选取各参考方式下 SM 与 Delay 的交集。

结果打印到终端，并写入 `projects/bipolar/sm_vwm_comparison/`：

- `SM_vWM_summary.csv`：逐被试汇总。
- `SM_vWM_contacts.csv`：第一电极的共同/独有分类。
- `SM_vWM_pairs.csv`：相关双极配对与第一电极的 average SM_vWM 状态。
- `README.txt`：分类含义和数据来源。

再次运行覆盖同名输出。未纳入某参考方式的数据与非 SM_vWM 分类分别统计。
自动加载要求当前数据与生成索引时的数据和配置一致；旧索引只保存整数，
无法自动证明历史通道顺序一致。

可选：提供原始完整标签 CSV（`full_label` 列，保持原始顺序），跳过自动加载：

```powershell
python projects/bipolar/compare_sm_electrodes.py --bipolar-labels bipolar_labels.csv --average-labels average_labels.csv
```

## SM vWM gamma 曲线

先运行比较脚本生成新的 SM_vWM_contacts.csv，再运行：

```powershell
python projects/bipolar/plot_sm_overlap_gamma.py
```

只绘制 Auditory、Delay、Go、Resp 四个 epoch；结果在 `projects/bipolar/figs/sm_vwm_overlap_gamma/`。旧的全部 SM 输出不覆盖，也不会作为默认输入。
