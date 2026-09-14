# 刺激对照：第 1–2 步

在 ieeg 环境、仓库根目录运行：

```powershell
python projects/bipolar/prepare_esm_comparison.py
```

默认读取指定 Box Excel 的 `Site_level`，以及 bipolar 的 `Lex_twin_idxes_hg_bipolar.npy`。
自动复现现有分组脚本的标签顺序；这个过程复用比较脚本的加载函数，需要读取 FIF，可能较慢。
也可用 `--labels labels.csv` 提供生成索引时的完整 `full_label` 列，跳过 FIF 加载。
历史索引与当前通道顺序必须一致。

输出在 `projects/bipolar/esm_comparison/`：

- `bipolar_physiology.csv`：所有 bipolar 的生理类别、SM vWM 标记和原始索引。
- `esm_all_rows.csv`：保留所有刺激行及原始字段，加上匹配和生理分类。
- `esm_matched.csv` / `esm_unmatched.csv`：匹配成功及需核查的记录。
- `physiology_without_esm_match.csv`：本 Excel 中没有精确刺激匹配的生理电极对。
- `match_summary.csv`：逐被试、逐匹配状态计数。

规则：按被试和完整有序电极对匹配，兼容 `D0063_ROF6-7` 与 `D63-ROF6-ROF7`。
跨号配对不替代相邻配对；反向配对单列待核查。未匹配的生理分类保持缺失，不能算阴性。
保留 Excel 中 `stim_result`、`stim_behavior`、`strict_esm_pos` 和旧 `repeat_class`；新分类以 `liang_` 开头。

新类别来自你的保存索引，SM vWM 为 SM 与 Delay 交集；motor preparation 不等于普通 speech-response。
没有落入 Auditory、SM、Motor、Delay 的通道标记为 Other / unclassified，不断言无任何生理活动。

此脚本不做刺激结果重编码或推断统计。重复电极对不自动去重；Excel 行号不等于独立刺激位点。
正式统计仍需补充刺激位点 ID 及原始刺激映射。没有匹配到本 Excel 也不代表从未接受刺激。
