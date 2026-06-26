# Debug Session: attention-risk-v134

- Status: OPEN
- Goal: 跑通 `attention_risk_delivery/run_attention_risk_index_only.py`，基于 v1.3.4 数据输出全部年份预测结果
- Started At: 2026-06-26

## Hypotheses

1. 输入数据或依赖文件路径默认值与当前仓库布局不一致，导致启动阶段 `ensure_exists()` 失败。
2. 多进程 `spawn` 模式下，子进程加载模型/数据集时发生序列化或导入问题，导致预测阶段报错。
3. GPU 设备参数或 CUDA 可用性与脚本默认值不匹配，导致默认 `--device cuda` 在当前环境不可执行。
4. CPA 映射表字段格式与脚本预期不完全一致，导致预测完成后在结果拼接阶段失败。
5. 全量年份任务在输出阶段触发类型转换或 Excel 写出问题，导致最终文件未成功落盘。

## Plan

1. 使用 `uv run python` 直接复现默认全量运行。
2. 根据首个真实报错定位最小修复点。
3. 修复后先做小规模验证，再执行全量年份任务。
4. 检查输出文件与摘要文件是否生成。

## Evidence

- 复现命令：`uv run python attention_risk_delivery/run_attention_risk_index_only.py`
- 首个失败：`ImportError: cannot import name 'HANDOVER_DIR' from 'common'`
- 根因确认：`run_attention_risk_index_only.py` 仍依赖旧常量名，当前 `attention_risk_delivery/common.py` 只暴露 `ROOT_DIR`
- 最小修复：在 `common.py` 增加 `HANDOVER_DIR = ROOT_DIR` 向后兼容别名

## Verification

- 烟测命令：`uv run python attention_risk_delivery/run_attention_risk_index_only.py --limit 1`
- 烟测结果：成功输出 1 条记录，`device_used=cuda`
- 全量命令：`uv run python attention_risk_delivery/run_attention_risk_index_only.py`
- 全量结果：`task_count=61122`，`row_count=61122`，年份范围 `2007-2025`
- 输出目录：`data/output/attention_risk_delivery_re/attentionrnn_Dv132_Full_001/risk_index_only`
