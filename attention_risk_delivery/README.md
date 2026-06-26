# Attention Risk Delivery

自包含的 AttentionRNN 风险报告交付脚本目录。

## 目录说明

- `run_attention_batch_reports.py`
  - 主入口
  - 读取 `attentionrnn_Dv132_Full_001`
  - 使用 v1.3.4 数据批量生成预测、模型高风险科目得分和 Markdown 风险报告
- `attention_model.py`
  - AttentionRNN 模型定义
  - 数据适配
  - 模型加载与预测
- `formula_utils.py`
  - 公式字典解析
  - 高风险会计科目提取
  - 模型高风险科目聚合
- `reporting.py`
  - 同行基线
  - Z-Score 分析
  - Markdown/PDF 报告输出
- `common.py`
  - 路径、常量和通用工具函数
- `markdown_pdf.py`
  - 内置 Markdown AST PDF 渲染
  - 中文字体自动探测
  - 数值列美化与表格右对齐

## 默认输入

- 模型：`engineering_handover/data/input/models_attention/attentionrnn/attentionrnn.pkl`
- 数据：`data/fraud-sent-v1.3.4-processed.csv`
- 基线库：`engineering_handover/data/input/db_data/ReferenceCompanyAlignedData_with_sw.csv`
- 公式字典：`engineering_handover/data/input/db_data/公式字典.csv`
- 财报字典：`engineering_handover/data/input/db_data/财务报表字典表.csv`

## 默认输出

默认输出根目录：

- `engineering_handover/data/output/attention_risk_delivery`

每次运行会在以下目录生成产物：

- `<output_root>/<model_name>/<run_tag>/predictions/all_company_predictions.csv`
- `<output_root>/<model_name>/<run_tag>/risk_analysis/model_scores/*.csv`
- `<output_root>/<model_name>/<run_tag>/risk_analysis/jsons/*.json`
- `<output_root>/<model_name>/<run_tag>/risk_analysis/reports/*.md`
- `<output_root>/<model_name>/<run_tag>/risk_analysis/pdfs/*.pdf`
- `<output_root>/<model_name>/<run_tag>/shap/*.csv`

其中 `shap/*.csv` 默认按 `model_name + data_version + stkcd + year + explain config` 命名，可直接复用为缓存。

## 示例命令

生成 2024 和 2025 全量公司风险报告：

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py
```

使用 4 张卡并行，4 个 worker：

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py \
  --gpu-devices 0,1,2,3 \
  --workers 4
```

使用 4 张卡并行，8 或 16 个 worker：

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py \
  --gpu-devices 0,1,2,3 \
  --workers 8
```

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py \
  --gpu-devices 0,1,2,3 \
  --workers 16
```

说明：

- `--workers` 表示并发 worker 数，是多进程并发
- GPU 模式下会按 `--gpu-devices` 轮转分配 worker
- 例如 `--gpu-devices 0,1,2,3 --workers 8` 表示每张卡大约承载 2 个 worker
- worker 过多会增加显存压力，建议从 `4` 或 `8` 开始测试

仅跑单家公司做验证：

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py \
  --years 2024 \
  --stkcds 300900 \
  --run-tag sanity_check_300900_2024
```

显示真实标签：

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py \
  --show-true-label
```

强制重算 shap：

```bash
uv run python engineering_handover/attention_risk_delivery/run_attention_batch_reports.py \
  --force-recompute-shap
```
