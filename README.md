# 工程交接目录

本目录用于将 `visualization/app.py` 背后的两段核心能力拆成可独立执行的 pipeline，并整理成工程实践部门可直接接手的交接物。

## 目录结构

```text
engineering_handover/
├── README.md
├── AGENT_HANDOFF.md
├── PRD.md
├── configs/
│   ├── prediction_default.json
│   ├── risk_report_default.json
│   └── support_files.json
├── vendor/
│   ├── predict_scripts/
│   ├── portable_pipeline_studio/
│   └── src/
├── pipelines/
│   ├── prediction_pipeline.py
│   ├── risk_report_pipeline.py
│   └── shared.py
└── data/
    ├── input/
    └── output/
```

## 两条 Pipeline

### 1. 批量预测 Pipeline

用途：
- 读取源数据集
- 加载 TabPFN v3 模型
- 生成 2024-2025 等年份区间的预测结果
- 输出工程可交付的 `xlsx/csv/json`

运行命令：

```bash
uv run python engineering_handover/pipelines/prediction_pipeline.py
```

CPU 运行：

```bash
uv run python engineering_handover/pipelines/prediction_pipeline.py --device cpu
```

关键输出：
- `prediction_scores.xlsx`
- `prediction_scores.csv`
- `predictions.csv`
- `predictions_summary.json`
- `handover_manifest.json`

其中 `prediction_scores.xlsx` 第一张表使用中文交付列：
- `股票代码`
- `公司简称`
- `年份`
- `财务报告风险指数-TabPFNv3`
- `模型训练结果`
- `真实标签`
- `财务报告风险级别`

第二张表 `meta` 写入运行元信息，包括模型路径、数据路径、设备、年份范围、输出文件列表。

### 2. SHAP + 风险报告 Pipeline

用途：
- 输入 `stkcd` 或 `公司简称 + 年份`
- 加载同一模型与源数据
- 生成 SHAP `csv/xlsx/json`
- 从 SHAP 结果衔接高风险会计科目、Z-score、Markdown 风险报告
- 保存可审计的中间结果

运行命令：

```bash
uv run python engineering_handover/pipelines/risk_report_pipeline.py --stkcd 600107 --year 2024
```

按公司简称运行：

```bash
uv run python engineering_handover/pipelines/risk_report_pipeline.py --short-name 美尔雅 --year 2024
```

CPU 运行：

```bash
uv run python engineering_handover/pipelines/risk_report_pipeline.py --stkcd 600107 --year 2024 --device cpu
```

关键输出结构：

```text
.../risk_report_runs/<model_name>/<run_id>/
├── handover_manifest.json
├── shap/
│   ├── ablation_<stkcd>_<year>_sw_industry_l2_100.csv
│   ├── ablation_<stkcd>_<year>_sw_industry_l2_100.xlsx
│   └── ablation_<stkcd>_<year>_sw_industry_l2_100.json
└── risk_reports/
    └── ablation_<stkcd>_<year>_sw_industry_l2_100/
        ├── manifest.json
        ├── processed/
        │   ├── enriched_comparison/
        │   └── risk_scores_manual/
        └── risk_analysis/
            ├── annual_metrics/
            ├── benchmarks/
            ├── model_scores/
            ├── reports/
            └── zscore_reports/
```

## 默认配置

默认配置文件：
- `configs/prediction_default.json`
- `configs/risk_report_default.json`

默认支持文件映射：
- `configs/support_files.json`

当前版本已将运行所需的本地源码、支持文件和模型副本复制到 `engineering_handover` 目录内。

默认本地副本位置：
- 数据：`data/input/`
- 模型：`data/input/models/`
- vendored 源码：`vendor/`

## 交接原则

- 保留现有核心算法实现，不在交接层重写模型和 SHAP 逻辑
- 在交接层补齐统一 CLI、默认配置、输入输出契约、审计产物
- 交接目录优先加载 `vendor/` 和 `data/input/`，不再依赖仓库根目录中的同名源码和模型
- 支持 `cuda` 与 `cpu` 两种执行方式
- 输出目录只写到 `engineering_handover/data/output/`

## 建议阅读顺序

1. `AGENT_HANDOFF.md`
2. `PRD.md`
3. `configs/support_files.json`
4. `pipelines/prediction_pipeline.py`
5. `pipelines/risk_report_pipeline.py`


# v22版本刷v1.3.2数据
uv run python attention_risk_delivery/run_attention_risk_index_only.py --data-path /data/wk/code/github/fin_fraud_project/data/input/fraud-sent-v1.3.2-processed.csv --output-dir /data/wk/code/github/fin_fraud_project/data/output/attention_risk_delivery_re/attentionrnn_Dv132_Full_001/risk_index_only_v132 --output-stem attentionrnn_v132_财务报告风险指数


# v22版本刷v1.3.2.1数据
uv run python attention_risk_delivery/run_attention_risk_index_only.py --data-path /data/wk/code/github/fin_fraud_project/data/input/fraud-sent-v1.3.2-processed.csv --output-dir /data/wk/code/github/fin_fraud_project/data/output/attention_risk_delivery_re/attentionrnn_Dv132_Full_001/risk_index_only_v132 --output-stem attentionrnn_v132_财务报告风险指数