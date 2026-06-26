from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pandas as pd

from common import (
    CompanyInput,
    DEFAULT_FOCUS_ACCOUNT_FILTER_MODE,
    DICT_REQUIRED_CN_NAMES,
    MANDATORY_ZSCORE_ACCOUNTS,
    RISK_ACCOUNT_METRIC_CONFIG,
    SEVERITY_RANK,
    ensure_directory,
    format_number,
    format_pct,
    normalize_stock_code,
    normalize_text,
    safe_div,
    sanitize_filename_part,
    to_numeric,
)
from markdown_pdf import render_markdown_file


BENCHMARK_COLUMNS = [
    "year",
    "industry_mean",
    "industry_std",
    "industry_median",
    "peer_count",
    "risk_account",
    "metric_key",
    "metric_name",
    "baseline_type",
    "baseline_level",
    "baseline_name",
    "baseline_code",
]

ANOMALY_COLUMNS = [
    "report_date",
    "year",
    "stock_code",
    "stock_name",
    "risk_account",
    "raw_value",
    "metric_name",
    "company_metric",
    "industry_mean",
    "industry_std",
    "industry_median",
    "peer_count",
    "deviation_pct",
    "z_score",
    "abs_z_score",
    "severity",
    "severity_rank",
    "reason",
    "baseline_type",
    "baseline_level",
    "baseline_name",
    "baseline_code",
]

MODEL_RISK_LEVEL_ORDER = ["高", "较高", "中", "低", "较低"]
METRIC_FORMULA_BY_NAME = {
    "货币资金/总资产": "货币资金/资产总计",
    "固定资产/总资产": "固定资产净额/资产总计",
    "营业收入同比增长率": "(营业收入本年金额-营业收入上年金额)/营业收入上年金额",
    "应收账款/营业收入": "应收账款净额/营业收入",
    "其他应收款/总资产": "其他应收款净额/资产总计",
    "预付账款/总资产": "预付款项净额/资产总计",
    "存货/营业收入": "存货净额/营业收入",
    "营业成本/营业收入": "营业成本/营业收入",
    "商誉/总资产": "商誉净额/资产总计",
    "无形资产/总资产": "无形资产净额/资产总计",
    "销售费用/营业收入": "销售费用/营业收入",
    "管理费用/营业收入": "管理费用/营业收入",
    "营业外收入/利润总额": "(营业外收入-营业外支出)/利润总额",
    "海外营业收入/营业收入": "海外营业收入/营业收入",
}


@dataclass(frozen=True)
class ReportArtifacts:
    model_score_path: Path
    report_path: Path
    pdf_path: Path
    json_path: Path
    anomaly_df: pd.DataFrame
    baseline_meta: dict[str, object]


@lru_cache(maxsize=4)
def load_sw_mapping(reference_db_path_text: str) -> pd.DataFrame:
    frame = pd.read_csv(
        reference_db_path_text,
        low_memory=False,
        dtype={
            "Stkcd": "string",
            "SW L1": "string",
            "SW L1 Code": "string",
            "SW L2": "string",
            "SW L2 Code": "string",
            "SW L3": "string",
            "SW L3 Code": "string",
        },
    )
    frame["year"] = pd.to_datetime(frame["Accper"], errors="coerce", dayfirst=True).dt.year.astype("Int64")
    frame["Stkcd"] = frame["Stkcd"].map(normalize_stock_code)
    return frame


def build_financial_dictionary(dict_path_text: str) -> dict[str, str]:
    df = pd.read_csv(dict_path_text)
    required_columns = {"中文简称", "英文简称"}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"字典文件缺少列: {sorted(required_columns - set(df.columns))}")
    mapping = (
        df[["中文简称", "英文简称"]]
        .dropna(subset=["中文简称", "英文简称"])
        .drop_duplicates(subset=["中文简称"], keep="first")
        .set_index("中文简称")["英文简称"]
        .to_dict()
    )
    resolved: dict[str, str] = {}
    for cn_name, fallback_en in DICT_REQUIRED_CN_NAMES.items():
        resolved[cn_name] = normalize_text(mapping.get(cn_name)) or fallback_en
    return resolved


@lru_cache(maxsize=4)
def load_reference_data(reference_db_path_text: str, financial_dict_path_text: str) -> tuple[pd.DataFrame, dict[str, str]]:
    column_mapping = build_financial_dictionary(financial_dict_path_text)
    usecols = [
        "Stkcd",
        "Accper",
        "SW L1",
        "SW L1 Code",
        "SW L2",
        "SW L2 Code",
        "SW L3",
        "SW L3 Code",
        *list(dict.fromkeys(column_mapping.values())),
    ]
    df = pd.read_csv(reference_db_path_text, usecols=lambda col: col in usecols, dtype={"Stkcd": str})
    missing_columns = [column for column in usecols if column not in df.columns]
    if missing_columns:
        raise KeyError(f"基线数据缺少列: {missing_columns}")

    df["Stkcd"] = df["Stkcd"].map(normalize_stock_code)
    df["Accper"] = pd.to_datetime(df["Accper"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Accper"]).copy()
    df["year"] = df["Accper"].dt.year
    df = df[(df["Accper"].dt.month == 12) & (df["Accper"].dt.day == 31)].copy()
    for column in column_mapping.values():
        df[column] = df[column].map(to_numeric)
    return df, column_mapping


def build_zscore_risk_accounts(model_df: pd.DataFrame) -> list[str]:
    ordered_accounts = [normalize_text(value) for value in model_df["高风险会计科目"].tolist()]
    filtered = [account for account in ordered_accounts if account in RISK_ACCOUNT_METRIC_CONFIG]
    for account in MANDATORY_ZSCORE_ACCOUNTS:
        if account not in filtered:
            filtered.append(account)
    return list(dict.fromkeys(filtered))


def choose_baseline(reference_df: pd.DataFrame, target_code: str, baseline_year: int, sw_min_peer_count: int) -> tuple[dict[str, object], list[str]]:
    target_rows = reference_df[(reference_df["Stkcd"] == target_code) & (reference_df["year"] == baseline_year)]
    if target_rows.empty:
        raise RuntimeError(f"未找到 {target_code} 在 {baseline_year} 年的基线信息")

    target_row = target_rows.iloc[0]
    third_name = normalize_text(target_row["SW L3"])
    third_code = normalize_text(target_row["SW L3 Code"])
    second_name = normalize_text(target_row["SW L2"])
    second_code = normalize_text(target_row["SW L2 Code"])

    third_constituents = sorted(
        reference_df[
            (reference_df["year"] == baseline_year)
            & (reference_df["SW L3 Code"].map(normalize_text) == third_code)
        ]["Stkcd"].drop_duplicates().tolist()
    )

    use_second = len(third_constituents) < sw_min_peer_count
    if use_second:
        chosen_name = second_name
        chosen_code = second_code
        chosen_level = "二级"
        constituents = sorted(
            reference_df[
                (reference_df["year"] == baseline_year)
                & (reference_df["SW L2 Code"].map(normalize_text) == second_code)
            ]["Stkcd"].drop_duplicates().tolist()
        )
        rule = (
            f"申万三级行业 {third_name} 成分股 {len(third_constituents)} 家，小于 {sw_min_peer_count} 家，"
            f"回退使用申万二级行业 {second_name}"
        )
    else:
        chosen_name = third_name
        chosen_code = third_code
        chosen_level = "三级"
        constituents = third_constituents
        rule = f"申万三级行业 {third_name} 成分股 {len(third_constituents)} 家，使用申万三级行业作为基线"

    meta = {
        "baseline_year": baseline_year,
        "baseline_type": "申万行业",
        "baseline_level": chosen_level,
        "industry_name": chosen_name,
        "industry_code": chosen_code,
        "rule": rule,
        "declared_constituent_count": len(constituents),
    }
    return meta, constituents


def build_annual_metrics(
    reference_df: pd.DataFrame,
    stock_codes: list[str],
    company_name_map: dict[str, str],
    column_mapping: dict[str, str],
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    subset = reference_df[
        reference_df["Stkcd"].isin(stock_codes)
        & (reference_df["year"] >= min_year)
        & (reference_df["year"] <= max_year)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    revenue_col = column_mapping["营业收入"]
    operating_cost_col = column_mapping["营业成本"]
    cash_col = column_mapping["货币资金"]
    receivable_col = column_mapping["应收账款净额"]
    other_receivables_col = column_mapping["其他应收款净额"]
    prepayments_col = column_mapping["预付款项净额"]
    inventory_col = column_mapping["存货净额"]
    goodwill_col = column_mapping["商誉净额"]
    fixed_assets_col = column_mapping["固定资产净额"]
    intangible_assets_col = column_mapping["无形资产净额"]
    selling_expenses_col = column_mapping["销售费用"]
    administrative_expenses_col = column_mapping["管理费用"]
    total_assets_col = column_mapping["资产总计"]
    non_operating_income_col = column_mapping["营业外收入"]
    total_profit_col = column_mapping["利润总额"]
    overseas_revenue_col = column_mapping["营业收入(海外营业收入)"]

    subset["stock_code"] = subset["Stkcd"]
    subset["stock_name"] = subset["stock_code"].map(lambda code: company_name_map.get(code, code))
    subset["report_date"] = subset["Accper"].dt.strftime("%Y-%m-%d")
    subset["revenue"] = subset[revenue_col]
    subset["operating_cost"] = subset[operating_cost_col]
    subset["cash"] = subset[cash_col]
    subset["accounts_receivable"] = subset[receivable_col]
    subset["other_receivables"] = subset[other_receivables_col]
    subset["prepayments"] = subset[prepayments_col]
    subset["inventory"] = subset[inventory_col]
    subset["goodwill"] = subset[goodwill_col]
    subset["fixed_assets"] = subset[fixed_assets_col]
    subset["intangible_assets"] = subset[intangible_assets_col]
    subset["selling_expenses"] = subset[selling_expenses_col]
    subset["administrative_expenses"] = subset[administrative_expenses_col]
    subset["total_assets"] = subset[total_assets_col]
    subset["non_operating_income"] = subset[non_operating_income_col]
    subset["total_profit"] = subset[total_profit_col]
    subset["overseas_operating_revenue"] = subset[overseas_revenue_col]

    annual_df = subset[
        [
            "stock_code",
            "stock_name",
            "report_date",
            "year",
            "revenue",
            "operating_cost",
            "cash",
            "accounts_receivable",
            "other_receivables",
            "prepayments",
            "inventory",
            "goodwill",
            "fixed_assets",
            "intangible_assets",
            "selling_expenses",
            "administrative_expenses",
            "total_assets",
            "non_operating_income",
            "total_profit",
            "overseas_operating_revenue",
        ]
    ].sort_values(["stock_code", "year"]).reset_index(drop=True)

    annual_df["cash_to_total_assets"] = annual_df.apply(lambda row: safe_div(row["cash"], row["total_assets"]), axis=1)
    annual_df["operating_cost_to_revenue"] = annual_df.apply(lambda row: safe_div(row["operating_cost"], row["revenue"]), axis=1)
    annual_df["receivable_to_revenue"] = annual_df.apply(lambda row: safe_div(row["accounts_receivable"], row["revenue"]), axis=1)
    annual_df["other_receivables_to_total_assets"] = annual_df.apply(
        lambda row: safe_div(row["other_receivables"], row["total_assets"]),
        axis=1,
    )
    annual_df["prepayments_to_total_assets"] = annual_df.apply(lambda row: safe_div(row["prepayments"], row["total_assets"]), axis=1)
    annual_df["inventory_to_revenue"] = annual_df.apply(lambda row: safe_div(row["inventory"], row["revenue"]), axis=1)
    annual_df["goodwill_to_total_assets"] = annual_df.apply(lambda row: safe_div(row["goodwill"], row["total_assets"]), axis=1)
    annual_df["fixed_assets_to_total_assets"] = annual_df.apply(
        lambda row: safe_div(row["fixed_assets"], row["total_assets"]),
        axis=1,
    )
    annual_df["intangible_assets_to_total_assets"] = annual_df.apply(
        lambda row: safe_div(row["intangible_assets"], row["total_assets"]),
        axis=1,
    )
    annual_df["selling_expenses_to_revenue"] = annual_df.apply(
        lambda row: safe_div(row["selling_expenses"], row["revenue"]),
        axis=1,
    )
    annual_df["administrative_expenses_to_revenue"] = annual_df.apply(
        lambda row: safe_div(row["administrative_expenses"], row["revenue"]),
        axis=1,
    )
    annual_df["non_operating_income_to_total_profit"] = annual_df.apply(
        lambda row: safe_div(row["non_operating_income"], row["total_profit"]),
        axis=1,
    )
    annual_df["overseas_revenue_to_revenue"] = annual_df.apply(
        lambda row: safe_div(row["overseas_operating_revenue"], row["revenue"]),
        axis=1,
    )
    annual_df["revenue_yoy"] = annual_df.groupby("stock_code")["revenue"].pct_change(fill_method=None)
    return annual_df


def build_benchmarks(metrics_df: pd.DataFrame, target_code: str, baseline_meta: dict[str, object]) -> pd.DataFrame:
    peer_df = metrics_df[metrics_df["stock_code"] != target_code].copy()
    rows: list[dict[str, object]] = []
    for risk_account, config in RISK_ACCOUNT_METRIC_CONFIG.items():
        metric_key = config["metric_key"]
        subset = peer_df[["year", "stock_code", metric_key]].dropna()
        if subset.empty:
            continue
        summary = subset.groupby("year")[metric_key].agg(["mean", "std", "median", "count"]).reset_index()
        summary["risk_account"] = risk_account
        summary["metric_key"] = metric_key
        summary["metric_name"] = config["metric_name"]
        summary.rename(
            columns={
                "mean": "industry_mean",
                "std": "industry_std",
                "median": "industry_median",
                "count": "peer_count",
            },
            inplace=True,
        )
        rows.extend(summary.to_dict(orient="records"))
    benchmark_df = pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)
    if benchmark_df.empty:
        return benchmark_df
    benchmark_df["baseline_type"] = str(baseline_meta["baseline_type"])
    benchmark_df["baseline_level"] = str(baseline_meta["baseline_level"])
    benchmark_df["baseline_name"] = str(baseline_meta["industry_name"])
    benchmark_df["baseline_code"] = str(baseline_meta["industry_code"])
    return benchmark_df


def classify_severity(z_score: float | None, deviation_pct: float | None, direction: str) -> str:
    signal = 0.0
    if z_score is not None and pd.notna(z_score):
        signal = float(z_score)
    elif deviation_pct is not None and pd.notna(deviation_pct):
        signal = float(deviation_pct)
    if direction == "high" and signal <= 0:
        return "低"
    abs_signal = abs(signal)
    if abs_signal >= 4.0:
        return "高"
    if abs_signal >= 2.5:
        return "较高"
    if abs_signal >= 1.5:
        return "中"
    return "低"


def build_reason(risk_account: str, severity: str, deviation_pct: float | None, metric_name: str) -> str:
    if severity == "低":
        return f"{metric_name} 与行业均值存在轻微偏离"
    direction_text = "高于" if (deviation_pct or 0) > 0 else "低于"
    templates = {
        "货币资金": f"货币资金占总资产比{direction_text}行业均值，需关注资金沉淀、受限资金或资金真实性",
        "应收账款": f"应收账款占营收比{direction_text}行业均值，需关注收入确认质量与回款压力",
        "其他应收款": f"其他应收款占总资产比{direction_text}行业均值，需关注资金占用与往来款异常",
        "预付账款": f"预付账款占总资产比{direction_text}行业均值，需关注预付款异常与采购真实性",
        "存货": f"存货占营收比{direction_text}行业均值，需关注积压、跌价或备货异常",
        "固定资产": f"固定资产占总资产比{direction_text}行业均值，需关注资本开支、产能利用率与折旧压力",
        "商誉": f"商誉占总资产比{direction_text}行业均值，需关注并购溢价与减值风险",
        "无形资产": f"无形资产占总资产比{direction_text}行业均值，需关注资本化与减值测试合理性",
        "营业收入": f"营业收入同比增速{direction_text}行业均值，需结合订单、确认节奏和持续性复核",
        "营业成本": f"营业成本占营收比{direction_text}行业均值，需关注成本结转、毛利率波动与采购真实性",
        "营业外收入": f"营业外收入占利润总额比{direction_text}行业均值，需关注非经常性收益对利润的拉动",
        "销售费用": f"销售费用占营收比{direction_text}行业均值，需关注费用投放、渠道政策与收入匹配性",
        "管理费用": f"管理费用占营收比{direction_text}行业均值，需关注期间费用归集与管理开支异常",
        "海外营业收入": f"海外营业收入占营收比{direction_text}行业均值，需关注境外业务真实性与确认节奏",
    }
    return templates.get(risk_account, f"{metric_name}{direction_text}行业均值")


def build_anomaly_report(
    metrics_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    target_code: str,
    risk_accounts: list[str],
    baseline_meta: dict[str, object],
) -> pd.DataFrame:
    target_df = metrics_df[metrics_df["stock_code"] == target_code].copy()
    if target_df.empty or benchmark_df.empty or "risk_account" not in benchmark_df.columns:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    rows: list[dict[str, object]] = []
    for risk_account in risk_accounts:
        config = RISK_ACCOUNT_METRIC_CONFIG[risk_account]
        metric_key = config["metric_key"]
        raw_value_key = config["raw_value_key"]
        metric_rows = target_df[
            ["stock_code", "stock_name", "report_date", "year", raw_value_key, metric_key]
        ].dropna(subset=[metric_key])
        for _, metric_row in metric_rows.iterrows():
            year = int(metric_row["year"])
            bench = benchmark_df[(benchmark_df["risk_account"] == risk_account) & (benchmark_df["year"] == year)]
            if bench.empty:
                continue
            bench_row = bench.iloc[0]
            company_metric = float(metric_row[metric_key])
            industry_mean = to_numeric(bench_row["industry_mean"])
            industry_std = to_numeric(bench_row["industry_std"])
            deviation_pct = None if industry_mean in (None, 0) else company_metric / industry_mean - 1
            z_score = None if industry_std in (None, 0) or pd.isna(industry_std) else (company_metric - industry_mean) / industry_std
            severity = classify_severity(z_score, deviation_pct, config["direction"])
            abs_z_score = abs(z_score) if z_score is not None and pd.notna(z_score) else 0.0
            rows.append(
                {
                    "report_date": metric_row["report_date"],
                    "year": year,
                    "stock_code": metric_row["stock_code"],
                    "stock_name": metric_row["stock_name"],
                    "risk_account": risk_account,
                    "raw_value": metric_row[raw_value_key],
                    "metric_name": config["metric_name"],
                    "company_metric": company_metric,
                    "industry_mean": industry_mean,
                    "industry_std": industry_std,
                    "industry_median": to_numeric(bench_row["industry_median"]),
                    "peer_count": int(bench_row["peer_count"]),
                    "deviation_pct": deviation_pct,
                    "z_score": z_score,
                    "abs_z_score": abs_z_score,
                    "severity": severity,
                    "severity_rank": SEVERITY_RANK[severity],
                    "reason": build_reason(risk_account, severity, deviation_pct, config["metric_name"]),
                    "baseline_type": str(baseline_meta["baseline_type"]),
                    "baseline_level": str(baseline_meta["baseline_level"]),
                    "baseline_name": str(baseline_meta["industry_name"]),
                    "baseline_code": str(baseline_meta["industry_code"]),
                }
            )
    report_df = pd.DataFrame(rows, columns=ANOMALY_COLUMNS)
    if report_df.empty:
        return report_df
    return report_df.sort_values(
        ["year", "severity_rank", "abs_z_score", "risk_account"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_model_zscore_columns(model_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> pd.DataFrame:
    working = model_df.copy()
    lookup: dict[tuple[str, int], pd.Series] = {}
    for _, row in anomaly_df.iterrows():
        lookup[(normalize_text(row["risk_account"]), int(row["year"]))] = row
    for year in (2023, 2024, 2025):
        zscore_values: list[str] = []
        severity_values: list[str] = []
        for _, row in working.iterrows():
            matched = lookup.get((normalize_text(row["高风险会计科目"]), year))
            if matched is None:
                zscore_values.append("")
                severity_values.append("")
                continue
            zscore_values.append("-" if pd.isna(matched["z_score"]) else f"{float(matched['z_score']):.2f}")
            severity_values.append(normalize_text(matched["severity"]))
        working[f"{year} z-score"] = zscore_values
        working[f"{year} 等级"] = severity_values
    return working


def build_model_risk_view(model_df: pd.DataFrame, fraud_probability: float | None) -> pd.DataFrame:
    working = model_df.copy()
    working["高风险会计科目"] = working["高风险会计科目"].map(normalize_text)
    working["加权得分"] = pd.to_numeric(working["加权得分"], errors="coerce").fillna(0.0)
    total_score = float(working["加权得分"].sum())
    if total_score > 0:
        working["归一化后值"] = working["加权得分"] / total_score
    else:
        working["归一化后值"] = 0.0

    if working.empty:
        working["风险程度"] = pd.Series(dtype="string")
        working["风险程度排序"] = pd.Series(dtype="Int64")
        return working

    median_score = float(working["归一化后值"].quantile(0.50))
    upper_score = float(working["归一化后值"].quantile(0.75))

    def classify_model_risk(value: float) -> str:
        if fraud_probability is not None and fraud_probability > 0.5:
            if value >= upper_score:
                return "高"
            if value >= median_score:
                return "较高"
            return "中"
        if value >= median_score:
            return "低"
        return "较低"

    working["风险程度"] = working["归一化后值"].map(classify_model_risk)
    working["风险程度排序"] = working["风险程度"].map(
        {label: index for index, label in enumerate(MODEL_RISK_LEVEL_ORDER)}
    )
    return working.sort_values(
        ["风险程度排序", "归一化后值", "高风险会计科目"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_model_risk_groups(model_view_df: pd.DataFrame) -> list[dict[str, str]]:
    groups: list[dict[str, str]] = []
    for level in MODEL_RISK_LEVEL_ORDER:
        subset = model_view_df[model_view_df["风险程度"] == level]
        if subset.empty:
            continue
        accounts = "，".join(subset["高风险会计科目"].tolist())
        groups.append({"风险程度": level, "会计科目": accounts})
    return groups


def get_metric_factor_label(metric_name: str | None) -> str:
    normalized = normalize_text(metric_name)
    return normalized or "-"


def get_metric_formula(metric_name: str | None) -> str:
    normalized = normalize_text(metric_name)
    return METRIC_FORMULA_BY_NAME.get(normalized, normalized or "-")


def build_reason_text(reason: str | None) -> str:
    return normalize_text(reason) or "-"


def build_metric_formula_appendix_rows(anomaly_df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_factors: set[str] = set()
    if anomaly_df.empty or "metric_name" not in anomaly_df.columns:
        return rows
    for metric_name in anomaly_df["metric_name"].tolist():
        factor_label = get_metric_factor_label(metric_name)
        if factor_label == "-" or factor_label in seen_factors:
            continue
        seen_factors.add(factor_label)
        rows.append(
            {
                "risk_account_factor": factor_label,
                "risk_account_formula": get_metric_formula(metric_name),
            }
        )
    return rows


def resolve_focus_account_filter_mode(mode: str | None) -> str:
    normalized = normalize_text(mode).strip().lower()
    if normalized == "zscore":
        return "zscore"
    return "model"


def describe_focus_account_filter_mode(mode: str, analysis_start_year: int, analysis_end_year: int) -> str:
    if mode == "zscore":
        return f"近三年（{analysis_start_year}-{analysis_end_year}）出现过“高”或“较高”的会计科目"
    return "第二部分会计科目分析中风险程度为“高”或“较高”的会计科目"


def build_focus_anomaly_groups(
    model_view_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    *,
    min_year: int,
    max_year: int,
    filter_mode: str | None = None,
) -> list[dict[str, object]]:
    if anomaly_df.empty:
        return []
    resolved_mode = resolve_focus_account_filter_mode(filter_mode)
    if resolved_mode == "zscore":
        focus_df = anomaly_df[anomaly_df["severity"].isin(["高", "较高"])].copy()
    else:
        focus_accounts = {
            normalize_text(value)
            for value in model_view_df.loc[
                model_view_df["风险程度"].isin(["高", "较高"]),
                "高风险会计科目",
            ].tolist()
        }
        if not focus_accounts:
            return []
        focus_df = anomaly_df[
            anomaly_df["risk_account"].map(normalize_text).isin(focus_accounts)
        ].copy()
    focus_df = focus_df[
        focus_df["year"].astype(int).between(min_year, max_year)
    ].copy()
    if focus_df.empty:
        return []

    order_rows: list[dict[str, object]] = []
    for risk_account, group_df in focus_df.groupby("risk_account", sort=False):
        order_rows.append(
            {
                "risk_account": risk_account,
                "max_severity_rank": int(group_df["severity_rank"].max()),
                "latest_focus_year": int(group_df["year"].max()),
                "max_abs_z_score": float(group_df["abs_z_score"].max()),
            }
        )
    order_df = pd.DataFrame(order_rows).sort_values(
        ["max_severity_rank", "latest_focus_year", "max_abs_z_score", "risk_account"],
        ascending=[False, False, False, True],
    )

    groups: list[dict[str, object]] = []
    for risk_account in order_df["risk_account"].tolist():
        account_df = anomaly_df[anomaly_df["risk_account"] == risk_account].copy()
        account_df = account_df[
            account_df["year"].astype(int).between(min_year, max_year)
        ].sort_values(["year", "severity_rank", "abs_z_score"], ascending=[True, False, False])
        recent_years = (
            focus_df[focus_df["risk_account"] == risk_account]["year"]
            .dropna()
            .astype(int)
            .sort_values()
            .tolist()
        )
        groups.append(
            {
                "risk_account": normalize_text(risk_account),
                "recent_focus_years": recent_years,
                "rows": [row for _, row in account_df.iterrows()],
            }
        )
    return groups


def render_markdown_report(
    company: CompanyInput,
    model_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    risk_accounts_for_zscore: list[str],
    baseline_meta: dict[str, object],
    min_year: int,
    max_year: int,
    include_appendix: bool = False,
) -> str:
    model_view_df = build_model_risk_view(model_df, company.fraud_probability)
    model_groups = build_model_risk_groups(model_view_df)
    analysis_end_year = min(max_year, int(company.score_year))
    analysis_start_year = max(min_year, analysis_end_year - 2)
    is_low_risk_report = company.fraud_probability is not None and company.fraud_probability < 0.5
    display_anomaly_df = anomaly_df[
        anomaly_df["year"].astype(int).between(analysis_start_year, analysis_end_year)
    ].copy()
    lines = [
        f"# {company.company_name}({company.stock_code}) 风险评估报告",
        "",
        "## 第一部分：风险评估概览",
        "",
        f"- 评估年度: {company.score_year}",
        (
            f"- 财报风险: {company.fraud_probability:.6f}"
            if company.fraud_probability is not None
            else "- 财报风险: -"
        ),
        (
            f"- 风险等级: {company.risk_level}"
            if company.risk_level is not None
            else "- 风险等级: -"
        ),
        f"- 会计科目数量: {len(model_view_df)}",
        "",
        "---",
        "",
        "| 风险程度 | 会计科目 |",
        "|---|---|",
    ]
    for group in model_groups:
        lines.append(f"| {group['风险程度']} | {group['会计科目']} |")

    lines.extend(
        [
            "",
            "## 第二部分：会计科目分析",
            "",
            "| 会计科目 | SHAP值 | 风险程度 | 关联指标 |",
            "|---|---:|---|---|",
        ]
    )
    for _, row in model_view_df.iterrows():
        lines.append(
            f"| {row['高风险会计科目']} | {format_number(row['归一化后值'], 6)} | "
            f"{row['风险程度']} | {normalize_text(row.get('关联指标列表')) or '-'} |"
        )

    lines.extend(
        [
            "",
            "## 第三部分：行业基线分析",
            "",
            f"- 分析区间: {analysis_start_year}-{analysis_end_year}",
            f"- 参考基线: 申万{baseline_meta['baseline_level']}行业 {baseline_meta['industry_name']} ({baseline_meta['industry_code']})",
            f"- 基线规则: {baseline_meta['rule']}",
            f"- 基线样本数: {baseline_meta['declared_constituent_count']}",
            f"- 覆盖科目: {'、'.join(risk_accounts_for_zscore)}",
            "",
        ]
    )

    if display_anomaly_df.empty:
        lines.append("未生成 Z-Score 风险分析结果。")
        return "\n".join(lines) + "\n"

    focus_filter_mode = resolve_focus_account_filter_mode(DEFAULT_FOCUS_ACCOUNT_FILTER_MODE)
    focus_groups = build_focus_anomaly_groups(
        model_view_df,
        display_anomaly_df,
        min_year=analysis_start_year,
        max_year=analysis_end_year,
        filter_mode=focus_filter_mode,
    )
    appendix_formula_rows = (
        build_metric_formula_appendix_rows(display_anomaly_df) if include_appendix else []
    )
    if not is_low_risk_report:
        lines.extend(
            [
                "### 1. 异常科目",
                "",
                f"- 筛选规则: {describe_focus_account_filter_mode(focus_filter_mode, analysis_start_year, analysis_end_year)}",
                "",
            ]
        )
        if not focus_groups:
            if focus_filter_mode == "zscore":
                lines.append("近三年未识别出“高”或“较高”的会计科目。")
            else:
                lines.append("第二部分会计科目分析中未识别出风险程度为“高”或“较高”的会计科目。")
            lines.append("")
        for group in focus_groups:
            lines.extend(
                [
                    f"#### {group['risk_account']}",
                    "",
                    "| 年度 | 风险科目因子 | 公司指标 | 行业均值 | 偏离比例 | z-score | 偏离程度 | 说明 |",
                    "|---|---|---:|---:|---:|---:|---|---|",
                ]
            )
            for row in group["rows"]:
                lines.append(
                    f"| {int(row['year'])} | {get_metric_factor_label(row.get('metric_name'))} | {format_number(row['company_metric'])} | "
                    f"{format_number(row['industry_mean'])} | {format_pct(row['deviation_pct'])} | "
                    f"{format_number(row['z_score'], 2)} | {row['severity']} | {build_reason_text(row.get('reason'))} |"
                )
            lines.append("")

    lines.extend(["### 年度结果" if is_low_risk_report else "### 2. 年度结果", ""])
    for year in sorted(display_anomaly_df["year"].dropna().astype(int).unique(), reverse=True):
        year_df = display_anomaly_df[display_anomaly_df["year"].astype(int) == year].copy()
        lines.extend(
            [
                f"#### {year}年",
                "",
                "| 风险科目因子 | 公司指标 | 行业均值 | 偏离比例 | z-score | 偏离程度 | 说明 |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for _, row in year_df.iterrows():
            lines.append(
                f"| {get_metric_factor_label(row.get('metric_name'))} | {format_number(row['company_metric'])} | {format_number(row['industry_mean'])} | "
                f"{format_pct(row['deviation_pct'])} | {format_number(row['z_score'], 2)} | {row['severity']} | {build_reason_text(row.get('reason'))} |"
            )
        lines.append("")
    if appendix_formula_rows:
        lines.extend(
            [
                "## 附录：因子指标公式对照表",
                "",
                "| 风险科目因子 | 公式 |",
                "|---|---|",
            ]
        )
        for row in appendix_formula_rows:
            lines.append(f"| {row['risk_account_factor']} | {row['risk_account_formula']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_structured_report_payload(
    company: CompanyInput,
    model_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    risk_accounts_for_zscore: list[str],
    baseline_meta: dict[str, object],
    *,
    min_year: int,
    max_year: int,
    include_appendix: bool = False,
) -> dict[str, object]:
    model_view_df = build_model_risk_view(model_df, company.fraud_probability)
    model_groups = build_model_risk_groups(model_view_df)
    analysis_end_year = min(max_year, int(company.score_year))
    analysis_start_year = max(min_year, analysis_end_year - 2)
    display_anomaly_df = anomaly_df[
        anomaly_df["year"].astype(int).between(analysis_start_year, analysis_end_year)
    ].copy()
    focus_filter_mode = resolve_focus_account_filter_mode(DEFAULT_FOCUS_ACCOUNT_FILTER_MODE)
    focus_groups = build_focus_anomaly_groups(
        model_view_df,
        display_anomaly_df,
        min_year=analysis_start_year,
        max_year=analysis_end_year,
        filter_mode=focus_filter_mode,
    )
    appendix_formula_rows = (
        build_metric_formula_appendix_rows(display_anomaly_df) if include_appendix else []
    )
    annual_results: dict[str, list[dict[str, object]]] = {}
    if not display_anomaly_df.empty:
        for year in sorted(display_anomaly_df["year"].dropna().astype(int).unique(), reverse=True):
            year_df = display_anomaly_df[display_anomaly_df["year"].astype(int) == year].copy()
            annual_results[str(year)] = [
                {
                    "risk_account": normalize_text(row["risk_account"]),
                    "risk_account_factor": get_metric_factor_label(row.get("metric_name")),
                    "risk_account_formula": get_metric_formula(row.get("metric_name")),
                    "company_metric": _json_safe_value(to_numeric(row["company_metric"])),
                    "industry_mean": _json_safe_value(to_numeric(row["industry_mean"])),
                    "industry_std": _json_safe_value(to_numeric(row["industry_std"])),
                    "industry_median": _json_safe_value(to_numeric(row["industry_median"])),
                    "peer_count": _json_safe_value(int(row["peer_count"])) if pd.notna(row["peer_count"]) else None,
                    "deviation_pct": _json_safe_value(to_numeric(row["deviation_pct"])),
                    "z_score": _json_safe_value(to_numeric(row["z_score"])),
                    "severity": normalize_text(row["severity"]),
                    "deviation_degree": normalize_text(row["severity"]),
                    "reason": build_reason_text(row.get("reason")),
                }
                for _, row in year_df.iterrows()
            ]
    top_df = display_anomaly_df.copy().sort_values(["severity_rank", "abs_z_score", "year"], ascending=[False, False, False]).head(10)
    return {
        "company": {
            "company_name": company.company_name,
            "stock_code": company.stock_code,
            "score_year": company.score_year,
            "explainer": company.explainer,
        },
        "prediction": {
            "fraud_probability": _json_safe_value(company.fraud_probability),
            "predicted_label": _json_safe_value(company.predicted_label),
            "risk_level": _json_safe_value(company.risk_level),
            "risk_level_rule": "risk_level = min(10, int(fraud_probability * 10) + 1)",
        },
        "report": {
            "section_1_overview": {
                "score_year": int(company.score_year),
                "fraud_probability": _json_safe_value(company.fraud_probability),
                "risk_level": _json_safe_value(company.risk_level),
                "risk_account_count": int(len(model_view_df)),
                "risk_account_groups": model_groups,
            },
            "section_2_model_attribution": {
                "rows": [
                    {
                        "risk_account": normalize_text(row["高风险会计科目"]),
                        "risk_score": _json_safe_value(to_numeric(row["加权得分"])),
                        "normalized_risk_score": _json_safe_value(to_numeric(row["归一化后值"])),
                        "risk_tier": normalize_text(row["风险程度"]),
                        "related_metrics": normalize_text(row.get("关联指标列表")),
                        "formula_audit_info": normalize_text(row.get("公式匹配审查信息")),
                    }
                    for _, row in model_view_df.iterrows()
                ],
            },
            "section_3_peer_baseline_analysis": {
                "analysis_year_range": {"min_year": int(analysis_start_year), "max_year": int(analysis_end_year)},
                "focus_account_filter_mode": focus_filter_mode,
                "focus_account_filter_rule": describe_focus_account_filter_mode(
                    focus_filter_mode,
                    analysis_start_year,
                    analysis_end_year,
                ),
                "baseline": {
                    "baseline_type": normalize_text(baseline_meta["baseline_type"]),
                    "baseline_level": normalize_text(baseline_meta["baseline_level"]),
                    "industry_name": normalize_text(baseline_meta["industry_name"]),
                    "industry_code": normalize_text(baseline_meta["industry_code"]),
                    "declared_constituent_count": _json_safe_value(int(baseline_meta["declared_constituent_count"])),
                    "rule": normalize_text(baseline_meta["rule"]),
                },
                "risk_accounts_for_zscore": risk_accounts_for_zscore,
                "focus_account_timelines": [
                    {
                        "risk_account": group["risk_account"],
                        "recent_focus_years": group["recent_focus_years"],
                        "rows": [
                            {
                                "year": _json_safe_value(int(row["year"])) if pd.notna(row["year"]) else None,
                                "risk_account": normalize_text(row["risk_account"]),
                                "risk_account_factor": get_metric_factor_label(row.get("metric_name")),
                                "risk_account_formula": get_metric_formula(row.get("metric_name")),
                                "company_metric": _json_safe_value(to_numeric(row["company_metric"])),
                                "industry_mean": _json_safe_value(to_numeric(row["industry_mean"])),
                                "deviation_pct": _json_safe_value(to_numeric(row["deviation_pct"])),
                                "z_score": _json_safe_value(to_numeric(row["z_score"])),
                                "severity": normalize_text(row["severity"]),
                                "deviation_degree": normalize_text(row["severity"]),
                                "reason": build_reason_text(row.get("reason")),
                            }
                            for row in group["rows"]
                        ],
                    }
                    for group in focus_groups
                ],
                "annual_results": annual_results,
                "top_anomalies": [
                    {
                        "year": _json_safe_value(int(row["year"])) if pd.notna(row["year"]) else None,
                        "risk_account": normalize_text(row["risk_account"]),
                        "risk_account_factor": get_metric_factor_label(row.get("metric_name")),
                        "risk_account_formula": get_metric_formula(row.get("metric_name")),
                        "company_metric": _json_safe_value(to_numeric(row["company_metric"])),
                        "industry_mean": _json_safe_value(to_numeric(row["industry_mean"])),
                        "deviation_pct": _json_safe_value(to_numeric(row["deviation_pct"])),
                        "z_score": _json_safe_value(to_numeric(row["z_score"])),
                        "severity": normalize_text(row["severity"]),
                        "deviation_degree": normalize_text(row["severity"]),
                        "reason": build_reason_text(row.get("reason")),
                    }
                    for _, row in top_df.iterrows()
                ],
                "metric_formula_appendix": appendix_formula_rows,
            },
        },
    }


def create_company_input(
    company_name: str,
    stock_code: str,
    score_year: int,
    risk_score_path: Path,
    explainer: str,
    *,
    fraud_probability: float | None,
    predicted_label: int | None,
    risk_level: int | None,
) -> CompanyInput:
    stem = f"{sanitize_filename_part(company_name)}（{stock_code}）_{int(score_year)}_{sanitize_filename_part(explainer)}"
    return CompanyInput(
        stem=stem,
        company_name=company_name,
        stock_code=stock_code,
        score_year=int(score_year),
        risk_score_path=risk_score_path,
        explainer=explainer,
        fraud_probability=fraud_probability,
        predicted_label=predicted_label,
        risk_level=risk_level,
    )


def build_report_outputs(
    *,
    company_name: str,
    stock_code: str,
    score_year: int,
    model_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    column_mapping: dict[str, str],
    output_dir: Path,
    explainer: str,
    fraud_probability: float | None,
    predicted_label: int | None,
    risk_level: int | None,
    min_year: int,
    max_year: int,
    sw_min_peer_count: int,
    include_appendix: bool = False,
) -> ReportArtifacts:
    risk_accounts_for_zscore = build_zscore_risk_accounts(model_df)
    baseline_year = min(max_year, int(reference_df[reference_df["Stkcd"] == stock_code]["year"].max()))
    baseline_meta, constituents = choose_baseline(
        reference_df,
        target_code=stock_code,
        baseline_year=baseline_year,
        sw_min_peer_count=sw_min_peer_count,
    )
    company_name_map = {stock_code: company_name}
    metrics_df = build_annual_metrics(
        reference_df,
        stock_codes=constituents,
        company_name_map=company_name_map,
        column_mapping=column_mapping,
        min_year=min_year,
        max_year=max_year,
    )
    if metrics_df.empty:
        raise RuntimeError(f"{company_name}({stock_code}) 未生成年度指标数据")
    benchmark_df = build_benchmarks(metrics_df, target_code=stock_code, baseline_meta=baseline_meta)
    anomaly_df = build_anomaly_report(
        metrics_df,
        benchmark_df,
        target_code=stock_code,
        risk_accounts=risk_accounts_for_zscore,
        baseline_meta=baseline_meta,
    )

    model_scores_dir = ensure_directory(output_dir / "risk_analysis" / "model_scores")
    reports_dir = ensure_directory(output_dir / "risk_analysis" / "reports")
    pdf_dir = ensure_directory(output_dir / "risk_analysis" / "pdfs")
    json_dir = ensure_directory(output_dir / "risk_analysis" / "jsons")
    stem = f"{sanitize_filename_part(company_name)}（{stock_code}）_{int(score_year)}_{sanitize_filename_part(explainer)}"
    model_path = model_scores_dir / f"{stem}_模型高风险科目得分.csv"
    report_path = reports_dir / f"{stem}_风险评估报告.md"
    pdf_path = pdf_dir / f"{stem}_风险评估报告.pdf"
    json_path = json_dir / f"{stem}_风险评估报告.json"
    company = create_company_input(
        company_name,
        stock_code,
        score_year,
        model_path,
        explainer,
        fraud_probability=fraud_probability,
        predicted_label=predicted_label,
        risk_level=risk_level,
    )
    model_df.to_csv(model_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        render_markdown_report(
            company,
            model_df,
            anomaly_df,
            risk_accounts_for_zscore,
            baseline_meta,
            min_year=min_year,
            max_year=max_year,
            include_appendix=include_appendix,
        ),
        encoding="utf-8",
    )
    render_markdown_file(
        report_path,
        pdf_path,
        merge_columns=("年度,风险科目",),
        no_thousands_columns=("年度",),
        wrap_columns=("关联指标", "风险科目因子", "公式"),
        max_lines=("说明:2",),
    )
    json_path.write_text(
        json.dumps(
            build_structured_report_payload(
                company,
                model_df,
                anomaly_df,
                risk_accounts_for_zscore,
                baseline_meta,
                min_year=min_year,
                max_year=max_year,
                include_appendix=include_appendix,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReportArtifacts(
        model_score_path=model_path,
        report_path=report_path,
        pdf_path=pdf_path,
        json_path=json_path,
        anomaly_df=anomaly_df,
        baseline_meta=baseline_meta,
    )
