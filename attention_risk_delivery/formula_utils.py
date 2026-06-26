from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from common import MANDATORY_MODEL_ACCOUNTS, normalize_text


TEXT_REPLACEMENTS = {
    "（": "(",
    "）": ")",
    "／": "/",
    "—": "-",
    "－": "-",
    "–": "-",
    "＋": "+",
    "，": ",",
    "：": ":",
    "；": ";",
    "“": '"',
    "”": '"',
}

COMPOSITE_ACCOUNT_KEYWORDS = ("利润", "总额", "合计")
COMPOSITE_ACCOUNT_EXACT_NAMES = {
    "流动资产",
    "流动负债",
    "非流动资产",
    "非流动负债",
    "总资产",
    "总负债",
    "股东权益",
    "营业总收入",
    "营业利润",
    "净利润",
    "经营活动现金流量净额",
    "留存收益",
}

EXTRA_ACCOUNT_ALIASES: dict[str, tuple[str, ...]] = {
    "营业收入": ("营业总收入", "营业收入本年金额", "营业收入上年金额", "营业收入", "主营业务收入"),
    "营业成本": ("营业成本本年金额", "营业成本前年金额", "营业成本"),
    "税金及附加": ("税金及附加",),
    "净利润": ("净利润本年金额", "净利润上年金额", "净利润"),
    "所得税费用": ("所得税费用",),
    "财务费用": ("财务费用",),
    "营业利润": ("营业利润",),
    "利润总额": ("本年利润总额", "上年利润总额", "利润总额"),
    "流动资产": ("流动资产合计", "流动资产"),
    "流动负债": ("流动负债合计", "流动负债"),
    "总资产": ("资产总计", "总资产"),
    "总负债": ("负债合计", "总负债"),
    "股东权益": (
        "归属于母公司所有者权益合计期末值",
        "所有者权益合计本年期末余额",
        "所有者权益合计上年期末余额",
        "所有者权益合计",
        "所有者权益",
        "股东权益",
    ),
    "股本": ("实收资本(或股本)本年期末余额", "实收资本(或股本)", "实收资本", "股本"),
    "货币资金": ("现金的期末余额", "货币资金本年期末余额", "货币资金上年期末余额", "货币资金"),
    "应收账款": ("应收账款净额", "应收账款本年期末余额", "应收账款上年期末余额", "应收账款"),
    "应付账款": ("应付账款本年期末余额", "应付账款上年期末余额", "应付账款"),
    "应收票据": ("应收票据净额", "应收票据"),
    "应收款项融资": ("应收款项融资",),
    "应付票据": ("应付票据",),
    "预付账款": ("预付款项净额", "预付账款本年年末金额", "预付账款前年年末金额", "预付账款"),
    "其他应收款": ("其他应收款净额", "其他应收款"),
    "存货": ("存货净额", "存货"),
    "无形资产": ("无形资产净额", "无形资产"),
    "商誉": ("商誉净额", "商誉"),
    "经营活动现金流量净额": ("经营活动产生的现金流量净额", "经营活动现金流量净额"),
    "非流动资产": ("非流动资产合计", "非流动资产"),
    "非流动负债": ("非流动负债合计本年年末金额", "非流动负债合计大前年年末金额", "非流动负债合计", "非流动负债"),
    "固定资产净额": ("固定资产净额",),
    "固定资产折旧": ("固定资产折旧",),
    "油气资产折耗": ("油气资产折耗",),
    "生产性生物资产折旧": ("生产性生物资产折旧",),
    "无形资产摊销": ("无形资产摊销",),
    "长期待摊费用摊销": ("长期待摊费用摊销",),
    "递延所得税资产": ("递延所得税资产减少", "递延所得税资产"),
    "递延所得税负债": ("递延所得税负债增加", "递延所得税负债"),
    "投资收益": ("投资收益",),
    "公允价值变动收益": ("公允价值变动收益",),
    "短期借款": ("短期借款",),
    "长期借款": ("长期借款",),
    "销售费用": ("销售费用",),
    "管理费用": ("管理费用",),
    "留存收益": ("留存收益",),
}

ACCOUNT_MERGE_MAP = {
    "主营业务收入": "营业收入",
}


def normalize_formula_text(formula: str) -> str:
    normalized = formula
    for old, new in TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    return normalized.strip()


def normalize_formula_expression(formula: str) -> str:
    normalized = normalize_formula_text(formula)
    if "=" in normalized:
        normalized = normalized.split("=", 1)[1]
    return normalized.strip()


def should_keep_manual_risk_account(account_name: str) -> bool:
    account = normalize_text(account_name)
    if not account:
        return False
    if account in COMPOSITE_ACCOUNT_EXACT_NAMES:
        return False
    return not any(keyword in account for keyword in COMPOSITE_ACCOUNT_KEYWORDS)


def build_alias_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in EXTRA_ACCOUNT_ALIASES.items():
        merged_canonical = ACCOUNT_MERGE_MAP.get(canonical, canonical)
        all_aliases = {canonical, *aliases}
        for alias in all_aliases:
            pairs.append((alias, merged_canonical))
    pairs.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return pairs


def extract_accounts_from_formula(formula: str, alias_pairs: list[tuple[str, str]]) -> tuple[str, ...]:
    expression = normalize_formula_expression(formula)
    if not expression:
        return ()

    matches: list[tuple[int, str]] = []
    seen: set[str] = set()
    occupied = [False] * len(expression)

    for alias, canonical in alias_pairs:
        if canonical in seen:
            continue
        search_from = 0
        while True:
            start = expression.find(alias, search_from)
            if start < 0:
                break
            end = start + len(alias)
            if not any(occupied[start:end]):
                seen.add(canonical)
                matches.append((start, canonical))
                for index in range(start, end):
                    occupied[index] = True
                break
            search_from = start + 1
    matches.sort(key=lambda item: item[0])
    return tuple(canonical for _, canonical in matches)


def extract_risk_accounts_from_formula(formula: str, alias_pairs: list[tuple[str, str]]) -> tuple[str, ...]:
    accounts = tuple(dict.fromkeys(extract_accounts_from_formula(formula, alias_pairs)))
    if not accounts:
        return ()
    if any(separator in formula for separator in ("/", "／")) and len(accounts) == 2:
        return (accounts[0],)
    return accounts


@lru_cache(maxsize=4)
def load_formula_dictionary(formula_dict_path_text: str) -> pd.DataFrame:
    frame = pd.read_csv(formula_dict_path_text)
    frame = frame.copy()
    frame["indicator_number"] = pd.to_numeric(frame["指标名称"], errors="coerce").astype("Int64")
    return frame


@lru_cache(maxsize=4)
def build_feature_dictionary(formula_dict_path_text: str) -> pd.DataFrame:
    formula_df = load_formula_dictionary(formula_dict_path_text)
    alias_pairs = build_alias_pairs()
    rows: list[dict[str, Any]] = []
    for _, row in formula_df.iterrows():
        indicator_number = row.get("indicator_number")
        if pd.isna(indicator_number):
            continue
        indicator_number = int(indicator_number)
        feature_name_en = f"Ind{indicator_number}_"
        formula_text = normalize_text(row.get("指标计算公式"))
        accounts = extract_accounts_from_formula(formula_text, alias_pairs)
        risk_accounts = extract_risk_accounts_from_formula(formula_text, alias_pairs)
        filtered_accounts = tuple(account for account in risk_accounts if should_keep_manual_risk_account(account))
        risk_accounts = filtered_accounts if len(dict.fromkeys(accounts)) <= 2 else ()
        rows.append(
            {
                "indicator_number": indicator_number,
                "feature_prefix": feature_name_en,
                "feature_name_cn": normalize_text(row.get("指标键")) or normalize_text(row.get("指标名称_全称")),
                "feature_name_full_cn": normalize_text(row.get("指标名称_全称")) or normalize_text(row.get("指标键")),
                "dict_indicator_name": normalize_text(row.get("指标名称")),
                "feature_formula": formula_text,
                "feature_remark": normalize_text(row.get("备注")),
                "original_formula_account_count": len(dict.fromkeys(accounts)),
                "original_formula_high_risk_accounts": "、".join(risk_accounts),
            }
        )
    return pd.DataFrame(rows)


def enrich_shap_with_feature_dictionary(shap_frame: pd.DataFrame, formula_dict_path_text: str) -> pd.DataFrame:
    dictionary = build_feature_dictionary(formula_dict_path_text)
    working = shap_frame.copy()
    working["indicator_number"] = (
        working["feature_name_en"].astype(str).str.extract(r"Ind(\d+)_", expand=False).astype("Int64")
    )
    enriched = working.merge(dictionary, on="indicator_number", how="left")
    for column in [
        "feature_name_cn",
        "feature_name_full_cn",
        "feature_formula",
        "feature_remark",
        "original_formula_high_risk_accounts",
    ]:
        if column in enriched.columns:
            enriched[column] = enriched[column].fillna("")
    return enriched


def build_model_risk_scores(shap_frame: pd.DataFrame) -> pd.DataFrame:
    detail_df = build_model_risk_audit_table(shap_frame)
    if detail_df.empty:
        raise ValueError("未能从 SHAP 结果解析出高风险会计科目，请检查公式字典。")

    model_df = (
        detail_df.groupby("高风险会计科目", as_index=False)
        .agg(
            加权得分=("指标重要性-Shap", "sum"),
            覆盖指标数=("指标名称", "nunique"),
            关联指标列表=("指标名称", lambda values: "；".join(dict.fromkeys(values))),
            公式匹配审查信息=("公式匹配字典", lambda values: " || ".join(dict.fromkeys(str(value) for value in values if str(value).strip()))),
        )
        .sort_values(["加权得分", "覆盖指标数", "高风险会计科目"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    model_df["来源"] = "模型识别"

    existing_accounts = set(model_df["高风险会计科目"])
    extra_rows: list[dict[str, object]] = []
    for account in MANDATORY_MODEL_ACCOUNTS:
        if account in existing_accounts:
            continue
        extra_rows.append(
            {
                "高风险会计科目": account,
                "加权得分": 0.0,
                "覆盖指标数": 0,
                "关联指标列表": "报告补充",
                "公式匹配审查信息": "",
                "来源": "报告补充",
            }
        )
    if extra_rows:
        model_df = pd.concat([model_df, pd.DataFrame(extra_rows)], ignore_index=True)

    return model_df.sort_values(
        ["加权得分", "覆盖指标数", "高风险会计科目"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_model_risk_audit_table(shap_frame: pd.DataFrame) -> pd.DataFrame:
    working = shap_frame.copy()
    working["feature_display_name"] = working["feature_name_full_cn"].astype(str)
    working["importance_value"] = pd.to_numeric(working["abs_shap_value"], errors="coerce").fillna(0.0)
    risk_accounts_series = working["original_formula_high_risk_accounts"].fillna("").astype(str).str.strip()
    working = working.loc[risk_accounts_series.ne("")].copy()
    working["original_formula_high_risk_accounts"] = risk_accounts_series.loc[working.index]

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        accounts = [item.strip() for item in str(row["original_formula_high_risk_accounts"]).split("、") if item.strip()]
        for account in accounts:
            formula_dict_payload = {
                "指标键": normalize_text(row.get("feature_name_cn")),
                "指标名称": normalize_text(row.get("dict_indicator_name")),
                "指标名称_全称": normalize_text(row.get("feature_name_full_cn")),
                "指标计算公式": normalize_text(row.get("feature_formula")),
                "备注": normalize_text(row.get("feature_remark")),
                "匹配单科目": account,
            }
            rows.append(
                {
                    "高风险会计科目": account,
                    "指标名称": normalize_text(row["feature_display_name"]) or normalize_text(row["feature_name_en"]),
                    "指标英文前缀": normalize_text(row.get("feature_name_en")),
                    "指标重要性-Shap": float(row["importance_value"]),
                    "原公式匹配科目": normalize_text(row.get("original_formula_high_risk_accounts")),
                    "指标键": normalize_text(row.get("feature_name_cn")),
                    "指标名称_全称": normalize_text(row.get("feature_name_full_cn")),
                    "指标计算公式": normalize_text(row.get("feature_formula")),
                    "备注": normalize_text(row.get("feature_remark")),
                    "公式匹配字典": json.dumps(formula_dict_payload, ensure_ascii=False, sort_keys=True),
                }
            )

    return pd.DataFrame(rows)
