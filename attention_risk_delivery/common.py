from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DELIVERY_DIR = Path(__file__).resolve().parent


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ROOT_DIR = DELIVERY_DIR.parent

DEFAULT_MODEL_NAME = "attentionrnn_Dv132_Full_001"
DEFAULT_MODEL_PATH = first_existing_path(
    ROOT_DIR / "data" / "input" / "models_attention" / "attentionrnn" / "attentionrnn.pkl",
)
DEFAULT_DATA_PATH = first_existing_path(
    ROOT_DIR / "data" / "input" / "fraud-sent-v1.3.4-processed.csv",
)
DEFAULT_REFERENCE_DB_PATH = first_existing_path(
    ROOT_DIR / "data" / "input" / "db_data" / "ReferenceCompanyAlignedData_with_sw.csv",
)
DEFAULT_FORMULA_DICT_PATH = first_existing_path(
    ROOT_DIR / "data" / "input" / "db_data" / "公式字典.csv",
)
DEFAULT_FINANCIAL_DICT_PATH = first_existing_path(
    ROOT_DIR / "data" / "input" / "db_data" / "财务报表字典表.csv",
)
DEFAULT_OUTPUT_ROOT = first_existing_path(
    ROOT_DIR / "data" / "output" / "attentionrnn_risk_delivery",
)

DEFAULT_MIN_YEAR = 2021
DEFAULT_MAX_YEAR = 2025
DEFAULT_SW_MIN_PEER_COUNT = 10
DEFAULT_BACKGROUND_SIZE = 100
DEFAULT_BACKGROUND_LEVEL = 3
DEFAULT_FOCUS_ACCOUNT_FILTER_MODE: str | None = "model"

MANDATORY_ZSCORE_ACCOUNTS = ("固定资产", "海外营业收入")
MANDATORY_MODEL_ACCOUNTS: tuple[str, ...] = ()
SEVERITY_RANK = {"高": 4, "较高": 3, "中": 2, "低": 1}

DICT_REQUIRED_CN_NAMES = {
    "营业收入": "OperatingRevenue",
    "营业成本": "OperatingCost",
    "货币资金": "Cash",
    "应收账款净额": "NetAccountsReceivable",
    "其他应收款净额": "NetOtherReceivables",
    "预付款项净额": "NetPrepayments",
    "存货净额": "NetInventories",
    "商誉净额": "NetGoodwill",
    "固定资产净额": "NetFixedAssets",
    "无形资产净额": "NetIntangibleAssets",
    "销售费用": "SellingExpenses",
    "管理费用": "AdministrativeExpenses",
    "资产总计": "TotalAssets",
    "营业外收入": "NonOperatingIncome",
    "利润总额": "TotalProfit",
    "营业收入(海外营业收入)": "OverseasOperatingRevenue",
}

RISK_ACCOUNT_METRIC_CONFIG: dict[str, dict[str, str]] = {
    "货币资金": {
        "metric_key": "cash_to_total_assets",
        "metric_name": "货币资金/总资产",
        "direction": "high",
        "raw_value_key": "cash",
    },
    "固定资产": {
        "metric_key": "fixed_assets_to_total_assets",
        "metric_name": "固定资产/总资产",
        "direction": "high",
        "raw_value_key": "fixed_assets",
    },
    "营业收入": {
        "metric_key": "revenue_yoy",
        "metric_name": "营业收入同比增长率",
        "direction": "both",
        "raw_value_key": "revenue",
    },
    "应收账款": {
        "metric_key": "receivable_to_revenue",
        "metric_name": "应收账款/营业收入",
        "direction": "high",
        "raw_value_key": "accounts_receivable",
    },
    "其他应收款": {
        "metric_key": "other_receivables_to_total_assets",
        "metric_name": "其他应收款/总资产",
        "direction": "high",
        "raw_value_key": "other_receivables",
    },
    "预付账款": {
        "metric_key": "prepayments_to_total_assets",
        "metric_name": "预付账款/总资产",
        "direction": "high",
        "raw_value_key": "prepayments",
    },
    "存货": {
        "metric_key": "inventory_to_revenue",
        "metric_name": "存货/营业收入",
        "direction": "high",
        "raw_value_key": "inventory",
    },
    "营业成本": {
        "metric_key": "operating_cost_to_revenue",
        "metric_name": "营业成本/营业收入",
        "direction": "high",
        "raw_value_key": "operating_cost",
    },
    "商誉": {
        "metric_key": "goodwill_to_total_assets",
        "metric_name": "商誉/总资产",
        "direction": "high",
        "raw_value_key": "goodwill",
    },
    "无形资产": {
        "metric_key": "intangible_assets_to_total_assets",
        "metric_name": "无形资产/总资产",
        "direction": "high",
        "raw_value_key": "intangible_assets",
    },
    "销售费用": {
        "metric_key": "selling_expenses_to_revenue",
        "metric_name": "销售费用/营业收入",
        "direction": "high",
        "raw_value_key": "selling_expenses",
    },
    "管理费用": {
        "metric_key": "administrative_expenses_to_revenue",
        "metric_name": "管理费用/营业收入",
        "direction": "high",
        "raw_value_key": "administrative_expenses",
    },
    "营业外收入": {
        "metric_key": "non_operating_income_to_total_profit",
        "metric_name": "营业外收入/利润总额",
        "direction": "high",
        "raw_value_key": "non_operating_income",
    },
    "海外营业收入": {
        "metric_key": "overseas_revenue_to_revenue",
        "metric_name": "海外营业收入/营业收入",
        "direction": "high",
        "raw_value_key": "overseas_operating_revenue",
    },
}


@dataclass(frozen=True)
class CompanyInput:
    stem: str
    company_name: str
    stock_code: str
    score_year: int
    risk_score_path: Path
    explainer: str | None = None
    fraud_probability: float | None = None
    predicted_label: int | None = None
    risk_level: int | None = None


def resolve_repo_path(value: str | Path | None, *, default: Path | None = None) -> Path:
    raw_value: str | Path | None = value if value not in (None, "") else default
    if raw_value is None:
        raise ValueError("路径参数不能为空")
    path = Path(raw_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.absolute()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} 不存在: {path}")
    return path


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_stock_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits.zfill(6)
    return text


def sanitize_filename_part(text: object) -> str:
    value = str(text).strip()
    for old in '\\/:*?"<>|':
        value = value.replace(old, "_")
    return value.strip(" ._") or "未命名"


def to_numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "nan", "None", "False"}:
        return None
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def probability_to_risk_level(probability: float) -> int:
    level = int(float(probability) * 10.0) + 1
    return max(1, min(10, level))


def format_number(value: object, digits: int = 4) -> str:
    number = to_numeric(value)
    if number is None or math.isnan(number):
        return "-"
    return f"{number:,.{digits}f}"


def format_pct(value: object, digits: int = 2) -> str:
    number = to_numeric(value)
    if number is None or math.isnan(number):
        return "-"
    return f"{number:.{digits}%}"
