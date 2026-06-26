from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from common import normalize_stock_code, probability_to_risk_level


ATTENTION_FEATURE_COLUMNS = [f"Ind{index}_" for index in range(1, 153)]
LEGACY_LAYOUT_FEATURE_COLUMNS = [
    *[f"Ind{index}_" for index in range(1, 148)],
    *[f"Ind{index}_" for index in range(149, 153)],
]
LEGACY_CORE_COLUMNS = {
    "Stkcd",
    "ShortName",
    "Accper",
    "IndustryCode",
    "SW_L1",
    "SW_L1_Code",
    "SW_L2",
    "SW_L2_Code",
    "SW_L3",
    "SW_L3_Code",
    "Vio",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
}


@dataclass(frozen=True)
class LoadedAttentionBundle:
    model: torch.nn.Module
    scaler: StandardScaler
    feature_columns: list[str]
    threshold: float
    device_used: str


class Attention(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_in = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.linear_out = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, rnn_output: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        rnn_output_mapped = self.linear_in(rnn_output)
        attn_scores = torch.bmm(rnn_output_mapped, encoder_output.transpose(1, 2))
        attn_scores = self.softmax(attn_scores)
        context = torch.bmm(attn_scores, encoder_output)
        output = torch.cat((rnn_output, context), dim=2)
        return self.tanh(self.linear_out(output))


class AttentionRNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.rnn(x)
        attention_output = self.attention(rnn_output, rnn_output)
        return self.fc(attention_output[:, -1, :])


def register_attention_classes_for_torch_load() -> None:
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    setattr(main_module, "Attention", Attention)
    setattr(main_module, "AttentionRNN", AttentionRNN)


def parse_year_series(series: pd.Series) -> pd.Series:
    datetime_year = pd.to_datetime(series, errors="coerce").dt.year.astype("Int64")
    numeric_year = pd.to_numeric(series, errors="coerce").astype("Int64")
    return datetime_year.fillna(numeric_year)


def source_feature_columns(columns: list[object] | pd.Index) -> list[str]:
    column_names = [str(column) for column in columns]
    matched_by_prefix: list[str] = []
    for feature_name in ATTENTION_FEATURE_COLUMNS:
        candidates = [column_name for column_name in column_names if column_name.startswith(feature_name)]
        if len(candidates) == 1:
            matched_by_prefix.append(candidates[0])
        else:
            matched_by_prefix = []
            break
    if len(matched_by_prefix) == len(ATTENTION_FEATURE_COLUMNS):
        return matched_by_prefix
    features = [str(column) for column in columns if str(column) not in LEGACY_CORE_COLUMNS]
    if len(features) != len(ATTENTION_FEATURE_COLUMNS):
        raise ValueError(f"AttentionRNN 期望 152 个特征列，实际检测到 {len(features)} 个")
    return features


def build_source_feature_mapping(data_path_text: str) -> dict[str, str]:
    frame = pd.read_csv(data_path_text, nrows=1, low_memory=False)
    raw_feature_columns = source_feature_columns(frame.columns)
    return {feature_name: raw_name for feature_name, raw_name in zip(ATTENTION_FEATURE_COLUMNS, raw_feature_columns)}


def build_legacy_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ATTENTION_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少模型特征列: {missing[:10]}")
    # 兼容作者训练/预测口径：删除旧版第 148 维，再追加常数 0 的第 153 维。
    legacy_frame = frame.loc[:, LEGACY_LAYOUT_FEATURE_COLUMNS].copy()
    legacy_frame["Ind153_constant_"] = 0.0
    return legacy_frame


def resolve_runtime_device(preferred_device: str = "auto") -> str:
    if preferred_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_attention_bundle(model_path: Path, data_path_text: str, preferred_device: str = "auto") -> LoadedAttentionBundle:
    register_attention_classes_for_torch_load()
    dataset = load_attention_dataset(data_path_text)
    legacy_matrix = build_legacy_model_matrix(dataset)
    scaler = StandardScaler()
    # scaler.fit(dataset.loc[:, ATTENTION_FEATURE_COLUMNS].to_numpy(dtype=float)) # wk训练方式
    scaler.fit(legacy_matrix.to_numpy(dtype=float))
    device = resolve_runtime_device(preferred_device)
    model = torch.load(model_path, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()
    return LoadedAttentionBundle(
        model=model,
        scaler=scaler,
        feature_columns=list(ATTENTION_FEATURE_COLUMNS),
        threshold=0.5,
        device_used=device,
    )


@lru_cache(maxsize=4)
def get_model_bundle(model_path_text: str, data_path_text: str, preferred_device: str) -> LoadedAttentionBundle:
    return load_attention_bundle(Path(model_path_text), data_path_text, preferred_device=preferred_device)


@lru_cache(maxsize=4)
def load_attention_dataset(data_path_text: str) -> pd.DataFrame:
    source = pd.read_csv(
        data_path_text,
        low_memory=False,
        dtype={"Stkcd": "string", "ShortName": "string", "IndustryCode": "string"},
    )
    raw_feature_columns = source_feature_columns(source.columns)
    data: dict[str, pd.Series] = {
        "Stkcd": source["Stkcd"].map(normalize_stock_code),
        "ShortName": source["ShortName"],
        "IndustryCode": source["IndustryCode"],
        "year": parse_year_series(source["Accper"]),
        "Vio": pd.to_numeric(source["Vio"], errors="coerce"),
        "V1": pd.to_numeric(source["V1"], errors="coerce"),
        "V2": pd.to_numeric(source["V2"], errors="coerce"),
        "V3": pd.to_numeric(source["V3"], errors="coerce"),
        "V4": pd.to_numeric(source["V4"], errors="coerce"),
        "V5": pd.to_numeric(source["V5"], errors="coerce"),
    }
    for feature_name, raw_name in zip(ATTENTION_FEATURE_COLUMNS, raw_feature_columns):
        if raw_name not in source.columns:
            raise KeyError(f"输入数据缺少模型特征列: {raw_name}")
        data[feature_name] = pd.to_numeric(source[raw_name], errors="coerce")
    adapted = pd.DataFrame(data)
    # 与作者口径保持一致：模型数值输入遇到缺失时按 0 处理，而不是整行丢弃。
    numeric_fill_columns = ["Vio", "V1", "V2", "V3", "V4", "V5", *ATTENTION_FEATURE_COLUMNS]
    adapted.loc[:, numeric_fill_columns] = adapted.loc[:, numeric_fill_columns].fillna(0.0)
    adapted["ShortName"] = adapted["ShortName"].fillna("")
    adapted["IndustryCode"] = adapted["IndustryCode"].fillna("")
    adapted = adapted.dropna(subset=["Stkcd", "year"]).reset_index(drop=True)
    return adapted


def transform_features(frame: pd.DataFrame, bundle: LoadedAttentionBundle) -> torch.Tensor:
    legacy_matrix = build_legacy_model_matrix(frame)
    values = legacy_matrix.to_numpy(dtype=float)
    scaled = bundle.scaler.transform(values)
    tensor = torch.tensor(scaled, dtype=torch.float32, device=bundle.device_used)
    return tensor.view(-1, 1, scaled.shape[1])


def score_samples(frame: pd.DataFrame, bundle: LoadedAttentionBundle) -> list[float]:
    inputs = transform_features(frame, bundle)
    with torch.no_grad():
        logits = bundle.model(inputs)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return probabilities.astype(float).tolist()


def predict_single(sample: pd.Series, bundle: LoadedAttentionBundle, model_name: str) -> dict[str, Any]:
    sample_frame = pd.DataFrame([sample.to_dict()])
    probability = float(score_samples(sample_frame, bundle)[0])
    output: dict[str, Any] = {
        "model_name": model_name,
        "stkcd": normalize_stock_code(sample.get("Stkcd")),
        "short_name": sample.get("ShortName"),
        "year": int(sample.get("year")),
        "fraud_probability": probability,
        "predicted_label": int(probability >= bundle.threshold),
        "risk_level": probability_to_risk_level(probability),
    }
    true_label = pd.to_numeric(sample.get("Vio"), errors="coerce")
    output["true_label"] = int(true_label) if pd.notna(true_label) else None
    return output
