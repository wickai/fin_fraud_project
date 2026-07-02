from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from attention_model import (  # noqa: E402
    LoadedAttentionBundle,
    build_source_feature_mapping,
    get_model_bundle,
    load_attention_dataset,
    predict_single,
    score_samples,
)
from common import (  # noqa: E402
    DEFAULT_BACKGROUND_LEVEL,
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_FINANCIAL_DICT_PATH,
    DEFAULT_FORMULA_DICT_PATH,
    DEFAULT_MAX_YEAR,
    DEFAULT_MIN_YEAR,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFERENCE_DB_PATH,
    DEFAULT_SW_MIN_PEER_COUNT,
    ensure_directory,
    ensure_exists,
    normalize_stock_code,
    normalize_text,
    resolve_repo_path,
    sanitize_filename_part,
)
from formula_utils import build_model_risk_audit_table, build_model_risk_scores, enrich_shap_with_feature_dictionary  # noqa: E402
from reporting import build_report_outputs, load_reference_data, load_sw_mapping  # noqa: E402


@dataclass(frozen=True)
class TaskSpec:
    stkcd: str
    short_name: str
    year: int


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str
    model_path: Path
    data_path: Path
    data_version: str
    reference_db_path: Path
    formula_dict_path: Path
    financial_dict_path: Path
    output_root: Path
    run_tag: str
    device: str
    effective_device: str
    gpu_devices: tuple[str, ...]
    workers: int
    explainer: str
    background_strategy: str
    background_level: int
    background_size: int
    reference_statistic: str
    nsamples: int
    max_evals: int
    min_year: int
    max_year: int
    sw_min_peer_count: int
    include_appendix: bool
    show_true_label: bool
    force_recompute_shap: bool


def infer_data_version(data_path: Path, explicit_version: str | None) -> str:
    if explicit_version:
        return explicit_version
    stem = data_path.stem.lower()
    if "v1.3.4" in stem:
        return "v1.3.4"
    if "v134" in stem:
        return "v134"
    if "v1.3.2" in stem:
        return "v1.3.2"
    return stem.replace("fraud-sent-", "").replace("-processed", "") or "unknown"


def build_run_tag(data_version: str, years: list[int], explicit_tag: str | None) -> str:
    if explicit_tag:
        return explicit_tag
    year_part = "_".join(str(year) for year in sorted(dict.fromkeys(years)))
    return f"{data_version}_years_{year_part}"


def parse_gpu_devices(raw_value: str | None) -> list[str]:
    if raw_value in (None, ""):
        return []
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def resolve_effective_device(requested_device: str) -> str:
    import torch

    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已指定 --device cuda，但当前环境不可用 CUDA。")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_worker_device_slots(runtime: RuntimeConfig) -> list[str | None]:
    if runtime.effective_device != "cuda":
        return [None] * max(1, int(runtime.workers))
    gpu_devices = list(runtime.gpu_devices) or ["0"]
    worker_count = max(1, int(runtime.workers))
    return [gpu_devices[index % len(gpu_devices)] for index in range(worker_count)]


def select_background(
    dataset: pd.DataFrame,
    sw_df: pd.DataFrame,
    bundle: LoadedAttentionBundle,
    *,
    target_sample: pd.Series,
    strategy: str,
    background_size: int,
    level: int | None,
    min_peer_count: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    year = int(target_sample["year"])
    target_stkcd = normalize_stock_code(target_sample.get("Stkcd"))
    all_candidates = dataset.loc[dataset["Stkcd"].map(normalize_stock_code).ne(target_stkcd)].copy()
    same_year_candidates = all_candidates.loc[all_candidates["year"].eq(year)].copy()
    candidates = same_year_candidates.copy()
    stats: dict[str, Any] = {
        "background_strategy": strategy,
        "background_level_requested": level,
        "background_size_requested": background_size,
        "year": year,
        "target_excluded": True,
    }
    if strategy == "sw_industry":
        same_year_mapping = sw_df.loc[sw_df["year"].eq(year)].copy()
        all_year_mapping = sw_df.copy()
        target_rows = same_year_mapping.loc[same_year_mapping["Stkcd"].eq(target_stkcd)]
        start_level = int(level) if level is not None else 3
        fallback_levels = list(dict.fromkeys(level_num for level_num in range(start_level, 0, -1)))
        stats["fallback_chain"] = [f"SW L{level_num}" for level_num in fallback_levels]

        def _industry_candidates(level_num: int, *, across_year: bool) -> tuple[pd.DataFrame, str | None, int]:
            level_code_col = f"SW L{level_num} Code"
            if target_rows.empty or level_code_col not in target_rows.columns:
                return pd.DataFrame(columns=dataset.columns), None, 0
            industry_code = normalize_text(target_rows.iloc[0].get(level_code_col))
            if not industry_code:
                return pd.DataFrame(columns=dataset.columns), None, 0
            mapping_source = all_year_mapping if across_year else same_year_mapping
            if level_code_col not in mapping_source.columns:
                return pd.DataFrame(columns=dataset.columns), industry_code, 0
            peer_codes = (
                mapping_source.loc[
                    mapping_source[level_code_col].map(normalize_text).eq(industry_code),
                    "Stkcd",
                ]
                .dropna()
                .map(normalize_stock_code)
                .dropna()
                .unique()
                .tolist()
            )
            base_candidates = all_candidates if across_year else same_year_candidates
            filtered = base_candidates.loc[base_candidates["Stkcd"].map(normalize_stock_code).isin(peer_codes)].copy()
            peer_company_count = int(filtered["Stkcd"].map(normalize_stock_code).nunique())
            return filtered, industry_code, peer_company_count

        fallback_attempts: list[dict[str, Any]] = []
        selected_scope = "same_year_market"
        selected_level: int | None = None
        selected_industry_code: str | None = None
        fallback_choice = same_year_candidates.copy()
        fallback_company_count = int(fallback_choice["Stkcd"].map(normalize_stock_code).nunique())

        for level_num in fallback_levels:
            filtered, industry_code, peer_company_count = _industry_candidates(level_num, across_year=False)
            fallback_attempts.append(
                {
                    "scope": "same_year_sw_industry",
                    "level": level_num,
                    "industry_code": industry_code,
                    "peer_company_count": peer_company_count,
                }
            )
            if peer_company_count > 0:
                fallback_choice = filtered
                fallback_company_count = peer_company_count
                selected_scope = "same_year_sw_industry"
                selected_level = level_num
                selected_industry_code = industry_code
            if peer_company_count >= min_peer_count:
                candidates = filtered
                break
        else:
            same_year_market_count = int(same_year_candidates["Stkcd"].map(normalize_stock_code).nunique())
            fallback_attempts.append(
                {
                    "scope": "same_year_market",
                    "level": None,
                    "industry_code": None,
                    "peer_company_count": same_year_market_count,
                }
            )
            if same_year_market_count > 0:
                fallback_choice = same_year_candidates
                fallback_company_count = same_year_market_count
                selected_scope = "same_year_market"
                selected_level = None
                selected_industry_code = None
            if same_year_market_count >= min_peer_count:
                candidates = same_year_candidates.copy()
            else:
                for level_num in fallback_levels:
                    filtered, industry_code, peer_company_count = _industry_candidates(level_num, across_year=True)
                    fallback_attempts.append(
                        {
                            "scope": "cross_year_sw_industry",
                            "level": level_num,
                            "industry_code": industry_code,
                            "peer_company_count": peer_company_count,
                        }
                    )
                    if peer_company_count > 0:
                        fallback_choice = filtered
                        fallback_company_count = peer_company_count
                        selected_scope = "cross_year_sw_industry"
                        selected_level = level_num
                        selected_industry_code = industry_code
                    if peer_company_count >= min_peer_count:
                        candidates = filtered
                        break
                else:
                    all_market_count = int(all_candidates["Stkcd"].map(normalize_stock_code).nunique())
                    fallback_attempts.append(
                        {
                            "scope": "cross_year_market",
                            "level": None,
                            "industry_code": None,
                            "peer_company_count": all_market_count,
                        }
                    )
                    if all_market_count > 0:
                        fallback_choice = all_candidates.copy()
                        fallback_company_count = all_market_count
                        selected_scope = "cross_year_market"
                        selected_level = None
                        selected_industry_code = None
                    candidates = fallback_choice.copy()

        if candidates.empty and fallback_company_count > 0:
            candidates = fallback_choice.copy()
        stats["background_level"] = selected_level
        stats["industry_value"] = selected_industry_code
        stats["selected_scope"] = selected_scope
        stats["fallback_attempts"] = fallback_attempts
        stats["available_peer_count"] = int(candidates["Stkcd"].map(normalize_stock_code).nunique())
    else:
        stats["background_level"] = None
        stats["selected_scope"] = "same_year_market"
        stats["available_peer_count"] = int(candidates["Stkcd"].map(normalize_stock_code).nunique())

    candidates = candidates.drop_duplicates(subset=["Stkcd", "year"]).reset_index(drop=True)
    if len(candidates) > background_size:
        candidates = candidates.sample(n=background_size, random_state=random_state).reset_index(drop=True)
    stats["background_size_used"] = int(len(candidates))
    return candidates.loc[:, bundle.feature_columns], stats


def _extract_explanation_vector(explanation: Any) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values[0]


def compute_sampling_shap(
    score_fn,
    sample_frame: pd.DataFrame,
    background_frame: pd.DataFrame,
    *,
    nsamples: int,
) -> pd.DataFrame:
    import shap

    explainer = shap.SamplingExplainer(score_fn, background_frame)
    shap_values = explainer(sample_frame, nsamples=nsamples)
    vector = _extract_explanation_vector(shap_values)
    frame = pd.DataFrame(
        {
            "feature_name_en": sample_frame.columns.tolist(),
            "feature_value": sample_frame.iloc[0].tolist(),
            "shap_value": vector,
            "reference_value": np.nan,
            "score_after_ablation": np.nan,
            "base_probability": np.nan,
        }
    )
    frame["abs_shap_value"] = frame["shap_value"].abs()
    frame["direction"] = np.where(frame["shap_value"] >= 0, "positive", "negative")
    frame["computed_at"] = datetime.now().isoformat(timespec="seconds")
    return frame.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)


def compute_permutation_shap(
    score_fn,
    sample_frame: pd.DataFrame,
    background_frame: pd.DataFrame,
    *,
    max_evals: int,
) -> pd.DataFrame:
    import shap

    explainer = shap.PermutationExplainer(score_fn, background_frame)
    shap_values = explainer(sample_frame, max_evals=max_evals, silent=True)
    vector = _extract_explanation_vector(shap_values)
    frame = pd.DataFrame(
        {
            "feature_name_en": sample_frame.columns.tolist(),
            "feature_value": sample_frame.iloc[0].tolist(),
            "shap_value": vector,
            "reference_value": np.nan,
            "score_after_ablation": np.nan,
            "base_probability": np.nan,
        }
    )
    frame["abs_shap_value"] = frame["shap_value"].abs()
    frame["direction"] = np.where(frame["shap_value"] >= 0, "positive", "negative")
    frame["computed_at"] = datetime.now().isoformat(timespec="seconds")
    return frame.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)


def compute_ablation_importance(
    score_fn,
    sample_frame: pd.DataFrame,
    background_frame: pd.DataFrame,
    *,
    reference_statistic: str,
) -> pd.DataFrame:
    if reference_statistic == "mean":
        reference_series = background_frame.mean(numeric_only=False)
    else:
        reference_series = background_frame.median(numeric_only=False)
    feature_columns = sample_frame.columns.tolist()
    base_probability = float(score_fn(sample_frame)[0])
    repeated_frame = pd.concat([sample_frame.astype(object)] * len(feature_columns), ignore_index=True)
    for idx, feature_name in enumerate(feature_columns):
        repeated_frame.iat[idx, idx] = reference_series[feature_name]

    perturbed_scores = score_fn(repeated_frame)
    shap_values = base_probability - perturbed_scores
    frame = pd.DataFrame(
        {
            "feature_name_en": feature_columns,
            "feature_value": sample_frame.iloc[0].tolist(),
            "shap_value": shap_values,
            "reference_value": reference_series.reindex(feature_columns).tolist(),
            "score_after_ablation": perturbed_scores,
            "base_probability": base_probability,
        }
    )
    frame["abs_shap_value"] = frame["shap_value"].abs()
    frame["direction"] = np.where(frame["shap_value"] >= 0, "positive", "negative")
    frame["computed_at"] = datetime.now().isoformat(timespec="seconds")
    return frame.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)


def explain_single(
    sample: pd.Series,
    background_frame: pd.DataFrame,
    *,
    bundle: LoadedAttentionBundle,
    model_name: str,
    explainer: str,
    background_strategy: str,
    background_level: int | None,
    background_size_requested: int,
    reference_statistic: str,
    nsamples: int,
    max_evals: int,
    formula_dict_path_text: str,
    feature_mapping: dict[str, str],
) -> pd.DataFrame:
    sample_frame = pd.DataFrame([sample.to_dict()]).loc[:, bundle.feature_columns]
    background_frame = background_frame.loc[:, bundle.feature_columns]

    def score_fn(values: Any) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
        else:
            frame = pd.DataFrame(values, columns=bundle.feature_columns)
        if frame.empty:
            return np.empty((0,), dtype=float)
        return np.asarray(score_samples(frame, bundle), dtype=float)

    if explainer == "sampling":
        shap_frame = compute_sampling_shap(score_fn, sample_frame, background_frame, nsamples=nsamples)
    elif explainer == "permutation":
        shap_frame = compute_permutation_shap(score_fn, sample_frame, background_frame, max_evals=max_evals)
    else:
        shap_frame = compute_ablation_importance(
            score_fn,
            sample_frame,
            background_frame,
            reference_statistic=reference_statistic,
        )

    shap_frame.insert(0, "model_name", model_name)
    shap_frame.insert(1, "stkcd", normalize_stock_code(sample.get("Stkcd")))
    shap_frame.insert(2, "short_name", sample.get("ShortName"))
    shap_frame.insert(3, "year", int(sample.get("year")))
    shap_frame["background_strategy"] = background_strategy
    shap_frame["background_level"] = background_level
    shap_frame["background_size_requested"] = background_size_requested
    shap_frame["background_size_used"] = int(len(background_frame))
    shap_frame["source_feature_name"] = shap_frame["feature_name_en"].map(feature_mapping).fillna("")
    return enrich_shap_with_feature_dictionary(shap_frame, formula_dict_path_text)


def build_tasks(
    dataset: pd.DataFrame,
    *,
    years: list[int],
    stkcds_text: str | None,
    limit: int,
) -> list[TaskSpec]:
    working = dataset.loc[dataset["year"].isin([int(year) for year in years])].copy()
    if stkcds_text:
        allowed_codes = {normalize_stock_code(value) for value in stkcds_text.split(",") if value.strip()}
        working = working.loc[working["Stkcd"].map(normalize_stock_code).isin(allowed_codes)]
    working = working.drop_duplicates(subset=["Stkcd", "year"]).sort_values(["year", "Stkcd"]).reset_index(drop=True)
    if limit > 0:
        working = working.head(limit)
    return [
        TaskSpec(
            stkcd=normalize_stock_code(row["Stkcd"]) or "",
            short_name=normalize_text(row["ShortName"]) or "",
            year=int(row["year"]),
        )
        for _, row in working.iterrows()
        if normalize_stock_code(row["Stkcd"])
    ]


def resolve_sample(dataset: pd.DataFrame, task: TaskSpec) -> pd.Series:
    matched = dataset.loc[
        dataset["Stkcd"].map(normalize_stock_code).eq(task.stkcd) & dataset["year"].eq(int(task.year))
    ].reset_index(drop=True)
    if matched.empty:
        raise ValueError(f"未找到目标样本: stkcd={task.stkcd}, year={task.year}")
    return matched.iloc[0]


def build_shap_cache_path(output_dir: Path, runtime: RuntimeConfig, task: TaskSpec) -> Path:
    background_level = runtime.background_level if runtime.background_strategy == "sw_industry" else 0
    file_name = (
        f"{runtime.model_name}_{runtime.data_version}_{task.stkcd}_{task.year}_"
        f"{runtime.explainer}_{runtime.background_strategy}_l{background_level}_{runtime.background_size}.csv"
    )
    return output_dir / "shap" / file_name


def build_prediction_export_row(
    *,
    prediction: dict[str, Any],
    runtime: RuntimeConfig,
    generated_at: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generated_at": generated_at,
        "model_name": runtime.model_name,
        "data_version": runtime.data_version,
        "data_path": str(runtime.data_path),
        "run_tag": runtime.run_tag,
        "explainer": runtime.explainer,
        "background_strategy": runtime.background_strategy,
        "background_level": runtime.background_level if runtime.background_strategy == "sw_industry" else None,
        "background_size": runtime.background_size,
        "hide_true_label": not runtime.show_true_label,
        "stkcd": prediction["stkcd"],
        "short_name": prediction["short_name"],
        "year": prediction["year"],
        "fraud_probability": prediction["fraud_probability"],
        "predicted_label": prediction["predicted_label"],
        "risk_level": prediction["risk_level"],
        "artifacts_ready": False,
        "shap_cache_hit": None,
        "shap_path": "",
        "model_score_path": "",
        "report_path": "",
        "pdf_path": "",
        "json_path": "",
        "formula_audit_path": "",
    }
    if runtime.show_true_label:
        row["true_label"] = prediction.get("true_label")
    return row


def to_relative_output_path(path: Path, output_dir: Path) -> str:
    return os.path.relpath(os.fspath(path), os.fspath(output_dir))


def enrich_prediction_export_row(
    export_row: dict[str, Any],
    *,
    output_dir: Path,
    shap_path: Path,
    shap_cache_hit: bool,
    model_score_path: Path,
    report_path: Path,
    pdf_path: Path,
    json_path: Path,
    formula_audit_path: Path,
) -> dict[str, Any]:
    row = dict(export_row)
    row.update(
        {
            "artifacts_ready": True,
            "shap_cache_hit": shap_cache_hit,
            "shap_path": to_relative_output_path(shap_path, output_dir),
            "model_score_path": to_relative_output_path(model_score_path, output_dir),
            "report_path": to_relative_output_path(report_path, output_dir),
            "pdf_path": to_relative_output_path(pdf_path, output_dir),
            "json_path": to_relative_output_path(json_path, output_dir),
            "formula_audit_path": to_relative_output_path(formula_audit_path, output_dir),
        }
    )
    return row


def write_prediction_output(export_rows: list[dict[str, Any]], prediction_output: Path) -> None:
    frame = pd.DataFrame(export_rows)
    if not frame.empty:
        frame = frame.sort_values(["year", "stkcd"], ascending=[False, True]).reset_index(drop=True)
        output_dir = prediction_output.parent.parent.resolve()
        for column in frame.columns:
            if not column.endswith("_path"):
                continue
            frame[column] = frame[column].apply(
                lambda value: (
                    os.path.relpath(os.fspath(Path(value).expanduser()), os.fspath(output_dir))
                    if isinstance(value, str) and value.strip() and os.path.isabs(value)
                    else value
                )
            )
    frame.to_csv(prediction_output, index=False, encoding="utf-8-sig")


@lru_cache(maxsize=4)
def load_prediction_context(
    model_path_text: str,
    data_path_text: str,
    device: str,
) -> tuple[pd.DataFrame, LoadedAttentionBundle]:
    dataset = load_attention_dataset(data_path_text)
    bundle = get_model_bundle(model_path_text, data_path_text, device)
    return dataset, bundle


@lru_cache(maxsize=4)
def load_worker_context(
    model_path_text: str,
    data_path_text: str,
    reference_db_path_text: str,
    _formula_dict_path_text: str,
    financial_dict_path_text: str,
    device: str,
) -> tuple[pd.DataFrame, LoadedAttentionBundle, pd.DataFrame, dict[str, str], pd.DataFrame, dict[str, str]]:
    dataset = load_attention_dataset(data_path_text)
    bundle = get_model_bundle(model_path_text, data_path_text, device)
    reference_df, column_mapping = load_reference_data(reference_db_path_text, financial_dict_path_text)
    sw_df = load_sw_mapping(reference_db_path_text)
    feature_mapping = build_source_feature_mapping(data_path_text)
    return dataset, bundle, reference_df, column_mapping, sw_df, feature_mapping


def process_prediction_task(runtime: RuntimeConfig, task: TaskSpec, generated_at: str) -> dict[str, Any]:
    dataset, bundle = load_prediction_context(
        str(runtime.model_path),
        str(runtime.data_path),
        runtime.device,
    )
    sample = resolve_sample(dataset, task)
    prediction = predict_single(sample, bundle, model_name=runtime.model_name)
    company_name = normalize_text(sample.get("ShortName")) or task.stkcd
    prediction["short_name"] = company_name
    export_row = build_prediction_export_row(
        prediction=prediction,
        runtime=runtime,
        generated_at=generated_at,
    )
    return {
        "task": asdict(task),
        "short_name": company_name,
        "export_row": export_row,
        "device_used": bundle.device_used,
    }


def process_artifact_task(runtime: RuntimeConfig, output_dir: Path, task: TaskSpec, export_row: dict[str, Any]) -> dict[str, Any]:
    dataset, bundle, reference_df, column_mapping, sw_df, feature_mapping = load_worker_context(
        str(runtime.model_path),
        str(runtime.data_path),
        str(runtime.reference_db_path),
        str(runtime.formula_dict_path),
        str(runtime.financial_dict_path),
        runtime.device,
    )
    sample = resolve_sample(dataset, task)
    prediction = {
        "fraud_probability": float(export_row["fraud_probability"]),
        "predicted_label": int(export_row["predicted_label"]),
        "risk_level": int(export_row["risk_level"]),
    }
    company_name = normalize_text(sample.get("ShortName")) or normalize_text(export_row.get("short_name")) or task.stkcd

    shap_path = build_shap_cache_path(output_dir, runtime, task)
    shap_cache_hit = shap_path.exists() and not runtime.force_recompute_shap
    if shap_cache_hit:
        shap_frame = pd.read_csv(shap_path)
    else:
        background_level = runtime.background_level if runtime.background_strategy == "sw_industry" else None
        background_frame, _background_stats = select_background(
            dataset,
            sw_df,
            bundle,
            target_sample=sample,
            strategy=runtime.background_strategy,
            background_size=runtime.background_size,
            level=background_level,
            min_peer_count=runtime.sw_min_peer_count,
        )
        resolved_background_level = _background_stats.get("background_level", background_level)
        resolved_background_strategy = str(_background_stats.get("selected_scope", runtime.background_strategy))
        shap_frame = explain_single(
            sample,
            background_frame,
            bundle=bundle,
            model_name=runtime.model_name,
            explainer=runtime.explainer,
            background_strategy=resolved_background_strategy,
            background_level=resolved_background_level,
            background_size_requested=runtime.background_size,
            reference_statistic=runtime.reference_statistic,
            nsamples=runtime.nsamples,
            max_evals=runtime.max_evals,
            formula_dict_path_text=str(runtime.formula_dict_path),
            feature_mapping=feature_mapping,
        )
        shap_frame.to_csv(shap_path, index=False, encoding="utf-8-sig")

    shap_frame["fraud_probability"] = prediction["fraud_probability"]
    shap_frame["predicted_label"] = prediction["predicted_label"]
    shap_frame["risk_level"] = prediction["risk_level"]
    audit_dir = ensure_directory(output_dir / "risk_analysis" / "audits")
    audit_stem = f"{sanitize_filename_part(company_name)}（{task.stkcd}）_{int(task.year)}_{sanitize_filename_part(runtime.explainer)}"
    formula_audit_path = audit_dir / f"{audit_stem}_公式匹配审查表.csv"
    formula_audit_df = build_model_risk_audit_table(shap_frame)
    formula_audit_df.to_csv(formula_audit_path, index=False, encoding="utf-8-sig")
    model_df = build_model_risk_scores(shap_frame)
    report_artifacts = build_report_outputs(
        company_name=company_name,
        stock_code=task.stkcd,
        score_year=task.year,
        model_df=model_df,
        reference_df=reference_df,
        column_mapping=column_mapping,
        output_dir=output_dir,
        explainer=runtime.explainer,
        fraud_probability=prediction["fraud_probability"],
        predicted_label=prediction["predicted_label"],
        risk_level=prediction["risk_level"],
        min_year=runtime.min_year,
        max_year=runtime.max_year,
        sw_min_peer_count=runtime.sw_min_peer_count,
        include_appendix=runtime.include_appendix,
    )
    updated_export_row = enrich_prediction_export_row(
        export_row,
        output_dir=output_dir,
        shap_path=shap_path,
        shap_cache_hit=shap_cache_hit,
        model_score_path=report_artifacts.model_score_path,
        report_path=report_artifacts.report_path,
        pdf_path=report_artifacts.pdf_path,
        json_path=report_artifacts.json_path,
        formula_audit_path=formula_audit_path,
    )
    return {
        "task": asdict(task),
        "short_name": company_name,
        "export_row": updated_export_row,
        "device_used": bundle.device_used,
    }


def _prediction_worker_loop(
    worker_index: int,
    gpu_id: str | None,
    runtime: RuntimeConfig,
    generated_at: str,
    task_queue,
    result_queue,
) -> None:
    if gpu_id is not None and runtime.effective_device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        try:
            payload = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if payload is None:
            break
        task = TaskSpec(**payload)
        try:
            result = process_prediction_task(runtime, task, generated_at)
            result_queue.put(
                {
                    "ok": True,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "result": result,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "ok": False,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "task": asdict(task),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def _artifact_worker_loop(
    worker_index: int,
    gpu_id: str | None,
    runtime: RuntimeConfig,
    output_dir_text: str,
    task_queue,
    result_queue,
) -> None:
    if gpu_id is not None and runtime.effective_device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    output_dir = Path(output_dir_text)
    while True:
        try:
            payload = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if payload is None:
            break
        task = TaskSpec(**payload["task"])
        export_row = dict(payload["export_row"])
        try:
            result = process_artifact_task(runtime, output_dir, task, export_row)
            result_queue.put(
                {
                    "ok": True,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "result": result,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "ok": False,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "task": asdict(task),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def run_prediction_tasks_parallel(
    runtime: RuntimeConfig,
    tasks: list[TaskSpec],
    generated_at: str,
) -> tuple[list[dict[str, Any]], str]:
    worker_slots = build_worker_device_slots(runtime)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []

    for worker_index, gpu_id in enumerate(worker_slots):
        process = ctx.Process(
            target=_prediction_worker_loop,
            args=(worker_index, gpu_id, runtime, generated_at, task_queue, result_queue),
        )
        process.start()
        workers.append(process)

    for task in tasks:
        task_queue.put(asdict(task))
    for _ in workers:
        task_queue.put(None)

    export_rows: list[dict[str, Any]] = []
    device_used_summary = runtime.effective_device
    errors: list[dict[str, Any]] = []
    for completed in range(1, len(tasks) + 1):
        message = result_queue.get()
        if message["ok"]:
            result = message["result"]
            export_rows.append(result["export_row"])
            device_used_summary = result["device_used"]
            print(
                f"[预测 {completed}/{len(tasks)}] worker={message['worker_index']} gpu={message['gpu_id']} "
                f"完成 {result['short_name']}({result['task']['stkcd']}) {result['task']['year']}"
            )
        else:
            errors.append(message)

    for process in workers:
        process.join()

    if errors:
        first_error = errors[0]
        raise RuntimeError(
            "预测阶段存在失败。\n"
            f"worker={first_error['worker_index']}\n"
            f"gpu={first_error['gpu_id']}\n"
            f"task={first_error['task']}\n"
            f"error={first_error['error']}\n"
            f"traceback=\n{first_error['traceback']}"
        )
    return export_rows, device_used_summary


def run_artifact_tasks_parallel(
    runtime: RuntimeConfig,
    output_dir: Path,
    export_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    worker_slots = build_worker_device_slots(runtime)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []

    for worker_index, gpu_id in enumerate(worker_slots):
        process = ctx.Process(
            target=_artifact_worker_loop,
            args=(worker_index, gpu_id, runtime, str(output_dir), task_queue, result_queue),
        )
        process.start()
        workers.append(process)

    for export_row in export_rows:
        task_queue.put(
            {
                "task": {
                    "stkcd": export_row["stkcd"],
                    "short_name": export_row.get("short_name", ""),
                    "year": int(export_row["year"]),
                },
                "export_row": export_row,
            }
        )
    for _ in workers:
        task_queue.put(None)

    updated_rows: list[dict[str, Any]] = []
    device_used_summary = runtime.effective_device
    errors: list[dict[str, Any]] = []
    for completed in range(1, len(export_rows) + 1):
        message = result_queue.get()
        if message["ok"]:
            result = message["result"]
            updated_rows.append(result["export_row"])
            device_used_summary = result["device_used"]
            print(
                f"[报告 {completed}/{len(export_rows)}] worker={message['worker_index']} gpu={message['gpu_id']} "
                f"完成 {result['short_name']}({result['task']['stkcd']}) {result['task']['year']}"
            )
        else:
            errors.append(message)

    for process in workers:
        process.join()

    if errors:
        first_error = errors[0]
        raise RuntimeError(
            "报告阶段存在失败。\n"
            f"worker={first_error['worker_index']}\n"
            f"gpu={first_error['gpu_id']}\n"
            f"task={first_error['task']}\n"
            f"error={first_error['error']}\n"
            f"traceback=\n{first_error['traceback']}"
        )
    return updated_rows, device_used_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量生成 AttentionRNN 风险报告交付产物。")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--data-version", default=None, help="默认从 data_path 自动推断")
    parser.add_argument("--reference-db-path", type=Path, default=DEFAULT_REFERENCE_DB_PATH)
    parser.add_argument("--formula-dict-path", type=Path, default=DEFAULT_FORMULA_DICT_PATH)
    parser.add_argument("--financial-dict-path", type=Path, default=DEFAULT_FINANCIAL_DICT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None, help="默认使用 data_version + years 生成稳定目录")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--stkcds", default=None, help="可选，逗号分隔，仅处理这些公司")
    parser.add_argument("--limit", type=int, default=0, help="调试用，仅处理前 N 家")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-devices", default="0,1,2,3", help="GPU 列表，逗号分隔；仅 GPU 模式生效")
    parser.add_argument("--workers", type=int, default=4, help="并发 worker 数，可设为 4/8/16")
    parser.add_argument("--explainer", choices=["ablation", "sampling", "permutation"], default="ablation")
    parser.add_argument("--background-strategy", choices=["sw_industry", "random"], default="sw_industry")
    parser.add_argument("--background-level", type=int, choices=[1, 2, 3], default=DEFAULT_BACKGROUND_LEVEL)
    parser.add_argument("--background-size", type=int, default=DEFAULT_BACKGROUND_SIZE)
    parser.add_argument("--reference-statistic", choices=["median", "mean"], default="median")
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--max-evals", type=int, default=305)
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR)
    parser.add_argument("--sw-min-peer-count", type=int, default=DEFAULT_SW_MIN_PEER_COUNT)
    parser.add_argument("--include-appendix", action="store_true", help="默认隐藏附录，显式开启后才输出")
    parser.add_argument("--show-true-label", action="store_true", help="默认隐藏真实标签")
    parser.add_argument("--force-recompute-shap", action="store_true", help="忽略本地 shap 缓存，强制重算")
    return parser


def resolve_runtime(args: argparse.Namespace) -> RuntimeConfig:
    model_path = ensure_exists(resolve_repo_path(args.model_path), "AttentionRNN 模型文件")
    data_path = ensure_exists(resolve_repo_path(args.data_path), "输入数据文件")
    reference_db_path = ensure_exists(resolve_repo_path(args.reference_db_path), "风险报告基线数据文件")
    formula_dict_path = ensure_exists(resolve_repo_path(args.formula_dict_path), "公式字典文件")
    financial_dict_path = ensure_exists(resolve_repo_path(args.financial_dict_path), "财报字典文件")
    output_root = ensure_directory(resolve_repo_path(args.output_root))
    data_version = infer_data_version(data_path, args.data_version)
    run_tag = build_run_tag(data_version, args.years, args.run_tag)
    effective_device = resolve_effective_device(args.device)
    gpu_devices = tuple(parse_gpu_devices(args.gpu_devices))
    if effective_device == "cuda" and not gpu_devices:
        gpu_devices = ("0",)
    workers = max(1, int(args.workers))
    return RuntimeConfig(
        model_name=args.model_name,
        model_path=model_path,
        data_path=data_path,
        data_version=data_version,
        reference_db_path=reference_db_path,
        formula_dict_path=formula_dict_path,
        financial_dict_path=financial_dict_path,
        output_root=output_root,
        run_tag=run_tag,
        device=args.device,
        effective_device=effective_device,
        gpu_devices=gpu_devices,
        workers=workers,
        explainer=args.explainer,
        background_strategy=args.background_strategy,
        background_level=int(args.background_level),
        background_size=int(args.background_size),
        reference_statistic=args.reference_statistic,
        nsamples=int(args.nsamples),
        max_evals=int(args.max_evals),
        min_year=int(args.min_year),
        max_year=int(args.max_year),
        sw_min_peer_count=int(args.sw_min_peer_count),
        include_appendix=bool(args.include_appendix),
        show_true_label=bool(args.show_true_label),
        force_recompute_shap=bool(args.force_recompute_shap),
    )


def main() -> int:
    args = build_parser().parse_args()
    runtime = resolve_runtime(args)
    output_dir = ensure_directory(runtime.output_root / runtime.model_name / runtime.run_tag)
    ensure_directory(output_dir / "predictions")
    ensure_directory(output_dir / "risk_analysis" / "model_scores")
    ensure_directory(output_dir / "risk_analysis" / "reports")
    ensure_directory(output_dir / "shap")

    dataset = load_attention_dataset(str(runtime.data_path))
    tasks = build_tasks(dataset, years=args.years, stkcds_text=args.stkcds, limit=int(args.limit))
    if not tasks:
        raise ValueError("未生成可执行任务，请检查 years 或 stkcds 过滤条件。")

    generated_at = datetime.now().isoformat(timespec="seconds")
    prediction_output = output_dir / "predictions" / "all_company_predictions.csv"
    print(f"预测阶段启动，共 {len(tasks)} 个公司-年份任务。")
    prediction_rows, device_used = run_prediction_tasks_parallel(runtime, tasks, generated_at)
    write_prediction_output(prediction_rows, prediction_output)
    print(f"预测阶段完成，已写出 {prediction_output}")

    print("报告阶段启动，将基于已落盘的预测结果生成 model_scores 和 Markdown 报告。")
    export_rows, device_used = run_artifact_tasks_parallel(runtime, output_dir, prediction_rows)
    write_prediction_output(export_rows, prediction_output)

    print(
        pd.Series(
            {
                "tasks": len(tasks),
                "prediction_output": str(prediction_output),
                "model_scores_dir": str(output_dir / "risk_analysis" / "model_scores"),
                "reports_dir": str(output_dir / "risk_analysis" / "reports"),
                "shap_dir": str(output_dir / "shap"),
                "device_used": device_used,
                "effective_device": runtime.effective_device,
                "workers": runtime.workers,
                "gpu_devices": ",".join(runtime.gpu_devices),
            }
        ).to_json(force_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
