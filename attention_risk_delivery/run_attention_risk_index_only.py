from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from attention_model import get_model_bundle, load_attention_dataset, parse_year_series, predict_single
from common import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    HANDOVER_DIR,
    ensure_directory,
    ensure_exists,
    normalize_stock_code,
    normalize_text,
    resolve_repo_path,
)


DEFAULT_CPA_DICT_PATH = HANDOVER_DIR / "data" / "input" / "db_data" / "A股上市公司CPA审计意见及内控意见.xlsx"
DEFAULT_OUTPUT_DIR = HANDOVER_DIR / "data" / "output" / "attention_risk_delivery_re" / DEFAULT_MODEL_NAME / "risk_index_only"


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
    cpa_dict_path: Path
    output_dir: Path
    output_stem: str
    data_version: str
    device: str
    effective_device: str
    gpu_devices: tuple[str, ...]
    workers: int


def infer_data_version(data_path: Path) -> str:
    stem = data_path.stem.lower()
    if "v1.3.4" in stem:
        return "v1.3.4"
    return data_path.stem


def parse_gpu_devices(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def resolve_effective_device(requested_device: str) -> str:
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


def build_tasks(
    dataset: pd.DataFrame,
    *,
    years: list[int] | None,
    stkcds_text: str | None,
    limit: int,
) -> list[TaskSpec]:
    working = dataset.copy()
    if years:
        working = working.loc[working["year"].isin([int(year) for year in years])].copy()
    if stkcds_text:
        allowed_codes = {normalize_stock_code(value) for value in stkcds_text.split(",") if value.strip()}
        working = working.loc[working["Stkcd"].map(normalize_stock_code).isin(allowed_codes)].copy()
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


@lru_cache(maxsize=8)
def load_prediction_context(model_path_text: str, data_path_text: str, device: str):
    dataset = load_attention_dataset(data_path_text)
    bundle = get_model_bundle(model_path_text, data_path_text, device)
    return dataset, bundle


def process_prediction_task(runtime: RuntimeConfig, task: TaskSpec) -> dict[str, Any]:
    dataset, bundle = load_prediction_context(
        str(runtime.model_path),
        str(runtime.data_path),
        runtime.device,
    )
    sample = resolve_sample(dataset, task)
    prediction = predict_single(sample, bundle, model_name=runtime.model_name)
    return {
        "stkcd": prediction["stkcd"],
        "short_name": normalize_text(sample.get("ShortName")) or task.short_name or task.stkcd,
        "year": int(prediction["year"]),
        "fraud_probability": float(prediction["fraud_probability"]),
        "predicted_label": int(prediction["predicted_label"]),
        "risk_level": int(prediction["risk_level"]),
        "true_label": prediction.get("true_label"),
        "device_used": bundle.device_used,
    }


def prediction_worker_loop(worker_index: int, gpu_id: str | None, runtime: RuntimeConfig, task_queue, result_queue) -> None:
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
            result = process_prediction_task(runtime, task)
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


def run_prediction_tasks_parallel(runtime: RuntimeConfig, tasks: list[TaskSpec]) -> tuple[list[dict[str, Any]], str]:
    worker_slots = build_worker_device_slots(runtime)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []

    for worker_index, gpu_id in enumerate(worker_slots):
        process = ctx.Process(
            target=prediction_worker_loop,
            args=(worker_index, gpu_id, runtime, task_queue, result_queue),
        )
        process.start()
        workers.append(process)

    for task in tasks:
        task_queue.put(asdict(task))
    for _ in workers:
        task_queue.put(None)

    prediction_rows: list[dict[str, Any]] = []
    device_used_summary = runtime.effective_device
    errors: list[dict[str, Any]] = []

    for completed in range(1, len(tasks) + 1):
        message = result_queue.get()
        if message["ok"]:
            result = message["result"]
            prediction_rows.append(result)
            device_used_summary = result["device_used"]
            print(
                f"[预测 {completed}/{len(tasks)}] worker={message['worker_index']} gpu={message['gpu_id']} "
                f"完成 {result['short_name']}({result['stkcd']}) {result['year']}"
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
    return prediction_rows, device_used_summary


def load_cpa_mapping(cpa_dict_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(cpa_dict_path)
    required_columns = {"Stkcd", "Accper", "CPA"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"CPA 字典缺少必需列: {sorted(missing_columns)}")
    frame = frame.copy()
    frame["stkcd"] = frame["Stkcd"].map(normalize_stock_code)
    frame["year"] = parse_year_series(frame["Accper"])
    frame["会计师事务所"] = frame["CPA"].map(normalize_text)
    frame = frame.dropna(subset=["stkcd", "year"]).copy()
    frame["year"] = frame["year"].astype(int)
    frame = frame.drop_duplicates(subset=["stkcd", "year"], keep="last")
    return frame.loc[:, ["stkcd", "year", "会计师事务所"]].reset_index(drop=True)


def format_stock_code_for_excel(stock_code: str) -> int | str:
    digits = "".join(ch for ch in str(stock_code) if ch.isdigit())
    if digits:
        return int(digits)
    return stock_code


def build_output_frame(prediction_rows: list[dict[str, Any]], cpa_mapping: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(prediction_rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "股票编码",
                "公司简称",
                "会计师事务所",
                "会计年份",
                "财务报告风险指数",
                "模型训练结果",
                "监管发现结果",
                "财务报告风险级别",
            ]
        )
    frame = frame.merge(cpa_mapping, how="left", on=["stkcd", "year"])
    output = pd.DataFrame(
        {
            "股票编码": frame["stkcd"].map(format_stock_code_for_excel),
            "公司简称": frame["short_name"].map(normalize_text),
            "会计师事务所": frame["会计师事务所"].fillna(""),
            "会计年份": frame["year"].astype(int),
            "财务报告风险指数": frame["fraud_probability"].astype(float),
            "模型训练结果": frame["predicted_label"].astype("Int64"),
            "监管发现结果": pd.to_numeric(frame["true_label"], errors="coerce").astype("Int64"),
            "财务报告风险级别": frame["risk_level"].astype("Int64"),
        }
    )
    output["_stkcd_sort"] = frame["stkcd"].map(normalize_text)
    output = output.sort_values(["_stkcd_sort", "会计年份"], ascending=[True, False]).drop(columns=["_stkcd_sort"])
    return output.reset_index(drop=True)


def write_outputs(frame: pd.DataFrame, output_dir: Path, output_stem: str) -> tuple[Path, Path]:
    ensure_directory(output_dir)
    xlsx_path = output_dir / f"{output_stem}.xlsx"
    csv_path = output_dir / f"{output_stem}.csv"
    frame.to_excel(xlsx_path, index=False)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return xlsx_path, csv_path


def write_summary(
    *,
    runtime: RuntimeConfig,
    tasks: list[TaskSpec],
    frame: pd.DataFrame,
    device_used: str,
    xlsx_path: Path,
    csv_path: Path,
) -> Path:
    summary_path = runtime.output_dir / f"{runtime.output_stem}_summary.json"
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": runtime.model_name,
        "data_version": runtime.data_version,
        "data_path": str(runtime.data_path),
        "cpa_dict_path": str(runtime.cpa_dict_path),
        "output_dir": str(runtime.output_dir),
        "xlsx_path": str(xlsx_path),
        "csv_path": str(csv_path),
        "task_count": len(tasks),
        "row_count": int(len(frame)),
        "device": runtime.device,
        "effective_device": runtime.effective_device,
        "device_used": device_used,
        "workers": runtime.workers,
        "gpu_devices": list(runtime.gpu_devices),
        "columns": frame.columns.tolist(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="仅使用 AttentionRNN 生成财务报告风险指数结果表。")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="默认固定为 v1.3.4 数据")
    parser.add_argument("--cpa-dict-path", type=Path, default=DEFAULT_CPA_DICT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="attentionrnn_v134_财务报告风险指数")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="默认输出数据集中全部年份")
    parser.add_argument("--stkcds", default=None, help="可选，逗号分隔，仅处理这些公司")
    parser.add_argument("--limit", type=int, default=0, help="调试用，仅处理前 N 条任务")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu-devices", default="0,1,2,3", help="GPU 列表，逗号分隔；GPU 模式下轮转分配")
    parser.add_argument("--workers", type=int, default=4, help="并发 worker 数，默认按多卡并行")
    return parser


def resolve_runtime(args: argparse.Namespace) -> RuntimeConfig:
    model_path = ensure_exists(resolve_repo_path(args.model_path), "AttentionRNN 模型文件")
    data_path = ensure_exists(resolve_repo_path(args.data_path), "输入数据文件")
    cpa_dict_path = ensure_exists(resolve_repo_path(args.cpa_dict_path), "CPA 字典文件")
    output_dir = ensure_directory(resolve_repo_path(args.output_dir))
    effective_device = resolve_effective_device(args.device)
    gpu_devices = tuple(parse_gpu_devices(args.gpu_devices))
    if effective_device == "cuda" and not gpu_devices:
        gpu_devices = ("0",)
    return RuntimeConfig(
        model_name=args.model_name,
        model_path=model_path,
        data_path=data_path,
        cpa_dict_path=cpa_dict_path,
        output_dir=output_dir,
        output_stem=str(args.output_stem).strip() or "attentionrnn_v134_财务报告风险指数",
        data_version=infer_data_version(data_path),
        device=args.device,
        effective_device=effective_device,
        gpu_devices=gpu_devices,
        workers=max(1, int(args.workers)),
    )


def main() -> int:
    args = build_parser().parse_args()
    runtime = resolve_runtime(args)

    dataset = load_attention_dataset(str(runtime.data_path))
    tasks = build_tasks(dataset, years=args.years, stkcds_text=args.stkcds, limit=int(args.limit))
    if not tasks:
        raise ValueError("未生成可执行任务，请检查 years 或 stkcds 过滤条件。")

    print(f"预测阶段启动，共 {len(tasks)} 个公司-年份任务。")
    prediction_rows, device_used = run_prediction_tasks_parallel(runtime, tasks)
    cpa_mapping = load_cpa_mapping(runtime.cpa_dict_path)
    output_frame = build_output_frame(prediction_rows, cpa_mapping)
    xlsx_path, csv_path = write_outputs(output_frame, runtime.output_dir, runtime.output_stem)
    summary_path = write_summary(
        runtime=runtime,
        tasks=tasks,
        frame=output_frame,
        device_used=device_used,
        xlsx_path=xlsx_path,
        csv_path=csv_path,
    )

    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "row_count": int(len(output_frame)),
                "xlsx_path": str(xlsx_path),
                "csv_path": str(csv_path),
                "summary_path": str(summary_path),
                "device_used": device_used,
                "effective_device": runtime.effective_device,
                "workers": runtime.workers,
                "gpu_devices": list(runtime.gpu_devices),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
