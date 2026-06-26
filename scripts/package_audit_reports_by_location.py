#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/input/会计师事务所的审计客户")
DEFAULT_BATCH_DIR = Path(
    "data/output/attentionrnn_risk_delivery/attentionrnn_Dv132_Full_001/batch_gpu_2024_2025"
)
DEFAULT_OUTPUT_ROOT = Path("data/output/audit_report_packages")

REPORT_PATH_COLUMN = "匹配PDF相对路径"


@dataclass(frozen=True)
class PredictionRecord:
    stock_code: str
    short_name: str
    year: int
    pdf_rel_path: str
    pdf_abs_path: Path


@dataclass(frozen=True)
class WorkbookInfo:
    source_path: Path
    prefix: str
    location: str
    firm_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按地点/事务所整理风险评估 PDF，并回写带相对路径的 xlsx 后打包为 zip。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="审计客户 xlsx 所在目录。",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=DEFAULT_BATCH_DIR,
        help="包含 predictions/all_company_predictions.csv 与 risk_analysis/pdfs 的批次目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="整理输出根目录。",
    )
    parser.add_argument(
        "--locations",
        nargs="*",
        default=None,
        help="仅处理指定地点；不传则处理全部地点，例如 --locations 辽宁 黑龙江。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标地点目录/zip 已存在则覆盖。",
    )
    return parser.parse_args()


def normalize_stock_code(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(\d{6})", text)
    if not match:
        return None
    return match.group(1)


def normalize_company_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", "", text)
    return text.replace("*", "")


def normalize_firm_name(value: str) -> str:
    return str(value).replace("(", "（").replace(")", "）").strip()


def parse_workbook_info(path: Path) -> Optional[WorkbookInfo]:
    if not path.name.endswith(".xlsx"):
        return None
    stem = path.stem
    match = re.match(r"(?P<prefix>\d+)\.(?P<location>[^-]+)-(?P<firm>.+)", stem)
    if not match:
        return None
    return WorkbookInfo(
        source_path=path,
        prefix=match.group("prefix"),
        location=match.group("location"),
        firm_name=match.group("firm"),
    )


def discover_workbooks(input_dir: Path, locations: Optional[Sequence[str]]) -> Dict[str, List[WorkbookInfo]]:
    wanted = set(locations or [])
    grouped: Dict[str, List[WorkbookInfo]] = {}
    for path in sorted(input_dir.glob("*.xlsx")):
        info = parse_workbook_info(path)
        if info is None:
            continue
        prefix_num = int(info.prefix)
        if prefix_num < 1 or prefix_num > 40:
            continue
        if wanted and info.location not in wanted:
            continue
        grouped.setdefault(info.location, []).append(info)
    return grouped


def load_predictions(batch_dir: Path) -> Tuple[Dict[Tuple[str, int], PredictionRecord], Dict[Tuple[str, int], PredictionRecord]]:
    predictions_path = batch_dir / "predictions" / "all_company_predictions.csv"
    df = pd.read_csv(predictions_path, dtype={"stkcd": str})

    by_code_year: Dict[Tuple[str, int], PredictionRecord] = {}
    by_name_year: Dict[Tuple[str, int], PredictionRecord] = {}

    for row in df.itertuples(index=False):
        stock_code = normalize_stock_code(getattr(row, "stkcd", None))
        short_name = normalize_company_name(getattr(row, "short_name", ""))
        year = int(getattr(row, "year"))
        pdf_rel_path = str(getattr(row, "pdf_path", "")).strip()
        if not pdf_rel_path:
            continue
        pdf_abs_path = batch_dir / pdf_rel_path
        if not pdf_abs_path.exists():
            continue
        record = PredictionRecord(
            stock_code=stock_code or "",
            short_name=short_name,
            year=year,
            pdf_rel_path=pdf_rel_path,
            pdf_abs_path=pdf_abs_path,
        )
        if stock_code:
            by_code_year[(stock_code, year)] = record
        if short_name:
            by_name_year[(short_name, year)] = record
    return by_code_year, by_name_year


def resolve_prediction(
    row: pd.Series,
    year: int,
    by_code_year: Dict[Tuple[str, int], PredictionRecord],
    by_name_year: Dict[Tuple[str, int], PredictionRecord],
) -> Optional[PredictionRecord]:
    code = normalize_stock_code(row.get("证券代码"))
    if code:
        record = by_code_year.get((code, year))
        if record is not None:
            return record
    name = normalize_company_name(row.get("证券名称"))
    if name:
        return by_name_year.get((name, year))
    return None


def write_workbook_with_links(
    workbook: WorkbookInfo,
    location_dir: Path,
    firm_dir: Path,
    by_code_year: Dict[Tuple[str, int], PredictionRecord],
    by_name_year: Dict[Tuple[str, int], PredictionRecord],
    copied_pdfs: Dict[Path, Path],
) -> Tuple[int, int, Path]:
    workbook_output = location_dir / workbook.source_path.name
    excel_file = pd.ExcelFile(workbook.source_path)
    matched_count = 0
    row_count = 0

    with pd.ExcelWriter(workbook_output, engine="openpyxl") as writer:
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(workbook.source_path, sheet_name=sheet_name)
            year_match = re.fullmatch(r"(20\d{2})年", str(sheet_name))
            if year_match and "证券代码" in df.columns:
                year = int(year_match.group(1))
                pdf_paths: List[str] = []
                for _, row in df.iterrows():
                    row_count += 1
                    record = resolve_prediction(row, year, by_code_year, by_name_year)
                    if record is None:
                        pdf_paths.append("")
                        continue
                    relative_pdf = copied_pdfs.get(record.pdf_abs_path)
                    if relative_pdf is None:
                        destination = firm_dir / record.pdf_abs_path.name
                        if not destination.exists():
                            shutil.copy2(record.pdf_abs_path, destination)
                        relative_pdf = destination.relative_to(location_dir)
                        copied_pdfs[record.pdf_abs_path] = relative_pdf
                    pdf_paths.append(relative_pdf.as_posix())
                    matched_count += 1
                df[REPORT_PATH_COLUMN] = pdf_paths
            df.to_excel(writer, sheet_name=str(sheet_name), index=False)

    return row_count, matched_count, workbook_output


def ensure_clean_target(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"目标已存在，请使用 --overwrite 覆盖: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def make_zip_archive(location_dir: Path, overwrite: bool) -> Path:
    zip_path = location_dir.with_suffix(".zip")
    ensure_clean_target(zip_path, overwrite=overwrite)
    archive_base = zip_path.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", root_dir=location_dir.parent, base_dir=location_dir.name)
    return zip_path


def process_location(
    location: str,
    workbooks: Sequence[WorkbookInfo],
    output_root: Path,
    by_code_year: Dict[Tuple[str, int], PredictionRecord],
    by_name_year: Dict[Tuple[str, int], PredictionRecord],
    overwrite: bool,
) -> Dict[str, object]:
    location_dir = output_root / location
    ensure_clean_target(location_dir, overwrite=overwrite)
    location_dir.mkdir(parents=True, exist_ok=True)

    copied_pdfs: Dict[Path, Path] = {}
    workbook_summaries: List[Dict[str, object]] = []
    total_rows = 0
    total_matches = 0

    for workbook in sorted(workbooks, key=lambda item: int(item.prefix)):
        firm_dir = location_dir / workbook.firm_name
        firm_dir.mkdir(parents=True, exist_ok=True)
        row_count, matched_count, workbook_output = write_workbook_with_links(
            workbook=workbook,
            location_dir=location_dir,
            firm_dir=firm_dir,
            by_code_year=by_code_year,
            by_name_year=by_name_year,
            copied_pdfs=copied_pdfs,
        )
        total_rows += row_count
        total_matches += matched_count
        workbook_summaries.append(
            {
                "source_xlsx": str(workbook.source_path),
                "output_xlsx": str(workbook_output),
                "firm_name": workbook.firm_name,
                "rows": row_count,
                "matched_rows": matched_count,
            }
        )

    zip_path = make_zip_archive(location_dir, overwrite=overwrite)
    return {
        "location": location,
        "location_dir": str(location_dir),
        "zip_path": str(zip_path),
        "workbooks": workbook_summaries,
        "pdf_count": len(copied_pdfs),
        "row_count": total_rows,
        "matched_row_count": total_matches,
    }


def iter_summary_lines(results: Iterable[Dict[str, object]]) -> Iterable[str]:
    for result in results:
        yield (
            f"{result['location']}: xlsx={len(result['workbooks'])}, "
            f"pdf={result['pdf_count']}, matched_rows={result['matched_row_count']}/{result['row_count']}, "
            f"zip={result['zip_path']}"
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    batch_dir = args.batch_dir.resolve()
    output_root = args.output_root.resolve()

    workbooks_by_location = discover_workbooks(input_dir=input_dir, locations=args.locations)
    if not workbooks_by_location:
        selected = ",".join(args.locations) if args.locations else "全部地点"
        raise SystemExit(f"未找到可处理的 xlsx，选择条件: {selected}")

    by_code_year, by_name_year = load_predictions(batch_dir=batch_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for location, workbooks in sorted(workbooks_by_location.items()):
        result = process_location(
            location=location,
            workbooks=workbooks,
            output_root=output_root,
            by_code_year=by_code_year,
            by_name_year=by_name_year,
            overwrite=args.overwrite,
        )
        results.append(result)

    print("处理完成:")
    for line in iter_summary_lines(results):
        print(f"- {line}")


if __name__ == "__main__":
    main()
