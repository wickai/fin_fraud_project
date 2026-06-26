from __future__ import annotations

import html
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Iterable

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


DEFAULT_CODE_FONT = "Courier"
DEFAULT_CJK_FONT = "STSong-Light"
DEFAULT_LATIN_FONT = "Helvetica"
NUMERIC_RE = re.compile(
    r"""
    ^[+-]?
    (?:
        (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?
    %?
    $
    """,
    re.VERBOSE,
)
COMMON_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJKSC-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
COMMON_CJK_BOLD_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Bold.otf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJKSC-Bold.otf",
]
COMMON_LATIN_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
COMMON_LATIN_BOLD_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


@dataclass(frozen=True)
class FontBundle:
    cjk_regular: str
    cjk_bold: str
    latin_regular: str
    latin_bold: str


@dataclass
class Theme:
    title_color: str = "#122b45"
    heading_color: str = "#1e3a5f"
    body_color: str = "#243447"
    muted_color: str = "#5a6b7c"
    header_bg: str = "#eaf1fb"
    row_bg: str = "#f8fbff"
    grid_color: str = "#cbd5e1"
    code_bg: str = "#f4f7fb"
    link_color: str = "#2463c5"


@dataclass
class RenderContext:
    fonts: FontBundle
    styles: dict[str, ParagraphStyle]
    theme: Theme
    available_width: float


@dataclass
class TableCell:
    markup: str
    plain_text: str
    is_header: bool


@dataclass(frozen=True)
class PdfTableRuleSet:
    merge_columns: tuple[tuple[str, ...], ...] = ()
    no_thousands_columns: tuple[str, ...] = ()
    wrap_columns: tuple[str, ...] = ()
    max_lines: dict[str, int] | None = None


DEFAULT_PDF_TABLE_RULES = PdfTableRuleSet()


def first_existing_path(paths: Iterable[str | Path]) -> Path | None:
    for raw in paths:
        path = Path(raw)
        if path.exists():
            return path
    return None


def safe_register_ttfont(path: Path, prefix: str) -> str | None:
    font_name = f"{prefix}-{path.stem}"
    try:
        pdfmetrics.getFont(font_name)
        return font_name
    except KeyError:
        pass
    try:
        registerFont(TTFont(font_name, str(path)))
        return font_name
    except Exception:
        return None


def register_cid_font(name: str) -> str:
    try:
        pdfmetrics.getFont(name)
        return name
    except KeyError:
        registerFont(UnicodeCIDFont(name))
        return name


def resolve_fonts() -> FontBundle:
    cjk_regular_path = first_existing_path(COMMON_CJK_FONT_PATHS)
    cjk_bold_path = first_existing_path(COMMON_CJK_BOLD_FONT_PATHS)
    latin_regular_path = first_existing_path(COMMON_LATIN_FONT_PATHS)
    latin_bold_path = first_existing_path(COMMON_LATIN_BOLD_FONT_PATHS)

    cjk_regular = safe_register_ttfont(cjk_regular_path, "CJKRegular") if cjk_regular_path else None
    cjk_bold = safe_register_ttfont(cjk_bold_path, "CJKBold") if cjk_bold_path else None
    latin_regular = safe_register_ttfont(latin_regular_path, "LatinRegular") if latin_regular_path else None
    latin_bold = safe_register_ttfont(latin_bold_path, "LatinBold") if latin_bold_path else None

    if cjk_regular is None:
        cjk_regular = register_cid_font(DEFAULT_CJK_FONT)
    if cjk_bold is None:
        cjk_bold = cjk_regular
    if latin_regular is None:
        latin_regular = DEFAULT_LATIN_FONT
    if latin_bold is None:
        latin_bold = latin_regular

    return FontBundle(
        cjk_regular=cjk_regular,
        cjk_bold=cjk_bold,
        latin_regular=latin_regular,
        latin_bold=latin_bold,
    )


def build_styles(fonts: FontBundle, theme: Theme) -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "AstPdfTitle",
            parent=sample["Title"],
            fontName=fonts.cjk_bold,
            fontSize=22,
            leading=28,
            textColor=colors.HexColor(theme.title_color),
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "AstPdfH1",
            parent=sample["Heading1"],
            fontName=fonts.cjk_bold,
            fontSize=17,
            leading=23,
            textColor=colors.HexColor(theme.heading_color),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "AstPdfH2",
            parent=sample["Heading2"],
            fontName=fonts.cjk_bold,
            fontSize=14,
            leading=19,
            textColor=colors.HexColor(theme.heading_color),
            spaceBefore=7,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "AstPdfH3",
            parent=sample["Heading3"],
            fontName=fonts.cjk_bold,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor(theme.heading_color),
            spaceBefore=5,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "AstPdfBody",
            parent=sample["BodyText"],
            fontName=fonts.cjk_regular,
            fontSize=10.4,
            leading=15.5,
            textColor=colors.HexColor(theme.body_color),
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "AstPdfBullet",
            parent=sample["BodyText"],
            fontName=fonts.cjk_regular,
            fontSize=10.3,
            leading=15.3,
            textColor=colors.HexColor(theme.body_color),
            leftIndent=14,
            bulletIndent=2,
            spaceAfter=3,
        ),
        "table_head": ParagraphStyle(
            "AstPdfTableHead",
            parent=sample["BodyText"],
            fontName=fonts.cjk_bold,
            fontSize=7.9,
            leading=9.2,
            textColor=colors.HexColor(theme.body_color),
            wordWrap="CJK",
        ),
        "table_body": ParagraphStyle(
            "AstPdfTableBody",
            parent=sample["BodyText"],
            fontName=fonts.cjk_regular,
            fontSize=8.2,
            leading=10.1,
            textColor=colors.HexColor(theme.body_color),
            wordWrap="CJK",
        ),
        "table_body_nowrap": ParagraphStyle(
            "AstPdfTableBodyNoWrap",
            parent=sample["BodyText"],
            fontName=fonts.cjk_regular,
            fontSize=8.2,
            leading=10.1,
            textColor=colors.HexColor(theme.body_color),
            splitLongWords=False,
        ),
        "code": ParagraphStyle(
            "AstPdfCode",
            parent=sample["Code"],
            fontName=fonts.latin_regular,
            fontSize=8.6,
            leading=10.6,
            backColor=colors.HexColor(theme.code_bg),
            leftIndent=6,
            rightIndent=6,
            borderPadding=6,
            borderColor=colors.HexColor(theme.grid_color),
            borderWidth=0.5,
            spaceAfter=6,
        ),
    }


def build_context() -> RenderContext:
    fonts = resolve_fonts()
    theme = Theme()
    return RenderContext(
        fonts=fonts,
        styles=build_styles(fonts, theme),
        theme=theme,
        available_width=A4[0] - 28 * mm,
    )


def is_numeric_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(NUMERIC_RE.match(stripped))


def beautify_numeric_text(text: str) -> str:
    stripped = text.strip().replace(",", "")
    suffix = "%" if stripped.endswith("%") else ""
    if suffix:
        stripped = stripped[:-1]
    try:
        number = Decimal(stripped)
    except InvalidOperation:
        return text

    if suffix:
        precision = 2
    elif number == number.to_integral_value():
        return f"{int(number):,d}{suffix}"
    elif abs(number) >= Decimal("1000"):
        precision = 2
    elif abs(number) >= Decimal("1"):
        precision = 4
    else:
        precision = 6

    quantized = number.quantize(Decimal(1).scaleb(-precision))
    rendered = f"{quantized:,.{precision}f}".rstrip("0").rstrip(".")
    if rendered == "-0":
        rendered = "0"
    return rendered + suffix


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def normalize_header_label(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def parse_merge_columns_specs(specs: Iterable[str] | None) -> tuple[tuple[str, ...], ...]:
    if not specs:
        return ()
    parsed: list[tuple[str, ...]] = []
    for raw in specs:
        columns = tuple(normalize_header_label(part) for part in str(raw).split(",") if normalize_header_label(part))
        if len(columns) >= 2:
            parsed.append(columns)
    return tuple(parsed)


def parse_no_thousands_specs(specs: Iterable[str] | None) -> tuple[str, ...]:
    if not specs:
        return ()
    return tuple(
        normalize_header_label(part)
        for raw in specs
        for part in str(raw).split(",")
        if normalize_header_label(part)
    )


def parse_wrap_columns_specs(specs: Iterable[str] | None) -> tuple[str, ...]:
    if not specs:
        return ()
    return tuple(
        normalize_header_label(part)
        for raw in specs
        for part in str(raw).split(",")
        if normalize_header_label(part)
    )


def parse_max_lines_specs(specs: Iterable[str] | None) -> dict[str, int]:
    parsed: dict[str, int] = {}
    if not specs:
        return parsed
    for raw in specs:
        text = str(raw).strip()
        if ":" not in text:
            continue
        column, value = text.split(":", 1)
        header = normalize_header_label(column)
        if not header:
            continue
        try:
            parsed[header] = max(1, int(value.strip()))
        except ValueError:
            continue
    return parsed


def build_pdf_table_rules(
    *,
    merge_columns: Iterable[str] | None = None,
    no_thousands_columns: Iterable[str] | None = None,
    wrap_columns: Iterable[str] | None = None,
    max_lines: Iterable[str] | None = None,
) -> PdfTableRuleSet:
    return PdfTableRuleSet(
        merge_columns=parse_merge_columns_specs(merge_columns),
        no_thousands_columns=parse_no_thousands_specs(no_thousands_columns),
        wrap_columns=parse_wrap_columns_specs(wrap_columns),
        max_lines=parse_max_lines_specs(max_lines),
    )


def inline_plain_text(inline_token) -> str:
    children = inline_token.children or []
    parts: list[str] = []
    for token in children:
        if token.type in {"text", "code_inline", "html_inline"}:
            parts.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
    return "".join(parts)


def inline_to_markup(inline_token, ctx: RenderContext) -> str:
    children = inline_token.children or []
    pieces: list[str] = []
    for token in children:
        token_type = token.type
        if token_type == "text":
            pieces.append(escape_text(token.content))
        elif token_type == "code_inline":
            pieces.append(
                f'<font face="{ctx.fonts.latin_regular}" backcolor="{ctx.theme.code_bg}">{escape_text(token.content)}</font>'
            )
        elif token_type in {"softbreak", "hardbreak"}:
            pieces.append("<br/>")
        elif token_type == "strong_open":
            pieces.append(f'<font face="{ctx.fonts.cjk_bold}"><b>')
        elif token_type == "strong_close":
            pieces.append("</b></font>")
        elif token_type == "em_open":
            pieces.append("<i>")
        elif token_type == "em_close":
            pieces.append("</i>")
        elif token_type == "link_open":
            pieces.append(f'<font color="{ctx.theme.link_color}">')
        elif token_type == "link_close":
            pieces.append("</font>")
        elif token_type == "html_inline":
            pieces.append(escape_text(token.content))
        else:
            pieces.append(escape_text(token.content or ""))
    return "".join(pieces) or "&nbsp;"


def extract_simple_inline(tokens, idx: int, ctx: RenderContext) -> tuple[str, int]:
    if idx < len(tokens) and tokens[idx].type == "inline":
        return inline_to_markup(tokens[idx], ctx), idx + 1
    return "&nbsp;", idx


def parse_list(tokens, idx: int, story: list, ctx: RenderContext) -> int:
    while idx < len(tokens) and tokens[idx].type != "bullet_list_close":
        if tokens[idx].type != "list_item_open":
            idx += 1
            continue
        idx += 1
        parts: list[str] = []
        while idx < len(tokens) and tokens[idx].type != "list_item_close":
            if tokens[idx].type == "paragraph_open":
                markup, idx = extract_simple_inline(tokens, idx + 1, ctx)
                parts.append(markup)
                if idx < len(tokens) and tokens[idx].type == "paragraph_close":
                    idx += 1
                continue
            idx += 1
        story.append(Paragraph("<br/>".join(parts) if parts else "&nbsp;", ctx.styles["bullet"], bulletText="•"))
        if idx < len(tokens) and tokens[idx].type == "list_item_close":
            idx += 1
    return idx + 1 if idx < len(tokens) else idx


def build_column_widths(rows: list[list[TableCell]], available_width: float) -> list[float]:
    col_count = max(len(row) for row in rows)
    header_labels = [normalize_header_label(cell.plain_text) for cell in rows[0]] if rows else []
    preferred_weights = {
        "年度/风险科目": 2.7,
        "年度风险科目": 2.7,
        "高风险会计科目": 2.1,
        "会计科目": 2.1,
        "SHAP值": 1.3,
        "风险科目因子": 2.9,
        "公司指标": 1.15,
        "行业均值": 1.15,
        "偏离比例": 1.25,
        "z-score": 1.1,
        "偏离程度": 1.55,
        "异常等级": 1.0,
        "说明": 4.1,
        "关联指标列表": 4.2,
        "关联指标": 4.2,
        "公式": 5.2,
    }
    weights = [1.0] * col_count
    for col_index in range(col_count):
        max_length = 1
        for row in rows:
            if col_index >= len(row):
                continue
            text = row[col_index].plain_text.strip()
            max_length = max(max_length, min(len(text), 42))
        weights[col_index] = max(1.0, min(float(max_length), 26.0))
        if col_index < len(header_labels):
            header_label = header_labels[col_index]
            preferred = preferred_weights.get(header_label)
            if preferred is not None:
                weights[col_index] = max(weights[col_index], preferred)
    total = sum(weights)
    return [available_width * weight / total for weight in weights]


def detect_numeric_columns(rows: list[list[TableCell]], *, excluded_headers: set[str] | None = None) -> set[int]:
    if len(rows) <= 1:
        return set()
    body_rows = rows[1:]
    col_count = max(len(row) for row in rows)
    excluded_headers = excluded_headers or set()
    numeric_columns: set[int] = set()
    for col_index in range(col_count):
        header_label = normalize_header_label(rows[0][col_index].plain_text) if col_index < len(rows[0]) else ""
        if header_label in excluded_headers:
            continue
        seen = 0
        numeric = 0
        for row in body_rows:
            if col_index >= len(row):
                continue
            text = row[col_index].plain_text.strip()
            if not text or text == "-":
                continue
            seen += 1
            if is_numeric_text(text):
                numeric += 1
        if seen and numeric / seen >= 0.7:
            numeric_columns.add(col_index)
    return numeric_columns


def cell_markup(cell: TableCell, is_numeric_column: bool, ctx: RenderContext) -> str:
    text = cell.plain_text.strip()
    if is_numeric_column and is_numeric_text(text):
        pretty = beautify_numeric_text(text)
        font_name = ctx.fonts.latin_bold if cell.is_header else ctx.fonts.latin_regular
        return f'<font face="{font_name}">{escape_text(pretty)}</font>'
    return cell.markup or "&nbsp;"


def merge_table_columns(rows: list[list[TableCell]], merge_headers: tuple[str, ...]) -> list[list[TableCell]]:
    if not rows or len(merge_headers) < 2:
        return rows
    header_labels = [normalize_header_label(cell.plain_text) for cell in rows[0]]
    try:
        start_index = header_labels.index(merge_headers[0])
    except ValueError:
        return rows
    expected_indexes = list(range(start_index, start_index + len(merge_headers)))
    if expected_indexes[-1] >= len(header_labels):
        return rows
    actual_headers = tuple(header_labels[index] for index in expected_indexes)
    if actual_headers != merge_headers:
        return rows

    merged_header_text = "/".join(merge_headers)
    merged_rows: list[list[TableCell]] = []
    for row_index, row in enumerate(rows):
        if len(row) <= expected_indexes[-1]:
            merged_rows.append(row)
            continue
        selected = [row[index] for index in expected_indexes]
        if row_index == 0:
            merged_plain = merged_header_text
            merged_markup = escape_text(merged_header_text)
        else:
            merged_plain = " ".join(cell.plain_text.strip() for cell in selected if cell.plain_text.strip())
            merged_markup = escape_text(merged_plain) if merged_plain else "&nbsp;"
        merged_cell = TableCell(
            markup=merged_markup,
            plain_text=merged_plain,
            is_header=selected[0].is_header,
        )
        merged_row = row[:start_index] + [merged_cell] + row[expected_indexes[-1] + 1 :]
        merged_rows.append(merged_row)
    return merged_rows


def apply_table_rules(rows: list[list[TableCell]], rules: PdfTableRuleSet) -> list[list[TableCell]]:
    transformed = rows
    for merge_headers in rules.merge_columns:
        transformed = merge_table_columns(transformed, merge_headers)
    return transformed


def truncate_text_to_lines(text: str, style: ParagraphStyle, width: float, max_lines: int) -> str:
    cleaned = text.strip()
    if not cleaned or max_lines <= 0:
        return cleaned
    line_height = float(style.leading or style.fontSize * 1.2)
    max_height = line_height * max_lines + 0.1
    if Paragraph(escape_text(cleaned), style).wrap(width, 10_000)[1] <= max_height:
        return cleaned

    low = 0
    high = len(cleaned)
    best = "..."
    while low <= high:
        mid = (low + high) // 2
        candidate_core = cleaned[:mid].rstrip("，。；、,.!?！？:： ")
        candidate = (candidate_core + "...") if candidate_core else "..."
        if Paragraph(escape_text(candidate), style).wrap(width, 10_000)[1] <= max_height:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def parse_table(tokens, idx: int, story: list, ctx: RenderContext, table_rules: PdfTableRuleSet) -> int:
    rows: list[list[TableCell]] = []
    current_row: list[TableCell] | None = None
    in_header = False
    while idx < len(tokens) and tokens[idx].type != "table_close":
        token = tokens[idx]
        if token.type == "thead_open":
            in_header = True
        elif token.type == "tbody_open":
            in_header = False
        elif token.type == "tr_open":
            current_row = []
        elif token.type in {"th_open", "td_open"} and current_row is not None:
            inline_token = tokens[idx + 1] if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline" else None
            current_row.append(
                TableCell(
                    markup=inline_to_markup(inline_token, ctx) if inline_token else "&nbsp;",
                    plain_text=inline_plain_text(inline_token) if inline_token else "",
                    is_header=in_header,
                )
            )
        elif token.type == "tr_close" and current_row is not None:
            rows.append(current_row)
            current_row = None
        idx += 1

    if rows:
        rows = apply_table_rules(rows, table_rules)
        numeric_columns = detect_numeric_columns(
            rows,
            excluded_headers=set(table_rules.no_thousands_columns),
        )
        widths = build_column_widths(rows, ctx.available_width)
        header_labels = [normalize_header_label(cell.plain_text) for cell in rows[0]] if rows else []
        data = []
        for row_index, row in enumerate(rows):
            normalized = row + [TableCell("&nbsp;", "", False)] * (len(widths) - len(row))
            paragraph_row = []
            for col_index, cell in enumerate(normalized):
                header_label = header_labels[col_index] if col_index < len(header_labels) else ""
                if cell.is_header:
                    style = ctx.styles["table_head"]
                else:
                    wrap_columns = set(table_rules.wrap_columns)
                    if header_label == "说明" or header_label in wrap_columns:
                        style = ctx.styles["table_body"]
                    else:
                        style = ctx.styles["table_body_nowrap"]
                rendered_markup = cell_markup(cell, col_index in numeric_columns, ctx)
                if row_index > 0 and col_index < len(header_labels):
                    max_lines = (table_rules.max_lines or {}).get(header_label)
                    if max_lines:
                        truncated = truncate_text_to_lines(cell.plain_text, style, widths[col_index], max_lines)
                        rendered_markup = escape_text(truncated) or "&nbsp;"
                paragraph_row.append(Paragraph(rendered_markup, style))
            data.append(paragraph_row)

        table = Table(data, colWidths=widths, repeatRows=1)
        style_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ctx.theme.header_bg)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(ctx.theme.row_bg)]),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor(ctx.theme.grid_color)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for col_index in numeric_columns:
            style_commands.append(("ALIGN", (col_index, 1), (col_index, -1), "RIGHT"))
            style_commands.append(("ALIGN", (col_index, 0), (col_index, 0), "RIGHT"))
        table.setStyle(TableStyle(style_commands))
        story.append(table)
        story.append(Spacer(1, 7))
    return idx + 1


def markdown_to_pdf_bytes(markdown_text: str, title: str, ctx: RenderContext, table_rules: PdfTableRuleSet | None = None) -> bytes:
    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(markdown_text)
    resolved_table_rules = table_rules or DEFAULT_PDF_TABLE_RULES
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
    )
    story: list = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.type == "heading_open":
            level = int(token.tag[1]) if token.tag.startswith("h") else 2
            inline_token = tokens[idx + 1] if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline" else None
            style = ctx.styles["title"] if level == 1 else ctx.styles["h1"] if level == 2 else ctx.styles["h2"] if level == 3 else ctx.styles["h3"]
            story.append(Paragraph(inline_to_markup(inline_token, ctx), style))
            idx += 3
            continue
        if token.type == "paragraph_open":
            markup, idx = extract_simple_inline(tokens, idx + 1, ctx)
            story.append(Paragraph(markup, ctx.styles["body"]))
            if idx < len(tokens) and tokens[idx].type == "paragraph_close":
                idx += 1
            continue
        if token.type == "bullet_list_open":
            idx = parse_list(tokens, idx + 1, story, ctx)
            story.append(Spacer(1, 3))
            continue
        if token.type == "table_open":
            idx = parse_table(tokens, idx + 1, story, ctx, resolved_table_rules)
            continue
        if token.type in {"fence", "code_block"}:
            story.append(Preformatted(token.content.rstrip("\n"), ctx.styles["code"]))
            idx += 1
            continue
        if token.type == "hr":
            story.append(Spacer(1, 8))
            idx += 1
            continue
        idx += 1
    doc.build(story)
    return buffer.getvalue()


def render_markdown_file(
    md_path: Path,
    pdf_path: Path,
    *,
    merge_columns: Iterable[str] | None = None,
    no_thousands_columns: Iterable[str] | None = None,
    wrap_columns: Iterable[str] | None = None,
    max_lines: Iterable[str] | None = None,
) -> None:
    ctx = build_context()
    markdown_text = md_path.read_text(encoding="utf-8")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    table_rules = build_pdf_table_rules(
        merge_columns=merge_columns,
        no_thousands_columns=no_thousands_columns,
        wrap_columns=wrap_columns,
        max_lines=max_lines,
    )
    pdf_path.write_bytes(markdown_to_pdf_bytes(markdown_text, md_path.stem, ctx, table_rules))
