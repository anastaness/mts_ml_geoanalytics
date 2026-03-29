from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


PALETTE = {
    "red": "#fe0032",
    "purple": "#6351b7",
    "purple_2": "#7d6dd8",
    "white": "#ffffff",
    "black": "#17171c",
    "bg": "#f3f1fb",
}


def read_excel_book(xlsx_path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(xlsx_path, sheet_name=None)


def _sheet(book: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    return book.get(name, pd.DataFrame()).copy()


def _metrics_from_two_cols(
    df: pd.DataFrame,
    key_col: str = "metric",
    value_col: str = "value",
) -> dict[str, Any]:
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return {}
    out: dict[str, Any] = {}
    for _, row in df.iterrows():
        out[str(row[key_col])] = row[value_col]
    return out


def _sum_column(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _safe_int(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        return int(value)
    except Exception:
        return int(float(value))


def _safe_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _fmt_int(value: Any) -> str:
    return f"{_safe_int(value):,}".replace(",", " ")


def _fmt_pct(part: float, whole: float, ndigits: int = 1) -> str:
    if whole <= 0:
        return "0,0%"
    return f"{(part / whole) * 100:.{ndigits}f}%".replace(".", ",")


def _count_rows_with_missing(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df.isna().any(axis=1).sum())


def _share_rows_with_missing(df: pd.DataFrame) -> float:
    if df.empty or len(df) == 0:
        return 0.0
    return float(df.isna().any(axis=1).mean())


def _quality_score(
    total_rows: int,
    rows_with_missing: int,
    invalid_geom: int,
    null_geom: int,
    empty_geom: int,
    signature_dups: int,
    invalid_holes: int,
) -> int:
    if total_rows <= 0:
        return 0

    penalty = 0.0
    penalty += (rows_with_missing / total_rows) * 32
    penalty += (invalid_geom / total_rows) * 32
    penalty += (null_geom / total_rows) * 14
    penalty += (empty_geom / total_rows) * 10
    penalty += (signature_dups / total_rows) * 8
    penalty += (invalid_holes / total_rows) * 4

    return max(0, min(100, round(100 - penalty)))


def _risk_label(score: int) -> str:
    if score >= 85:
        return "Низкий риск"
    if score >= 70:
        return "Умеренный риск"
    if score >= 50:
        return "Повышенный риск"
    return "Высокий риск"


def _build_dataset(
    *,
    name: str,
    df: pd.DataFrame,
    rows: int,
    cols: int,
    invalid_geom: int,
    null_geom: int,
    empty_geom: int,
    parse_success_rate: float,
    signature_dups: int,
    invalid_holes: int,
    numeric_flags: int,
    area_mismatch_count: int = 0,
) -> dict[str, Any]:
    rows_with_missing = _count_rows_with_missing(df)
    rows_with_missing_share = _share_rows_with_missing(df)
    total_missing_cells = _sum_column(
        pd.DataFrame({"na_count": df.isna().sum() if not df.empty else []}),
        "na_count",
    )

    score = _quality_score(
        total_rows=rows,
        rows_with_missing=rows_with_missing,
        invalid_geom=invalid_geom,
        null_geom=null_geom,
        empty_geom=empty_geom,
        signature_dups=signature_dups,
        invalid_holes=invalid_holes,
    )

    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "rows_fmt": _fmt_int(rows),
        "cols_fmt": _fmt_int(cols),
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_fmt": _fmt_int(rows_with_missing),
        "rows_with_missing_pct": _fmt_pct(rows_with_missing, rows),
        "rows_with_missing_share": rows_with_missing_share,
        "total_missing_cells": total_missing_cells,
        "total_missing_cells_fmt": _fmt_int(total_missing_cells),
        "invalid_geom": invalid_geom,
        "invalid_geom_fmt": _fmt_int(invalid_geom),
        "invalid_geom_pct": _fmt_pct(invalid_geom, rows),
        "null_geom": null_geom,
        "null_geom_fmt": _fmt_int(null_geom),
        "null_geom_pct": _fmt_pct(null_geom, rows),
        "empty_geom": empty_geom,
        "empty_geom_fmt": _fmt_int(empty_geom),
        "empty_geom_pct": _fmt_pct(empty_geom, rows),
        "parse_success_rate": parse_success_rate,
        "parse_success_pct_text": f"{parse_success_rate * 100:.1f}%".replace(".", ","),
        "signature_dups": signature_dups,
        "signature_dups_fmt": _fmt_int(signature_dups),
        "signature_dups_pct": _fmt_pct(signature_dups, rows),
        "invalid_holes": invalid_holes,
        "invalid_holes_fmt": _fmt_int(invalid_holes),
        "invalid_holes_pct": _fmt_pct(invalid_holes, rows),
        "numeric_flags": numeric_flags,
        "numeric_flags_fmt": _fmt_int(numeric_flags),
        "area_mismatch_count": area_mismatch_count,
        "area_mismatch_fmt": _fmt_int(area_mismatch_count),
        "score": score,
        "risk_label": _risk_label(score),
    }


def build_context(book: dict[str, pd.DataFrame], project_title: str, logo_path: str) -> dict[str, Any]:
    shapes = _metrics_from_two_cols(_sheet(book, "ovw_dataset_shapes"))
    source_a = _sheet(book, "ovw_source_A_head")
    source_b = _sheet(book, "ovw_source_B_head")
    geom = _metrics_from_two_cols(_sheet(book, "geo_basic_quality"))
    parse_metrics = _metrics_from_two_cols(_sheet(book, "qlt_wkt_parse"))
    basic_numeric = _metrics_from_two_cols(_sheet(book, "qlt_basic_numeric"))
    dup_sig_a = _metrics_from_two_cols(_sheet(book, "dup_sig_A_summary"))
    dup_sig_b = _metrics_from_two_cols(_sheet(book, "dup_sig_B_summary"))
    holes_a = _metrics_from_two_cols(_sheet(book, "holes_A_summary"))
    holes_b = _metrics_from_two_cols(_sheet(book, "holes_B_summary"))
    area_thresholds = _metrics_from_two_cols(_sheet(book, "area_diff_thresholds"))

    rows_a = _safe_int(shapes.get("source_A_rows", 0))
    cols_a = _safe_int(shapes.get("source_A_cols", 0))
    rows_b = _safe_int(shapes.get("source_B_rows", 0))
    cols_b = _safe_int(shapes.get("source_B_cols", 0))

    invalid_a = _safe_int(geom.get("A_invalid_geometry", 0))
    invalid_b = _safe_int(geom.get("B_invalid_geometry", 0))
    null_a = _safe_int(geom.get("A_geometry_null", 0))
    null_b = _safe_int(geom.get("B_geometry_null", 0))
    empty_a = _safe_int(geom.get("A_empty_geometry", 0))
    empty_b = _safe_int(geom.get("B_empty_geometry", 0))

    parse_success_a = _safe_float(parse_metrics.get("A_parse_success_rate", 0.0))
    parse_success_b = _safe_float(parse_metrics.get("B_parse_success_rate", 0.0))

    sig_dup_a = _safe_int(dup_sig_a.get("candidate_duplicate_geometries", 0))
    sig_dup_b = _safe_int(dup_sig_b.get("candidate_duplicate_geometries", 0))

    invalid_holes_a = _safe_int(holes_a.get("invalid_holes", 0))
    invalid_holes_b = _safe_int(holes_b.get("invalid_holes", 0))

    numeric_flags_a = (
        _safe_int(basic_numeric.get("A_area_le_0", 0))
        + _safe_int(basic_numeric.get("A_floor_min_gt_floor_max", 0))
    )
    numeric_flags_b = (
        _safe_int(basic_numeric.get("B_stairs_le_0", 0))
        + _safe_int(basic_numeric.get("B_stairs_gt_100", 0))
        + _safe_int(basic_numeric.get("B_avg_floor_height_le_0", 0))
        + _safe_int(basic_numeric.get("B_avg_floor_height_gt_10", 0))
        + _safe_int(basic_numeric.get("B_height_le_0", 0))
        + _safe_int(basic_numeric.get("B_height_gt_300", 0))
    )

    area_mismatch_a = _safe_int(area_thresholds.get("area_diff_pct_gt_10pct", 0))

    dataset_a = _build_dataset(
        name="Источник A",
        df=source_a,
        rows=rows_a,
        cols=cols_a,
        invalid_geom=invalid_a,
        null_geom=null_a,
        empty_geom=empty_a,
        parse_success_rate=parse_success_a,
        signature_dups=sig_dup_a,
        invalid_holes=invalid_holes_a,
        numeric_flags=numeric_flags_a,
        area_mismatch_count=area_mismatch_a,
    )
    dataset_b = _build_dataset(
        name="Источник B",
        df=source_b,
        rows=rows_b,
        cols=cols_b,
        invalid_geom=invalid_b,
        null_geom=null_b,
        empty_geom=empty_b,
        parse_success_rate=parse_success_b,
        signature_dups=sig_dup_b,
        invalid_holes=invalid_holes_b,
        numeric_flags=numeric_flags_b,
    )

    total_rows = rows_a + rows_b
    total_rows_with_missing = dataset_a["rows_with_missing"] + dataset_b["rows_with_missing"]
    total_invalid = invalid_a + invalid_b
    total_sig_dups = sig_dup_a + sig_dup_b

    overview_cards = [
        {
            "label": "Всего объектов",
            "value": _fmt_int(total_rows),
            "note": "Общий объем двух источников",
            "tone": "dark",
        },
        {
            "label": "Строки с пропусками",
            "value": _fmt_int(total_rows_with_missing),
            "note": "Количество объектов, где есть хотя бы один пропуск",
            "tone": "light",
        },
        {
            "label": "Некорректные геометрии",
            "value": _fmt_int(total_invalid),
            "note": "Объекты, требующие геометрической очистки",
            "tone": "red",
        },
        {
            "label": "Подозрительные дубликаты",
            "value": _fmt_int(total_sig_dups),
            "note": "Кандидаты в дубликаты по геометрической сигнатуре",
            "tone": "light",
        },
    ]

    compare_rows = [
        {
            "label": "Объем данных",
            "a": rows_a,
            "b": rows_b,
            "a_fmt": _fmt_int(rows_a),
            "b_fmt": _fmt_int(rows_b),
            "max": max(rows_a, rows_b, 1),
        },
        {
            "label": "Строки с пропусками",
            "a": dataset_a["rows_with_missing"],
            "b": dataset_b["rows_with_missing"],
            "a_fmt": dataset_a["rows_with_missing_fmt"],
            "b_fmt": dataset_b["rows_with_missing_fmt"],
            "max": max(dataset_a["rows_with_missing"], dataset_b["rows_with_missing"], 1),
        },
        {
            "label": "Некорректные геометрии",
            "a": invalid_a,
            "b": invalid_b,
            "a_fmt": _fmt_int(invalid_a),
            "b_fmt": _fmt_int(invalid_b),
            "max": max(invalid_a, invalid_b, 1),
        },
        {
            "label": "Подозрительные дубликаты",
            "a": sig_dup_a,
            "b": sig_dup_b,
            "a_fmt": _fmt_int(sig_dup_a),
            "b_fmt": _fmt_int(sig_dup_b),
            "max": max(sig_dup_a, sig_dup_b, 1),
        },
    ]

    reader_notes = [
        "Главный приоритет — геометрическая корректность, потому что именно она определяет надежность пространственного сопоставления объектов.",
        "Пропуски оценены по числу строк, затронутых неполнотой, а не по сумме всех пустых ячеек.",
        "Подозрительные дубликаты и ошибки в контурах полезно разбирать до объединения источников, чтобы не переносить шум в итоговый датасет.",
    ]

    context = {
        "project_title": project_title,
        "palette": PALETTE,
        "logo_path": logo_path,
        "overview_cards": overview_cards,
        "datasets": [dataset_a, dataset_b],
        "compare_rows": compare_rows,
        "reader_notes": reader_notes,
        "summary_text": "Краткий обзор качества двух источников перед дальнейшей очисткой и объединением данных.",
        "hero_total": _fmt_int(total_rows),
    }
    return context


def render_html(template_dir: Path, template_name: str, context: dict[str, Any], output_html: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_name)
    html = template.render(**context)
    output_html.write_text(html, encoding="utf-8")


async def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1654, "height": 1169}, device_scale_factor=1)
            await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            await page.pdf(
                path=str(pdf_path),
                format="A3",
                landscape=True,
                print_background=True,
                margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
                prefer_css_page_size=True,
            )
            await browser.close()
    except Exception as e:
        raise RuntimeError(
            "Не удалось собрать PDF через Playwright/Chromium. Сначала установите браузер командой: python -m playwright install chromium"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Собирает PDF-дашборд из EDA xlsx")
    parser.add_argument("--xlsx", required=True, help="Путь к Excel-файлу с результатами EDA")
    parser.add_argument("--template-dir", default=str(Path(__file__).parent / "templates"))
    parser.add_argument("--template-name", default="eda_dashboard.html.j2")
    parser.add_argument("--project-title", default="EDA Quality Dashboard")
    parser.add_argument("--logo", default="logo.png", help="Путь к logo.png")
    parser.add_argument("--out-html", default="eda_dashboard.html")
    parser.add_argument("--out-pdf", default="eda_dashboard.pdf")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Не найден xlsx: {xlsx_path}")

    template_dir = Path(args.template_dir)
    out_html = Path(args.out_html)
    out_pdf = Path(args.out_pdf)
    logo_path = Path(args.logo).resolve().as_uri()

    book = read_excel_book(xlsx_path)
    context = build_context(book, args.project_title, logo_path)
    render_html(template_dir, args.template_name, context, out_html)
    asyncio.run(html_to_pdf(out_html, out_pdf))

    print(f"HTML сохранен в: {out_html}")
    print(f"PDF сохранен в: {out_pdf}")


if __name__ == "__main__":
    main()
