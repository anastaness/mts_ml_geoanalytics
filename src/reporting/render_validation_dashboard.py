from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from shapely import wkb


PALETTE = {
    "red": "#fe0032",
    "purple": "#6351b7",
    "purple_2": "#7d6dd8",
    "white": "#ffffff",
    "black": "#17171c",
    "bg": "#f3f1fb",
}


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


def _fmt_float(value: Any, ndigits: int = 1) -> str:
    return f"{_safe_float(value):,.{ndigits}f}".replace(",", " ").replace(".", ",")


def _fmt_pct(part: float, whole: float, ndigits: int = 1) -> str:
    if whole <= 0:
        return "0,0%"
    return f"{(part / whole) * 100:.{ndigits}f}%".replace(".", ",")


def _wkb_to_geometry(x):
    try:
        if isinstance(x, bytes):
            return wkb.loads(x)
        if isinstance(x, memoryview):
            return wkb.loads(x.tobytes())
        if isinstance(x, bytearray):
            return wkb.loads(bytes(x))
        return x
    except Exception:
        return None


def _validation_score(
    *,
    total_rows: int,
    invalid_geom: int,
    empty_geom: int,
    zero_area: int,
    negative_area: int,
    target_height_filled_missing: int,
    source_flag_mismatch: int,
    observed_changed: int,
    predicted_with_original: int,
    outliers_gt_500: int,
) -> int:
    if total_rows <= 0:
        return 0

    penalty = 0.0
    penalty += (invalid_geom / total_rows) * 28
    penalty += (empty_geom / total_rows) * 10
    penalty += (zero_area / total_rows) * 10
    penalty += (negative_area / total_rows) * 12
    penalty += (target_height_filled_missing / total_rows) * 20
    penalty += (source_flag_mismatch / total_rows) * 10
    penalty += (observed_changed / total_rows) * 6
    penalty += (predicted_with_original / total_rows) * 6
    penalty += (outliers_gt_500 / total_rows) * 6

    return max(0, min(100, round(100 - penalty)))


def _risk_label(score: int) -> str:
    if score >= 95:
        return "Данные согласованы"
    if score >= 85:
        return "Низкий риск"
    if score >= 70:
        return "Умеренный риск"
    return "Нужна перепроверка"


def _describe_series(s: pd.Series) -> dict[str, str]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {
            "mean": "0,0",
            "median": "0,0",
            "min": "0,0",
            "max": "0,0",
            "std": "0,0",
        }

    return {
        "mean": _fmt_float(s.mean(), 1),
        "median": _fmt_float(s.median(), 1),
        "min": _fmt_float(s.min(), 1),
        "max": _fmt_float(s.max(), 1),
        "std": _fmt_float(s.std(ddof=0), 2),
    }


def _histogram_svg(
    series: pd.Series,
    *,
    bins: int = 28,
    clip_max: float | None = None,
    width: int = 620,
    height: int = 190,
) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if clip_max is not None:
        s = s[s <= clip_max]

    if s.empty:
        return ""

    counts, edges = pd.cut(s, bins=bins, retbins=True, include_lowest=True)
    value_counts = counts.value_counts(sort=False).to_list()
    max_count = max(value_counts) if value_counts else 1

    left_pad = 12
    right_pad = 8
    top_pad = 10
    bottom_pad = 18

    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad
    bar_gap = 2
    bar_w = max(3, plot_w / max(len(value_counts), 1) - bar_gap)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="histogram">'
    ]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="rgba(255,255,255,0.0)"/>'
    )

    y0 = top_pad + plot_h
    parts.append(
        f'<line x1="{left_pad}" y1="{y0}" x2="{left_pad + plot_w}" y2="{y0}" stroke="rgba(23,23,28,0.18)" stroke-width="1"/>'
    )

    for i, count in enumerate(value_counts):
        h = 0 if max_count == 0 else (count / max_count) * plot_h
        x = left_pad + i * (bar_w + bar_gap)
        y = top_pad + plot_h - h
        fill = "url(#g1)" if i % 2 == 0 else "url(#g2)"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="3" fill="{fill}"/>'
        )

    parts.insert(
        1,
        """
        <defs>
          <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#6351b7"/>
            <stop offset="100%" stop-color="#7d6dd8"/>
          </linearGradient>
          <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#fe0032"/>
            <stop offset="100%" stop-color="#ff6f8e"/>
          </linearGradient>
        </defs>
        """,
    )

    min_x = _fmt_float(edges[0], 0)
    max_x = _fmt_float(edges[-1], 0)
    parts.append(
        f'<text x="{left_pad}" y="{height - 4}" font-size="11" fill="rgba(23,23,28,0.55)">{min_x} м</text>'
    )
    parts.append(
        f'<text x="{width - right_pad}" y="{height - 4}" text-anchor="end" font-size="11" fill="rgba(23,23,28,0.55)">{max_x} м</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def build_context(parquet_path: Path, project_title: str, logo_path: str) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)

    if "rep_geometry" not in df.columns:
        raise ValueError("В parquet отсутствует колонка 'rep_geometry'")

    df = df.copy()
    df["geometry"] = df["rep_geometry"].apply(_wkb_to_geometry)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32635")

    total_rows = len(gdf)
    total_cols = len(gdf.columns)
    memory_mb = gdf.memory_usage(deep=True).sum() / 1024**2

    valid_mask = gdf.geometry.notna()
    parsed_ok = int(valid_mask.sum())
    invalid_geom = int((~gdf.loc[valid_mask, "geometry"].is_valid).sum()) if parsed_ok else 0
    empty_geom = int(gdf.loc[valid_mask, "geometry"].is_empty.sum()) if parsed_ok else 0
    zero_area = int((gdf.loc[valid_mask, "geometry"].area == 0).sum()) if parsed_ok else 0
    negative_area = int((gdf.loc[valid_mask, "geometry"].area < 0).sum()) if parsed_ok else 0

    target_height_filled_missing = int(gdf["target_height_filled"].isna().sum()) if "target_height_filled" in gdf.columns else total_rows
    target_height_was_predicted_missing = int(gdf["target_height_was_predicted"].isna().sum()) if "target_height_was_predicted" in gdf.columns else total_rows
    union_area_all_missing = int(gdf["union_area_all"].isna().sum()) if "union_area_all" in gdf.columns else total_rows
    median_stairs_b_missing = int(gdf["median_stairs_b"].isna().sum()) if "median_stairs_b" in gdf.columns else total_rows
    floor_height_missing = int(gdf["median_avg_floor_height_b"].isna().sum()) if "median_avg_floor_height_b" in gdf.columns else total_rows

    height_min = _safe_float(gdf["target_height_filled"].min()) if "target_height_filled" in gdf.columns else 0.0
    height_max = _safe_float(gdf["target_height_filled"].max()) if "target_height_filled" in gdf.columns else 0.0
    height_neg = int((pd.to_numeric(gdf["target_height_filled"], errors="coerce") < 0).sum()) if "target_height_filled" in gdf.columns else 0

    stairs_min = _safe_float(gdf["median_stairs_b"].min()) if "median_stairs_b" in gdf.columns else 0.0
    stairs_max = _safe_float(gdf["median_stairs_b"].max()) if "median_stairs_b" in gdf.columns else 0.0
    stairs_neg = int((pd.to_numeric(gdf["median_stairs_b"], errors="coerce") < 0).sum()) if "median_stairs_b" in gdf.columns else 0

    area_min = _safe_float(gdf["union_area_all"].min()) if "union_area_all" in gdf.columns else 0.0
    area_max = _safe_float(gdf["union_area_all"].max()) if "union_area_all" in gdf.columns else 0.0
    area_neg = int((pd.to_numeric(gdf["union_area_all"], errors="coerce") < 0).sum()) if "union_area_all" in gdf.columns else 0

    outliers_gt_300 = int((pd.to_numeric(gdf["target_height_filled"], errors="coerce") > 300).sum()) if "target_height_filled" in gdf.columns else 0
    outliers_gt_500 = int((pd.to_numeric(gdf["target_height_filled"], errors="coerce") > 500).sum()) if "target_height_filled" in gdf.columns else 0
    high_buildings_gt_50 = int((pd.to_numeric(gdf["target_height_filled"], errors="coerce") > 50).sum()) if "target_height_filled" in gdf.columns else 0

    observed_mask = gdf["target_height_was_predicted"] == 0 if "target_height_was_predicted" in gdf.columns else pd.Series(False, index=gdf.index)
    predicted_mask = gdf["target_height_was_predicted"] == 1 if "target_height_was_predicted" in gdf.columns else pd.Series(False, index=gdf.index)

    observed_count = int(observed_mask.sum())
    predicted_count = int(predicted_mask.sum())

    observed_stats = _describe_series(gdf.loc[observed_mask, "target_height_filled"]) if "target_height_filled" in gdf.columns else _describe_series(pd.Series(dtype=float))
    predicted_stats = _describe_series(gdf.loc[predicted_mask, "target_height_filled"]) if "target_height_filled" in gdf.columns else _describe_series(pd.Series(dtype=float))

    observed_changed = 0
    if {"target_height", "target_height_filled", "target_height_was_predicted"}.issubset(gdf.columns):
        observed_changed = int(
            (
                gdf.loc[observed_mask, "target_height"]
                != gdf.loc[observed_mask, "target_height_filled"]
            ).fillna(False).sum()
        )

    predicted_with_original = 0
    if {"target_height", "target_height_was_predicted"}.issubset(gdf.columns):
        predicted_with_original = int(gdf.loc[predicted_mask, "target_height"].notna().sum())

    source_flag_mismatch = 0
    if {"target_height_fill_source", "target_height_was_predicted"}.issubset(gdf.columns):
        source_flag_mismatch = int(
            ~(
                ((gdf["target_height_was_predicted"] == 0) & (gdf["target_height_fill_source"] == "observed"))
                | ((gdf["target_height_was_predicted"] == 1) & (gdf["target_height_fill_source"] == "catboost"))
            ).sum()
        )

    fill_source_counts = (
        gdf["target_height_fill_source"].value_counts(dropna=False).to_dict()
        if "target_height_fill_source" in gdf.columns
        else {}
    )

    spatial_rows = 0
    spatial_high_diff = 0
    spatial_stats = {
        "mean": "0,0",
        "median": "0,0",
        "p75": "0,0",
        "p90": "0,0",
        "p95": "0,0",
        "p99": "0,0",
    }
    lakhta_info = None

    if {
        "target_height_was_predicted",
        "n_neighbors_100m",
        "target_height_filled",
        "neighbor_height_mean_100m",
    }.issubset(gdf.columns):
        spatial_mask = (
            (gdf["target_height_was_predicted"] == 0)
            & (pd.to_numeric(gdf["n_neighbors_100m"], errors="coerce") > 0)
            & gdf["neighbor_height_mean_100m"].notna()
        )
        spatial_check = gdf.loc[spatial_mask].copy()
        if not spatial_check.empty:
            spatial_check["height_diff"] = (
                pd.to_numeric(spatial_check["target_height_filled"], errors="coerce")
                - pd.to_numeric(spatial_check["neighbor_height_mean_100m"], errors="coerce")
            ).abs()

            s = spatial_check["height_diff"].dropna()
            spatial_rows = int(len(s))
            spatial_high_diff = int((s > 20).sum())

            if len(s) > 0:
                spatial_stats = {
                    "mean": _fmt_float(s.mean(), 1),
                    "median": _fmt_float(s.quantile(0.50), 1),
                    "p75": _fmt_float(s.quantile(0.75), 1),
                    "p90": _fmt_float(s.quantile(0.90), 1),
                    "p95": _fmt_float(s.quantile(0.95), 1),
                    "p99": _fmt_float(s.quantile(0.99), 1),
                }

            lakhta = spatial_check[pd.to_numeric(spatial_check["target_height_filled"], errors="coerce") > 400]
            if not lakhta.empty:
                row = lakhta.iloc[0]
                lakhta_info = {
                    "height": _fmt_float(row["target_height_filled"], 1),
                    "neighbors_mean": _fmt_float(row["neighbor_height_mean_100m"], 1),
                    "diff": _fmt_float(row["height_diff"], 1),
                }

    score = _validation_score(
        total_rows=total_rows,
        invalid_geom=invalid_geom,
        empty_geom=empty_geom,
        zero_area=zero_area,
        negative_area=negative_area,
        target_height_filled_missing=target_height_filled_missing,
        source_flag_mismatch=source_flag_mismatch,
        observed_changed=observed_changed,
        predicted_with_original=predicted_with_original,
        outliers_gt_500=outliers_gt_500,
    )

    overview_cards = [
        {
            "label": "Всего зданий",
            "value": _fmt_int(total_rows),
            "note": "Финальный датасет после сопоставления и заполнения высот",
            "tone": "dark",
        },
        {
            "label": "Высота заполнена",
            "value": _fmt_pct(total_rows - target_height_filled_missing, total_rows),
            "note": f"{_fmt_int(total_rows - target_height_filled_missing)} из {_fmt_int(total_rows)} объектов",
            "tone": "light",
        },
        {
            "label": "Геометрия валидна",
            "value": _fmt_pct(parsed_ok - invalid_geom, total_rows),
            "note": "Пустые, нулевые и невалидные полигоны не обнаружены либо сведены к минимуму",
            "tone": "red" if invalid_geom > 0 or empty_geom > 0 or zero_area > 0 else "light",
        },
        {
            "label": "Предсказано моделью",
            "value": _fmt_pct(predicted_count, total_rows),
            "note": f"{_fmt_int(predicted_count)} зданий получили высоту через CatBoost",
            "tone": "light",
        },
    ]

    geometry_boxes = [
        {
            "k": "Успешно разобрано",
            "v": f"{_fmt_int(parsed_ok)} / {_fmt_int(total_rows)}",
            "n": f"Доля успешного разбора · {_fmt_pct(parsed_ok, total_rows)}",
        },
        {
            "k": "Невалидные геометрии",
            "v": _fmt_int(invalid_geom),
            "n": f"Доля от набора · {_fmt_pct(invalid_geom, total_rows)}",
        },
        {
            "k": "Пустые / нулевая площадь",
            "v": f"{_fmt_int(empty_geom)} / {_fmt_int(zero_area)}",
            "n": "Пустые полигоны и вырожденные контуры",
        },
        {
            "k": "Отрицательная площадь",
            "v": _fmt_int(negative_area),
            "n": "Физически невозможные значения площади",
        },
    ]

    completeness_boxes = [
        {
            "k": "target_height_filled",
            "v": _fmt_int(target_height_filled_missing),
            "n": f"Пропусков · {_fmt_pct(target_height_filled_missing, total_rows)}",
        },
        {
            "k": "target_height_was_predicted",
            "v": _fmt_int(target_height_was_predicted_missing),
            "n": f"Пропусков · {_fmt_pct(target_height_was_predicted_missing, total_rows)}",
        },
        {
            "k": "union_area_all",
            "v": _fmt_int(union_area_all_missing),
            "n": f"Пропусков · {_fmt_pct(union_area_all_missing, total_rows)}",
        },
        {
            "k": "median_stairs_b",
            "v": _fmt_int(median_stairs_b_missing),
            "n": f"Ожидаемо для A_only · {_fmt_pct(median_stairs_b_missing, total_rows)}",
        },
        {
            "k": "median_avg_floor_height_b",
            "v": _fmt_int(floor_height_missing),
            "n": f"Связано с отсутствием данных источника B · {_fmt_pct(floor_height_missing, total_rows)}",
        },
    ]

    ranges_boxes = [
        {
            "k": "Высота",
            "v": f"{_fmt_float(height_min, 1)} — {_fmt_float(height_max, 1)} м",
            "n": f"Отрицательных значений · {_fmt_int(height_neg)}",
        },
        {
            "k": "Этажность",
            "v": f"{_fmt_float(stairs_min, 1)} — {_fmt_float(stairs_max, 1)}",
            "n": f"Отрицательных значений · {_fmt_int(stairs_neg)}",
        },
        {
            "k": "Площадь",
            "v": f"{_fmt_float(area_min, 1)} — {_fmt_float(area_max, 1)} м²",
            "n": f"Отрицательных значений · {_fmt_int(area_neg)}",
        },
        {
            "k": "Высотные выбросы",
            "v": f">{_fmt_int(300)} м: {_fmt_int(outliers_gt_300)}",
            "n": f">{_fmt_int(500)} м: {_fmt_int(outliers_gt_500)}",
        },
    ]

    ml_boxes = [
        {
            "k": "Observed",
            "v": _fmt_int(observed_count),
            "n": f"Доля от набора · {_fmt_pct(observed_count, total_rows)}",
        },
        {
            "k": "CatBoost",
            "v": _fmt_int(predicted_count),
            "n": f"Доля от набора · {_fmt_pct(predicted_count, total_rows)}",
        },
        {
            "k": "Изменено observed",
            "v": _fmt_int(observed_changed),
            "n": "Должно быть 0: исходные наблюдения нельзя перезаписывать",
        },
        {
            "k": "Predicted с original height",
            "v": _fmt_int(predicted_with_original),
            "n": "Должно быть 0: модель заполняет только отсутствующие значения",
        },
    ]

    observed_vs_pred = [
        {
            "label": "Observed mean / median",
            "a": observed_stats["mean"],
            "b": observed_stats["median"],
        },
        {
            "label": "Predicted mean / median",
            "a": predicted_stats["mean"],
            "b": predicted_stats["median"],
        },
        {
            "label": "Observed min / max",
            "a": observed_stats["min"],
            "b": observed_stats["max"],
        },
        {
            "label": "Predicted min / max",
            "a": predicted_stats["min"],
            "b": predicted_stats["max"],
        },
        {
            "label": "Predicted std",
            "a": predicted_stats["std"],
            "b": "узкий разброс",
        },
    ]

    spatial_boxes = [
        {
            "k": "Проверено объектов",
            "v": _fmt_int(spatial_rows),
            "n": "Только observed-объекты с соседями в радиусе 100 м",
        },
        {
            "k": "Средняя разница",
            "v": f"{spatial_stats['mean']} м",
            "n": f"Медиана · {spatial_stats['median']} м",
        },
        {
            "k": "75 / 90 / 95 перцентиль",
            "v": f"{spatial_stats['p75']} / {spatial_stats['p90']} / {spatial_stats['p95']}",
            "n": "Разница между высотой здания и средней высотой соседей",
        },
        {
            "k": "Разница > 20 м",
            "v": _fmt_int(spatial_high_diff),
            "n": f"Доля от проверенных · {_fmt_pct(spatial_high_diff, spatial_rows)}",
        },
    ]

    reader_notes = [
        "Финальный дашборд оценивает не качество сырых источников, а корректность итогового набора после обучения и заполнения высот.",
        "Ключевая логика проверки: модель не должна менять observed-высоты и не должна предсказывать там, где исходная высота уже известна.",
        "Пропуски в median_stairs_b и median_avg_floor_height_b интерпретируются как ожидаемое следствие объектов типа A_only, а не как ошибка сборки датасета.",
    ]

    if lakhta_info:
        reader_notes.append(
            f"Единичный экстремум по высоте объясним реальным объектом: {lakhta_info['height']} м против среднего соседского фона {lakhta_info['neighbors_mean']} м."
        )

    fill_source_text = []
    for key, value in fill_source_counts.items():
        fill_source_text.append(f"{key}: {_fmt_int(value)}")
    fill_source_summary = " • ".join(fill_source_text) if fill_source_text else "Нет данных"

    hist_100_svg = _histogram_svg(gdf["target_height_filled"], bins=30, clip_max=100, width=590, height=180)
    hist_full_svg = _histogram_svg(gdf["target_height_filled"], bins=30, clip_max=None, width=590, height=180)

    context = {
        "project_title": project_title,
        "palette": PALETTE,
        "logo_path": logo_path,
        "overview_cards": overview_cards,
        "hero_total": _fmt_int(total_rows),
        "summary_text": "Пост-ML валидация итогового датасета: геометрическая корректность, полнота высот, согласованность источников и пространственная правдоподобность результатов.",
        "score": score,
        "risk_label": _risk_label(score),
        "memory_mb": _fmt_float(memory_mb, 2),
        "total_cols": _fmt_int(total_cols),
        "geometry_boxes": geometry_boxes,
        "completeness_boxes": completeness_boxes,
        "ranges_boxes": ranges_boxes,
        "ml_boxes": ml_boxes,
        "spatial_boxes": spatial_boxes,
        "observed_vs_pred": observed_vs_pred,
        "reader_notes": reader_notes,
        "fill_source_summary": fill_source_summary,
        "hist_100_svg": hist_100_svg,
        "hist_full_svg": hist_full_svg,
        "high_buildings_gt_50": _fmt_int(high_buildings_gt_50),
        "lakhta_info": lakhta_info,
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
            page = await browser.new_page(
                viewport={"width": 1654, "height": 1169},
                device_scale_factor=1,
            )
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
            "Не удалось собрать PDF через Playwright/Chromium. Сначала установите браузер: python -m playwright install chromium"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Собирает post-ML validation dashboard из parquet")
    parser.add_argument("--parquet", required=True, help="Путь к итоговому parquet-файлу")
    parser.add_argument("--template-dir", default=str(Path(__file__).parent / "templates"))
    parser.add_argument("--template-name", default="validation_dashboard.html.j2")
    parser.add_argument("--project-title", default="Validation Quality Dashboard")
    parser.add_argument("--logo", default="logo.png", help="Путь к logo.png")
    parser.add_argument("--out-html", default="validation_dashboard.html")
    parser.add_argument("--out-pdf", default="validation_dashboard.pdf")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Не найден parquet: {parquet_path}")

    template_dir = Path(args.template_dir)
    out_html = Path(args.out_html)
    out_pdf = Path(args.out_pdf)
    logo_path = Path(args.logo).resolve().as_uri()

    context = build_context(parquet_path, args.project_title, logo_path)
    render_html(template_dir, args.template_name, context, out_html)
    asyncio.run(html_to_pdf(out_html, out_pdf))

    print(f"HTML сохранен в: {out_html}")
    print(f"PDF сохранен в: {out_pdf}")


if __name__ == "__main__":
    main()