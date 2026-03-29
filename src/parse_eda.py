from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import wkt

warnings.filterwarnings("ignore", category=UserWarning)


OUTPUT_XLSX = Path("../data/interim/eda_outputs_refactored.xlsx")
RANDOM_STATE = 42

SAMPLE_WKT_SIZE = 1000
SAMPLE_NEIGHBORS_SIZE = 3000
SAMPLE_NEARDUP_SIZE = 2000

CRS_GEOGRAPHIC = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

def safe_load_wkt(x):
    try:
        return wkt.loads(x) if pd.notna(x) else None
    except Exception:
        return None


def parse_wkt_series(series: pd.Series) -> gpd.GeoSeries:
    try:
        return gpd.GeoSeries.from_wkt(series, crs=CRS_GEOGRAPHIC)
    except Exception:
        return gpd.GeoSeries(series.apply(safe_load_wkt), crs=CRS_GEOGRAPHIC)


def normalize_series_to_df(series: pd.Series, index_name="index", value_name="value") -> pd.DataFrame:
    out = series.rename(value_name).reset_index()
    out.columns = [index_name, value_name]
    return out


def metrics_dict_to_df(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])


def describe_df(df: pd.DataFrame, percentiles=None) -> pd.DataFrame:
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.5, 0.95, 0.99]

    if df.empty or df.shape[1] == 0:
        return pd.DataFrame({"feature": [], "count": [], "mean": [], "std": []})

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return pd.DataFrame({"feature": [], "count": [], "mean": [], "std": []})

    out = (
        numeric_df.describe(percentiles=percentiles)
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return out


def safe_top_value_counts(series: pd.Series, top_n=20, index_name="value", value_name="count") -> pd.DataFrame:
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=[index_name, value_name])
    return normalize_series_to_df(series.value_counts(dropna=False).head(top_n), index_name, value_name)


def safe_select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.DataFrame()
    return df[existing].copy()


def df_or_placeholder(df: pd.DataFrame, message="Нет данных для вывода") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"message": [message]})
    return df


def sanitize_sheet_name(name: str, used_names: set[str]) -> str:
    bad = ['\\', '/', '*', '?', ':', '[', ']']
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip()
    if len(name) > 31:
        name = name[:31]

    base = name
    i = 1
    while name in used_names:
        suffix = f"_{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1

    used_names.add(name)
    return name


def geometry_hash(geom) -> str | None:
    if geom is None:
        return None
    try:
        return hashlib.md5(geom.wkb).hexdigest()
    except Exception:
        return None


def add_sheet(result_tables, section, sheet_name, description, df):
    result_tables.append({
        "section": section,
        "sheet_name": sheet_name,
        "description": description,
        "df": df_or_placeholder(df),
    })


def count_intersecting_neighbors(geom, full_gdf, sindex):
    if geom is None or geom.is_empty:
        return np.nan
    candidate_idx = list(sindex.intersection(geom.bounds))
    if not candidate_idx:
        return 0
    candidates = full_gdf.iloc[candidate_idx]
    return int(candidates.intersects(geom).sum() - 1)


def build_geom_signature(gdf, area_decimals=2, bounds_decimals=2):
    bounds = gdf.geometry.bounds.round(bounds_decimals)
    area = gdf.geometry.area.round(area_decimals)
    geom_type = gdf.geometry.geom_type.astype(str)

    signature = (
        geom_type + "|" +
        area.astype(str) + "|" +
        bounds["minx"].astype(str) + "|" +
        bounds["miny"].astype(str) + "|" +
        bounds["maxx"].astype(str) + "|" +
        bounds["maxy"].astype(str)
    )
    return signature


def analyze_geometry_duplicates(gdf: gpd.GeoDataFrame, dataset_name: str):
    geom_hash_series = gdf.geometry.apply(geometry_hash)
    dup_mask = geom_hash_series.duplicated(keep=False)

    total = len(gdf)
    dup_count = int(dup_mask.sum())
    unique_dup_groups = int(geom_hash_series[dup_mask].nunique())
    group_sizes = geom_hash_series[dup_mask].value_counts()

    summary = metrics_dict_to_df({
        "dataset": dataset_name,
        "total_objects": total,
        "duplicate_geometries": dup_count,
        "duplicate_share": dup_count / total if total > 0 else np.nan,
        "unique_duplicate_shapes": unique_dup_groups,
    })

    return geom_hash_series, dup_mask, summary, group_sizes


def collect_duplicate_examples(gdf, hash_series, dup_mask, cols, n_groups=5):
    dup_hashes = hash_series[dup_mask].value_counts().index[:n_groups]
    chunks = []

    for i, hash_val in enumerate(dup_hashes, start=1):
        subset = gdf[hash_series == hash_val].copy()
        subset = safe_select(subset, cols)
        subset.insert(0, "duplicate_group", i)
        subset.insert(1, "geom_hash", hash_val)
        chunks.append(subset)

    if chunks:
        return pd.concat(chunks, ignore_index=True)

    return pd.DataFrame(columns=["duplicate_group", "geom_hash"] + cols)


def analyze_geometric_duplicate_candidates(gdf, dataset_name):
    gdf_local = gdf.copy()
    gdf_local["geom_signature"] = build_geom_signature(gdf_local)

    dup_mask = gdf_local["geom_signature"].duplicated(keep=False)
    total = len(gdf_local)
    dup_count = int(dup_mask.sum())
    dup_groups = int(gdf_local.loc[dup_mask, "geom_signature"].nunique())

    summary = metrics_dict_to_df({
        "dataset": dataset_name,
        "total_objects": total,
        "candidate_duplicate_geometries": dup_count,
        "candidate_duplicate_share": dup_count / total if total > 0 else np.nan,
        "unique_candidate_groups": dup_groups,
    })

    group_sizes = gdf_local.loc[dup_mask, "geom_signature"].value_counts()
    return gdf_local, dup_mask, summary, group_sizes


def check_near_duplicates(gdf, dataset_name, dist_thresh=1.0, overlap_thresh=0.8, max_examples=100):
    if gdf.empty:
        return (
            metrics_dict_to_df({"dataset": dataset_name, "near_duplicate_pairs_in_sample": 0}),
            pd.DataFrame()
        )

    sample_n = min(SAMPLE_NEARDUP_SIZE, len(gdf))
    work_gdf = (
        gdf.sample(sample_n, random_state=RANDOM_STATE)
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "source_index"})
    )

    sindex = work_gdf.sindex
    count = 0
    examples = []
    visited_pairs = set()

    for i in range(len(work_gdf)):
        geom = work_gdf.geometry.iloc[i]

        if geom is None or geom.is_empty or geom.area == 0:
            continue

        candidate_idx = list(sindex.intersection(geom.buffer(dist_thresh).bounds))
        candidate_idx = [j for j in candidate_idx if j != i]

        for j in candidate_idx:
            pair = tuple(sorted((i, j)))
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)

            geom_j = work_gdf.geometry.iloc[j]
            if geom_j is None or geom_j.is_empty or geom_j.area == 0:
                continue

            inter = geom.intersection(geom_j)
            if inter.is_empty:
                continue

            ratio_i = inter.area / geom.area
            ratio_j = inter.area / geom_j.area

            if ratio_i > overlap_thresh and ratio_j > overlap_thresh:
                count += 1

                if len(examples) < max_examples:
                    row = {
                        "pos_i": i,
                        "pos_j": j,
                        "source_index_i": work_gdf["source_index"].iloc[i],
                        "source_index_j": work_gdf["source_index"].iloc[j],
                        "ratio_i": ratio_i,
                        "ratio_j": ratio_j,
                        "inter_area": inter.area,
                    }

                    if "id" in work_gdf.columns:
                        row["id_i"] = work_gdf["id"].iloc[i]
                        row["id_j"] = work_gdf["id"].iloc[j]

                    examples.append(row)

    summary = metrics_dict_to_df({
        "dataset": dataset_name,
        "sample_size": sample_n,
        "dist_thresh": dist_thresh,
        "overlap_thresh": overlap_thresh,
        "near_duplicate_pairs_in_sample": count,
    })

    return summary, pd.DataFrame(examples)


def check_invalid_holes(gdf, dataset_name):
    bad = 0
    bad_examples = []

    for idx, geom in gdf.geometry.items():
        if geom is None:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly_id, poly in enumerate(polygons):
            exterior = poly.exterior

            for hole_id, interior in enumerate(poly.interiors):
                reason = None

                if not interior.is_ring:
                    reason = "interior_not_ring"
                elif interior.crosses(exterior):
                    reason = "interior_crosses_exterior"

                if reason is not None:
                    bad += 1
                    row = {
                        "row_index": idx,
                        "polygon_idx": poly_id,
                        "hole_idx": hole_id,
                        "reason": reason,
                    }
                    if "id" in gdf.columns:
                        row["id"] = gdf.loc[idx, "id"]
                    bad_examples.append(row)

    summary = metrics_dict_to_df({
        "dataset": dataset_name,
        "invalid_holes": bad,
    })

    return summary, pd.DataFrame(bad_examples)

SOURCE_A_PATH, SOURCE_B_PATH = Path("../data/raw/cup_it_example_src_A.csv"), Path("../data/raw/cup_it_example_src_B.csv")

source_A = pd.read_csv(
    SOURCE_A_PATH,
    encoding="utf-8-sig",
    sep=",",
    quotechar='"',
    low_memory=False
)

source_B = pd.read_csv(
    SOURCE_B_PATH,
    encoding="utf-8-sig",
    sep=",",
    quotechar='"',
    low_memory=False
)

for df in (source_A, source_B):
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

result_tables = []

add_sheet(
    result_tables,
    "overview",
    "ovw_dataset_shapes",
    "Размерности исходных датасетов A и B",
    metrics_dict_to_df({
        "source_A_rows": len(source_A),
        "source_A_cols": source_A.shape[1],
        "source_B_rows": len(source_B),
        "source_B_cols": source_B.shape[1],
    })
)

add_sheet(
    result_tables,
    "overview",
    "ovw_source_A_head",
    "Первые 5 строк источника A",
    source_A.head(5)
)

add_sheet(
    result_tables,
    "overview",
    "ovw_source_B_head",
    "Первые 5 строк источника B",
    source_B.head(5)
)

add_sheet(
    result_tables,
    "overview",
    "ovw_source_A_na",
    "Количество пропусков по колонкам в источнике A",
    normalize_series_to_df(source_A.isna().sum(), "column", "na_count")
)

add_sheet(
    result_tables,
    "overview",
    "ovw_source_B_na",
    "Количество пропусков по колонкам в источнике B",
    normalize_series_to_df(source_B.isna().sum(), "column", "na_count")
)

geom_type_A_raw = source_A["geometry"].astype(str).str.split().str[0].value_counts(dropna=False).head(10)
geom_type_B_raw = source_B["wkt"].astype(str).str.split().str[0].value_counts(dropna=False).head(10)

add_sheet(
    result_tables,
    "numeric",
    "num_A_raw_geom_types",
    "Топ-10 сырых типов геометрии в поле geometry источника A",
    normalize_series_to_df(geom_type_A_raw, "geom_type", "count")
)

add_sheet(
    result_tables,
    "numeric",
    "num_B_raw_geom_types",
    "Топ-10 сырых типов геометрии в поле wkt источника B",
    normalize_series_to_df(geom_type_B_raw, "geom_type", "count")
)

num_cols_A = [c for c in ["area_sq_m", "gkh_floor_count_min", "gkh_floor_count_max"] if c in source_A.columns]
num_cols_B = [c for c in ["number", "stairs", "avg_floor_height", "height"] if c in source_B.columns]

add_sheet(
    result_tables,
    "numeric",
    "num_A_describe",
    "Описательная статистика по числовым полям источника A",
    describe_df(source_A[num_cols_A], [0.01, 0.05, 0.5, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "numeric",
    "num_B_describe",
    "Описательная статистика по числовым полям источника B",
    describe_df(source_B[num_cols_B], [0.01, 0.05, 0.5, 0.95, 0.99])
)

sample_A = source_A["geometry"].sample(min(SAMPLE_WKT_SIZE, len(source_A)), random_state=RANDOM_STATE)
sample_B = source_B["wkt"].sample(min(SAMPLE_WKT_SIZE, len(source_B)), random_state=RANDOM_STATE)

parsed_A = sample_A.apply(safe_load_wkt)
parsed_B = sample_B.apply(safe_load_wkt)

add_sheet(
    result_tables,
    "quality",
    "qlt_wkt_parse",
    "Успешность парсинга WKT по выборке",
    metrics_dict_to_df({
        "A_parse_success_rate": parsed_A.notna().mean(),
        "B_parse_success_rate": parsed_B.notna().mean(),
        "A_sample_size": len(sample_A),
        "B_sample_size": len(sample_B),
    })
)

add_sheet(
    result_tables,
    "quality",
    "qlt_basic_numeric",
    "Базовые проверки качества числовых полей",
    metrics_dict_to_df({
        "A_area_le_0": int((source_A["area_sq_m"] <= 0).sum()) if "area_sq_m" in source_A.columns else np.nan,
        "A_floor_min_gt_floor_max": int((
            source_A["gkh_floor_count_min"].notna() &
            source_A["gkh_floor_count_max"].notna() &
            (source_A["gkh_floor_count_min"] > source_A["gkh_floor_count_max"])
        ).sum()) if {"gkh_floor_count_min", "gkh_floor_count_max"}.issubset(source_A.columns) else np.nan,
        "B_stairs_le_0": int((source_B["stairs"] <= 0).sum()) if "stairs" in source_B.columns else np.nan,
        "B_stairs_gt_100": int((source_B["stairs"] > 100).sum()) if "stairs" in source_B.columns else np.nan,
        "B_avg_floor_height_le_0": int((source_B["avg_floor_height"] <= 0).sum()) if "avg_floor_height" in source_B.columns else np.nan,
        "B_avg_floor_height_gt_10": int((source_B["avg_floor_height"] > 10).sum()) if "avg_floor_height" in source_B.columns else np.nan,
        "B_height_le_0": int((source_B["height"] <= 0).sum()) if "height" in source_B.columns else np.nan,
        "B_height_gt_300": int((source_B["height"] > 300).sum()) if "height" in source_B.columns else np.nan,
    })
)

b = source_B.copy()

if {"stairs", "height"}.issubset(b.columns):
    mask = b["stairs"].notna() & b["height"].notna() & (b["stairs"] != 0)
    b.loc[mask, "height_per_floor"] = b.loc[mask, "height"] / b.loc[mask, "stairs"]

if {"stairs", "avg_floor_height", "height"}.issubset(b.columns):
    mask2 = b["stairs"].notna() & b["avg_floor_height"].notna() & b["height"].notna()
    b.loc[mask2, "height_from_stairs"] = b.loc[mask2, "stairs"] * b.loc[mask2, "avg_floor_height"]
    b.loc[mask2, "height_error_abs"] = (b.loc[mask2, "height"] - b.loc[mask2, "height_from_stairs"]).abs()

add_sheet(
    result_tables,
    "derived_height",
    "drv_height_per_floor_desc",
    "Статистика по признаку height_per_floor",
    describe_df(safe_select(b, ["height_per_floor"]), [0.01, 0.05, 0.5, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "derived_height",
    "drv_height_per_floor_chk",
    "Проверки по реалистичности высоты на этаж",
    metrics_dict_to_df({
        "height_per_floor_lt_2": int((b["height_per_floor"] < 2).sum()) if "height_per_floor" in b.columns else np.nan,
        "height_per_floor_gt_6": int((b["height_per_floor"] > 6).sum()) if "height_per_floor" in b.columns else np.nan,
    })
)

add_sheet(
    result_tables,
    "derived_height",
    "drv_height_error_desc",
    "Статистика абсолютной ошибки между height и stairs * avg_floor_height",
    describe_df(safe_select(b, ["height_error_abs"]), [0.5, 0.9, 0.95, 0.99])
)

if "avg_floor_height" in source_B.columns:
    add_sheet(
        result_tables,
        "derived_height",
        "drv_avg_floor_height_top",
        "Топ значений avg_floor_height",
        normalize_series_to_df(source_B["avg_floor_height"].value_counts(dropna=False).head(20), "avg_floor_height", "count")
    )


addr_cols_B = [
    "subject", "district", "type", "locality", "type_street",
    "name_street", "number", "letter", "fraction", "housing", "building"
]
addr_cols_B_existing = [c for c in addr_cols_B if c in source_B.columns]

if addr_cols_B_existing:
    addr_na_share = ((source_B[addr_cols_B_existing].isna().mean() * 100).round(2)).sort_values(ascending=False)
else:
    addr_na_share = pd.Series(dtype=float)

has_basic_addr = (
    source_B["name_street"].notna() & source_B["number"].notna()
    if {"name_street", "number"}.issubset(source_B.columns)
    else pd.Series([False] * len(source_B))
)

if "tags" in source_A.columns:
    add_sheet(
        result_tables,
        "categorical",
        "cat_A_tags_top30",
        "Топ-30 значений поля tags в источнике A",
        normalize_series_to_df(source_A["tags"].value_counts(dropna=False).head(30), "tags", "count")
    )

if "purpose_of_building" in source_B.columns:
    add_sheet(
        result_tables,
        "categorical",
        "cat_B_purpose_top30",
        "Топ-30 назначений зданий в источнике B",
        normalize_series_to_df(source_B["purpose_of_building"].value_counts(dropna=False).head(30), "purpose_of_building", "count")
    )

add_sheet(
    result_tables,
    "categorical",
    "cat_B_addr_na_share",
    "Доля пропусков по адресным полям источника B, %",
    normalize_series_to_df(addr_na_share, "column", "na_share_pct")
)

add_sheet(
    result_tables,
    "categorical",
    "cat_B_basic_addr_cov",
    "Покрытие базового адреса: name_street + number",
    metrics_dict_to_df({
        "B_with_basic_address_share": has_basic_addr.mean(),
        "B_with_basic_address_count": int(has_basic_addr.sum()),
    })
)


geom_A = parse_wkt_series(source_A["geometry"])
geom_B = parse_wkt_series(source_B["wkt"])

gdf_A = gpd.GeoDataFrame(source_A.copy(), geometry=geom_A, crs=CRS_GEOGRAPHIC)
gdf_B = gpd.GeoDataFrame(source_B.copy(), geometry=geom_B, crs=CRS_GEOGRAPHIC)

geom_basic_df = metrics_dict_to_df({
    "A_geometry_null": int(gdf_A.geometry.isna().sum()),
    "B_geometry_null": int(gdf_B.geometry.isna().sum()),
    "A_invalid_geometry": int((~gdf_A.geometry.is_valid.fillna(False)).sum()),
    "B_invalid_geometry": int((~gdf_B.geometry.is_valid.fillna(False)).sum()),
    "A_empty_geometry": int(gdf_A.geometry.is_empty.fillna(False).sum()),
    "B_empty_geometry": int(gdf_B.geometry.is_empty.fillna(False).sum()),
})

add_sheet(
    result_tables,
    "geometry_basic",
    "geo_basic_quality",
    "Базовые показатели качества геометрий",
    geom_basic_df
)

add_sheet(
    result_tables,
    "geometry_basic",
    "geo_A_types",
    "Типы геометрий в A после парсинга",
    normalize_series_to_df(gdf_A.geom_type.value_counts(dropna=False), "geom_type", "count")
)

add_sheet(
    result_tables,
    "geometry_basic",
    "geo_B_types",
    "Типы геометрий в B после парсинга",
    normalize_series_to_df(gdf_B.geom_type.value_counts(dropna=False), "geom_type", "count")
)


gdf_A_m = gdf_A.to_crs(CRS_METRIC).copy()
gdf_B_m = gdf_B.to_crs(CRS_METRIC).copy()

gdf_A_m["centroid"] = gdf_A_m.geometry.centroid
gdf_B_m["centroid"] = gdf_B_m.geometry.centroid

gdf_A_m["centroid_x"] = gdf_A_m["centroid"].x
gdf_A_m["centroid_y"] = gdf_A_m["centroid"].y
gdf_B_m["centroid_x"] = gdf_B_m["centroid"].x
gdf_B_m["centroid_y"] = gdf_B_m["centroid"].y

bounds_A = gdf_A_m.geometry.bounds
bounds_B = gdf_B_m.geometry.bounds

gdf_A_m["bbox_width_m"] = bounds_A["maxx"] - bounds_A["minx"]
gdf_A_m["bbox_height_m"] = bounds_A["maxy"] - bounds_A["miny"]
gdf_A_m["bbox_area_m2"] = gdf_A_m["bbox_width_m"] * gdf_A_m["bbox_height_m"]

gdf_B_m["bbox_width_m"] = bounds_B["maxx"] - bounds_B["minx"]
gdf_B_m["bbox_height_m"] = bounds_B["maxy"] - bounds_B["miny"]
gdf_B_m["bbox_area_m2"] = gdf_B_m["bbox_width_m"] * gdf_B_m["bbox_height_m"]

add_sheet(
    result_tables,
    "geom_stats",
    "gst_centroid_nulls",
    "Количество пустых центроидов",
    metrics_dict_to_df({
        "A_centroid_null": int(gdf_A_m["centroid"].isna().sum()),
        "B_centroid_null": int(gdf_B_m["centroid"].isna().sum()),
    })
)

add_sheet(
    result_tables,
    "geom_stats",
    "gst_A_bbox_stats",
    "Статистика bbox-метрик для A",
    describe_df(gdf_A_m[["bbox_width_m", "bbox_height_m", "bbox_area_m2"]], [0.01, 0.05, 0.5, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "geom_stats",
    "gst_B_bbox_stats",
    "Статистика bbox-метрик для B",
    describe_df(gdf_B_m[["bbox_width_m", "bbox_height_m", "bbox_area_m2"]], [0.01, 0.05, 0.5, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "geom_stats",
    "gst_A_centroid_head",
    "Примеры центроидов объектов A",
    safe_select(gdf_A_m.head(5), ["id", "centroid_x", "centroid_y"])
)

add_sheet(
    result_tables,
    "geom_stats",
    "gst_B_centroid_head",
    "Примеры центроидов объектов B",
    safe_select(gdf_B_m.head(5), ["id", "centroid_x", "centroid_y"])
)


gdf_A_m["geom_area_m2"] = gdf_A_m.geometry.area

if "area_sq_m" in gdf_A_m.columns:
    gdf_A_m["area_diff_abs"] = (gdf_A_m["geom_area_m2"] - gdf_A_m["area_sq_m"]).abs()
    gdf_A_m["area_diff_pct"] = np.where(
        gdf_A_m["area_sq_m"] > 0,
        gdf_A_m["area_diff_abs"] / gdf_A_m["area_sq_m"],
        np.nan
    )
else:
    gdf_A_m["area_diff_abs"] = np.nan
    gdf_A_m["area_diff_pct"] = np.nan

add_sheet(
    result_tables,
    "area_checks",
    "area_compare_stats",
    "Сравнение табличной и геометрической площади в A",
    describe_df(
        safe_select(gdf_A_m, ["area_sq_m", "geom_area_m2", "area_diff_abs", "area_diff_pct"]),
        [0.5, 0.9, 0.95, 0.99]
    )
)

add_sheet(
    result_tables,
    "area_checks",
    "area_diff_thresholds",
    "Количество объектов A с большими расхождениями по площади",
    metrics_dict_to_df({
        "area_diff_pct_gt_10pct": int((gdf_A_m["area_diff_pct"] > 0.10).sum()),
        "area_diff_pct_gt_25pct": int((gdf_A_m["area_diff_pct"] > 0.25).sum()),
        "area_diff_pct_gt_50pct": int((gdf_A_m["area_diff_pct"] > 0.50).sum()),
    })
)

top_area_mismatch = gdf_A_m.sort_values("area_diff_pct", ascending=False)
add_sheet(
    result_tables,
    "area_checks",
    "area_top_mismatch",
    "Топ подозрительных объектов A по расхождению площади",
    safe_select(top_area_mismatch.head(10), ["id", "title", "area_sq_m", "geom_area_m2", "area_diff_abs", "area_diff_pct"])
)


gdf_A_m["perimeter_m"] = gdf_A_m.geometry.length
gdf_B_m["perimeter_m"] = gdf_B_m.geometry.length
gdf_B_m["geom_area_m2"] = gdf_B_m.geometry.area

gdf_A_m["compactness"] = np.where(
    gdf_A_m["perimeter_m"] > 0,
    4 * np.pi * gdf_A_m["geom_area_m2"] / (gdf_A_m["perimeter_m"] ** 2),
    np.nan
)

gdf_B_m["compactness"] = np.where(
    gdf_B_m["perimeter_m"] > 0,
    4 * np.pi * gdf_B_m["geom_area_m2"] / (gdf_B_m["perimeter_m"] ** 2),
    np.nan
)

gdf_A_m["rectangularity"] = np.where(
    gdf_A_m["bbox_area_m2"] > 0,
    gdf_A_m["geom_area_m2"] / gdf_A_m["bbox_area_m2"],
    np.nan
)

gdf_B_m["rectangularity"] = np.where(
    gdf_B_m["bbox_area_m2"] > 0,
    gdf_B_m["geom_area_m2"] / gdf_B_m["bbox_area_m2"],
    np.nan
)

gdf_A_m["elongation"] = np.where(
    gdf_A_m[["bbox_width_m", "bbox_height_m"]].min(axis=1) > 0,
    gdf_A_m[["bbox_width_m", "bbox_height_m"]].max(axis=1) / gdf_A_m[["bbox_width_m", "bbox_height_m"]].min(axis=1),
    np.nan
)

gdf_B_m["elongation"] = np.where(
    gdf_B_m[["bbox_width_m", "bbox_height_m"]].min(axis=1) > 0,
    gdf_B_m[["bbox_width_m", "bbox_height_m"]].max(axis=1) / gdf_B_m[["bbox_width_m", "bbox_height_m"]].min(axis=1),
    np.nan
)

shape_cols = ["geom_area_m2", "perimeter_m", "compactness", "rectangularity", "elongation"]

add_sheet(
    result_tables,
    "shape_features",
    "shp_A_stats",
    "Статистика shape-признаков для A",
    describe_df(gdf_A_m[shape_cols], [0.01, 0.05, 0.5, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "shape_features",
    "shp_B_stats",
    "Статистика shape-признаков для B",
    describe_df(gdf_B_m[shape_cols], [0.01, 0.05, 0.5, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "shape_features",
    "shp_A_elongated_top",
    "Наиболее вытянутые объекты A",
    safe_select(
        gdf_A_m.sort_values("elongation", ascending=False).head(10),
        ["id", "title", "geom_area_m2", "elongation", "compactness", "rectangularity"]
    )
)

add_sheet(
    result_tables,
    "shape_features",
    "shp_B_elongated_top",
    "Наиболее вытянутые объекты B",
    safe_select(
        gdf_B_m.sort_values("elongation", ascending=False).head(10),
        ["id", "purpose_of_building", "geom_area_m2", "elongation", "compactness", "rectangularity"]
    )
)


sample_n_A = min(SAMPLE_NEIGHBORS_SIZE, len(gdf_A_m))
sample_n_B = min(SAMPLE_NEIGHBORS_SIZE, len(gdf_B_m))

A_sample = gdf_A_m.sample(sample_n_A, random_state=RANDOM_STATE).copy()
B_sample = gdf_B_m.sample(sample_n_B, random_state=RANDOM_STATE).copy()

sindex_A = gdf_A_m.sindex
sindex_B = gdf_B_m.sindex

A_sample["neighbors_cnt"] = A_sample.geometry.apply(lambda geom: count_intersecting_neighbors(geom, gdf_A_m, sindex_A))
B_sample["neighbors_cnt"] = B_sample.geometry.apply(lambda geom: count_intersecting_neighbors(geom, gdf_B_m, sindex_B))

add_sheet(
    result_tables,
    "neighbors",
    "nbr_A_stats",
    "Статистика числа пересекающихся соседей для выборки A",
    describe_df(A_sample[["neighbors_cnt"]], [0.5, 0.9, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "neighbors",
    "nbr_B_stats",
    "Статистика числа пересекающихся соседей для выборки B",
    describe_df(B_sample[["neighbors_cnt"]], [0.5, 0.9, 0.95, 0.99])
)

add_sheet(
    result_tables,
    "neighbors",
    "nbr_A_top_connected",
    "Объекты A с наибольшим числом пересечений в выборке",
    safe_select(
        A_sample.sort_values("neighbors_cnt", ascending=False).head(10),
        ["id", "title", "geom_area_m2", "neighbors_cnt"]
    )
)

add_sheet(
    result_tables,
    "neighbors",
    "nbr_B_top_connected",
    "Объекты B с наибольшим числом пересечений в выборке",
    safe_select(
        B_sample.sort_values("neighbors_cnt", ascending=False).head(10),
        ["id", "purpose_of_building", "geom_area_m2", "neighbors_cnt"]
    )
)

hash_A, dup_A, dup_summary_A, dup_group_sizes_A = analyze_geometry_duplicates(gdf_A, "A")
hash_B, dup_B, dup_summary_B, dup_group_sizes_B = analyze_geometry_duplicates(gdf_B, "B")

add_sheet(
    result_tables,
    "duplicates_exact",
    "dup_exact_A_summary",
    "Сводка по точным дубликатам геометрии в A",
    dup_summary_A
)

add_sheet(
    result_tables,
    "duplicates_exact",
    "dup_exact_B_summary",
    "Сводка по точным дубликатам геометрии в B",
    dup_summary_B
)

add_sheet(
    result_tables,
    "duplicates_exact",
    "dup_exact_A_groups_top",
    "Топ размеров групп точных дубликатов в A",
    normalize_series_to_df(dup_group_sizes_A.head(10), "geom_hash", "count")
)

add_sheet(
    result_tables,
    "duplicates_exact",
    "dup_exact_B_groups_top",
    "Топ размеров групп точных дубликатов в B",
    normalize_series_to_df(dup_group_sizes_B.head(10), "geom_hash", "count")
)

add_sheet(
    result_tables,
    "duplicates_exact",
    "dup_exact_A_examples",
    "Примеры точных дубликатов геометрии в A",
    collect_duplicate_examples(
        gdf_A,
        hash_A,
        dup_A,
        [c for c in ["id", "title", "area_sq_m"] if c in gdf_A.columns]
    )
)

add_sheet(
    result_tables,
    "duplicates_exact",
    "dup_exact_B_examples",
    "Примеры точных дубликатов геометрии в B",
    collect_duplicate_examples(
        gdf_B,
        hash_B,
        dup_B,
        [c for c in ["id", "purpose_of_building", "height", "stairs"] if c in gdf_B.columns]
    )
)

gdf_A_sig, dupA_sig, sig_summary_A, sig_group_sizes_A = analyze_geometric_duplicate_candidates(gdf_A_m, "A")
gdf_B_sig, dupB_sig, sig_summary_B, sig_group_sizes_B = analyze_geometric_duplicate_candidates(gdf_B_m, "B")

add_sheet(
    result_tables,
    "duplicates_signature",
    "dup_sig_A_summary",
    "Сводка по кандидатам в дубликаты по сигнатуре для A",
    sig_summary_A
)

add_sheet(
    result_tables,
    "duplicates_signature",
    "dup_sig_B_summary",
    "Сводка по кандидатам в дубликаты по сигнатуре для B",
    sig_summary_B
)

add_sheet(
    result_tables,
    "duplicates_signature",
    "dup_sig_A_groups_top",
    "Топ размеров групп по сигнатуре для A",
    normalize_series_to_df(sig_group_sizes_A.head(10), "geom_signature", "count")
)

add_sheet(
    result_tables,
    "duplicates_signature",
    "dup_sig_B_groups_top",
    "Топ размеров групп по сигнатуре для B",
    normalize_series_to_df(sig_group_sizes_B.head(10), "geom_signature", "count")
)


neardup_summary_A, neardup_examples_A = check_near_duplicates(gdf_A_m, "A")
neardup_summary_B, neardup_examples_B = check_near_duplicates(gdf_B_m, "B")

add_sheet(
    result_tables,
    "near_duplicates",
    "near_dup_A_summary",
    "Сводка по near-duplicates в выборке A",
    neardup_summary_A
)

add_sheet(
    result_tables,
    "near_duplicates",
    "near_dup_B_summary",
    "Сводка по near-duplicates в выборке B",
    neardup_summary_B
)

add_sheet(
    result_tables,
    "near_duplicates",
    "near_dup_A_examples",
    "Примеры near-duplicates в выборке A",
    neardup_examples_A
)

add_sheet(
    result_tables,
    "near_duplicates",
    "near_dup_B_examples",
    "Примеры near-duplicates в выборке B",
    neardup_examples_B
)

holes_summary_A, holes_examples_A = check_invalid_holes(gdf_A_m, "A")
holes_summary_B, holes_examples_B = check_invalid_holes(gdf_B_m, "B")

add_sheet(
    result_tables,
    "invalid_holes",
    "holes_A_summary",
    "Сводка по некорректным отверстиям в A",
    holes_summary_A
)

add_sheet(
    result_tables,
    "invalid_holes",
    "holes_B_summary",
    "Сводка по некорректным отверстиям в B",
    holes_summary_B
)

add_sheet(
    result_tables,
    "invalid_holes",
    "holes_A_examples",
    "Примеры некорректных отверстий в A",
    holes_examples_A
)

add_sheet(
    result_tables,
    "invalid_holes",
    "holes_B_examples",
    "Примеры некорректных отверстий в B",
    holes_examples_B
)

gdf_A_m_export = gdf_A_m.drop(columns=["geometry", "centroid"], errors="ignore").copy()
gdf_B_m_export = gdf_B_m.drop(columns=["geometry", "centroid"], errors="ignore").copy()

add_sheet(
    result_tables,
    "flat_exports",
    "flat_A_metrics",
    "Плоская таблица A со всеми рассчитанными метриками без geometry",
    gdf_A_m_export
)

add_sheet(
    result_tables,
    "flat_exports",
    "flat_B_metrics",
    "Плоская таблица B со всеми рассчитанными метриками без geometry",
    gdf_B_m_export
)

sheet_index_rows = []
for item in result_tables:
    df = item["df"]
    sheet_index_rows.append({
        "section": item["section"],
        "sheet_name": item["sheet_name"],
        "description": item["description"],
        "rows": len(df),
        "cols": len(df.columns),
    })

sheet_index_df = pd.DataFrame(sheet_index_rows)

used_sheet_names = set()

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    index_sheet_name = sanitize_sheet_name("sheet_index", used_sheet_names)
    sheet_index_df.to_excel(writer, sheet_name=index_sheet_name, index=False)

    for item in result_tables:
        safe_name = sanitize_sheet_name(item["sheet_name"], used_sheet_names)
        item["df"].to_excel(writer, sheet_name=safe_name, index=False)

print(f"Готово: результаты сохранены в {OUTPUT_XLSX}")