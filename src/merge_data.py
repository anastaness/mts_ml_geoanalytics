import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from pathlib import Path

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

DATA_DIR = Path("../data/interim")
OUTPUT_DIR = Path("../data/interim")
SOURCE_A_PATH = DATA_DIR / "A_clean_final1.parquet"
SOURCE_B_PATH = DATA_DIR / "B_clean_final1.parquet"

TARGET_CRS = "EPSG:32635"

gdf_a = gpd.read_parquet(SOURCE_A_PATH)
gdf_b = gpd.read_parquet(SOURCE_B_PATH)

print("Исходные размеры после загрузки parquet:")
print(f"A: {len(gdf_a):,}")
print(f"B: {len(gdf_b):,}")

# ---------- фильтр по релевантным зданиям ----------
if "is_relevant_building" in gdf_a.columns:
    gdf_a = gdf_a[gdf_a["is_relevant_building"]].copy()

if "is_relevant_building" in gdf_b.columns:
    gdf_b = gdf_b[gdf_b["is_relevant_building"]].copy()

print("\nПосле фильтра is_relevant_building:")
print(f"A: {len(gdf_a):,}")
print(f"B: {len(gdf_b):,}")

# ---------- перевод в метрическую CRS ----------
gdf_a = gdf_a.to_crs(TARGET_CRS)
gdf_b = gdf_b.to_crs(TARGET_CRS)

# ---------- служебные идентификаторы ----------
gdf_a = gdf_a.reset_index(drop=False).rename(columns={"index": "orig_index_a"})
gdf_b = gdf_b.reset_index(drop=False).rename(columns={"index": "orig_index_b"})

gdf_a["src"] = "A"
gdf_b["src"] = "B"

gdf_a["uid"] = "A_" + gdf_a.index.astype(str)
gdf_b["uid"] = "B_" + gdf_b.index.astype(str)

# ---------- базовые геометрические признаки ----------
gdf_a["area_m2"] = gdf_a.geometry.area
gdf_b["area_m2"] = gdf_b.geometry.area

gdf_a["perimeter_m"] = gdf_a.geometry.length
gdf_b["perimeter_m"] = gdf_b.geometry.length

gdf_a["centroid"] = gdf_a.geometry.centroid
gdf_b["centroid"] = gdf_b.geometry.centroid

print("\nБазовая статистика площадей:")
print(gdf_a["area_m2"].describe(percentiles=[0.25, 0.5, 0.75]))
print()
print(gdf_b["area_m2"].describe(percentiles=[0.25, 0.5, 0.75]))

# Словари для быстрого доступа
a_geom = gdf_a.set_index("uid").geometry.to_dict()
b_geom = gdf_b.set_index("uid").geometry.to_dict()

a_area = gdf_a.set_index("uid")["area_m2"].to_dict()
b_area = gdf_b.set_index("uid")["area_m2"].to_dict()

# Для удобства оставим укороченные таблицы
a_small = gdf_a[["uid", "geometry", "area_m2", "perimeter_m"]].copy()
b_small = gdf_b[["uid", "geometry", "area_m2", "perimeter_m"]].copy()

print("Технические таблицы подготовлены.")
print(f"a_small: {a_small.shape}")
print(f"b_small: {b_small.shape}")

buffer_tol_m = 0.5

a_buffered = a_small.copy()
a_buffered["geometry"] = a_buffered.geometry.buffer(buffer_tol_m)

candidate_pairs = gpd.sjoin(
    a_buffered,
    b_small,
    how="inner",
    predicate="intersects"
).reset_index(drop=True)

# Переименуем колонки в более удобный вид
candidate_pairs = candidate_pairs.rename(columns={
    "uid_left": "uid_a",
    "uid_right": "uid_b",
    "area_m2_left": "area_a",
    "area_m2_right": "area_b",
    "perimeter_m_left": "perimeter_a",
    "perimeter_m_right": "perimeter_b"
})

print("Количество кандидатных пар A-B после spatial join:")
print(f"{len(candidate_pairs):,}")

candidate_pairs.head()

def calc_pair_metrics(uid_a: str, uid_b: str) -> dict:
    ga = a_geom[uid_a]
    gb = b_geom[uid_b]

    inter_geom = ga.intersection(gb)
    inter_area = inter_geom.area

    union_area = ga.union(gb).area
    iou = inter_area / union_area if union_area > 0 else 0.0

    area_a = a_area[uid_a]
    area_b = b_area[uid_b]

    overlap_a = inter_area / area_a if area_a > 0 else 0.0
    overlap_b = inter_area / area_b if area_b > 0 else 0.0

    dist_m = ga.distance(gb)
    centroid_dist_m = ga.centroid.distance(gb.centroid)

    return {
        "intersection_area": inter_area,
        "union_area": union_area,
        "iou": iou,
        "overlap_a": overlap_a,
        "overlap_b": overlap_b,
        "dist_m": dist_m,
        "centroid_dist_m": centroid_dist_m,
    }

metrics = candidate_pairs[["uid_a", "uid_b"]].apply(
    lambda row: pd.Series(calc_pair_metrics(row["uid_a"], row["uid_b"])),
    axis=1
)

candidate_pairs = pd.concat([candidate_pairs, metrics], axis=1)

print("Метрики рассчитаны.")
print(candidate_pairs[["iou", "overlap_a", "overlap_b", "dist_m"]].describe())

candidate_pairs.head()

IOU_THR = 0.15
OVERLAP_THR = 0.60
DIST_THR_M = 1.0
MIN_REL_INTERSECTION = 0.05

candidate_pairs["min_area"] = candidate_pairs[["area_a", "area_b"]].min(axis=1)
candidate_pairs["rel_intersection_min"] = np.where(
    candidate_pairs["min_area"] > 0,
    candidate_pairs["intersection_area"] / candidate_pairs["min_area"],
    0.0
)

edge_mask = (
    (candidate_pairs["iou"] >= IOU_THR) |
    (candidate_pairs["overlap_a"] >= OVERLAP_THR) |
    (candidate_pairs["overlap_b"] >= OVERLAP_THR) |
    (
        (candidate_pairs["dist_m"] <= DIST_THR_M) &
        (candidate_pairs["rel_intersection_min"] >= MIN_REL_INTERSECTION)
    )
)

edges_ab = candidate_pairs[edge_mask].copy()

print("После фильтра пространственных связей осталось рёбер:")
print(f"{len(edges_ab):,}")

print("\nРаспределение по основным условиям:")
print(f"IoU >= {IOU_THR}: {(candidate_pairs['iou'] >= IOU_THR).sum():,}")
print(f"overlap_a >= {OVERLAP_THR}: {(candidate_pairs['overlap_a'] >= OVERLAP_THR).sum():,}")
print(f"overlap_b >= {OVERLAP_THR}: {(candidate_pairs['overlap_b'] >= OVERLAP_THR).sum():,}")
print(f"distance <= {DIST_THR_M} м и rel_intersection >= {MIN_REL_INTERSECTION}: {(((candidate_pairs['dist_m'] <= DIST_THR_M) & (candidate_pairs['rel_intersection_min'] >= MIN_REL_INTERSECTION))).sum():,}")

edges_ab.head()

G = nx.Graph()

# Добавляем вершины
for uid in gdf_a["uid"]:
    G.add_node(uid, src="A")

for uid in gdf_b["uid"]:
    G.add_node(uid, src="B")

# Добавляем рёбра
for _, row in edges_ab.iterrows():
    G.add_edge(
        row["uid_a"],
        row["uid_b"],
        iou=row["iou"],
        overlap_a=row["overlap_a"],
        overlap_b=row["overlap_b"],
        dist_m=row["dist_m"],
        intersection_area=row["intersection_area"],
    )

components = list(nx.connected_components(G))

print("Граф построен.")
print(f"Количество компонент связности: {len(components):,}")

component_sizes = pd.Series([len(c) for c in components])
print("\nРаспределение размеров компонент:")
print(component_sizes.value_counts().sort_index().head(20))

edge_lookup = edges_ab.set_index(["uid_a", "uid_b"])[
    ["iou", "overlap_a", "overlap_b", "dist_m", "intersection_area"]
].to_dict("index")

def get_component_edge_metrics(nodes):
    nodes = list(nodes)
    a_nodes = [n for n in nodes if n.startswith("A_")]
    b_nodes = [n for n in nodes if n.startswith("B_")]

    rows = []
    for a in a_nodes:
        for b in b_nodes:
            key = (a, b)
            if key in edge_lookup:
                row = edge_lookup[key].copy()
                row["uid_a"] = a
                row["uid_b"] = b
                rows.append(row)

    if not rows:
        return {
            "n_edges_ab": 0,
            "max_iou": np.nan,
            "mean_iou": np.nan,
            "max_overlap_a": np.nan,
            "max_overlap_b": np.nan,
            "min_dist_m": np.nan,
        }

    tmp = pd.DataFrame(rows)
    return {
        "n_edges_ab": len(tmp),
        "max_iou": tmp["iou"].max(),
        "mean_iou": tmp["iou"].mean(),
        "max_overlap_a": tmp["overlap_a"].max(),
        "max_overlap_b": tmp["overlap_b"].max(),
        "min_dist_m": tmp["dist_m"].min(),
    }

def classify_match_type(n_a, n_b):
    if n_a > 0 and n_b == 0:
        return "A_only"
    if n_b > 0 and n_a == 0:
        return "B_only"
    if n_a == 1 and n_b == 1:
        return "1:1"
    if n_a == 1 and n_b > 1:
        return "1:n"
    if n_a > 1 and n_b == 1:
        return "n:1"
    if n_a > 1 and n_b > 1:
        return "n:n"
    return "unknown"

component_rows = []

for comp_id, comp_nodes in enumerate(components, start=1):
    comp_nodes = sorted(comp_nodes)
    a_nodes = [n for n in comp_nodes if n.startswith("A_")]
    b_nodes = [n for n in comp_nodes if n.startswith("B_")]

    metrics_dict = get_component_edge_metrics(comp_nodes)

    component_rows.append({
        "component_id": comp_id,
        "uids_a": a_nodes,
        "uids_b": b_nodes,
        "n_a": len(a_nodes),
        "n_b": len(b_nodes),
        "match_type": classify_match_type(len(a_nodes), len(b_nodes)),
        **metrics_dict
    })

components_df = pd.DataFrame(component_rows)

print("Таблица компонент построена.")
print(components_df["match_type"].value_counts(dropna=False))

components_df.head()

a_geom_by_uid = gdf_a.set_index("uid").geometry.to_dict()
b_geom_by_uid = gdf_b.set_index("uid").geometry.to_dict()

a_area_by_uid = gdf_a.set_index("uid")["area_m2"].to_dict()
b_area_by_uid = gdf_b.set_index("uid")["area_m2"].to_dict()

def safe_union(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    return unary_union(geoms)

def get_union_area(geoms):
    union_geom = safe_union(geoms)
    if union_geom is None:
        return np.nan
    return union_geom.area

components_df["sum_area_a"] = components_df["uids_a"].apply(
    lambda ids: sum(a_area_by_uid[x] for x in ids) if ids else 0.0
)

components_df["sum_area_b"] = components_df["uids_b"].apply(
    lambda ids: sum(b_area_by_uid[x] for x in ids) if ids else 0.0
)

components_df["union_area_a"] = components_df["uids_a"].apply(
    lambda ids: get_union_area([a_geom_by_uid[x] for x in ids]) if ids else np.nan
)

components_df["union_area_b"] = components_df["uids_b"].apply(
    lambda ids: get_union_area([b_geom_by_uid[x] for x in ids]) if ids else np.nan
)

components_df["union_area_all"] = components_df.apply(
    lambda row: get_union_area(
        [a_geom_by_uid[x] for x in row["uids_a"]] +
        [b_geom_by_uid[x] for x in row["uids_b"]]
    ),
    axis=1
)

print("Агрегированные геометрические признаки компонент добавлены.")
components_df.head()

def pick_geometry_for_component(uids_a, uids_b, match_type):
    geom_a = safe_union([a_geom_by_uid[x] for x in uids_a]) if uids_a else None
    geom_b = safe_union([b_geom_by_uid[x] for x in uids_b]) if uids_b else None

    if geom_a is None and geom_b is None:
        return None, "none"

    if geom_a is not None and geom_b is None:
        return geom_a, "A"

    if geom_b is not None and geom_a is None:
        return geom_b, "B"

    # Если связь простая 1:1, в качестве репрезентативной геометрии
    # берём B как источник, содержащий целевую высоту.
    if match_type == "1:1":
        return geom_b, "B"

    area_a = geom_a.area if geom_a is not None else np.nan
    area_b = geom_b.area if geom_b is not None else np.nan

    # Простое правило:
    # если геометрии сопоставимы по площади, предпочитаем B;
    # если B сильно отличается, берём A как более "основной" контур.
    ratio = min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) > 0 else 0

    if ratio >= 0.70:
        return geom_b, "B"
    else:
        return geom_a, "A"

picked = components_df.apply(
    lambda row: pick_geometry_for_component(row["uids_a"], row["uids_b"], row["match_type"]),
    axis=1
)

components_df["rep_geometry"] = [x[0] for x in picked]
components_df["geometry_source"] = [x[1] for x in picked]

components_gdf = gpd.GeoDataFrame(
    components_df.copy(),
    geometry="rep_geometry",
    crs=TARGET_CRS
)

print("Итоговая репрезентативная геометрия выбрана.")
print(components_gdf["geometry_source"].value_counts(dropna=False))
components_gdf.head()

b_attr = gdf_b.set_index("uid")[["height", "stairs", "avg_floor_height", "purpose_of_building"]].copy()

def aggregate_b_attributes(uids_b):
    if not uids_b:
        return pd.Series({
            "n_b_with_height": 0,
            "median_height_b": np.nan,
            "median_stairs_b": np.nan,
            "median_avg_floor_height_b": np.nan,
            "mode_purpose_b": np.nan,
        })

    tmp = b_attr.loc[uids_b].copy()

    purpose_mode = tmp["purpose_of_building"].mode()
    purpose_value = purpose_mode.iloc[0] if len(purpose_mode) > 0 else np.nan

    return pd.Series({
        "n_b_with_height": tmp["height"].notna().sum(),
        "median_height_b": tmp["height"].median(),
        "median_stairs_b": tmp["stairs"].median(),
        "median_avg_floor_height_b": tmp["avg_floor_height"].median(),
        "mode_purpose_b": purpose_value,
    })

b_agg = components_gdf["uids_b"].apply(aggregate_b_attributes)
components_gdf = pd.concat([components_gdf, b_agg], axis=1)

print("Атрибуты B агрегированы на уровень компонент.")
components_gdf[
    ["match_type", "n_b_with_height", "median_height_b", "median_stairs_b", "mode_purpose_b"]
].head()

def assign_match_confidence(row):
    if row["match_type"] in {"A_only", "B_only"}:
        return "none"

    max_iou = row["max_iou"]
    max_oa = row["max_overlap_a"]
    max_ob = row["max_overlap_b"]

    if (
        pd.notna(max_iou) and max_iou >= 0.50
    ) or (
        pd.notna(max_oa) and max_oa >= 0.80
    ) or (
        pd.notna(max_ob) and max_ob >= 0.80
    ):
        return "high"

    if (
        pd.notna(max_iou) and max_iou >= 0.15
    ) or (
        pd.notna(max_oa) and max_oa >= 0.60
    ) or (
        pd.notna(max_ob) and max_ob >= 0.60
    ):
        return "medium"

    return "low"

components_gdf["match_confidence"] = components_gdf.apply(assign_match_confidence, axis=1)

print("Распределение уверенности:")
print(components_gdf["match_confidence"].value_counts(dropna=False))

components_gdf["target_height"] = components_gdf["median_height_b"]
components_gdf["target_height_is_observed"] = components_gdf["target_height"].notna().astype("int8")

def assign_height_source(row):
    if pd.notna(row["median_height_b"]):
        return "B_observed"
    return "missing"

def assign_height_source_detail(row):
    if pd.isna(row["median_height_b"]):
        return "missing"

    # Детализация полезна для последующего анализа качества обучающей выборки.
    if row["match_type"] == "B_only":
        return "B_only_direct"
    if row["match_type"] == "1:1":
        return "B_from_1to1_match"
    if row["match_type"] == "1:n":
        return "B_from_1ton_match"
    if row["match_type"] == "n:1":
        return "B_from_nto1_match"
    if row["match_type"] == "n:n":
        return "B_from_nton_match"
    return "B_observed_other"

components_gdf["target_height_source"] = components_gdf.apply(assign_height_source, axis=1)
components_gdf["target_height_source_detail"] = components_gdf.apply(assign_height_source_detail, axis=1)

# Дополнительный полезный флаг:
# можно ли считать observed target относительно надежным для обучения.
# Например, самые надежные наблюдения — B_only и 1:1/high.
def assign_target_reliability(row):
    if row["target_height_is_observed"] == 0:
        return "none"

    if row["match_type"] == "B_only":
        return "high"

    if row["match_type"] == "1:1" and row["match_confidence"] == "high":
        return "high"

    if row["match_confidence"] in {"high", "medium"}:
        return "medium"

    return "low"

components_gdf["target_height_reliability"] = components_gdf.apply(assign_target_reliability, axis=1)

print("Поля target/source добавлены.")
print(
    components_gdf[
        [
            "target_height",
            "target_height_is_observed",
            "target_height_source",
            "target_height_source_detail",
            "target_height_reliability",
        ]
    ].head()
)

components_gdf = components_gdf.set_geometry("rep_geometry").copy()

# Центроиды храним отдельно как служебную геометрию для поиска соседей
components_centroids = components_gdf.copy()
components_centroids["geometry"] = components_centroids.geometry.centroid

# Пространственный индекс
sindex = components_centroids.sindex

def compute_neighbor_features(gdf_centroids, radii=(50, 100)):
    """
    Считает признаки локального окружения для каждой компоненты:
    - число соседей в радиусе;
    - число соседей с известной высотой;
    - статистики высот соседей.
    """
    rows = []

    geom_arr = gdf_centroids.geometry.values
    height_arr = gdf_centroids["target_height"].values
    comp_id_arr = gdf_centroids["component_id"].values

    for i, geom in enumerate(geom_arr):
        row = {"component_id": comp_id_arr[i]}

        for r in radii:
            # Кандидаты через spatial index по bbox буфера
            bbox = geom.buffer(r).bounds
            candidate_idx = list(sindex.intersection(bbox))

            # Убираем сам объект
            candidate_idx = [j for j in candidate_idx if j != i]

            if not candidate_idx:
                row[f"n_neighbors_{r}m"] = 0
                row[f"n_neighbors_obs_height_{r}m"] = 0
                row[f"neighbor_height_mean_{r}m"] = np.nan
                row[f"neighbor_height_median_{r}m"] = np.nan
                row[f"neighbor_height_min_{r}m"] = np.nan
                row[f"neighbor_height_max_{r}m"] = np.nan
                row[f"neighbor_height_std_{r}m"] = np.nan
                row[f"neighbor_height_q25_{r}m"] = np.nan
                row[f"neighbor_height_q75_{r}m"] = np.nan
                continue

            # Точное расстояние
            dists = gdf_centroids.geometry.iloc[candidate_idx].distance(geom)
            near_idx = [candidate_idx[k] for k, d in enumerate(dists) if d <= r]

            row[f"n_neighbors_{r}m"] = len(near_idx)

            if len(near_idx) == 0:
                row[f"n_neighbors_obs_height_{r}m"] = 0
                row[f"neighbor_height_mean_{r}m"] = np.nan
                row[f"neighbor_height_median_{r}m"] = np.nan
                row[f"neighbor_height_min_{r}m"] = np.nan
                row[f"neighbor_height_max_{r}m"] = np.nan
                row[f"neighbor_height_std_{r}m"] = np.nan
                row[f"neighbor_height_q25_{r}m"] = np.nan
                row[f"neighbor_height_q75_{r}m"] = np.nan
                continue

            neighbor_heights = pd.Series(height_arr[near_idx]).dropna()

            row[f"n_neighbors_obs_height_{r}m"] = len(neighbor_heights)

            if len(neighbor_heights) == 0:
                row[f"neighbor_height_mean_{r}m"] = np.nan
                row[f"neighbor_height_median_{r}m"] = np.nan
                row[f"neighbor_height_min_{r}m"] = np.nan
                row[f"neighbor_height_max_{r}m"] = np.nan
                row[f"neighbor_height_std_{r}m"] = np.nan
                row[f"neighbor_height_q25_{r}m"] = np.nan
                row[f"neighbor_height_q75_{r}m"] = np.nan
            else:
                row[f"neighbor_height_mean_{r}m"] = neighbor_heights.mean()
                row[f"neighbor_height_median_{r}m"] = neighbor_heights.median()
                row[f"neighbor_height_min_{r}m"] = neighbor_heights.min()
                row[f"neighbor_height_max_{r}m"] = neighbor_heights.max()
                row[f"neighbor_height_std_{r}m"] = neighbor_heights.std()
                row[f"neighbor_height_q25_{r}m"] = neighbor_heights.quantile(0.25)
                row[f"neighbor_height_q75_{r}m"] = neighbor_heights.quantile(0.75)

        rows.append(row)

    return pd.DataFrame(rows)

neighbor_features = compute_neighbor_features(components_centroids, radii=(50, 100))

components_gdf = components_gdf.merge(
    neighbor_features,
    on="component_id",
    how="left"
)

print("Соседские признаки добавлены.")
print(
    components_gdf[
        [
            "component_id",
            "n_neighbors_50m",
            "n_neighbors_obs_height_50m",
            "neighbor_height_median_50m",
            "n_neighbors_100m",
            "n_neighbors_obs_height_100m",
            "neighbor_height_median_100m",
        ]
    ].head()
)

merged_buildings = components_gdf.copy()

# Упорядочим основные поля
main_cols = [
    "component_id",
    "match_type",
    "match_confidence",
    "geometry_source",

    "target_height",
    "target_height_is_observed",
    "target_height_source",
    "target_height_source_detail",
    "target_height_reliability",

    "n_a",
    "n_b",
    "uids_a",
    "uids_b",

    "n_edges_ab",
    "max_iou",
    "mean_iou",
    "max_overlap_a",
    "max_overlap_b",
    "min_dist_m",

    "sum_area_a",
    "sum_area_b",
    "union_area_a",
    "union_area_b",
    "union_area_all",

    "n_b_with_height",
    "median_height_b",
    "median_stairs_b",
    "median_avg_floor_height_b",
    "mode_purpose_b",

    "n_neighbors_50m",
    "n_neighbors_obs_height_50m",
    "neighbor_height_mean_50m",
    "neighbor_height_median_50m",
    "neighbor_height_min_50m",
    "neighbor_height_max_50m",
    "neighbor_height_std_50m",
    "neighbor_height_q25_50m",
    "neighbor_height_q75_50m",

    "n_neighbors_100m",
    "n_neighbors_obs_height_100m",
    "neighbor_height_mean_100m",
    "neighbor_height_median_100m",
    "neighbor_height_min_100m",
    "neighbor_height_max_100m",
    "neighbor_height_std_100m",
    "neighbor_height_q25_100m",
    "neighbor_height_q75_100m",

    "rep_geometry",
]

other_cols = [c for c in merged_buildings.columns if c not in main_cols]
merged_buildings = merged_buildings[main_cols + other_cols]

print("Итоговый датасет оформлен.")
print(merged_buildings.shape)
merged_buildings.head()

total_components = len(merged_buildings)
matched_components = (~merged_buildings["match_type"].isin(["A_only", "B_only"])).sum()
a_only_components = (merged_buildings["match_type"] == "A_only").sum()
b_only_components = (merged_buildings["match_type"] == "B_only").sum()
with_height_from_b = merged_buildings["median_height_b"].notna().sum()

print("=" * 70)
print("СВОДКА ПО РЕЗУЛЬТАТАМ MERGE")
print("=" * 70)

print(f"Всего компонент: {total_components:,}")
print(f"Смешанных (matched) компонент: {matched_components:,}")
print(f"Только A: {a_only_components:,}")
print(f"Только B: {b_only_components:,}")
print(f"Компонент с наблюдаемой высотой из B: {with_height_from_b:,}")

print("\nТипы матчей:")
print(merged_buildings["match_type"].value_counts(dropna=False))

print("\nConfidence:")
print(merged_buildings["match_confidence"].value_counts(dropna=False))

print("\nКомпоненты с высотой из B по типам матчей:")
print(
    merged_buildings.assign(has_height=merged_buildings["median_height_b"].notna())
    .groupby("match_type")["has_height"]
    .mean()
    .sort_values(ascending=False)
)

MERGED_PARQUET = OUTPUT_DIR / "merged_buildings_by_geometry.parquet"
MERGED_GPKG = OUTPUT_DIR / "merged_buildings_by_geometry.gpkg"
MERGED_CSV = OUTPUT_DIR / "merged_buildings_by_geometry.csv"

EDGES_CSV = OUTPUT_DIR / "merge_edges_ab.csv"
CANDIDATES_CSV = OUTPUT_DIR / "merge_candidate_pairs_metrics.csv"

# Итоговые merged-сущности
merged_buildings.to_parquet(MERGED_PARQUET)
merged_buildings.to_file(MERGED_GPKG, driver="GPKG")

merged_buildings.drop(columns=["rep_geometry"]).to_csv(MERGED_CSV, index=False)

# Таблицы связей и кандидатов
edges_ab.to_csv(EDGES_CSV, index=False)
candidate_pairs.to_csv(CANDIDATES_CSV, index=False)

print("Файлы сохранены:")
print(MERGED_PARQUET)
print(MERGED_GPKG)
print(MERGED_CSV)
print(EDGES_CSV)
print(CANDIDATES_CSV)