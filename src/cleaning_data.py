import pandas as pd
import geopandas as gpd
from shapely import wkt
from pathlib import Path

DATA_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/interim")
SOURCE_A_PATH = DATA_DIR / "cup_it_example_src_A.csv"
SOURCE_B_PATH = DATA_DIR / "cup_it_example_src_B.csv"

source_A = pd.read_csv(SOURCE_A_PATH, low_memory=False)
source_B = pd.read_csv(SOURCE_B_PATH, low_memory=False)

df_a = source_A.copy()
df_b = source_B.copy()

initial_count_A = len(df_a)
initial_count_B = len(df_b)

print(f"\nИсходные данные:")
print(f"  А: {initial_count_A} записей")
print(f"  Б: {initial_count_B} записей")

# Для датасета А
df_a['status'] = 'clean'           # пока все чистые
df_a['quality_flags'] = ''          # пока пусто
df_a['drop_reason'] = ''            # пока пусто

# Для датасета Б
df_b['status'] = 'clean'
df_b['quality_flags'] = ''
df_b['drop_reason'] = ''

print("Добавлены поля: status, quality_flags, drop_reason")
print(f"А: {df_a.columns.tolist()}")
print(f"B: {df_b.columns.tolist()}")

if 'Unnamed: 0' in df_a.columns:
    df_a = df_a.drop(columns=['Unnamed: 0'])

if 'Unnamed: 0' in df_b.columns:
    df_b = df_b.drop(columns=['Unnamed: 0'])
print(f"А: {df_a.columns.tolist()}")
print(f"B: {df_b.columns.tolist()}")

# словарь для статистики (счетчики)
stats = {
    'source': ['A', 'B'],
    'initial_count': [len(df_a), len(df_b)],           # исходное количество
    'broken_wkt': [0, 0],                              # битый WKT
    'empty_geometry': [0, 0],                         # пустая геометрия
    'repaired_geometry': [0, 0],                      # исправленная геометрия
    'zero_area_dropped': [0, 0],                      # нулевая площадь
    'small_utility_dropped': [0, 0],                  # маленькие объекты
    'duplicate_dropped': [0, 0],                      # дубликаты
    'records_kept': [len(df_a), len(df_b)]            # пока равно initial_count
}
print(pd.DataFrame(stats))

def safe_wkt(x):
    try:
        return wkt.loads(x) if pd.notna(x) else None
    except:
        return None

# Преобразование геометрии
df_a['geometry'] = df_a['geometry'].apply(safe_wkt)
df_b['geometry'] = df_b['wkt'].apply(safe_wkt)

# Считаем, сколько успешно преобразовалось
a_success = df_a['geometry'].notna().sum()
b_success = df_b['geometry'].notna().sum()
print(f"A: успешно {a_success} из {len(df_a)}")
print(f"B: успешно {b_success} из {len(df_b)}")

# Находим битые
a_broken = df_a[df_a['geometry'].isna()].index.tolist()
b_broken = df_b[df_b['geometry'].isna()].index.tolist()
print(f"\nБитых WKT в А: {len(a_broken)}")
print(f"Битых WKT в Б: {len(b_broken)}")

# Обновляем статусы
if len(a_broken) > 0:
    df_a.loc[a_broken, 'status'] = 'dropped'
    df_a.loc[a_broken, 'drop_reason'] = 'broken_wkt'
if len(b_broken) > 0:
    df_b.loc[b_broken, 'status'] = 'dropped'
    df_b.loc[b_broken, 'drop_reason'] = 'broken_wkt'

# Обновляем статистику
stats['broken_wkt'] = [len(a_broken), len(b_broken)]

# Удаляем битые объекты
df_a = df_a[df_a['geometry'].notna()].copy()
df_b = df_b[df_b['geometry'].notna()].copy()
stats['records_kept'] = [len(df_a), len(df_b)]


print("Пример геометрии из А:")
print(df_a['geometry'].iloc[0])
print("\nПример геометрии из Б:")
print(df_b['geometry'].iloc[0])

print("\nОбновленная статитистика")
print(pd.DataFrame(stats))

# Создаём GeoDataFrame
gdf_a = gpd.GeoDataFrame(df_a, geometry='geometry', crs='EPSG:4326')
gdf_b = gpd.GeoDataFrame(df_b, geometry='geometry', crs='EPSG:4326')

# Проверяем валидность и пустоту
a_invalid = ~gdf_a.geometry.is_valid
b_invalid = ~gdf_b.geometry.is_valid
a_empty = gdf_a.geometry.is_empty
b_empty = gdf_b.geometry.is_empty

# Обновляем статистику
stats['empty_geometry'] = [a_empty.sum(), b_empty.sum()]

# Функция для исправления геометрии
def repair_geometry(geom):
    if geom is None or geom.is_valid:
        return geom
    try:
        repaired = geom.buffer(0)
        if repaired.is_valid:
            return repaired
        repaired = make_valid(geom)
        if repaired.is_valid:
            return repaired
    except:
        pass
    return None

# Исправляем невалидные полигоны
repaired_a = 0
if a_invalid.sum() > 0:
    gdf_a.loc[a_invalid, 'geometry'] = gdf_a.loc[a_invalid, 'geometry'].apply(repair_geometry)
    repaired_a = (~gdf_a.loc[a_invalid, 'geometry'].is_valid).sum()
    repaired_a = a_invalid.sum() - repaired_a

repaired_b = 0
if b_invalid.sum() > 0:
    gdf_b.loc[b_invalid, 'geometry'] = gdf_b.loc[b_invalid, 'geometry'].apply(repair_geometry)
    repaired_b = (~gdf_b.loc[b_invalid, 'geometry'].is_valid).sum()
    repaired_b = b_invalid.sum() - repaired_b

stats['repaired_geometry'] = [repaired_a, repaired_b]

a_still_invalid = gdf_a[~gdf_a.geometry.is_valid].index
b_still_invalid = gdf_b[~gdf_b.geometry.is_valid].index

if len(a_still_invalid) > 0:
    gdf_a.loc[a_still_invalid, 'status'] = 'dropped'
    gdf_a.loc[a_still_invalid, 'drop_reason'] = 'invalid_geometry'

if len(b_still_invalid) > 0:
    gdf_b.loc[b_still_invalid, 'status'] = 'dropped'
    gdf_b.loc[b_still_invalid, 'drop_reason'] = 'invalid_geometry'

# Удаляем из датасетов
gdf_a = gdf_a[gdf_a.geometry.is_valid].copy()
gdf_b = gdf_b[gdf_b.geometry.is_valid].copy()

stats['records_kept'] = [len(gdf_a), len(gdf_b)]


print(f"A: осталось {len(gdf_a)} записей")
print(f"B: осталось {len(gdf_b)} записей")

print("Статистико")
print(pd.DataFrame(stats))

def check_invalid_holes_detailed(gdf, name):
    bad_polygons = []
    bad_holes_count = 0

    for idx, geom in gdf.geometry.items():
        if geom is None or geom.geom_type != "Polygon":
            continue
        exterior = geom.exterior
        poly_has_bad = False
        for interior in geom.interiors:
            if not interior.is_ring or interior.crosses(exterior):
                bad_holes_count += 1
                poly_has_bad = True
        if poly_has_bad:
            bad_polygons.append(idx)

    return bad_polygons, bad_holes_count

# Находим объекты с вырожденными отверстиями
a_bad_polygons, a_bad_holes = check_invalid_holes_detailed(gdf_a, "A")
b_bad_polygons, b_bad_holes = check_invalid_holes_detailed(gdf_b, "B")

print(f"A: полигонов с вырожденными отверстиями: {len(a_bad_polygons)}")
print(f"A: всего вырожденных отверстий: {a_bad_holes}")
print(f"B: полигонов с вырожденными отверстиями: {len(b_bad_polygons)}")
print(f"B: всего вырожденных отверстий: {b_bad_holes}")

# Помечаем и удаляем полигоны
if len(a_bad_polygons) > 0:
    gdf_a.loc[a_bad_polygons, 'status'] = 'dropped'
    gdf_a.loc[a_bad_polygons, 'drop_reason'] = 'invalid_holes'
    gdf_a = gdf_a.drop(index=a_bad_polygons).copy()
    print(f"\nУдалено из А: {len(a_bad_polygons)} объектов")

if len(b_bad_polygons) > 0:
    gdf_b.loc[b_bad_polygons, 'status'] = 'dropped'
    gdf_b.loc[b_bad_polygons, 'drop_reason'] = 'invalid_holes'
    gdf_b = gdf_b.drop(index=b_bad_polygons).copy()
    print(f"\nУдалено из Б: {len(b_bad_polygons)} объектов")

# Обновляем статистику
stats['records_kept'] = [len(gdf_a), len(gdf_b)]


print(f"A: {len(gdf_a)} записей")
print(f"B: {len(gdf_b)} записей")
print(pd.DataFrame(stats))

gdf_a = gdf_a.to_crs('EPSG:3857')
gdf_b = gdf_b.to_crs('EPSG:3857')

gdf_a['geom_area_m2'] = gdf_a.geometry.area
gdf_b['geom_area_m2'] = gdf_b.geometry.area

print(f"A: {len(gdf_a)} записей, площадь от {gdf_a['geom_area_m2'].min():.1f} до {gdf_a['geom_area_m2'].max():.1f} м²")
print(f"B: {len(gdf_b)} записей, площадь от {gdf_b['geom_area_m2'].min():.1f} до {gdf_b['geom_area_m2'].max():.1f} м²")

print(pd.DataFrame(stats))

# Проверка текущего количества
before_a = len(gdf_a)
before_b = len(gdf_b)
print(f"A: {before_a} записей")
print(f"B: {before_b} записей")

small_a = gdf_a[gdf_a['geom_area_m2'] < 15]
small_b = gdf_b[gdf_b['geom_area_m2'] < 15]
print(f"A: объектов < 15 м²: {len(small_a)}")
print(f"B: объектов < 15 м²: {len(small_b)}")

# Помечаем удаляемые объекты
gdf_a.loc[small_a.index, 'status'] = 'dropped'
gdf_a.loc[small_a.index, 'drop_reason'] = 'small_utility_object'

gdf_b.loc[small_b.index, 'status'] = 'dropped'
gdf_b.loc[small_b.index, 'drop_reason'] = 'small_utility_object'

# Удаляем
gdf_a = gdf_a[gdf_a['geom_area_m2'] >= 15].copy()
gdf_b = gdf_b[gdf_b['geom_area_m2'] >= 15].copy()


print(f"A: было {before_a}, стало {len(gdf_a)} (удалено {before_a - len(gdf_a)})")
print(f"B: было {before_b}, стало {len(gdf_b)} (удалено {before_b - len(gdf_b)})")

# Обновляем статистику
stats['small_utility_dropped'] = [before_a - len(gdf_a), before_b - len(gdf_b)]
stats['records_kept'] = [len(gdf_a), len(gdf_b)]

# Проверка минимальной площади после удаления
print(f"A: мин. площадь = {gdf_a['geom_area_m2'].min():.1f} м²")
print(f"B: мин. площадь = {gdf_b['geom_area_m2'].min():.1f} м²")

print(pd.DataFrame(stats))

# Поиск записей с min > max
mask = (gdf_a['gkh_floor_count_min'].notna() &
        gdf_a['gkh_floor_count_max'].notna() &
        (gdf_a['gkh_floor_count_min'] > gdf_a['gkh_floor_count_max']))

if mask.sum() > 0:
    # Сохраняем индексы исправленных записей
    repaired_indices = gdf_a[mask].index.tolist()

    # Меняем местами
    gdf_a.loc[mask, ['gkh_floor_count_min', 'gkh_floor_count_max']] = \
        gdf_a.loc[mask, ['gkh_floor_count_max', 'gkh_floor_count_min']].values

    # Обновляем статус для исправленных объектов
    gdf_a.loc[repaired_indices, 'status'] = 'repaired'
    gdf_a.loc[repaired_indices, 'quality_flags'] = gdf_a.loc[repaired_indices, 'quality_flags'] + 'swapped_min_max;'


    # Проверяем, что ошибок больше нет
    mask_after = (gdf_a['gkh_floor_count_min'].notna() &
                  gdf_a['gkh_floor_count_max'].notna() &
                  (gdf_a['gkh_floor_count_min'] > gdf_a['gkh_floor_count_max']))


# Обновляем статистику (считаем исправленные)
stats['repaired_count'] = [len(repaired_indices) if mask.sum() > 0 else 0, 0]


print(f"A: после исправления")

print(pd.DataFrame(stats))

# Поиск записей с avg_floor_height = 0
zero_mask = (gdf_b['avg_floor_height'] == 0)

if zero_mask.sum() > 0:
    median_by_purpose = gdf_b.groupby('purpose_of_building')['avg_floor_height'].median()
    # Заполняем медианой по типу здания
    repaired_indices = []

    for idx in gdf_b[zero_mask].index:
        purpose = gdf_b.loc[idx, 'purpose_of_building']
        if pd.notna(purpose) and purpose in median_by_purpose.index:
            new_value = median_by_purpose[purpose]
        else:
            new_value = gdf_b['avg_floor_height'].median()

        gdf_b.loc[idx, 'avg_floor_height'] = new_value
        repaired_indices.append(idx)

    # Обновляем статус для исправленных объектов
    gdf_b.loc[repaired_indices, 'status'] = 'repaired'
    gdf_b.loc[repaired_indices, 'quality_flags'] = gdf_b.loc[repaired_indices, 'quality_flags'] + 'filled_avg_floor_height;'


    # Проверяем, что нулей не осталось
    zero_mask_after = (gdf_b['avg_floor_height'] == 0)
    print(f"Осталось нулей: {zero_mask_after.sum()}")
else:
    print("Нулевых значений не найдено")

# Обновляем статистику (считаем исправленные в Б)
stats['repaired_count'] = [stats['repaired_count'][0], len(repaired_indices) if zero_mask.sum() > 0 else 0]


print(gdf_b['avg_floor_height'].describe())

print(pd.DataFrame(stats))

# Находим выбросы
stairs_bad = (gdf_b['stairs'] <= 0) & (gdf_b['stairs'].notna())
height_bad = (gdf_b['avg_floor_height'] > 10) & (gdf_b['avg_floor_height'].notna())
height_null = gdf_b['height'].isna()
bad_mask = stairs_bad | height_bad | height_null

# Помечаем удаляемые
if bad_mask.sum() > 0:
    gdf_b.loc[bad_mask, 'status'] = 'dropped'
    gdf_b.loc[stairs_bad & ~height_bad & ~height_null, 'drop_reason'] = 'stairs_invalid'
    gdf_b.loc[height_bad & ~stairs_bad & ~height_null, 'drop_reason'] = 'avg_floor_height_too_high'
    gdf_b.loc[height_null, 'drop_reason'] = 'missing_height'
    gdf_b.loc[stairs_bad & height_null, 'drop_reason'] = 'stairs_invalid;missing_height'
    gdf_b.loc[height_bad & height_null, 'drop_reason'] = 'avg_floor_height_too_high;missing_height'

    before = len(gdf_b)
    gdf_b = gdf_b[~bad_mask].copy()
    stats['records_kept'] = [len(gdf_a), len(gdf_b)]


print(f"A: {len(gdf_a)} записей")
print(f"B: {len(gdf_b)} записей")
print()
print(pd.DataFrame(stats))

# Создаём копию для проверки
b_check = gdf_b.copy()
mask = b_check['stairs'].notna() & b_check['height'].notna() & b_check['avg_floor_height'].notna()

# Считаем метрики
b_check.loc[mask, 'height_per_floor'] = b_check.loc[mask, 'height'] / b_check.loc[mask, 'stairs']
b_check.loc[mask, 'height_error_abs'] = abs(
    b_check.loc[mask, 'height'] - b_check.loc[mask, 'stairs'] * b_check.loc[mask, 'avg_floor_height']
)

# Находим аномалии
low_floor = (b_check['height_per_floor'] < 2).sum()
high_floor = (b_check['height_per_floor'] > 6).sum()
large_error = (b_check['height_error_abs'] > 10).sum()


print(f"Всего записей в Б: {len(gdf_b)}")
print(f"height_per_floor < 2: {low_floor}")
print(f"height_per_floor > 6: {high_floor}")
print(f"height_error_abs > 10: {large_error}")
print(f"\nАномалии составляют {large_error/len(gdf_b)*100:.3f}% от всех данных")

print(pd.DataFrame(stats)) # аномалий оч мало, можно оставить

# Проверка полных дубликатов (по WKT)

# Преобразуем геометрию в WKT-строки для сравнения
wkt_a = gdf_a.geometry.apply(lambda g: g.wkt if g is not None else None)
wkt_b = gdf_b.geometry.apply(lambda g: g.wkt if g is not None else None)

full_dup_a = wkt_a.duplicated().sum()
full_dup_b = wkt_b.duplicated().sum()

print(f"A: полных дубликатов: {full_dup_a}")
print(f"B: полных дубликатов: {full_dup_b}")

# Если есть полные дубликаты — удаляем
if full_dup_a > 0:
    dup_mask_a = wkt_a.duplicated(keep='first')
    gdf_a.loc[dup_mask_a, 'status'] = 'dropped'
    gdf_a.loc[dup_mask_a, 'drop_reason'] = 'duplicate'
    gdf_a = gdf_a[~dup_mask_a].copy()
    print(f"Удалено полных дубликатов из А: {full_dup_a}")

if full_dup_b > 0:
    dup_mask_b = wkt_b.duplicated(keep='first')
    gdf_b.loc[dup_mask_b, 'status'] = 'dropped'
    gdf_b.loc[dup_mask_b, 'drop_reason'] = 'duplicate'
    gdf_b = gdf_b[~dup_mask_b].copy()
    print(f"Удалено полных дубликатов из Б: {full_dup_b}")

# Почти-дубликаты не удаляем, так как они могут быть частями одного здания (например, дом с пристройкой, комплекс зданий).
# При сопоставлении они объединятся в компоненты связности.


print(f"A: {len(gdf_a)} записей")
print(f"B: {len(gdf_b)} записей")
print(f"Почти-дубликаты в А (46 объектов) оставлены с пометкой для последующего объединения")


# Обновляем статистику
stats['duplicate_dropped'] = [full_dup_a, full_dup_b]
stats['records_kept'] = [len(gdf_a), len(gdf_b)]

print(pd.DataFrame(stats))

# Заполнение purpose_of_building
print("\n--- 1. purpose_of_building ---")
before_purpose = gdf_b['purpose_of_building'].isna().sum()
print(f"Пропусков: {before_purpose}")

if before_purpose > 0:
    gdf_b.loc[gdf_b['purpose_of_building'].isna(), 'purpose_of_building'] = 'unknown'
    gdf_b.loc[gdf_b['purpose_of_building'].isna(), 'status'] = 'repaired'
    gdf_b.loc[gdf_b['purpose_of_building'].isna(), 'quality_flags'] = 'filled_purpose;'
    print(f"Заполнено {before_purpose} пропусков значением 'unknown'")

after_purpose = gdf_b['purpose_of_building'].isna().sum()
print(f"Пропусков после заполнения: {after_purpose}")

# Заполнение stairs медианой по типу здания
print("\n--- 2. stairs (медиана по purpose_of_building) ---")
before_stairs = gdf_b['stairs'].isna().sum()
print(f"Пропусков stairs до заполнения: {before_stairs}")

if before_stairs > 0:
    # Считаем медиану для каждого типа здания
    median_by_purpose = gdf_b.groupby('purpose_of_building')['stairs'].median()

    # Создаём маску пропусков
    mask_stairs = gdf_b['stairs'].isna()
    repaired_indices = gdf_b[mask_stairs].index.tolist()

    # Заполняем медианой по типу здания
    for idx in repaired_indices:
        purpose = gdf_b.loc[idx, 'purpose_of_building']
        if pd.notna(purpose) and purpose in median_by_purpose.index:
            new_value = median_by_purpose[purpose]
        else:
            new_value = gdf_b['stairs'].median()
        gdf_b.loc[idx, 'stairs'] = new_value

    # Обновляем статус
    gdf_b.loc[repaired_indices, 'status'] = 'repaired'
    gdf_b.loc[repaired_indices, 'quality_flags'] = gdf_b.loc[repaired_indices, 'quality_flags'] + 'filled_stairs;'

    print(f"Заполнено {before_stairs} пропусков stairs")

after_stairs = gdf_b['stairs'].isna().sum()
print(f"Пропусков stairs после заполнения: {after_stairs}")

print(f"B: purpose_of_building пропусков: {gdf_b['purpose_of_building'].isna().sum()}")
print(f"B: stairs пропусков: {gdf_b['stairs'].isna().sum()}")
print(f"B: unique purpose_of_building: {gdf_b['purpose_of_building'].nunique()}")
print(f"  из них 'unknown': {(gdf_b['purpose_of_building'] == 'unknown').sum()}")

# Обновляем статистику
# repaired_count уже был 4258, добавляем заполненные
stats['repaired_count'][1] = stats['repaired_count'][1] + before_purpose + before_stairs

print(pd.DataFrame(stats))

# Добавляем колонку
gdf_a['is_relevant_building'] = True
gdf_b['is_relevant_building'] = True

# Объекты с status = dropped
dropped_a = gdf_a[gdf_a['status'] == 'dropped']
dropped_b = gdf_b[gdf_b['status'] == 'dropped']
if len(dropped_a) > 0:
    gdf_a.loc[dropped_a.index, 'is_relevant_building'] = False
if len(dropped_b) > 0:
    gdf_b.loc[dropped_b.index, 'is_relevant_building'] = False

# Объекты, не являющиеся зданиями (теплицы)
greenhouses = gdf_b[gdf_b['purpose_of_building'] == 'Парники, оранжереи, теплицы']
if len(greenhouses) > 0:
    gdf_b.loc[greenhouses.index, 'is_relevant_building'] = False


print(f"A: релевантных зданий: {gdf_a['is_relevant_building'].sum()} / {len(gdf_a)}")
print(f"B: релевантных зданий: {gdf_b['is_relevant_building'].sum()} / {len(gdf_b)}")

print(pd.DataFrame(stats))

print("="*70)
print("ИТОГОВАЯ СТАТИСТИКА ПОСЛЕ ВСЕХ ЭТАПОВ ОЧИСТКИ")
print("="*70)

print("\n--- 1. Количество записей ---")
print(f"A: {len(gdf_a)} записей (было 171454, удалено {171454 - len(gdf_a)})")
print(f"B: {len(gdf_b)} записей (было 161076, удалено {161076 - len(gdf_b)})")

print("\n--- 2. Статусы объектов ---")
print("\nA:")
print(gdf_a['status'].value_counts())
print("\nB:")
print(gdf_b['status'].value_counts())

print("\n--- 3. Пропуски в ключевых полях (которые будем использовать) ---")
print("\nA (пропуски):")
a_cols = ['tags', 'geom_area_m2']
for col in a_cols:
    miss = gdf_a[col].isna().sum()
    print(f"  {col}: {miss} ({miss/len(gdf_a)*100:.2f}%)")

print("\nB (пропуски):")
b_cols = ['purpose_of_building', 'stairs', 'height', 'avg_floor_height', 'geom_area_m2']
for col in b_cols:
    miss = gdf_b[col].isna().sum()
    print(f"  {col}: {miss} ({miss/len(gdf_b)*100:.2f}%)")

print("\n--- 4. Пропуски в полях, которые НЕ используем (для информации) ---")
print("\nA (не используем):")
a_skip = ['title', 'gkh_address', 'gkh_floor_count_min', 'gkh_floor_count_max']
for col in a_skip:
    if col in gdf_a.columns:
        miss = gdf_a[col].isna().sum()
        print(f"  {col}: {miss} ({miss/len(gdf_a)*100:.2f}%)")

print("\nB (не используем):")
b_skip = ['type_street', 'name_street', 'letter', 'fraction', 'housing', 'building', 'subject', 'district']
for col in b_skip:
    if col in gdf_b.columns:
        miss = gdf_b[col].isna().sum()
        print(f"  {col}: {miss} ({miss/len(gdf_b)*100:.2f}%)")

print("\n--- 5. Категориальные признаки ---")
print(f"A: уникальных tags: {gdf_a['tags'].nunique()}")
print(f"B: уникальных purpose_of_building: {gdf_b['purpose_of_building'].nunique()}")
print(f"   из них 'unknown': {(gdf_b['purpose_of_building'] == 'unknown').sum()}")

print("\n--- 6. Числовые признаки (статистика) ---")
print("\nA (geom_area_m2):")
print(gdf_a['geom_area_m2'].describe())

print("\nB (height):")
print(gdf_b['height'].describe())

print("\nB (stairs):")
print(gdf_b['stairs'].describe())

print("\n--- 7. is_relevant_building ---")
print(f"A: релевантных: {gdf_a['is_relevant_building'].sum()} / {len(gdf_a)}")
print(f"B: релевантных: {gdf_b['is_relevant_building'].sum()} / {len(gdf_b)}")


print(pd.DataFrame(stats))

summary = pd.DataFrame({
    'Показатель': [
        'Исходное количество записей',
        'Удалено (маленькие объекты <15 м²)',
        'Удалено (stairs <= 0)',
        'Удалено (avg_floor_height > 10)',
        'Удалено (missing height)',
        'Всего удалено',
        'Исправлено (min > max в А)',
        'Исправлено (avg_floor_height = 0)',
        'Исправлено (purpose_of_building)',
        'Исправлено (stairs)',
        'Всего исправлено',
        'Осталось записей',
        'Релевантных зданий'
    ],
    'А': [
        stats['initial_count'][0],
        stats['small_utility_dropped'][0],
        0,
        0,
        0,
        stats['small_utility_dropped'][0],
        stats['repaired_count'][0],
        0,
        0,
        0,
        stats['repaired_count'][0],
        stats['records_kept'][0],
        gdf_a['is_relevant_building'].sum()
    ],
    'Б': [
        stats['initial_count'][1],
        stats['small_utility_dropped'][1],
        17,
        2,
        66,
        stats['small_utility_dropped'][1] + 17 + 2 + 66,
        0,
        4258,
        3964,
        3935,
        stats['repaired_count'][1],
        stats['records_kept'][1],
        gdf_b['is_relevant_building'].sum()
    ]
})

print(summary.to_string(index=False))

# GeoPackage
gdf_a.to_file(OUTPUT_DIR / "A_clean_final1.gpkg", driver="GPKG")
gdf_b.to_file(OUTPUT_DIR / "B_clean_final1.gpkg", driver="GPKG")

# Parquet
gdf_a.to_parquet(OUTPUT_DIR / "A_clean_final1.parquet")
gdf_b.to_parquet(OUTPUT_DIR / "B_clean_final1.parquet")

# CSV без геометрии
gdf_a.drop(columns=["geometry"]).to_csv(
    OUTPUT_DIR / "A_clean_final1.csv", index=False
)
gdf_b.drop(columns=["geometry"]).to_csv(
    OUTPUT_DIR / "B_clean_final1.csv", index=False
)

# Сохраняем статистику
summary.to_csv(OUTPUT_DIR / 'cleaning_statistics.csv', index=False)

print(f"A: {len(gdf_a)} записей, колонки: {list(gdf_a.columns)}")
print(f"B: {len(gdf_b)} записей, колонки: {list(gdf_b.columns)}")
