from pathlib import Path

import numpy as np
import pandas as pd

COL_DATE = "date"
COL_GRID = "grid_id"
COL_FIRE = "fire_occurrence"
COL_COUNT = "hotspot_count"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# 분석 범위를 먼저 격자로 잘라서 쓰기 위해 생성
def create_grid(bbox, size):
    west, south, east, north = bbox
    lons = np.arange(west, east, size)
    lats = np.arange(south, north, size)

    rows = []
    idx = 0
    for iy, lat in enumerate(lats):
        for ix, lon in enumerate(lons):
            rows.append({
                COL_GRID: f"G{idx:06d}",
                "ix": ix,
                "iy": iy,
                "lon_min": lon,
                "lon_max": min(lon + size, east),
                "lat_min": lat,
                "lat_max": min(lat + size, north),
                "lon_center": lon + size / 2,
                "lat_center": lat + size / 2,
            })
            idx += 1
    return pd.DataFrame(rows)


def build_base(grid, start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq="D")
    return pd.MultiIndex.from_product([dates, grid[COL_GRID]], names=[COL_DATE, COL_GRID]).to_frame(index=False)


def point_to_grid(df, bbox, size, lon_col, lat_col):
    west, south, east, north = bbox
    nx = int(np.ceil((east - west) / size))

    ok = (
        df[lon_col].between(west, east, inclusive="left")
        & df[lat_col].between(south, north, inclusive="left")
    )
    df = df.loc[ok].copy()
    if df.empty:
        return df

    ix = np.floor((df[lon_col] - west) / size).astype(int)
    iy = np.floor((df[lat_col] - south) / size).astype(int)
    df[COL_GRID] = (iy * nx + ix).map(lambda x: f"G{x:06d}")
    return df


def read_firms(folder, bbox, size):
    # 화재 강도보다 "그 날짜-격자에서 감지가 있었는지"를 먼저 보는 쪽으로 정리했다.
    # TODO: confidence 낮은 감지 나중에 걸러내기
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        print("FIRMS 파일 없음")
        return pd.DataFrame(columns=[COL_DATE, COL_GRID, COL_COUNT, COL_FIRE])

    parts = []
    for fp in files:
        df = pd.read_csv(fp)
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("acq_date") or cols.get("date")
        lat_col = cols.get("latitude")
        lon_col = cols.get("longitude")
        if not all([date_col, lat_col, lon_col]):
            continue

        out = df[[date_col, lat_col, lon_col]].copy()
        out.columns = [COL_DATE, "latitude", "longitude"]
        out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.normalize()
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
        out = out.dropna()
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=[COL_DATE, COL_GRID, COL_COUNT, COL_FIRE])

    fire = point_to_grid(pd.concat(parts, ignore_index=True), bbox, size, "longitude", "latitude")
    if fire.empty:
        return pd.DataFrame(columns=[COL_DATE, COL_GRID, COL_COUNT, COL_FIRE])

    fire = fire.groupby([COL_DATE, COL_GRID]).size().reset_index(name=COL_COUNT)
    fire[COL_FIRE] = (fire[COL_COUNT] > 0).astype(int)
    return fire


def _var_name(ds, names):
    lower = {k.lower(): k for k in ds.data_vars}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    for key in ds.data_vars:
        if any(name.lower() in key.lower() for name in names):
            return key
    return None


def read_era(folder, grid, bbox, windows):
    try:
        import xarray as xr
    except Exception as e:
        raise ImportError("xarray와 netcdf4가 필요") from e

    files = sorted(Path(folder).glob("*.nc"))
    if not files:
        print("ERA5 파일 없음")
        return pd.DataFrame()

    pts = grid[[COL_GRID, "lon_center", "lat_center"]].copy()
    frames = []

    for fp in files:
        ds = xr.open_dataset(fp)

        lat = next((n for n in ["latitude", "lat"] if n in ds.coords or n in ds.variables), None)
        lon = next((n for n in ["longitude", "lon"] if n in ds.coords or n in ds.variables), None)
        tim = next((n for n in ["time", "valid_time"] if n in ds.coords or n in ds.variables), None)  # valid_time은 최신 ERA5에서 바뀐 이름
        if not all([lat, lon, tim]):
            continue

        west, south, east, north = bbox
        lat_vals = ds[lat].values
        lon_vals = ds[lon].values
        lat_slice = slice(north, south) if lat_vals[0] > lat_vals[-1] else slice(south, north)
        lon_slice = slice(east, west) if lon_vals[0] > lon_vals[-1] else slice(west, east)
        ds = ds.sel({lat: lat_slice, lon: lon_slice})

        # 온도, 이슬점, 풍속, 강수, 토양수분 정도만 먼저 사용했다.
        vars_map = {
            "temp": _var_name(ds, ["t2m", "2m_temperature"]),
            "dewpoint": _var_name(ds, ["d2m", "2m_dewpoint_temperature"]),
            "u10": _var_name(ds, ["u10", "10m_u_component_of_wind"]),
            "v10": _var_name(ds, ["v10", "10m_v_component_of_wind"]),
            "precip": _var_name(ds, ["tp", "total_precipitation"]),
            "soil_moisture": _var_name(ds, ["swvl1", "volumetric_soil_water_layer_1"]),
        }
        vars_map = {k: v for k, v in vars_map.items() if v}
        if not vars_map:
            continue

        dates = pd.to_datetime(ds[tim].values).normalize()
        idx = pd.MultiIndex.from_product([dates, pts[COL_GRID]], names=[COL_DATE, COL_GRID])
        out = pd.DataFrame(index=idx).reset_index()

        lat_idx = xr.DataArray(pts["lat_center"].values, dims="point")
        lon_idx = xr.DataArray(pts["lon_center"].values, dims="point")

        for key, value in vars_map.items():
            arr = ds[value].sel({lat: lat_idx, lon: lon_idx}, method="nearest").values
            out[key] = arr.reshape(-1)

        # 단위 확인 안 하면 온도가 300도 넘게 나옴
        if "temp" in out and out["temp"].dropna().median() > 100:
            out["temp"] = out["temp"] - 273.15
        if "dewpoint" in out and out["dewpoint"].dropna().median() > 100:
            out["dewpoint"] = out["dewpoint"] - 273.15
        if "precip" in out and out["precip"].dropna().max() < 10:
            out["precip"] = out["precip"] * 1000.0
        if {"u10", "v10"}.issubset(out.columns):
            out["wind_speed"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)

        agg = {c: ("sum" if c == "precip" else "mean") for c in out.columns if c not in [COL_DATE, COL_GRID]}
        out = out.groupby([COL_DATE, COL_GRID], as_index=False).agg(agg)
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    era = pd.concat(frames, ignore_index=True)
    era = era.groupby([COL_DATE, COL_GRID], as_index=False).mean(numeric_only=True)
    era = era.sort_values([COL_GRID, COL_DATE]).reset_index(drop=True)

    use_cols = [c for c in ["temp", "dewpoint", "wind_speed", "soil_moisture", "precip"] if c in era.columns]
    for col in use_cols:
        for w in windows:
            if w <= 1:
                continue
            if col == "precip":
                name = f"{col}_rollsum_{w}"
                era[name] = era.groupby(COL_GRID)[col].transform(lambda s: s.rolling(w, min_periods=1).sum())
            else:
                name = f"{col}_rollmean_{w}"
                era[name] = era.groupby(COL_GRID)[col].transform(lambda s: s.rolling(w, min_periods=1).mean())

    return era


def _sample_array(arr, transform, xs, ys):
    inv = ~transform
    cols, rows = inv * (xs, ys)
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)

    ok = (
        (rows >= 0) & (rows < arr.shape[0]) &
        (cols >= 0) & (cols < arr.shape[1])
    )
    out = np.full(xs.shape[0], np.nan, dtype=float)
    out[ok] = arr[rows[ok], cols[ok]]
    return out


def read_dem(folder, grid):
    try:
        import rasterio
    except Exception as e:
        raise ImportError("rasterio가 필요") from e

    files = sorted(list(Path(folder).glob("*.hgt")) + list(Path(folder).glob("*.tif")))
    if not files:
        print("DEM 파일 없음")
        return grid[[COL_GRID]].copy()

    xs = grid["lon_center"].to_numpy()
    ys = grid["lat_center"].to_numpy()
    elev = np.full(len(grid), np.nan)
    slope = np.full(len(grid), np.nan)
    aspect = np.full(len(grid), np.nan)

    for fp in files:
        with rasterio.open(fp) as src:
            need = np.isnan(elev)
            b = src.bounds
            pick = need & (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
            if not pick.any():
                continue

            arr = src.read(1).astype("float32")
            xres = abs(src.transform.a)
            yres = abs(src.transform.e)
            gy, gx = np.gradient(arr, yres, xres)

            slp = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
            asp = np.degrees(np.arctan2(-gx, gy))
            asp = np.where(asp < 0, 90.0 - asp, 450.0 - asp)
            asp = np.mod(asp, 360.0)

            idx = np.where(pick)[0]
            elev[idx] = _sample_array(arr, src.transform, xs[idx], ys[idx])
            slope[idx] = _sample_array(slp, src.transform, xs[idx], ys[idx])
            aspect[idx] = _sample_array(asp, src.transform, xs[idx], ys[idx])

    out = grid[[COL_GRID]].copy()
    out["elevation"] = elev
    out["slope"] = slope
    out["aspect"] = aspect
    return out


def read_worldcover(folder, grid):
    try:
        import rasterio
    except Exception as e:
        raise ImportError("rasterio가 필요") from e

    files = sorted(Path(folder).glob("*Map.tif")) or sorted(Path(folder).glob("*.tif"))
    if not files:
        print("WorldCover 파일 없음")
        return grid[[COL_GRID]].copy()

    xs = grid["lon_center"].to_numpy()
    ys = grid["lat_center"].to_numpy()
    land = np.full(len(grid), np.nan)

    for fp in files:
        with rasterio.open(fp) as src:
            need = np.isnan(land)
            b = src.bounds
            pick = need & (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
            if not pick.any():
                continue
            vals = np.array([v[0] for v in src.sample(list(zip(xs[pick], ys[pick])))], dtype=float)
            land[np.where(pick)[0]] = vals

    out = grid[[COL_GRID]].copy()
    out["landcover"] = land
    return out


def build_dataset(cfg):
    p = cfg["paths"]
    d = cfg["data"]
    windows = cfg.get("features", {}).get("weather_roll_windows", [1, 3])

    out_dir = ensure_dir(p["processed_dir"])

    print("격자 범위 생성")
    grid = create_grid(d["bbox"], float(d["grid_size"]))
    base = build_base(grid, d["start_date"], d["end_date"])
    print(f"  {len(grid)}개 격자 / {d['start_date']} ~ {d['end_date']}")

    print("FIRMS 날짜-격자 정리")
    fire = read_firms(p["firms_dir"], d["bbox"], float(d["grid_size"]))
    print("ERA5 기상 변수 추가")
    era = read_era(p["era5_dir"], grid, d["bbox"], windows)
    print("DEM 지형 정보 읽기")
    dem = read_dem(p["dem_dir"], grid)
    print("WorldCover 토지피복 추가")
    wc = read_worldcover(p["worldcover_dir"], grid)

    print("전체 병합")
    data = base.merge(fire, on=[COL_DATE, COL_GRID], how="left")
    data[COL_COUNT] = data[COL_COUNT].fillna(0).astype(int)
    data[COL_FIRE] = data[COL_FIRE].fillna(0).astype(int)

    if not era.empty:
        data = data.merge(era, on=[COL_DATE, COL_GRID], how="left")
    if not dem.empty:
        data = data.merge(dem, on=COL_GRID, how="left")
    if not wc.empty:
        data = data.merge(wc, on=COL_GRID, how="left")

    grid.to_csv(out_dir / "grid_metadata.csv", index=False, encoding="utf-8-sig")
    path = out_dir / "wildfire_dataset.csv"
    data.sort_values([COL_DATE, COL_GRID]).to_csv(path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {path}")
    print(f"전체 행: {len(data):,} | 화재 발생: {int(data[COL_FIRE].sum()):,}")
    return str(path)


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        build_dataset(yaml.safe_load(f))
