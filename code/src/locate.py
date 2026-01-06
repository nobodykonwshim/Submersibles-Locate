from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .visualize import create_ocean_visualizations, create_visualizations


def _ensure_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    if "q" not in df.columns:
        df["q"] = config.get("default_quality", 1.0)
    if "depth_m" not in df.columns:
        df["depth_m"] = np.nan
    if "z_obs_km" not in df.columns:
        df["z_obs_km"] = -df["depth_m"].to_numpy(dtype=float) / 1000.0
    df["z_obs_km"] = np.nan_to_num(df["z_obs_km"], nan=0.0)
    return df


def _sample_ocean_at(x_km: float, y_km: float, ocean_state: Dict) -> Dict:
    x_grid = ocean_state["x_grid_km"]
    y_grid = ocean_state["y_grid_km"]

    x_clipped = float(np.clip(x_km, x_grid.min(), x_grid.max()))
    y_clipped = float(np.clip(y_km, y_grid.min(), y_grid.max()))

    def _bilinear(field: np.ndarray) -> float:
        ix = int(np.searchsorted(x_grid, x_clipped) - 1)
        iy = int(np.searchsorted(y_grid, y_clipped) - 1)
        ix = np.clip(ix, 0, len(x_grid) - 2)
        iy = np.clip(iy, 0, len(y_grid) - 2)

        x0, x1 = x_grid[ix], x_grid[ix + 1]
        y0, y1 = y_grid[iy], y_grid[iy + 1]
        tx = 0.0 if x1 == x0 else (x_clipped - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y_clipped - y0) / (y1 - y0)

        f00 = field[iy, ix]
        f01 = field[iy, ix + 1]
        f10 = field[iy + 1, ix]
        f11 = field[iy + 1, ix + 1]
        return float((f00 * (1 - tx) + f01 * tx) * (1 - ty) + (f10 * (1 - tx) + f11 * tx) * ty)

    return {
        "density": _bilinear(ocean_state["density"]),
        "current_u_kmh": _bilinear(ocean_state["current_u_kmh"]),
        "current_v_kmh": _bilinear(ocean_state["current_v_kmh"]),
        "bathymetry_m": _bilinear(ocean_state["bathymetry_m"]),
    }


def _compute_smoothed_estimates(df: pd.DataFrame, config: Dict, ocean_state: Dict) -> pd.DataFrame:
    alpha = config.get("smoothing_alpha", 0.35)
    res_alpha = config.get("residual_ewma_alpha", 0.25)
    base_density = config.get("ocean_density_base", 1025.0)
    density_span = max(config.get("ocean_density_variation", 3.0), 1e-6)

    x_hat: List[float] = []
    y_hat: List[float] = []
    z_hat: List[float] = []
    var_x: List[float] = []
    var_y: List[float] = []
    var_z: List[float] = []
    cov_xy: List[float] = []

    prev_x, prev_y, prev_z = (
        df.loc[0, "x_obs_km"],
        df.loc[0, "y_obs_km"],
        df.loc[0, "z_obs_km"],
    )
    prev_t = df.loc[0, "t_hours"]
    prev_vx = prev_vy = prev_vz = 0.05 ** 2
    prev_cxy = 0.0

    for _, row in df.iterrows():
        q = float(row.get("q", 1.0))
        dt_hours = float(row["t_hours"] - prev_t)
        dt_hours = max(dt_hours, 0.0)

        ocean_at_pos = _sample_ocean_at(prev_x, prev_y, ocean_state)
        density_factor = np.clip(abs(ocean_at_pos["density"] - base_density) / density_span, 0, 2.5)
        stability = np.clip(1.0 - 0.2 * density_factor, 0.2, 1.0)

        adv_x = prev_x + ocean_at_pos["current_u_kmh"] * dt_hours * stability
        adv_y = prev_y + ocean_at_pos["current_v_kmh"] * dt_hours * stability

        depth_target_km = -ocean_at_pos["bathymetry_m"] / 1000.0
        adv_z = prev_z + 0.05 * (depth_target_km - prev_z) * stability

        weight = alpha * q * stability
        weight = np.clip(weight, 0.05, 0.95)

        x_new = weight * row["x_obs_km"] + (1 - weight) * adv_x
        y_new = weight * row["y_obs_km"] + (1 - weight) * adv_y
        z_new = weight * row["z_obs_km"] + (1 - weight) * adv_z

        resid_x = row["x_obs_km"] - x_new
        resid_y = row["y_obs_km"] - y_new
        resid_z = row["z_obs_km"] - z_new

        var_x_new = res_alpha * (resid_x ** 2) + (1 - res_alpha) * prev_vx
        var_y_new = res_alpha * (resid_y ** 2) + (1 - res_alpha) * prev_vy
        var_z_new = res_alpha * (resid_z ** 2) + (1 - res_alpha) * prev_vz
        cov_xy_new = res_alpha * (resid_x * resid_y) + (1 - res_alpha) * prev_cxy

        x_hat.append(x_new)
        y_hat.append(y_new)
        z_hat.append(z_new)
        var_x.append(var_x_new)
        var_y.append(var_y_new)
        var_z.append(var_z_new)
        cov_xy.append(cov_xy_new)

        prev_x, prev_y, prev_z = x_new, y_new, z_new
        prev_t = float(row["t_hours"])
        prev_vx, prev_vy, prev_vz, prev_cxy = var_x_new, var_y_new, var_z_new, cov_xy_new

    out = df.copy()
    out["x_hat_km"] = x_hat
    out["y_hat_km"] = y_hat
    out["z_hat_km"] = z_hat
    out["sigma_x_km"] = np.sqrt(np.maximum(var_x, 0))
    out["sigma_y_km"] = np.sqrt(np.maximum(var_y, 0))
    out["sigma_z_km"] = np.sqrt(np.maximum(var_z, 0))
    denom = np.maximum(np.sqrt(np.array(var_x) * np.array(var_y)), 1e-8)
    out["rho_xy"] = np.clip(np.array(cov_xy) / denom, -0.99, 0.99)
    return out


def _compute_r95(track: pd.DataFrame, config: Dict) -> pd.Series:
    chi2_val = float(config.get("chi2_3d_95", 7.814727903))
    r95_min = float(config.get("r95_min_km", 0.1))

    sigma_x = track["sigma_x_km"].to_numpy()
    sigma_y = track["sigma_y_km"].to_numpy()
    sigma_z = track["sigma_z_km"].to_numpy()
    rho = track["rho_xy"].to_numpy()

    cov_mats = np.array(
        [
            [
                [sx ** 2, r * sx * sy, 0.0],
                [r * sx * sy, sy ** 2, 0.0],
                [0.0, 0.0, sz ** 2],
            ]
            for sx, sy, sz, r in zip(sigma_x, sigma_y, sigma_z, rho)
        ]
    )

    eigvals = np.linalg.eigvalsh(cov_mats)
    lambda_max = np.maximum(eigvals[:, 2], 0)
    r95 = np.sqrt(chi2_val * lambda_max)
    return np.maximum(r95, r95_min)


def _summarize(track: pd.DataFrame, config: Dict) -> Dict:
    r95 = track["R95_km"]
    search_center = {
        "center_x_km": track.iloc[-1]["x_hat_km"],
        "center_y_km": track.iloc[-1]["y_hat_km"],
        "center_z_km": track.iloc[-1]["z_hat_km"],
    }
    bbox = _compute_search_region(track, config)
    return {
        "n_obs": int(len(track)),
        "t_hours_min": float(track["t_hours"].min()),
        "t_hours_max": float(track["t_hours"].max()),
        "R95_km_mean": float(r95.mean()),
        "R95_km_p50": float(r95.quantile(0.5)),
        "R95_km_p90": float(r95.quantile(0.9)),
        "max_R95_km": float(r95.max()),
        "search_region": {
            **search_center,
            "bbox_width_km": bbox["bbox_width_km"],
            "bbox_height_km": bbox["bbox_height_km"],
            "depth_span_km": bbox["depth_span_km"],
        },
        "config": config,
    }


def _compute_search_region(track: pd.DataFrame, config: Dict) -> Dict:
    max_r95 = float(track["R95_km"].max())
    bbox_multiplier = float(config.get("bbox_multiplier_on_r95", 2.5))
    bbox_min = float(config.get("bbox_min_km", 2.0))
    half_width = bbox_multiplier * max_r95
    width = max(bbox_min, 2 * half_width)
    height = width
    center_x = float(track.iloc[-1]["x_hat_km"])
    center_y = float(track.iloc[-1]["y_hat_km"])
    depth_span = max(bbox_min, 2 * bbox_multiplier * max_r95)
    center_z = float(track.iloc[-1]["z_hat_km"])
    return {
        "center_x_km": center_x,
        "center_y_km": center_y,
        "center_z_km": center_z,
        "bbox_width_km": width,
        "bbox_height_km": height,
        "region_area_km2": width * height,
        "max_R95_km": max_r95,
        "depth_span_km": depth_span,
    }


def _compute_r95_timeseries(track: pd.DataFrame) -> pd.DataFrame:
    agg = (
        track.assign(t_hours_int=np.floor(track["t_hours"]).astype(int))
        .groupby("t_hours_int")["R95_km"]
        .mean()
        .reset_index()
        .rename(columns={"t_hours_int": "t_hours"})
    )
    agg["Area95_km2"] = np.pi * agg["R95_km"] ** 2
    return agg


def _depth_distribution(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    bins = config.get("depth_bins_m")
    if not bins or len(bins) < 2:
        raise ValueError("depth_bins_m must contain at least two bin edges")

    edges = np.array(bins, dtype=float)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
    if df["depth_m"].notna().any():
        counts, _ = np.histogram(df["depth_m"].dropna(), bins=edges)
        probs = counts.astype(float)
        if probs.sum() == 0:
            probs = np.ones_like(probs)
    else:
        prior = np.array(config.get("depth_prior_default", []), dtype=float)
        if len(prior) != len(edges) - 1:
            prior = np.ones(len(edges) - 1)
        probs = prior

    probabilities = probs / probs.sum()
    cumulative = probabilities.cumsum()
    return pd.DataFrame(
        {
            "bin_label": labels,
            "depth_min_m": edges[:-1],
            "depth_max_m": edges[1:],
            "probability": probabilities,
            "cumulative_prob": cumulative,
        }
    )


def _equipment_template() -> pd.DataFrame:
    data = [
        ["AUV-Scout", "AUV", 1500, 0.8, 5.0, 45, "medium", "high", "Autonomous search"],
        ["ROV-Workclass", "ROV", 2000, 0.3, 2.5, 60, "high", "very_high", "Tethered ops"],
        ["Towfish-SideScan", "Towfish", 800, 1.2, 4.5, 30, "low", "medium", "Side-scan sonar"],
    ]
    columns = [
        "equipment",
        "type",
        "max_depth_m",
        "sweep_width_km",
        "search_speed_kmh",
        "deploy_time_min",
        "opex_level",
        "capex_level",
        "notes",
    ]
    return pd.DataFrame(data, columns=columns)


def _write_interface_excel(output_dir: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "Locate_to_Member2_Interface.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"Excel interface written to {excel_path}")


def _write_sheet_csvs(output_dir: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    for name, df in sheets.items():
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)


def _normalize_field(field: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed = gaussian_filter(field, sigma=sigma)
    smoothed = smoothed - smoothed.mean()
    scale = np.max(np.abs(smoothed))
    return smoothed / scale if scale > 0 else smoothed


def _generate_ocean_state(config: Dict) -> Dict:
    grid = int(config.get("ocean_grid_size", 60))
    domain = float(config.get("ocean_domain_km", 30.0))
    base_density = float(config.get("ocean_density_base", 1025.0))
    density_variation = float(config.get("ocean_density_variation", 3.0))
    current_max = float(config.get("current_max_kmh", 5.0))
    bathy_max = float(config.get("bathymetry_max_depth_m", 3200.0))
    rng = np.random.default_rng(config.get("seed", 42))

    x_grid = np.linspace(-domain / 2, domain / 2, grid)
    y_grid = np.linspace(-domain / 2, domain / 2, grid)

    density_field = base_density + density_variation * _normalize_field(rng.normal(size=(grid, grid)))
    current_u = 0.6 * current_max * _normalize_field(rng.normal(size=(grid, grid)))
    current_v = 0.6 * current_max * _normalize_field(rng.normal(size=(grid, grid)))

    bathy_noise = _normalize_field(rng.normal(size=(grid, grid)), sigma=3.0)
    bathymetry = (0.5 + 0.5 * bathy_noise) * bathy_max

    return {
        "x_grid_km": x_grid,
        "y_grid_km": y_grid,
        "density": density_field,
        "current_u_kmh": current_u,
        "current_v_kmh": current_v,
        "bathymetry_m": bathymetry,
    }


def _save_ocean_state(ocean_state: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ocean_path = output_dir / "ocean_state.npz"
    np.savez_compressed(
        ocean_path,
        x_grid_km=ocean_state["x_grid_km"],
        y_grid_km=ocean_state["y_grid_km"],
        density=ocean_state["density"],
        current_u_kmh=ocean_state["current_u_kmh"],
        current_v_kmh=ocean_state["current_v_kmh"],
        bathymetry_m=ocean_state["bathymetry_m"],
    )
    return ocean_path


def run_locate(observations_path: Path, root_dir: Path, config: Dict) -> None:
    df = pd.read_csv(observations_path)
    if df.empty:
        raise ValueError("Observations file is empty")

    df = _ensure_columns(df, config)
    df = df.sort_values("t_hours").reset_index(drop=True)

    ocean_state = _generate_ocean_state(config)

    track = _compute_smoothed_estimates(df, config, ocean_state)
    track["R95_km"] = _compute_r95(track, config)

    track_out = track[
        [
            "t_hours",
            "x_obs_km",
            "y_obs_km",
            "z_obs_km",
            "q",
            "x_hat_km",
            "y_hat_km",
            "z_hat_km",
            "sigma_x_km",
            "sigma_y_km",
            "sigma_z_km",
            "rho_xy",
            "R95_km",
        ]
    ]

    output_dir = root_dir / "outputs/locate"
    output_dir.mkdir(parents=True, exist_ok=True)
    track_out.to_csv(output_dir / "track.csv", index=False)

    ocean_path = _save_ocean_state(ocean_state, output_dir)

    summary = _summarize(track_out, config)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    r95_ts = _compute_r95_timeseries(track_out)
    search_region = pd.DataFrame([_compute_search_region(track_out, config)])
    depth_dist = _depth_distribution(df, config)
    equip = _equipment_template()

    sheets = {
        "R95_TimeSeries": r95_ts,
        "SearchRegion": search_region,
        "DepthDistribution": depth_dist,
        "EquipmentTemplate": equip,
    }
    _write_interface_excel(output_dir, sheets)
    _write_sheet_csvs(output_dir, sheets)

    vis_outputs = create_visualizations(track_out, root_dir, ocean_state)
    vis_outputs.extend(create_ocean_visualizations(ocean_state, root_dir))

    print(f"Track written to {output_dir / 'track.csv'}")
    print(f"Summary written to {output_dir / 'summary.json'}")
    print(f"Ocean state written to {ocean_path}")
    for path in vis_outputs:
        print(f"Visualization written to {path}")
