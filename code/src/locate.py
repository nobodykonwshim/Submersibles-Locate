from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .visualize import create_visualizations


def _ensure_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    if "q" not in df.columns:
        df["q"] = config.get("default_quality", 1.0)
    if "depth_m" not in df.columns:
        df["depth_m"] = np.nan
    return df


def _compute_smoothed_estimates(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    alpha = config.get("smoothing_alpha", 0.35)
    res_alpha = config.get("residual_ewma_alpha", 0.25)

    x_hat = []
    y_hat = []
    var_x = []
    var_y = []
    cov_xy = []

    prev_x, prev_y = df.loc[0, "x_obs_km"], df.loc[0, "y_obs_km"]
    prev_vx = prev_vy = 0.05 ** 2
    prev_cxy = 0.0

    for _, row in df.iterrows():
        q = float(row.get("q", 1.0))
        weight = alpha * q
        x_new = weight * row["x_obs_km"] + (1 - weight) * prev_x
        y_new = weight * row["y_obs_km"] + (1 - weight) * prev_y

        resid_x = row["x_obs_km"] - x_new
        resid_y = row["y_obs_km"] - y_new

        var_x_new = res_alpha * (resid_x ** 2) + (1 - res_alpha) * prev_vx
        var_y_new = res_alpha * (resid_y ** 2) + (1 - res_alpha) * prev_vy
        cov_xy_new = res_alpha * (resid_x * resid_y) + (1 - res_alpha) * prev_cxy

        x_hat.append(x_new)
        y_hat.append(y_new)
        var_x.append(var_x_new)
        var_y.append(var_y_new)
        cov_xy.append(cov_xy_new)

        prev_x, prev_y = x_new, y_new
        prev_vx, prev_vy, prev_cxy = var_x_new, var_y_new, cov_xy_new

    out = df.copy()
    out["x_hat_km"] = x_hat
    out["y_hat_km"] = y_hat
    out["sigma_x_km"] = np.sqrt(np.maximum(var_x, 0))
    out["sigma_y_km"] = np.sqrt(np.maximum(var_y, 0))
    denom = np.maximum(np.sqrt(np.array(var_x) * np.array(var_y)), 1e-8)
    out["rho"] = np.clip(np.array(cov_xy) / denom, -0.99, 0.99)
    return out


def _compute_r95(track: pd.DataFrame, config: Dict) -> pd.Series:
    chi2_val = float(config.get("chi2_2d_95", 5.9914645471))
    r95_min = float(config.get("r95_min_km", 0.1))

    sigma_x = track["sigma_x_km"].to_numpy()
    sigma_y = track["sigma_y_km"].to_numpy()
    rho = track["rho"].to_numpy()

    cov_mats = np.array(
        [
            [[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]]
            for sx, sy, r in zip(sigma_x, sigma_y, rho)
        ]
    )

    eigvals = np.linalg.eigvalsh(cov_mats)
    lambda_max = np.maximum(eigvals[:, 1], 0)
    r95 = np.sqrt(chi2_val * lambda_max)
    return np.maximum(r95, r95_min)


def _summarize(track: pd.DataFrame, config: Dict) -> Dict:
    r95 = track["R95_km"]
    search_center = {
        "center_x_km": track.iloc[-1]["x_hat_km"],
        "center_y_km": track.iloc[-1]["y_hat_km"],
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
    return {
        "center_x_km": center_x,
        "center_y_km": center_y,
        "bbox_width_km": width,
        "bbox_height_km": height,
        "region_area_km2": width * height,
        "max_R95_km": max_r95,
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


def run_locate(observations_path: Path, root_dir: Path, config: Dict) -> None:
    df = pd.read_csv(observations_path)
    if df.empty:
        raise ValueError("Observations file is empty")

    df = _ensure_columns(df, config)
    df = df.sort_values("t_hours").reset_index(drop=True)

    track = _compute_smoothed_estimates(df, config)
    track["R95_km"] = _compute_r95(track, config)

    track_out = track[
        [
            "t_hours",
            "x_obs_km",
            "y_obs_km",
            "q",
            "x_hat_km",
            "y_hat_km",
            "sigma_x_km",
            "sigma_y_km",
            "rho",
            "R95_km",
        ]
    ]

    output_dir = root_dir / "outputs/locate"
    output_dir.mkdir(parents=True, exist_ok=True)
    track_out.to_csv(output_dir / "track.csv", index=False)

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

    vis_outputs = create_visualizations(track_out, root_dir)

    print(f"Track written to {output_dir / 'track.csv'}")
    print(f"Summary written to {output_dir / 'summary.json'}")
    for path in vis_outputs:
        print(f"Visualization written to {path}")
