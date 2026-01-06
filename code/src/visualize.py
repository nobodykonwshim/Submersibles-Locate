from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def _plot_r95_timeseries(track: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(track["t_hours"], track["R95_km"], color="tab:blue", linewidth=2, label="R95")
    ax.fill_between(track["t_hours"], track["R95_km"], color="tab:blue", alpha=0.1)
    ax.set_xlabel("t_hours")
    ax.set_ylabel("R95 (km)")
    ax.set_title("R95 over Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "r95_timeseries.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_track_static(track: pd.DataFrame, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")

    t_norm = (track["t_hours"] - track["t_hours"].min()) / (
        track["t_hours"].max() - track["t_hours"].min() + 1e-6
    )

    line = ax.plot3D(
        track["x_hat_km"],
        track["y_hat_km"],
        track["z_hat_km"],
        color="tab:orange",
        linewidth=2,
        label="Estimated track",
    )
    scatter = ax.scatter(
        track["x_hat_km"],
        track["y_hat_km"],
        track["z_hat_km"],
        c=t_norm,
        cmap=cmap,
        s=24,
        label="Time-coded",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.1)
    cbar.set_label("t_hours (normalized)")

    ax.set_xlabel("x_hat_km")
    ax.set_ylabel("y_hat_km")
    ax.set_zlabel("z_hat_km (negative = depth)")
    ax.set_title("3D Track (static)")
    ax.view_init(elev=18, azim=35)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "track_static_3d.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _animate_track(track: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Track Evolution (3D with depth)")

    padding = 0.2
    x_min, x_max = track["x_hat_km"].min(), track["x_hat_km"].max()
    y_min, y_max = track["y_hat_km"].min(), track["y_hat_km"].max()
    z_min, z_max = track["z_hat_km"].min(), track["z_hat_km"].max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    z_range = z_max - z_min if z_max > z_min else 1.0

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    ax.set_xlabel("x_hat_km")
    ax.set_ylabel("y_hat_km")
    ax.set_zlabel("z_hat_km (negative = depth)")
    ax.grid(True)
    ax.view_init(elev=22, azim=35)

    (line,) = ax.plot([], [], [], color="tab:orange", linewidth=2, label="Track")
    scatter = ax.scatter([], [], [], color="tab:red", s=30, depthshade=True)
    time_text = ax.text2D(0.02, 0.92, "", transform=ax.transAxes, fontsize=10)
    ax.legend(loc="upper right")

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return line, scatter, time_text

    def update(frame_idx: int):
        x_data = track["x_hat_km"].iloc[: frame_idx + 1]
        y_data = track["y_hat_km"].iloc[: frame_idx + 1]
        z_data = track["z_hat_km"].iloc[: frame_idx + 1]

        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
        scatter._offsets3d = ([x_data.iloc[-1]], [y_data.iloc[-1]], [z_data.iloc[-1]])

        current_time = track["t_hours"].iloc[frame_idx]
        time_text.set_text(f"t = {current_time:.2f} h")
        return line, scatter, time_text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(track),
        interval=80,
        blit=False,
    )

    gif_path = output_dir / "track_animation_3d.gif"
    anim.save(gif_path, writer="pillow", dpi=100)
    plt.close(fig)
    return gif_path


def _plot_ocean_density_currents(ocean_state: Dict, output_dir: Path) -> Path:
    x_grid = ocean_state["x_grid_km"]
    y_grid = ocean_state["y_grid_km"]
    density = ocean_state["density"]
    u = ocean_state["current_u_kmh"]
    v = ocean_state["current_v_kmh"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    im = ax.imshow(density, origin="lower", extent=extent, cmap="cividis")
    fig.colorbar(im, ax=ax, label="Density (kg/m^3)")

    step = max(1, len(x_grid) // 15)
    ax.quiver(
        x_grid[::step],
        y_grid[::step],
        u[::step, ::step],
        v[::step, ::step],
        color="white",
        scale=30,
        width=0.004,
        alpha=0.8,
    )
    ax.set_xlabel("x_km")
    ax.set_ylabel("y_km")
    ax.set_title("Ocean density & surface currents")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ocean_density_currents.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_bathymetry(ocean_state: Dict, output_dir: Path) -> Path:
    x_grid = ocean_state["x_grid_km"]
    y_grid = ocean_state["y_grid_km"]
    bathy = ocean_state["bathymetry_m"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    im = ax.imshow(bathy, origin="lower", extent=extent, cmap="Blues_r")
    fig.colorbar(im, ax=ax, label="Depth (m)")
    ax.set_xlabel("x_km")
    ax.set_ylabel("y_km")
    ax.set_title("Seafloor bathymetry")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ocean_bathymetry.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def create_visualizations(track: pd.DataFrame, root_dir: Path, ocean_state: Dict) -> List[Path]:
    vis_dir = root_dir / "outputs/locate/visualizations"
    outputs: List[Path] = []
    outputs.append(_plot_r95_timeseries(track, vis_dir))
    outputs.append(_plot_track_static(track, vis_dir))
    outputs.append(_animate_track(track, vis_dir))
    return outputs


def create_ocean_visualizations(ocean_state: Dict, root_dir: Path) -> List[Path]:
    vis_dir = root_dir / "outputs/locate/visualizations"
    outputs: List[Path] = []
    outputs.append(_plot_ocean_density_currents(ocean_state, vis_dir))
    outputs.append(_plot_bathymetry(ocean_state, vis_dir))
    return outputs
