from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def _plot_r95_timeseries(track: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(track["t_hours"], track["R95_km"], color="tab:blue", linewidth=2, label="R95")
    ax.fill_between(track["t_hours"], track["R95_km"], color="tab:blue", alpha=0.1)
    ax.set_xlabel("t_hours")
    ax.set_ylabel("R95 (km)")
    ax.set_title("R95 随时间变化")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "r95_timeseries.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _animate_track(track: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("定位轨迹演化 (x_hat, y_hat)")

    padding = 0.2
    x_min, x_max = track["x_hat_km"].min(), track["x_hat_km"].max()
    y_min, y_max = track["y_hat_km"].min(), track["y_hat_km"].max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_xlabel("x_hat_km")
    ax.set_ylabel("y_hat_km")
    ax.grid(True, linestyle="--", alpha=0.4)

    (line,) = ax.plot([], [], color="tab:orange", linewidth=2, label="x_hat/y_hat")
    scatter = ax.scatter([], [], color="tab:red", zorder=3)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
    ax.legend(loc="upper right")

    def init():
        line.set_data([], [])
        scatter.set_offsets([])
        time_text.set_text("")
        return line, scatter, time_text

    def update(frame_idx: int):
        x_data = track["x_hat_km"].iloc[: frame_idx + 1]
        y_data = track["y_hat_km"].iloc[: frame_idx + 1]
        line.set_data(x_data, y_data)
        scatter.set_offsets([[x_data.iloc[-1], y_data.iloc[-1]]])
        current_time = track["t_hours"].iloc[frame_idx]
        time_text.set_text(f"t = {current_time:.2f} h")
        return line, scatter, time_text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(track),
        interval=80,
        blit=True,
    )

    gif_path = output_dir / "track_animation.gif"
    anim.save(gif_path, writer="pillow", dpi=100)
    plt.close(fig)
    return gif_path


def create_visualizations(track: pd.DataFrame, root_dir: Path) -> List[Path]:
    vis_dir = root_dir / "outputs/locate/visualizations"
    outputs: List[Path] = []
    outputs.append(_plot_r95_timeseries(track, vis_dir))
    outputs.append(_animate_track(track, vis_dir))
    return outputs
