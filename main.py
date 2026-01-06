import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "code"))
from src.locate import run_locate  # noqa: E402


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sample_data(config: dict, output_path: Path) -> None:
    rng = np.random.default_rng(config.get("seed", 42))
    n = 120
    t_hours = np.linspace(0, 48, n)
    steps = rng.normal(loc=0.0, scale=0.06, size=(n, 2)).cumsum(axis=0)
    slow_current = rng.normal(loc=0.0, scale=0.015, size=(n, 2)).cumsum(axis=0)
    base_traj = np.column_stack([0.6 * np.cos(t_hours / 12), 0.6 * np.sin(t_hours / 12)])
    noise = rng.normal(scale=0.04, size=(n, 2))
    data = base_traj + steps * 0.25 + slow_current * 0.35 + noise

    q = np.clip(rng.normal(loc=0.85, scale=0.1, size=n), 0.5, 1.0)
    depth_wave = 350 + 200 * np.sin(t_hours / 6) + rng.normal(scale=40, size=n)
    depth_choices = np.clip(depth_wave, 30, 1800)

    df = pd.DataFrame(
        {
            "t_hours": t_hours,
            "x_obs_km": data[:, 0],
            "y_obs_km": data[:, 1],
            "q": q,
            "depth_m": depth_choices,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample data written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate module runner")
    parser.add_argument("--stage", choices=["locate"], default="locate", help="Stage to run")
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate sample observations.csv in data/processed_v1/",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="Config path")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.generate_sample_data:
        output_path = ROOT / "data/processed_v1/observations.csv"
        generate_sample_data(config, output_path)

    if args.stage == "locate":
        observations_path = ROOT / "data/processed_v1/observations.csv"
        if not observations_path.exists():
            raise FileNotFoundError(
                f"Observations file not found at {observations_path}. Run with --generate-sample-data first."
            )
        run_locate(observations_path, ROOT, config)
        print("Locate stage completed. Outputs are under outputs/locate/")


if __name__ == "__main__":
    main()
