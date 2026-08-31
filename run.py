#!/usr/bin/env python3
"""GlucoMind-14 simulator entry point.

Usage:
    python run.py                     # Run a 3-day test trajectory
    python run.py --days 7            # Custom simulation duration
    python run.py --population 100    # Run population study
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    Parameters, N_STATES, STATE_NAMES,
    simulate_dense, get_bgm_times,
    generate_mdi_schedule, sample_bgm_observations,
    generate_population, compute_population_metrics,
)


def load_config(path: str) -> dict:
    """Load and validate a YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("rtol", "atol", "max_step", "dt_dense", "days"):
        v = cfg.get("simulation", {}).get(k)
        if v is not None:
            cfg["simulation"][k] = float(v) if isinstance(v, str) else v
    return cfg


def run_test(config: dict, output_dir: str, days: int = 3) -> None:
    """Run a single trajectory test and save the plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Parameters()
    t_span = (0.0, days * 24.0)
    dt = config["simulation"]["dt_dense"]

    events = generate_mdi_schedule(days=days, **config["schedule"])
    bgm_times = get_bgm_times(events, t_span)
    state0 = np.zeros(N_STATES)
    state0[0] = 120.0
    state0[1] = 90.0

    t_sim, y_sim = simulate_dense(
        state0, events, p, t_span,
        dt=dt,
        rtol=config["simulation"]["rtol"],
        atol=config["simulation"]["atol"],
    )

    obs = sample_bgm_observations(bgm_times, t_sim, y_sim, p, seed=0)

    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_sim, y_sim[:, 0], "b-", linewidth=1, label="Gp (plasma glucose)")
    axes[0].plot(t_sim, y_sim[:, 1], "c-", linewidth=1, label="Gi (interstitial glucose)")
    axes[0].axhline(180, color="r", linestyle="--", alpha=0.5, label="Renal threshold")
    axes[0].axhline(70, color="orange", linestyle=":", alpha=0.5, label="Hypo threshold")
    axes[0].scatter(bgm_times, obs, color="red", s=20, zorder=5, label="BGM readings")
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_sim, y_sim[:, 9], "g-", linewidth=1, label="Ip (plasma insulin)")
    axes[1].plot(t_sim, y_sim[:, 10], "m-", linewidth=1, label="X (remote insulin action)")
    axes[1].set_ylabel("Insulin (mU/L)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_sim, y_sim[:, 2], "brown", linewidth=1, label="Qsto1")
    axes[2].plot(t_sim, y_sim[:, 3], "orange", linewidth=1, label="Qsto2")
    axes[2].set_ylabel("Stomach content (g)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.savefig(
        f"{output_dir}/figures/test_trajectory.png", dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Test trajectory saved to {output_dir}/figures/test_trajectory.png")
    print(f"  Simulation: {len(t_sim)} time points over {days} days")
    print(f"  Gp range: [{y_sim[:, 0].min():.1f}, {y_sim[:, 0].max():.1f}] mg/dL")
    print(f"  BGM readings: {len(obs)}")


def run_population(config: dict, output_dir: str, n_patients: int) -> None:
    """Run a population study and print summary."""
    days = config["simulation"]["days"]
    dt = config["simulation"]["dt_dense"]
    rtol = config["simulation"]["rtol"]
    atol = config["simulation"]["atol"]
    t_span = (0.0, days * 24.0)

    print(f"Generating {n_patients} virtual patients...")
    population = generate_population(n_patients, seed=config["population"]["seed"])

    events = generate_mdi_schedule(days=days, **config["schedule"])

    gp_traces = []
    for p in population:
        t_sim, y_sim = simulate_dense(
            np.zeros(N_STATES), events, p, t_span,
            dt=dt, rtol=rtol, atol=atol,
        )
        gp_traces.append(y_sim[:, 0])

    all_gp = np.array(gp_traces)
    metrics = compute_population_metrics(all_gp, t_sim, config["schedule"]["meal_times_h"])

    print(f"\n{'Metric':<25} {'Mean':<10} {'SD':<10} {'P5':<10} {'P95':<10}")
    print("-" * 65)
    for name, (mean, sd, p5, p95) in metrics.items():
        print(f"{name:<25} {mean:<10.2f} {sd:<10.2f} {p5:<10.2f} {p95:<10.2f}")

    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd
    rows = [
        {"metric": k, "mean": v[0], "sd": v[1], "p5": v[2], "p95": v[3]}
        for k, v in metrics.items()
    ]
    pd.DataFrame(rows).to_csv(f"{output_dir}/population_summary.csv", index=False)
    print(f"\nSaved to {output_dir}/population_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="GlucoMind-14 T1D Simulator")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Simulation days (overrides config)",
    )
    parser.add_argument(
        "--population", "-n", type=int, default=None,
        help="Run population study with N patients",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.days is not None:
        config["simulation"]["days"] = args.days

    os.makedirs(args.output, exist_ok=True)

    if args.population is not None:
        run_population(config, args.output, args.population)
    else:
        days = args.days or int(config["simulation"]["days"])
        run_test(config, args.output, days)

    print("\nDone.")


if __name__ == "__main__":
    main()
