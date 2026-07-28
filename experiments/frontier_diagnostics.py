#!/usr/bin/env python3
"""Targeted diagnostics for the three coverage-limited scaling frontiers.

This script complements ``multi_active_replay.py`` with two experiments that
are especially useful in the paper:

1. a rank-aligned width sweep that exposes the capacity-to-coverage transition;
2. an input-only estimator of the saturation epoch, evaluated against replay
   curves and against the label-free singleton-anchor output.

The training protocol is the same example-norm-clipped with-replacement replay
used in the main experiment suite.  All population risks include the analytic
power-law tail beyond the simulated ambient dimension.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numba import njit

# Allow execution both from the repository root and as a module.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multi_active_replay import (  # noqa: E402
    analytic_tail,
    coverage_floor,
    generate_sparse_dataset,
)


@njit(cache=True)
def _train_diagnostics(
    theta: np.ndarray,
    eigenvalues: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    epochs: np.ndarray,
    eta: float,
    seed: int,
    anchor_mask: np.ndarray,
    clip_radius: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return last, Polyak-average, and anchor-masked-average risks."""
    np.random.seed(seed)
    n = len(labels)
    d = len(theta)
    max_epoch = int(epochs[-1])

    w = np.zeros(d, dtype=np.float64)
    accumulated = np.zeros(d, dtype=np.float64)
    last_accounted = np.zeros(d, dtype=np.int64)
    total_updates = 0

    last_risk = np.empty(len(epochs), dtype=np.float64)
    average_risk = np.empty(len(epochs), dtype=np.float64)
    anchor_risk = np.empty(len(epochs), dtype=np.float64)
    snapshot_index = 0
    order = np.empty(n, dtype=np.int64)

    for epoch in range(1, max_epoch + 1):
        for t in range(n):
            order[t] = np.random.randint(0, n)

        for position in range(n):
            i = order[position]
            lo = indptr[i]
            hi = indptr[i + 1]

            error = -labels[i]
            norm_sq = 0.0
            for k in range(lo, hi):
                j = indices[k]
                value = values[k]
                error += value * w[j]
                norm_sq += value * value

            step = eta
            if step * norm_sq > clip_radius:
                step = clip_radius / norm_sq

            for k in range(lo, hi):
                j = indices[k]
                accumulated[j] += (
                    total_updates - last_accounted[j] + 1
                ) * w[j]
                last_accounted[j] = total_updates + 1

            for k in range(lo, hi):
                j = indices[k]
                w[j] -= step * error * values[k]

            total_updates += 1

        if snapshot_index < len(epochs) and epoch == epochs[snapshot_index]:
            risk_last = 0.0
            risk_average = 0.0
            risk_anchor = 0.0
            for j in range(d):
                residual = w[j] - theta[j]
                risk_last += eigenvalues[j] * residual * residual

                coordinate_sum = (
                    accumulated[j]
                    + (total_updates - last_accounted[j]) * w[j]
                )
                averaged_value = coordinate_sum / total_updates
                averaged_residual = averaged_value - theta[j]
                risk_average += eigenvalues[j] * averaged_residual * averaged_residual

                masked_value = averaged_value if anchor_mask[j] else 0.0
                masked_residual = masked_value - theta[j]
                risk_anchor += eigenvalues[j] * masked_residual * masked_residual

            last_risk[snapshot_index] = risk_last
            average_risk[snapshot_index] = risk_average
            anchor_risk[snapshot_index] = risk_anchor
            snapshot_index += 1
            if snapshot_index == len(epochs):
                break

    return last_risk, average_risk, anchor_risk


def _anchor_mask(data: dict[str, np.ndarray | float], n: int) -> np.ndarray:
    """Distribution-aware label-free anchor mask used in the sharp theorem."""
    probabilities = np.asarray(data["p"], dtype=np.float64)
    indptr = np.asarray(data["indptr"], dtype=np.int64)
    indices = np.asarray(data["indices"], dtype=np.int32)
    row_sizes = np.diff(indptr)
    singleton_rows = np.flatnonzero(row_sizes == 1)
    singleton_coordinates = np.empty(len(singleton_rows), dtype=np.int32)
    for k, row in enumerate(singleton_rows):
        singleton_coordinates[k] = indices[indptr[row]]
    singleton_counts = np.bincount(
        singleton_coordinates, minlength=len(probabilities)
    )
    pi0 = float(np.exp(np.log1p(-probabilities).sum()))
    kappa = 0.5 * pi0
    return singleton_counts >= kappa * n * probabilities


def _input_only_prediction(
    data: dict[str, np.ndarray | float], n: int, eta: float, min_count: int = 8
) -> tuple[float, float, float, int]:
    """Estimate q^2(p)=exp(c) p^r from input frequencies and amplitudes."""
    indices = np.asarray(data["indices"], dtype=np.int32)
    values = np.asarray(data["values"], dtype=np.float64)
    d = len(np.asarray(data["p"]))
    counts = np.bincount(indices, minlength=d)
    sum_squares = np.bincount(indices, weights=values * values, minlength=d)
    eligible = counts >= min_count
    if int(eligible.sum()) < 3:
        raise RuntimeError(
            f"only {int(eligible.sum())} coordinates have at least {min_count} occurrences"
        )
    p_hat = counts[eligible] / float(n)
    q2_hat = sum_squares[eligible] / counts[eligible]
    slope, intercept = np.polyfit(np.log(p_hat), np.log(q2_hat), 1)
    predicted_epoch = math.exp(-float(intercept)) * n ** float(slope) / eta
    return float(slope), float(intercept), float(predicted_epoch), int(eligible.sum())


def _log_interpolated_crossing(
    epochs: np.ndarray, ratio: np.ndarray, threshold: float = 1.25
) -> float:
    """First log-interpolated epoch with risk/floor <= threshold."""
    below = np.flatnonzero(ratio <= threshold)
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(epochs[0])
    x0, x1 = np.log(float(epochs[i - 1])), np.log(float(epochs[i]))
    y0, y1 = np.log(float(ratio[i - 1])), np.log(float(ratio[i]))
    if y1 == y0:
        return float(epochs[i])
    x = x0 + (math.log(threshold) - y0) * (x1 - x0) / (y1 - y0)
    return float(math.exp(x))


def run_capacity_sweep(out: Path, seeds: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    n, epochs, eta = 32768, np.asarray([256], dtype=np.int64), 0.2
    s, a, b, mu = 1.5, 2.0, 2.0, 1.0
    for width in (64, 128, 256, 512, 1024, 2048, 4096):
        for seed in seeds:
            data = generate_sparse_dataset(n, width, s, a, b, mu, seed)
            mask = np.ones(width, dtype=np.bool_)
            last, average, _ = _train_diagnostics(
                np.asarray(data["theta"]),
                np.asarray(data["lambda"]),
                np.asarray(data["indptr"]),
                np.asarray(data["indices"]),
                np.asarray(data["values"]),
                np.asarray(data["labels"]),
                epochs,
                eta,
                seed + 100_003,
                mask,
            )
            tail = analytic_tail(b, width)
            rows.append(
                {
                    "m": width,
                    "n": n,
                    "K": int(epochs[0]),
                    "seed": seed,
                    "last": float(last[0] + tail),
                    "average": float(average[0] + tail),
                    "capacity_tail": tail,
                    "coverage_floor": coverage_floor(data, n, b, width),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "capacity_sweep.csv", index=False)
    return frame


def run_stopping_sweep(out: Path, seeds: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    configs = (
        ("default", 2.0, 1.5, 2.0),
        ("slow-tail", 2.4, 1.5, 2.0),
        ("steep-frequency", 2.5, 2.0, 2.5),
    )
    ns = (256, 512, 1024, 2048, 4096, 8192)
    epochs = np.asarray(
        (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        dtype=np.int64,
    )
    eta, mu, d = 0.2, 1.0, 8192
    for name, a, s, b in configs:
        for n in ns:
            for seed in seeds:
                data = generate_sparse_dataset(n, d, s, a, b, mu, seed)
                mask = _anchor_mask(data, n)
                slope, intercept, predicted, fitted_coordinates = _input_only_prediction(
                    data, n, eta
                )
                last, average, anchor = _train_diagnostics(
                    np.asarray(data["theta"]),
                    np.asarray(data["lambda"]),
                    np.asarray(data["indptr"]),
                    np.asarray(data["indices"]),
                    np.asarray(data["values"]),
                    np.asarray(data["labels"]),
                    epochs,
                    eta,
                    seed + 100_003,
                    mask,
                )
                tail = analytic_tail(b, d)
                floor = coverage_floor(data, n, b, d)
                average = average + tail
                anchor = anchor + tail
                last = last + tail
                measured = _log_interpolated_crossing(epochs, average / floor)
                for epoch, r_last, r_average, r_anchor in zip(
                    epochs, last, average, anchor
                ):
                    rows.append(
                        {
                            "config": name,
                            "a": a,
                            "s": s,
                            "b": b,
                            "n": n,
                            "K": int(epoch),
                            "seed": seed,
                            "last": float(r_last),
                            "average": float(r_average),
                            "anchor_average": float(r_anchor),
                            "floor": floor,
                            "estimated_exponent": slope,
                            "estimated_intercept": intercept,
                            "predicted_saturation": predicted,
                            "measured_saturation": measured,
                            "fitted_coordinates": fitted_coordinates,
                            "anchor_fraction": float(mask.mean()),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "stopping_rule_sweep.csv", index=False)
    return frame


def _slope(x: pd.Series, y: pd.Series) -> float:
    return float(np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)[0])


def summarize(capacity: pd.DataFrame, stopping: pd.DataFrame, out: Path) -> dict:
    capacity_mean = capacity.groupby("m", as_index=False).mean(numeric_only=True)
    capacity_core = capacity_mean[capacity_mean["m"] <= 256]
    summary: dict[str, object] = {
        "capacity": {
            "predicted_slope": -1.0,
            "measured_average_slope": _slope(
                capacity_core["m"], capacity_core["average"]
            ),
            "measured_tail_slope": _slope(
                capacity_core["m"], capacity_core["capacity_tail"]
            ),
            "transition_width": 1024,
        }
    }

    final = stopping[stopping["K"] == stopping["K"].max()]
    configs: dict[str, object] = {}
    for name, group in stopping.groupby("config"):
        one_per_dataset = group.drop_duplicates(["n", "seed"])
        medians = one_per_dataset.groupby("n", as_index=False)[
            ["predicted_saturation", "measured_saturation"]
        ].median()
        finite = one_per_dataset[
            np.isfinite(one_per_dataset["measured_saturation"])
        ].copy()
        factor = np.maximum(
            finite["predicted_saturation"] / finite["measured_saturation"],
            finite["measured_saturation"] / finite["predicted_saturation"],
        )
        a = float(group["a"].iloc[0])
        s = float(group["s"].iloc[0])
        b = float(group["b"].iloc[0])
        final_group = final[final["config"] == name]
        aggregate_final = final_group.groupby("n", as_index=False)[
            ["average", "anchor_average"]
        ].mean()
        configs[name] = {
            "a": a,
            "s": s,
            "b": b,
            "predicted_saturation_exponent": a / s - 1.0,
            "mean_estimated_exponent": float(
                one_per_dataset["estimated_exponent"].mean()
            ),
            "standard_error_estimated_exponent": float(
                one_per_dataset["estimated_exponent"].std(ddof=1)
                / math.sqrt(len(one_per_dataset))
            ),
            "predicted_epoch_slope": _slope(
                medians["n"], medians["predicted_saturation"]
            ),
            "measured_epoch_slope": _slope(
                medians["n"], medians["measured_saturation"]
            ),
            "median_epoch_factor_error": float(np.median(factor)),
            "predicted_coverage_slope": -(b - 1.0) / s,
            "plain_coverage_slope": _slope(
                aggregate_final["n"], aggregate_final["average"]
            ),
            "anchor_coverage_slope": _slope(
                aggregate_final["n"], aggregate_final["anchor_average"]
            ),
        }
    summary["stopping_rule"] = configs
    path = out / "frontier_diagnostics_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--mode", choices=("full", "capacity", "stopping", "summarize"), default="full"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seeds))

    capacity_path = args.out / "capacity_sweep.csv"
    stopping_path = args.out / "stopping_rule_sweep.csv"
    if args.mode in ("full", "capacity"):
        capacity = run_capacity_sweep(args.out, seeds)
    else:
        capacity = pd.read_csv(capacity_path)
    if args.mode in ("full", "stopping"):
        stopping = run_stopping_sweep(args.out, seeds)
    else:
        stopping = pd.read_csv(stopping_path)
    result = summarize(capacity, stopping, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
