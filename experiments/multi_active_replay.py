#!/usr/bin/env python3
"""Sparse multi-active replay experiments for coverage-limited scaling laws.

The generator samples independent Bernoulli activations exactly, but constructs
the design in sparse CSR form.  The main optimizer is with-replacement replay
SGD with per-example norm clipping,

    gamma_i = min(eta, clip_radius / ||x_i||^2),

which is the algorithm analyzed in the accompanying proof.  The script also
supports cyclic and independently reshuffled replay, and an unclipped stress
test.

Examples
--------
Quick validation:
    python experiments/multi_active_replay.py --suite quick

Reproduce all checked-in sweeps:
    python experiments/multi_active_replay.py --suite full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import zeta


@njit(cache=True)
def _labels_from_csr(
    theta: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    n = len(indptr) - 1
    labels = np.empty(n, dtype=np.float64)
    for i in range(n):
        value = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            value += values[k] * theta[indices[k]]
        labels[i] = value
    return labels


@njit(cache=True)
def train_snapshots(
    theta: np.ndarray,
    eigenvalues: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    epochs: np.ndarray,
    eta: float,
    seed: int,
    protocol_code: int,
    clipped: bool,
    clip_radius: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return population risks of the last and Polyak-averaged iterates.

    protocol_code: 0 = cyclic, 1 = fresh reshuffling each epoch,
                   2 = with-replacement sampling.
    The average is over the pre-update iterates w_0,...,w_{T-1}.
    """
    np.random.seed(seed)
    n = len(labels)
    d = len(theta)
    max_epoch = int(epochs[-1])

    w = np.zeros(d, dtype=np.float64)

    # Sparse lazy accumulation of pre-update iterates.
    accumulated = np.zeros(d, dtype=np.float64)
    last_accounted = np.zeros(d, dtype=np.int64)
    total_updates = 0

    last_risk = np.empty(len(epochs), dtype=np.float64)
    average_risk = np.empty(len(epochs), dtype=np.float64)
    snapshot_index = 0
    order = np.arange(n)

    for epoch in range(1, max_epoch + 1):
        if protocol_code == 1:
            # Fisher--Yates shuffle.
            for t in range(n - 1, 0, -1):
                r = np.random.randint(0, t + 1)
                tmp = order[t]
                order[t] = order[r]
                order[r] = tmp
        elif protocol_code == 2:
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
            if clipped and step * norm_sq > clip_radius:
                step = clip_radius / norm_sq

            # Account for the current pre-update iterate only on coordinates
            # that are about to change.
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
            for j in range(d):
                residual = w[j] - theta[j]
                risk_last += eigenvalues[j] * residual * residual

                coordinate_sum = (
                    accumulated[j]
                    + (total_updates - last_accounted[j]) * w[j]
                )
                averaged_value = coordinate_sum / total_updates
                averaged_residual = averaged_value - theta[j]
                risk_average += (
                    eigenvalues[j] * averaged_residual * averaged_residual
                )

            last_risk[snapshot_index] = risk_last
            average_risk[snapshot_index] = risk_average
            snapshot_index += 1
            if snapshot_index == len(epochs):
                break

    return last_risk, average_risk


def generate_sparse_dataset(
    n: int,
    d: int,
    s: float,
    a: float,
    b: float,
    expected_active: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Generate independent Bernoulli sparse features in exact CSR form.

    For each coordinate j, first draw its Binomial(n,p_j) occupancy and then a
    uniformly random subset of rows of that size.  This is distributionally
    identical to drawing all n*d independent Bernoulli masks, but costs
    O(d + nnz) rather than O(nd) memory.
    """
    if not (n > 0 and d > 0 and s > 1 and a >= s and b > 1):
        raise ValueError("require n,d>0, s>1, a>=s, and b>1")

    rng = np.random.default_rng(seed)
    ranks = np.arange(1, d + 1, dtype=np.float64)
    activation_prefactor = expected_active / float(zeta(s, 1.0))
    if not 0.0 < activation_prefactor < 1.0:
        raise ValueError(
            "expected_active / zeta(s) must lie in (0,1); "
            f"got {activation_prefactor}"
        )

    probabilities = activation_prefactor * ranks ** (-s)
    eigenvalues = ranks ** (-a)
    conditional_variances = eigenvalues / probabilities
    target_energy = ranks ** (-b)
    teacher = np.sqrt(target_energy / eigenvalues) * rng.choice(
        np.array([-1.0, 1.0]), size=d
    )

    coordinate_counts = rng.binomial(n, probabilities)
    nnz = int(coordinate_counts.sum())
    rows = np.empty(nnz, dtype=np.int64)
    columns = np.empty(nnz, dtype=np.int32)

    cursor = 0
    for j, count in enumerate(coordinate_counts):
        count_int = int(count)
        if count_int == 0:
            continue
        selected_rows = rng.choice(n, size=count_int, replace=False)
        rows[cursor : cursor + count_int] = selected_rows
        columns[cursor : cursor + count_int] = j
        cursor += count_int

    ordering = np.argsort(rows, kind="stable")
    sorted_rows = rows[ordering]
    indices = columns[ordering]
    row_counts = np.bincount(sorted_rows, minlength=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])

    signs = rng.choice(np.array([-1.0, 1.0]), size=nnz)
    values = signs * np.sqrt(conditional_variances[indices])
    labels = _labels_from_csr(teacher, indptr, indices, values)

    return {
        "p": probabilities,
        "lambda": eigenvalues,
        "q2": conditional_variances,
        "g": target_energy,
        "theta": teacher,
        "indptr": indptr,
        "indices": indices,
        "values": values,
        "labels": labels,
        "activation_prefactor": float(activation_prefactor),
    }


def analytic_tail(b: float, d: int) -> float:
    return float(zeta(b, d + 1))


def coverage_floor(data: dict[str, np.ndarray | float], n: int, b: float, d: int) -> float:
    probabilities = np.asarray(data["p"])
    target_energy = np.asarray(data["g"])
    return float(target_energy @ np.power(1.0 - probabilities, n)) + analytic_tail(b, d)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")


def _run_one(
    *,
    n: int,
    d: int,
    s: float,
    a: float,
    b: float,
    mu: float,
    eta: float,
    epochs: Sequence[int],
    seed: int,
    protocol_code: int,
    clipped: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    data = generate_sparse_dataset(n, d, s, a, b, mu, seed)
    tail = analytic_tail(b, d)
    last, average = train_snapshots(
        np.asarray(data["theta"]),
        np.asarray(data["lambda"]),
        np.asarray(data["indptr"]),
        np.asarray(data["indices"]),
        np.asarray(data["values"]),
        np.asarray(data["labels"]),
        np.asarray(epochs, dtype=np.int64),
        eta,
        seed + 100_003,
        protocol_code,
        clipped,
    )
    return last + tail, average + tail, coverage_floor(data, n, b, d)


def run_all_data(
    *,
    out: Path,
    ns: Sequence[int],
    epochs: Sequence[int],
    seeds: Iterable[int],
    d: int,
    s: float = 1.5,
    a: float = 2.0,
    b: float = 2.0,
    mu: float = 1.0,
    eta: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    protocols = (
        ("with_replacement_clipped", 2, True),
        ("reshuffle_clipped", 1, True),
        ("reshuffle_unclipped", 1, False),
    )
    for n in ns:
        for seed in seeds:
            for name, code, clipped in protocols:
                last, average, floor = _run_one(
                    n=n, d=d, s=s, a=a, b=b, mu=mu, eta=eta,
                    epochs=epochs, seed=seed, protocol_code=code,
                    clipped=clipped,
                )
                for epoch, risk_last, risk_average in zip(epochs, last, average):
                    rows.append({
                        "n": n, "K": epoch, "seed": seed, "protocol": name,
                        "last": float(risk_last),
                        "average": float(risk_average),
                        "floor": floor, "eta": eta, "mu": mu,
                        "s": s, "a": a, "b": b, "d": d,
                    })
        print(f"all-data: n={n}")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "all_data_sweep.csv")
    return frame


def run_progressive(
    *,
    out: Path,
    total_updates: Sequence[int],
    seeds: Iterable[int],
    d: int,
    s: float = 1.5,
    a: float = 2.0,
    b: float = 2.0,
    mu: float = 1.0,
    eta: float = 0.2,
    subset_prefactor: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_updates in total_updates:
        n = max(32, int(subset_prefactor * (eta * target_updates) ** (s / a)))
        n = min(n, target_updates)
        epochs = max(1, target_updates // n)
        effective_updates = n * epochs
        for seed in seeds:
            last, average, floor = _run_one(
                n=n, d=d, s=s, a=a, b=b, mu=mu, eta=eta,
                epochs=(epochs,), seed=seed, protocol_code=2, clipped=True,
            )
            rows.append({
                "T_target": target_updates, "T_eff": effective_updates,
                "N": n, "K": epochs, "seed": seed,
                "last": float(last[0]), "average": float(average[0]),
                "floor": floor, "s": s, "a": a, "b": b, "mu": mu,
                "eta": eta, "subset_prefactor": subset_prefactor, "d": d,
            })
        print(f"progressive: T={target_updates}, N={n}, K={epochs}")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "progressive_schedule_sweep.csv")
    return frame


def run_fixed_compute(
    *,
    out: Path,
    total_updates: int,
    unique_counts: Sequence[int],
    seeds: Iterable[int],
    d: int,
    eta: float = 0.2,
    mu: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for n in unique_counts:
        if total_updates % n != 0:
            raise ValueError("each unique count must divide total_updates")
        epochs = total_updates // n
        for seed in seeds:
            last, average, floor = _run_one(
                n=n, d=d, s=1.5, a=2.0, b=2.0, mu=mu, eta=eta,
                epochs=(epochs,), seed=seed, protocol_code=2, clipped=True,
            )
            rows.append({
                "T": total_updates, "N": n, "K": epochs, "seed": seed,
                "last": float(last[0]), "average": float(average[0]),
                "floor": floor,
            })
        print(f"fixed-compute: N={n}, K={epochs}")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "fixed_compute_sweep.csv")
    return frame


def run_source_exponents(
    *,
    out: Path,
    source_exponents: Sequence[float],
    ns: Sequence[int],
    seeds: Iterable[int],
    d: int,
    epochs: int = 512,
    eta: float = 0.2,
    mu: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for b in source_exponents:
        for n in ns:
            for seed in seeds:
                last, average, floor = _run_one(
                    n=n, d=d, s=1.5, a=2.0, b=b, mu=mu, eta=eta,
                    epochs=(epochs,), seed=seed, protocol_code=2, clipped=True,
                )
                rows.append({
                    "b": b, "n": n, "K": epochs, "seed": seed,
                    "last": float(last[0]), "average": float(average[0]),
                    "floor": floor,
                })
        print(f"source exponent: b={b}")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "source_exponent_sweep.csv")
    return frame


def run_coactivation(
    *,
    out: Path,
    mus: Sequence[float],
    n: int,
    epochs: Sequence[int],
    seeds: Iterable[int],
    d: int,
    eta: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    protocols = (("with_replacement", 2), ("reshuffle", 1), ("cyclic", 0))
    for mu in mus:
        for seed in seeds:
            for name, code in protocols:
                last, average, floor = _run_one(
                    n=n, d=d, s=1.5, a=2.0, b=2.0, mu=mu, eta=eta,
                    epochs=epochs, seed=seed, protocol_code=code, clipped=True,
                )
                for epoch, risk_last, risk_average in zip(epochs, last, average):
                    rows.append({
                        "mu": mu, "n": n, "K": epoch, "seed": seed,
                        "protocol": name, "last": float(risk_last),
                        "average": float(risk_average), "floor": floor,
                    })
        print(f"coactivation: mu={mu}")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "coactivation_sweep.csv")
    return frame


def run_clipping_stress(
    *,
    out: Path,
    mus: Sequence[float],
    etas: Sequence[float],
    n: int,
    epochs: Sequence[int],
    seeds: Iterable[int],
    d: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mu in mus:
        for eta in etas:
            for seed in seeds:
                for clipped in (True, False):
                    last, average, floor = _run_one(
                        n=n, d=d, s=1.5, a=2.0, b=2.0, mu=mu, eta=eta,
                        epochs=epochs, seed=seed, protocol_code=1,
                        clipped=clipped,
                    )
                    for epoch, risk_last, risk_average in zip(
                        epochs, last, average
                    ):
                        rows.append({
                            "mu": mu, "eta": eta, "clip": clipped,
                            "K": epoch, "seed": seed,
                            "last": float(risk_last),
                            "average": float(risk_average), "floor": floor,
                        })
            print(f"clipping stress: mu={mu}, eta={eta}")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "clipping_stress.csv")
    return frame


def run_spectrum_grid(
    *,
    out: Path,
    configs: Sequence[tuple[float, float, float, float]],
    ns: Sequence[int],
    epochs: Sequence[int],
    seeds: Iterable[int],
    d: int,
    eta: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for a, s, b, mu in configs:
        for n in ns:
            for seed in seeds:
                last, average, floor = _run_one(
                    n=n, d=d, s=s, a=a, b=b, mu=mu, eta=eta,
                    epochs=epochs, seed=seed, protocol_code=2, clipped=True,
                )
                for epoch, risk_last, risk_average in zip(
                    epochs, last, average
                ):
                    rows.append({
                        "a": a, "s": s, "b": b, "mu": mu,
                        "n": n, "K": epoch, "seed": seed,
                        "last": float(risk_last),
                        "average": float(risk_average), "floor": floor,
                    })
        print(f"spectrum: (a,s,b)=({a},{s},{b})")
    frame = pd.DataFrame(rows)
    _write_frame(frame, out / "spectrum_sweep.csv")
    return frame


def _log_slope(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)[0])


def summarize(out: Path) -> dict[str, object]:
    summary: dict[str, object] = {}

    all_path = out / "all_data_sweep.csv"
    if all_path.exists():
        frame = pd.read_csv(all_path)
        frame = frame[frame["protocol"] == "with_replacement_clipped"]
        aggregate = frame.groupby(["n", "K"])[
            ["last", "average", "floor"]
        ].mean().reset_index()
        eta = float(frame["eta"].iloc[0])
        a = float(frame["a"].iloc[0])
        s = float(frame["s"].iloc[0])
        b = float(frame["b"].iloc[0])
        design = np.column_stack([
            (eta * aggregate["n"] * aggregate["K"]) ** (-(b - 1.0) / a),
            aggregate["n"] ** (-(b - 1.0) / s),
        ])
        fits: dict[str, object] = {}
        for column in ("average", "last"):
            coefficients = np.linalg.lstsq(
                design, aggregate[column], rcond=None
            )[0]
            prediction = design @ coefficients
            residual = np.sum((aggregate[column] - prediction) ** 2)
            total = np.sum(
                (aggregate[column] - aggregate[column].mean()) ** 2
            )
            fits[column] = {
                "optimization_coefficient": float(coefficients[0]),
                "coverage_coefficient": float(coefficients[1]),
                "r_squared": float(1.0 - residual / total),
                "median_relative_error": float(np.median(
                    np.abs(prediction - aggregate[column]) / aggregate[column]
                )),
            }
        final = aggregate[aggregate["K"] == aggregate["K"].max()]
        largest_n = aggregate[aggregate["n"] == aggregate["n"].max()]
        early = largest_n[largest_n["K"] <= 8]
        summary["all_data"] = {
            "two_term_fit": fits,
            "predicted_optimization_slope": -(b - 1.0) / a,
            "measured_early_average_slope": _log_slope(
                early["K"], early["average"]
            ),
            "predicted_coverage_slope": -(b - 1.0) / s,
            "measured_large_epoch_average_slope": _log_slope(
                final["n"], final["average"]
            ),
            "measured_large_epoch_last_slope": _log_slope(
                final["n"], final["last"]
            ),
        }

    progressive_path = out / "progressive_schedule_sweep.csv"
    if progressive_path.exists():
        frame = pd.read_csv(progressive_path)
        aggregate = frame.groupby("T_eff")[
            ["last", "average", "floor"]
        ].mean().reset_index()
        summary["progressive_schedule"] = {
            column + "_slope": _log_slope(aggregate["T_eff"], aggregate[column])
            for column in ("last", "average", "floor")
        }

    fixed_path = out / "fixed_compute_sweep.csv"
    if fixed_path.exists():
        frame = pd.read_csv(fixed_path)
        aggregate = frame.groupby("N")[
            ["last", "average", "floor"]
        ].mean().reset_index()
        summary["fixed_compute"] = {
            "total_updates": int(frame["T"].iloc[0]),
            "risk_ratio_fewest_over_most_unique": float(
                aggregate["last"].iloc[0] / aggregate["last"].iloc[-1]
            ),
        }

    source_path = out / "source_exponent_sweep.csv"
    if source_path.exists():
        frame = pd.read_csv(source_path)
        slopes: dict[str, object] = {}
        for exponent, group in frame.groupby("b"):
            aggregate = group.groupby("n")[
                ["last", "average", "floor"]
            ].mean().reset_index()
            slopes[str(exponent)] = {
                "predicted": -(float(exponent) - 1.0) / 1.5,
                "last": _log_slope(aggregate["n"], aggregate["last"]),
                "average": _log_slope(aggregate["n"], aggregate["average"]),
            }
        summary["source_exponents"] = slopes

    spectrum_path = out / "spectrum_sweep.csv"
    if spectrum_path.exists():
        frame = pd.read_csv(spectrum_path)
        configs: dict[str, object] = {}
        for (a, s, b, mu), group in frame.groupby(["a", "s", "b", "mu"]):
            aggregate = group.groupby(["n", "K"])[
                ["last", "average", "floor"]
            ].mean().reset_index()
            design = np.column_stack([
                (0.2 * aggregate["n"] * aggregate["K"])
                ** (-(b - 1.0) / a),
                aggregate["n"] ** (-(b - 1.0) / s),
            ])
            coefficients = np.linalg.lstsq(
                design, aggregate["average"], rcond=None
            )[0]
            prediction = design @ coefficients
            residual = np.sum((aggregate["average"] - prediction) ** 2)
            total = np.sum(
                (aggregate["average"] - aggregate["average"].mean()) ** 2
            )
            final = aggregate[aggregate["K"] == aggregate["K"].max()]
            largest_n = aggregate[aggregate["n"] == aggregate["n"].max()]
            early = largest_n[largest_n["K"] <= 8]
            key = f"a={a:g},s={s:g},b={b:g},mu={mu:g}"
            configs[key] = {
                "predicted_optimization_slope": -(b - 1.0) / a,
                "measured_optimization_slope": _log_slope(
                    early["K"], early["average"]
                ),
                "predicted_coverage_slope": -(b - 1.0) / s,
                "measured_coverage_slope": _log_slope(
                    final["n"], final["average"]
                ),
                "two_term_r_squared": float(1.0 - residual / total),
                "two_term_median_relative_error": float(np.median(
                    np.abs(prediction - aggregate["average"])
                    / aggregate["average"]
                )),
            }
        summary["spectrum_grid"] = configs

    run_group_columns = {
        "all_data_sweep.csv": ["n", "seed", "protocol", "eta", "mu", "s", "a", "b", "d"],
        "progressive_schedule_sweep.csv": ["T_target", "seed"],
        "fixed_compute_sweep.csv": ["T", "N", "seed"],
        "source_exponent_sweep.csv": ["b", "n", "seed"],
        "coactivation_sweep.csv": ["mu", "n", "seed", "protocol"],
        "clipping_stress.csv": ["mu", "eta", "clip", "seed"],
        "spectrum_sweep.csv": ["a", "s", "b", "mu", "n", "seed"],
    }
    checkpoints = 0
    runs = 0
    for filename, columns in run_group_columns.items():
        path = out / filename
        if path.exists():
            frame = pd.read_csv(path)
            checkpoints += len(frame)
            runs += len(frame[columns].drop_duplicates())
    summary["reproducibility"] = {
        "risk_checkpoints": checkpoints,
        "training_runs": runs,
    }

    (out / "multi_active_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_quick(out: Path, seeds: int) -> None:
    seed_range = range(seeds)
    run_all_data(
        out=out, ns=(256, 1024), epochs=(1, 4, 16, 64),
        seeds=seed_range, d=2048,
    )
    run_progressive(
        out=out, total_updates=(2**12, 2**14, 2**16),
        seeds=seed_range, d=2048,
    )


def run_full(out: Path) -> None:
    run_all_data(
        out=out,
        ns=(256, 512, 1024, 2048, 4096, 8192),
        epochs=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        seeds=range(5), d=4096,
    )
    run_progressive(
        out=out, total_updates=tuple(2**k for k in range(12, 21)),
        seeds=range(5), d=8192,
    )
    run_fixed_compute(
        out=out, total_updates=2**18,
        unique_counts=(256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
        seeds=range(5), d=8192,
    )
    run_source_exponents(
        out=out, source_exponents=(1.4, 1.7, 2.0, 2.4),
        ns=(256, 512, 1024, 2048, 4096, 8192),
        seeds=range(5), d=8192,
    )
    run_coactivation(
        out=out, mus=(0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.35),
        n=2048, epochs=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        seeds=range(10), d=4096,
    )
    run_clipping_stress(
        out=out, mus=(1.0, 1.75, 2.25), etas=(0.2, 0.4, 0.8, 1.2),
        n=2048, epochs=(1, 2, 4, 8, 16, 32, 64),
        seeds=range(5), d=4096,
    )
    run_spectrum_grid(
        out=out,
        configs=((2.0, 1.5, 2.0, 1.0),
                 (2.4, 1.5, 2.0, 1.0),
                 (2.5, 2.0, 2.5, 1.0)),
        ns=(512, 2048, 8192),
        epochs=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        seeds=range(5), d=8192,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("quick", "full", "summarize"),
        default="quick",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/results")
    )
    parser.add_argument(
        "--seeds", type=int, default=2,
        help="number of seeds for the quick suite",
    )
    args = parser.parse_args()

    started = time.time()
    if args.suite == "quick":
        run_quick(args.out, args.seeds)
    elif args.suite == "full":
        run_full(args.out)
    result = summarize(args.out)
    print(json.dumps(result, indent=2))
    print(f"elapsed_seconds={time.time() - started:.2f}")


if __name__ == "__main__":
    main()
