#!/usr/bin/env python3
"""Create paper figures from the checked-in sparse replay sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Render text through LaTeX/NewTX so AAAI figures contain embedded Type 1
# fonts rather than Matplotlib's default Type 3 or Identity-H fonts.
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext,newtxmath}"


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def risk_vs_epochs(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "all_data_sweep.csv")
    frame = frame[frame["protocol"] == "with_replacement_clipped"]
    aggregate = frame.groupby(["n", "K"])[
        ["average", "floor"]
    ].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    for n in (512, 2048, 8192):
        group = aggregate[aggregate["n"] == n]
        line, = ax.loglog(
            group["K"], group["average"], marker="o", markersize=3,
            label=rf"$n={n}$",
        )
        ax.axhline(
            group["floor"].iloc[0], linestyle="--", linewidth=1,
            color=line.get_color(),
        )
    ax.set_xlabel("epochs $K$")
    ax.set_ylabel("population excess risk")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "risk_vs_epochs.pdf")


def scaling_collapse(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "all_data_sweep.csv")
    frame = frame[frame["protocol"] == "with_replacement_clipped"]
    aggregate = frame.groupby(["n", "K"])[
        ["average", "floor", "a", "s", "b", "eta"]
    ].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    for n, group in aggregate.groupby("n"):
        a = float(group["a"].iloc[0])
        s = float(group["s"].iloc[0])
        b = float(group["b"].iloc[0])
        eta = float(group["eta"].iloc[0])
        x = eta * group["K"] * n ** (1.0 - a / s)
        y = group["average"] * n ** ((b - 1.0) / s)
        ax.loglog(x, y, marker="o", markersize=2.8, label=rf"$n={n}$")
    ax.set_xlabel(r"normalized epochs $\eta K n^{1-a/s}$")
    ax.set_ylabel(r"normalized risk $n^{(b-1)/s}\mathcal{E}$")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "scaling_collapse.pdf")


def progressive_schedule(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "progressive_schedule_sweep.csv")
    aggregate = frame.groupby("T_eff")[
        ["last", "average", "floor"]
    ].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    ax.loglog(
        aggregate["T_eff"], aggregate["average"], marker="o",
        label="Polyak average",
    )
    ax.loglog(
        aggregate["T_eff"], aggregate["last"], marker="s",
        label="last iterate",
    )
    reference = aggregate["average"].iloc[-1] * (
        aggregate["T_eff"] / aggregate["T_eff"].iloc[-1]
    ) ** (-0.5)
    ax.loglog(
        aggregate["T_eff"], reference, linestyle="--",
        label=r"$T^{-1/2}$ reference",
    )
    ax.set_xlabel("total updates $T$")
    ax.set_ylabel("population excess risk")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "progressive_schedule.pdf")


def fixed_compute(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "fixed_compute_sweep.csv")
    aggregate = frame.groupby("N")[["last", "average", "floor"]].mean().reset_index()
    total_updates = int(frame["T"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    ax.loglog(aggregate["N"], aggregate["last"], marker="o", label="last iterate")
    ax.loglog(
        aggregate["N"], aggregate["floor"], marker="s",
        linestyle="--", label="unseen-feature floor",
    )
    ax.set_xlabel("unique examples $N$")
    ax.set_ylabel("population excess risk")
    ax.set_title(rf"fixed compute: $T={total_updates:,}$ updates")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "fixed_compute.pdf")


def source_exponents(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "source_exponent_sweep.csv")
    predicted: list[float] = []
    measured: list[float] = []
    labels: list[str] = []
    for b, group in frame.groupby("b"):
        aggregate = group.groupby("n")["average"].mean().reset_index()
        predicted.append(-(float(b) - 1.0) / 1.5)
        measured.append(float(np.polyfit(
            np.log(aggregate["n"]), np.log(aggregate["average"]), 1
        )[0]))
        labels.append(rf"$b={b:g}$")

    fig, ax = plt.subplots(figsize=(4.0, 3.25))
    ax.scatter(predicted, measured, s=38)
    lo = min(predicted + measured) - 0.04
    hi = max(predicted + measured) + 0.04
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    for x, y, label in zip(predicted, measured, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted coverage slope")
    ax.set_ylabel("measured coverage slope")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    save(fig, figures / "source_exponents.pdf")


def clipping_stress(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "clipping_stress.csv")
    frame = frame[(frame["mu"] == 2.25) & (frame["eta"] == 1.2)]
    aggregate = frame.groupby(["clip", "K"])["last"].median().reset_index()

    # Plot log10 risk directly.  This keeps the stable clipped trajectory visible
    # while showing the hundreds-of-orders-of-magnitude unclipped divergence.
    display_cap = 300.0
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    for clipped, group in aggregate.groupby("clip"):
        group = group.sort_values("K").copy()
        values = group["last"].to_numpy(dtype=float)
        log_values = np.full_like(values, display_cap)
        finite_positive = np.isfinite(values) & (values > 0.0)
        log_values[finite_positive] = np.minimum(
            np.log10(values[finite_positive]), display_cap
        )
        ax.semilogx(
            group["K"], log_values, marker="o",
            label="clipped" if clipped else "unclipped",
        )
    ax.set_xlabel("epochs $K$")
    ax.set_ylabel(r"$\log_{10}$ population risk")
    ax.set_ylim(-3.0, 310.0)
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "clipping_stress.pdf")



def capacity_frontier(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "capacity_sweep.csv")
    aggregate = frame.groupby("m")[[
        "average", "capacity_tail", "coverage_floor"
    ]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    ax.loglog(
        aggregate["m"], aggregate["average"], marker="o",
        label="clipped replay",
    )
    ax.loglog(
        aggregate["m"], aggregate["capacity_tail"], linestyle="--",
        label="capacity tail",
    )
    ax.loglog(
        aggregate["m"], aggregate["coverage_floor"], linestyle=":",
        label="coverage floor",
    )
    reference = aggregate["average"].iloc[1] * (
        aggregate["m"] / aggregate["m"].iloc[1]
    ) ** (-1.0)
    ax.loglog(
        aggregate["m"], reference, linestyle="-.",
        label=r"$m^{-1}$ reference",
    )
    ax.set_xlabel("model width $m$")
    ax.set_ylabel("population excess risk")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "capacity_frontier.pdf")


def stopping_rule(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "stopping_rule_sweep.csv")
    one = frame.drop_duplicates(["config", "n", "seed"])
    aggregate = one.groupby(["config", "n"])[[
        "predicted_saturation", "measured_saturation"
    ]].median().reset_index()
    aggregate = aggregate[
        np.isfinite(aggregate["predicted_saturation"])
        & np.isfinite(aggregate["measured_saturation"])
    ]

    fig, ax = plt.subplots(figsize=(4.25, 3.25))
    markers = ("o", "s", "^")
    for marker, (name, group) in zip(markers, aggregate.groupby("config")):
        ax.loglog(
            group["predicted_saturation"], group["measured_saturation"],
            marker=marker, linestyle="none", label=name.replace("-", " "),
        )
    lo = min(
        float(aggregate["predicted_saturation"].min()),
        float(aggregate["measured_saturation"].min()),
    )
    hi = max(
        float(aggregate["predicted_saturation"].max()),
        float(aggregate["measured_saturation"].max()),
    )
    ax.loglog([lo, hi], [lo, hi], linestyle="--", linewidth=1,
              label="perfect prediction")
    ax.set_xlabel(r"input-only predicted $K_{\rm sat}$")
    ax.set_ylabel(r"measured $K_{\rm sat}$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    save(fig, figures / "stopping_rule.pdf")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/results")
    )
    parser.add_argument(
        "--figures", type=Path, default=Path("paper/figures")
    )
    args = parser.parse_args()

    risk_vs_epochs(args.results, args.figures)
    scaling_collapse(args.results, args.figures)
    progressive_schedule(args.results, args.figures)
    fixed_compute(args.results, args.figures)
    source_exponents(args.results, args.figures)
    clipping_stress(args.results, args.figures)
    capacity_frontier(args.results, args.figures)
    stopping_rule(args.results, args.figures)


if __name__ == "__main__":
    main()
