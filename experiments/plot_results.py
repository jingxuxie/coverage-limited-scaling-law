#!/usr/bin/env python3
"""Create paper figures from the checked-in sparse replay sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes


# Render text through LaTeX/NewTX so AAAI figures contain embedded Type 1
# fonts rather than Matplotlib's default Type 3 or Identity-H fonts.
COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X")

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 11.0,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=COLORS),
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.55,
    "ytick.minor.width": 0.55,
    "lines.linewidth": 1.65,
    "lines.markersize": 4.2,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.55,
    "legend.borderaxespad": 0.35,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axis(ax: Axes) -> None:
    """Apply the shared open-frame academic style to one axes."""
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both", which="both", top=False, right=False, color="#333333",
    )
    ax.grid(
        True, which="major", color="#C9CED3", linewidth=0.45, alpha=0.62,
    )
    ax.margins(x=0.025)


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.align_labels()
    fig.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={
            "Creator": "coverage-limited-scaling-law",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def risk_vs_epochs(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "all_data_sweep.csv")
    frame = frame[frame["protocol"] == "with_replacement_clipped"]
    aggregate = frame.groupby(["n", "K"], as_index=False).agg(
        average=("average", "mean"),
        average_se=("average", "sem"),
        floor=("floor", "mean"),
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.65))
    for index, n in enumerate((512, 2048, 8192)):
        group = aggregate[aggregate["n"] == n]
        epochs = group["K"].to_numpy(dtype=float)
        mean = group["average"].to_numpy(dtype=float)
        se = group["average_se"].fillna(0.0).to_numpy(dtype=float)
        line, = ax.loglog(
            epochs,
            mean,
            marker=MARKERS[index],
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=rf"$n={n}$",
        )
        ax.fill_between(
            epochs,
            np.maximum(mean - se, np.finfo(float).tiny),
            mean + se,
            color=line.get_color(),
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.axhline(
            group["floor"].iloc[0],
            linestyle=(0, (4, 2)),
            linewidth=1.05,
            color=line.get_color(),
            alpha=0.82,
        )
    ax.set_xlabel("epochs $K$")
    ax.set_ylabel("population excess risk")
    ax.legend()
    style_axis(ax)
    save(fig, figures / "risk_vs_epochs.pdf")


def scaling_collapse(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "all_data_sweep.csv")
    frame = frame[frame["protocol"] == "with_replacement_clipped"]
    aggregate = frame.groupby(["n", "K"], as_index=False).agg(
        average=("average", "mean"),
        average_se=("average", "sem"),
        floor=("floor", "mean"),
        a=("a", "first"),
        s=("s", "first"),
        b=("b", "first"),
        eta=("eta", "first"),
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.65))
    for index, (n, group) in enumerate(aggregate.groupby("n")):
        a = float(group["a"].iloc[0])
        s = float(group["s"].iloc[0])
        b = float(group["b"].iloc[0])
        eta = float(group["eta"].iloc[0])
        x = eta * group["K"] * n ** (1.0 - a / s)
        y = group["average"] * n ** ((b - 1.0) / s)
        y_se = group["average_se"].fillna(0.0) * n ** ((b - 1.0) / s)
        line, = ax.loglog(
            x,
            y,
            marker=MARKERS[index % len(MARKERS)],
            markerfacecolor="white",
            markeredgewidth=0.75,
            label=rf"$n={n}$",
        )
        ax.fill_between(
            x,
            np.maximum(y - y_se, np.finfo(float).tiny),
            y + y_se,
            color=line.get_color(),
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
    ax.set_xlabel(r"normalized epochs $\eta K n^{1-a/s}$")
    ax.set_ylabel(r"normalized risk $n^{(b-1)/s}\mathcal{E}$")
    ax.legend(ncol=2)
    style_axis(ax)
    save(fig, figures / "scaling_collapse.pdf")


def progressive_schedule(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "progressive_schedule_sweep.csv")
    aggregate = frame.groupby("T_eff")[
        ["last", "average", "floor"]
    ].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 2.65))
    ax.loglog(
        aggregate["T_eff"],
        aggregate["average"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Polyak average",
    )
    ax.loglog(
        aggregate["T_eff"],
        aggregate["last"],
        marker="s",
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="last iterate",
    )
    reference = aggregate["average"].iloc[-1] * (
        aggregate["T_eff"] / aggregate["T_eff"].iloc[-1]
    ) ** (-0.5)
    ax.loglog(
        aggregate["T_eff"],
        reference,
        color="#555555",
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        label=r"$T_{\rm eff}^{-1/2}$ slope guide",
    )
    ax.set_xlabel(r"actual updates $T_{\rm eff}$")
    ax.set_ylabel("population excess risk")
    ax.legend()
    style_axis(ax)
    save(fig, figures / "progressive_schedule.pdf")


def fixed_compute(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "fixed_compute_sweep.csv")
    aggregate = frame.groupby("N")[["last", "average", "floor"]].mean().reset_index()
    total_updates = int(frame["T"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.8, 2.65))
    ax.loglog(
        aggregate["N"],
        aggregate["last"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="last iterate",
    )
    ax.loglog(
        aggregate["N"],
        aggregate["floor"],
        marker="s",
        markerfacecolor="white",
        markeredgewidth=0.8,
        linestyle=(0, (4, 2)),
        label="unseen-feature floor",
    )
    ax.set_xlabel("unique examples $N$")
    ax.set_ylabel("population excess risk")
    ax.set_title(
        rf"Fixed compute: $T={total_updates:,}$ updates",
        loc="left",
        pad=6,
    )
    ax.legend()
    style_axis(ax)
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

    fig, ax = plt.subplots(figsize=(4.0, 2.65))
    ax.scatter(
        predicted,
        measured,
        s=40,
        color=COLORS[0],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )
    lo = min(predicted + measured) - 0.04
    hi = max(predicted + measured) + 0.04
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#555555",
        linestyle=(0, (4, 2)),
        linewidth=1.1,
    )
    for x, y, label in zip(predicted, measured, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9.5,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted coverage slope")
    ax.set_ylabel("measured coverage slope")
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax)
    save(fig, figures / "source_exponents.pdf")


def clipping_stress(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "clipping_stress.csv")
    frame = frame[(frame["mu"] == 2.25) & (frame["eta"] == 1.2)]
    aggregate = frame.groupby(["clip", "K"])["last"].median().reset_index()

    # Plot log10 risk directly.  This keeps the stable clipped trajectory visible
    # while showing the hundreds-of-orders-of-magnitude unclipped divergence.
    display_cap = 300.0
    fig, ax = plt.subplots(figsize=(4.8, 2.65))
    for clipped, group in aggregate.groupby("clip"):
        group = group.sort_values("K").copy()
        values = group["last"].to_numpy(dtype=float)
        log_values = np.full_like(values, display_cap)
        finite_positive = np.isfinite(values) & (values > 0.0)
        log_values[finite_positive] = np.minimum(
            np.log10(values[finite_positive]), display_cap
        )
        ax.semilogx(
            group["K"],
            log_values,
            color=COLORS[0] if clipped else COLORS[1],
            marker="o" if clipped else "s",
            markerfacecolor="white",
            markeredgewidth=0.8,
            label="clipped" if clipped else "unclipped",
        )
    ax.set_xlabel("epochs $K$")
    ax.set_ylabel(r"$\log_{10}$ population risk")
    ax.set_ylim(-3.0, 310.0)
    ax.legend()
    style_axis(ax)
    save(fig, figures / "clipping_stress.pdf")


def capacity_frontier(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "capacity_sweep.csv")
    aggregate = frame.groupby("m")[[
        "average", "capacity_tail", "coverage_floor"
    ]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 2.65))
    ax.loglog(
        aggregate["m"],
        aggregate["average"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Polyak-averaged clipped replay",
    )
    ax.loglog(
        aggregate["m"],
        aggregate["capacity_tail"],
        linestyle=(0, (4, 2)),
        label="capacity tail",
    )
    ax.loglog(
        aggregate["m"],
        aggregate["coverage_floor"],
        linestyle=(0, (1, 1.6)),
        label="capacity + coverage floor",
    )
    ax.set_xlabel("model width $m$")
    ax.set_ylabel("population excess risk")
    ax.legend()
    style_axis(ax)
    save(fig, figures / "capacity_frontier.pdf")


def stopping_rule(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "stopping_rule_sweep.csv")
    one = frame.drop_duplicates(["config", "n", "seed"])
    aggregate = one.groupby(["config", "n"], as_index=False).agg(
        predicted_saturation=("predicted_saturation", "median"),
        measured_saturation=("measured_saturation", "median"),
        a=("a", "first"),
        s=("s", "first"),
    )
    aggregate = aggregate[
        np.isfinite(aggregate["predicted_saturation"])
        & np.isfinite(aggregate["measured_saturation"])
    ]

    fig, ax = plt.subplots(figsize=(4.25, 2.65))
    markers = ("o", "s", "^")
    for index, (marker, (_name, group)) in enumerate(
        zip(markers, aggregate.groupby("config"))
    ):
        a = float(group["a"].iloc[0])
        s = float(group["s"].iloc[0])
        ax.loglog(
            group["predicted_saturation"],
            group["measured_saturation"],
            color=COLORS[index],
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=5.0,
            linestyle="none",
            label=rf"$(a,s)=({a:g},{s:g})$",
        )
    lo = min(
        float(aggregate["predicted_saturation"].min()),
        float(aggregate["measured_saturation"].min()),
    )
    hi = max(
        float(aggregate["predicted_saturation"].max()),
        float(aggregate["measured_saturation"].max()),
    )
    ax.loglog(
        [lo, hi],
        [lo, hi],
        color="#555555",
        linestyle=(0, (4, 2)),
        linewidth=1.1,
        label="uncalibrated identity",
    )
    ax.loglog(
        [lo, hi],
        [2.0 * lo, 2.0 * hi],
        color="#888888",
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        label=r"$2\times$ and $\frac{1}{2}\times$",
    )
    ax.loglog(
        [lo, hi],
        [0.5 * lo, 0.5 * hi],
        color="#888888",
        linestyle=(0, (2, 2)),
        linewidth=0.9,
    )
    ax.set_xlabel(r"input-only predicted $K_{\rm sat}$")
    ax.set_ylabel(r"measured $K_{\rm sat}$")
    ax.legend()
    style_axis(ax)
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
