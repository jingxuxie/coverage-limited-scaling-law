#!/usr/bin/env python3
"""Exact pilots for the one-hot and singleton-filtered replay theorems."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gamma, zeta


def log_pi0(c: float, s: float, tol: float = 1e-15) -> float:
    """Return log prod_j(1-c j^{-s}) via its convergent zeta series."""
    total = 0.0
    ell = 1
    while True:
        term = c**ell * float(zeta(s * ell, 1.0)) / ell
        total += term
        if abs(term) <= tol:
            return -total
        ell += 1
        if ell > 100_000:
            raise RuntimeError("pi_0 series did not converge")


def slope(xs: list[float], ys: list[float]) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ExactRisk:
    def __init__(
        self,
        *,
        s: float,
        a: float,
        b: float,
        eta: float,
        m: int,
        activation: str,
        expected_active: float,
    ) -> None:
        if not (s > 1 and b > 1 and a >= s and 0 < eta < 1 and m > 0):
            raise ValueError("require s>1, b>1, a>=s, 0<eta<1, and m>0")
        self.s, self.a, self.b, self.eta, self.m = s, a, b, eta, m
        self.activation = activation
        j = np.arange(1, m + 1, dtype=np.float64)
        self.q2 = j ** (-(a - s))
        self.g = j ** (-b) / float(zeta(b, 1.0))
        self.tail = float(zeta(b, m + 1) / zeta(b, 1.0))
        self.log_update = np.log1p(-eta * self.q2)

        if activation == "one_hot":
            c = 1.0 / float(zeta(s, 1.0))
            self.sample_rate = c * j ** (-s)
            self.rate_prefactor = c
            self.metadata = {"activation_prefactor": c}
        elif activation == "singleton_filtered":
            c = expected_active / float(zeta(s, 1.0))
            if not 0 < c < 1:
                raise ValueError("expected_active / zeta(s) must lie in (0,1)")
            pi0 = math.exp(log_pi0(c, s))
            p = c * j ** (-s)
            self.sample_rate = pi0 * p / (1.0 - p)
            self.rate_prefactor = pi0 * c
            self.metadata = {
                "activation_prefactor": c,
                "all_inactive_probability": pi0,
                "singleton_rate_prefactor": pi0 * c,
            }
        else:
            raise ValueError("unknown activation mode")

    def risk(self, n: int, epochs: int) -> float:
        residual = np.exp(2.0 * epochs * self.log_update)
        learned_per_draw = self.sample_rate * (1.0 - residual)
        return float(self.g @ np.exp(n * np.log1p(-learned_per_draw))) + self.tail

    def sharp_floor_constant(self) -> float:
        cg = 1.0 / float(zeta(self.b, 1.0))
        return (
            cg
            / self.s
            * float(gamma((self.b - 1.0) / self.s))
            * self.rate_prefactor ** (-(self.b - 1.0) / self.s)
        )


def run_mode(model: ExactRisk, out: Path) -> dict[str, object]:
    epochs = [2**k for k in range(16)]
    ns = [512, 4096, 32768]
    epoch_rows: list[dict[str, object]] = []
    early_slopes: dict[str, float] = {}
    plateau_ratios: dict[str, float] = {}
    for n in ns:
        risks = [model.risk(n, k) for k in epochs]
        for k, risk in zip(epochs, risks):
            epoch_rows.append({"n": n, "epochs": k, "risk": f"{risk:.16e}"})
        early_slopes[str(n)] = slope(epochs[:4], risks[:4])
        plateau_ratios[str(n)] = risks[-1] / n ** (-(model.b - 1) / model.s)
    write_csv(out / "epoch_sweep.csv", epoch_rows)

    coverage_ns = [2**k for k in range(7, 17)]
    coverage_risks = [model.risk(n, epochs[-1]) for n in coverage_ns]
    write_csv(
        out / "coverage_sweep.csv",
        [
            {"n": n, "epochs": epochs[-1], "risk": f"{risk:.16e}"}
            for n, risk in zip(coverage_ns, coverage_risks)
        ],
    )

    total = 2**20
    reuse = [1, 4, 16, 64, 256, 1024, 4096]
    fixed = [
        {
            "total_updates": total,
            "n_unique": total // k,
            "epochs": k,
            "risk": f"{model.risk(total // k, k):.16e}",
        }
        for k in reuse
    ]
    write_csv(out / "fixed_compute_sweep.csv", fixed)

    summary: dict[str, object] = {
        "mode": model.activation,
        "s": model.s,
        "a": model.a,
        "b": model.b,
        "eta": model.eta,
        "m": model.m,
        **model.metadata,
        "predicted_optimization_slope": -(model.b - 1) / model.a,
        "fitted_early_slopes": early_slopes,
        "predicted_coverage_slope": -(model.b - 1) / model.s,
        "fitted_coverage_slope": slope(coverage_ns[-6:], coverage_risks[-6:]),
        "sharp_predicted_floor_constant": model.sharp_floor_constant(),
        "measured_floor_ratios": plateau_ratios,
        "fixed_compute_degradation": float(fixed[-1]["risk"]) / float(fixed[0]["risk"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=float, default=1.5)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.125)
    parser.add_argument("--m", type=int, default=200_000)
    parser.add_argument("--expected-active", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    summaries = {}
    for mode in ("one_hot", "singleton_filtered"):
        model = ExactRisk(
            s=args.s,
            a=args.a,
            b=args.b,
            eta=args.eta,
            m=args.m,
            activation=mode,
            expected_active=args.expected_active,
        )
        summaries[mode] = run_mode(model, args.out / mode)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
