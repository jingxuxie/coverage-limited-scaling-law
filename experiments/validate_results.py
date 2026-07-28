#!/usr/bin/env python3
"""Validate the checked-in scaling-law evidence against preregistered tolerances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/results/multi_active_summary.json"),
    )
    parser.add_argument(
        "--diagnostics-summary",
        type=Path,
        default=Path("experiments/results/frontier_diagnostics_summary.json"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    diagnostics = json.loads(
        args.diagnostics_summary.read_text(encoding="utf-8")
    )

    all_data = summary["all_data"]
    fit = all_data["two_term_fit"]["average"]
    require(fit["r_squared"] >= 0.95, "two-term fit R^2 fell below 0.95")
    require(
        fit["median_relative_error"] <= 0.15,
        "two-term median relative error exceeded 15%",
    )
    require(
        abs(
            all_data["measured_early_average_slope"]
            - all_data["predicted_optimization_slope"]
        )
        <= 0.08,
        "optimization-limited slope missed prediction by more than 0.08",
    )
    require(
        abs(
            all_data["measured_large_epoch_average_slope"]
            - all_data["predicted_coverage_slope"]
        )
        <= 0.03,
        "coverage-limited slope missed prediction by more than 0.03",
    )

    progressive = summary["progressive_schedule"]
    require(
        abs(progressive["average_slope"] + 0.5) <= 0.03,
        "progressive averaged-iterate slope is not close to -1/2",
    )
    require(
        abs(progressive["last_slope"] + 0.5) <= 0.03,
        "progressive last-iterate slope is not close to -1/2",
    )

    fixed = summary["fixed_compute"]
    require(
        fixed["risk_ratio_fewest_over_most_unique"] >= 10.0,
        "fresh-data advantage at fixed compute fell below 10x",
    )

    for exponent, result in summary["source_exponents"].items():
        require(
            abs(result["average"] - result["predicted"]) <= 0.03,
            f"source exponent b={exponent} missed by more than 0.03",
        )

    for config, result in summary["spectrum_grid"].items():
        require(
            result["two_term_r_squared"] >= 0.95,
            f"spectrum-grid two-term fit failed for {config}",
        )
        require(
            abs(
                result["measured_optimization_slope"]
                - result["predicted_optimization_slope"]
            )
            <= 0.07,
            f"spectrum-grid optimization slope failed for {config}",
        )
        require(
            abs(
                result["measured_coverage_slope"]
                - result["predicted_coverage_slope"]
            )
            <= 0.06,
            f"spectrum-grid coverage slope failed for {config}",
        )

    capacity = diagnostics["capacity"]
    require(
        abs(capacity["measured_average_slope"] - capacity["predicted_slope"])
        <= 0.03,
        "capacity-frontier slope missed prediction by more than 0.03",
    )
    require(
        abs(capacity["measured_tail_slope"] - capacity["predicted_slope"])
        <= 0.02,
        "analytic capacity-tail slope missed prediction by more than 0.02",
    )

    for config, result in diagnostics["stopping_rule"].items():
        require(
            abs(
                result["mean_estimated_exponent"]
                - result["predicted_saturation_exponent"]
            )
            <= 0.04,
            f"input-only saturation exponent failed for {config}",
        )
        require(
            result["median_epoch_factor_error"] <= 1.8,
            f"input-only saturation factor error exceeded 1.8x for {config}",
        )

    reproducibility = summary["reproducibility"]
    require(
        reproducibility["risk_checkpoints"] == 4285,
        "unexpected number of checked-in risk checkpoints",
    )
    require(
        reproducibility["training_runs"] == 670,
        "unexpected number of checked-in training runs",
    )

    report = {
        "status": "passed",
        "two_term_r_squared": fit["r_squared"],
        "median_relative_error": fit["median_relative_error"],
        "optimization_slope_error": abs(
            all_data["measured_early_average_slope"]
            - all_data["predicted_optimization_slope"]
        ),
        "coverage_slope_error": abs(
            all_data["measured_large_epoch_average_slope"]
            - all_data["predicted_coverage_slope"]
        ),
        "fixed_compute_fresh_data_gain": fixed[
            "risk_ratio_fewest_over_most_unique"
        ],
        "capacity_slope_error": abs(
            capacity["measured_average_slope"] - capacity["predicted_slope"]
        ),
        "combined_training_runs": reproducibility["training_runs"] + 125,
        "combined_risk_checkpoints": reproducibility["risk_checkpoints"] + 1115,
        **reproducibility,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
