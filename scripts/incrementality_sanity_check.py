"""Sanity-check experiment incrementality with a difference in proportions.

Attribution scores can look persuasive even when a campaign does not create
incremental conversions. This small utility compares treated and control
conversion rates and reports an approximate 95% confidence interval for lift.
"""

from __future__ import annotations

from math import sqrt


def incrementality_check(
    treated_conversions: int,
    treated_total: int,
    control_conversions: int,
    control_total: int,
    *,
    z_value: float = 1.96,
) -> dict[str, float | bool]:
    if treated_total <= 0 or control_total <= 0:
        raise ValueError("group totals must be positive")
    if not 0 <= treated_conversions <= treated_total:
        raise ValueError("treated conversions must be within group size")
    if not 0 <= control_conversions <= control_total:
        raise ValueError("control conversions must be within group size")

    treated_rate = treated_conversions / treated_total
    control_rate = control_conversions / control_total
    absolute_lift = treated_rate - control_rate
    relative_lift = absolute_lift / control_rate if control_rate > 0 else float("inf")

    standard_error = sqrt(
        treated_rate * (1 - treated_rate) / treated_total
        + control_rate * (1 - control_rate) / control_total
    )
    ci_low = absolute_lift - z_value * standard_error
    ci_high = absolute_lift + z_value * standard_error

    return {
        "treated_rate": treated_rate,
        "control_rate": control_rate,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "positive_incrementality": ci_low > 0,
    }


if __name__ == "__main__":
    result = incrementality_check(
        treated_conversions=486,
        treated_total=10000,
        control_conversions=402,
        control_total=10000,
    )
    print("Incrementality sanity check")
    for metric, value in result.items():
        if isinstance(value, bool):
            print(f"{metric:>26}: {value}")
        else:
            print(f"{metric:>26}: {value:.4f}")
