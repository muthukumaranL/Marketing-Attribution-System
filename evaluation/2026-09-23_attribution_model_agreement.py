"""Compare channel-credit agreement between attribution models."""

from __future__ import annotations

import math


def attribution_model_agreement(
    model_a: dict[str, float],
    model_b: dict[str, float],
) -> dict[str, object]:
    """Compare normalized credit shares with total variation and rank agreement."""
    channels = sorted(set(model_a) | set(model_b))
    if not channels:
        raise ValueError("at least one channel is required")
    for credits in (model_a, model_b):
        if any(value < 0 or not math.isfinite(value) for value in credits.values()):
            raise ValueError("credits must be finite and non-negative")
        if sum(credits.values()) <= 0:
            raise ValueError("each model must assign positive total credit")

    total_a, total_b = sum(model_a.values()), sum(model_b.values())
    share_a = {c: model_a.get(c, 0.0) / total_a for c in channels}
    share_b = {c: model_b.get(c, 0.0) / total_b for c in channels}
    deltas = {c: share_b[c] - share_a[c] for c in channels}
    tv_distance = 0.5 * sum(abs(delta) for delta in deltas.values())

    order_a = sorted(channels, key=lambda c: (-share_a[c], c))
    order_b = sorted(channels, key=lambda c: (-share_b[c], c))
    rank_a = {c: i + 1 for i, c in enumerate(order_a)}
    rank_b = {c: i + 1 for i, c in enumerate(order_b)}
    mean_rank_shift = sum(abs(rank_a[c] - rank_b[c]) for c in channels) / len(channels)
    largest = sorted(channels, key=lambda c: abs(deltas[c]), reverse=True)

    return {
        "total_variation_distance": float(tv_distance),
        "mean_absolute_rank_shift": float(mean_rank_shift),
        "same_top_channel": order_a[0] == order_b[0],
        "largest_credit_shifts": [{"channel": c, "share_a": share_a[c], "share_b": share_b[c], "delta": deltas[c]} for c in largest],
    }


if __name__ == "__main__":
    markov = {"Search": 0.40, "Email": 0.25, "Social": 0.20, "Display": 0.15}
    shapley = {"Search": 0.33, "Email": 0.31, "Social": 0.22, "Display": 0.14}
    print(attribution_model_agreement(markov, shapley))
