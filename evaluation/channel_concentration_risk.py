"""Measure how concentrated attributed conversions are across channels.

Run:
    python evaluation/channel_concentration_risk.py

The audit reports channel shares, Herfindahl-Hirschman Index (HHI), effective
channel count, and the top-channel share. It helps distinguish diversified
attribution from a portfolio that depends heavily on one channel.
"""
from __future__ import annotations


def concentration_metrics(attribution: dict[str, float]) -> dict[str, float]:
    if not attribution or any(v < 0 for v in attribution.values()):
        raise ValueError("attribution values must be non-negative and non-empty")
    total = sum(attribution.values())
    if total <= 0:
        raise ValueError("attribution total must be positive")
    shares = {k: v / total for k, v in attribution.items()}
    hhi = sum(share * share for share in shares.values())
    return {
        "hhi": hhi,
        "effective_channel_count": 1.0 / hhi,
        "top_channel_share": max(shares.values()),
    }


if __name__ == "__main__":
    demo = {
        "Paid Search": 3400,
        "Organic Search": 2200,
        "Email": 1500,
        "Social": 900,
        "Display": 500,
    }
    metrics = concentration_metrics(demo)
    total = sum(demo.values())
    print("Channel shares:")
    for channel, value in sorted(demo.items(), key=lambda item: item[1], reverse=True):
        print(f"- {channel}: {value / total:.1%}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
