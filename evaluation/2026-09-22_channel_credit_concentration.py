"""Measure concentration and effective diversity of channel attribution credit."""

from __future__ import annotations

import math


def attribution_concentration(channel_credit: dict[str, float]) -> dict[str, object]:
    """Return normalized shares, HHI, entropy, and effective channel count."""
    if not channel_credit:
        raise ValueError("channel_credit must not be empty")
    if any(value < 0 or not math.isfinite(value) for value in channel_credit.values()):
        raise ValueError("channel credits must be finite and non-negative")
    total = sum(channel_credit.values())
    if total <= 0:
        raise ValueError("total attribution credit must be positive")

    shares = {channel: value / total for channel, value in channel_credit.items()}
    hhi = sum(share * share for share in shares.values())
    entropy = -sum(share * math.log(share) for share in shares.values() if share > 0)
    normalized_entropy = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
    effective_channels = math.exp(entropy)
    dominant_channel = max(shares, key=shares.get)

    return {
        "channel_shares": shares,
        "hhi": hhi,
        "normalized_entropy": normalized_entropy,
        "effective_channel_count": effective_channels,
        "dominant_channel": dominant_channel,
        "dominant_share": shares[dominant_channel],
    }


if __name__ == "__main__":
    credits = {"Paid Search": 0.42, "Organic": 0.25, "Email": 0.18, "Social": 0.10, "Display": 0.05}
    print(attribution_concentration(credits))