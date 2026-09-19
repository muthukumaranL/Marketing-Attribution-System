"""Monitor whether channel attribution shares move materially over time."""

from __future__ import annotations

from math import log


def channel_credit_drift(
    reference: dict[str, float],
    candidate: dict[str, float],
    *,
    epsilon: float = 1e-9,
) -> dict[str, object]:
    """Compare normalized channel credit distributions with JSD and share deltas."""
    if not reference or not candidate or epsilon <= 0:
        raise ValueError("credit mappings must be non-empty and epsilon positive")
    channels = sorted(set(reference) | set(candidate))
    if any(reference.get(c, 0) < 0 or candidate.get(c, 0) < 0 for c in channels):
        raise ValueError("channel credits must be non-negative")
    ref_total, cur_total = sum(reference.values()), sum(candidate.values())
    if ref_total <= 0 or cur_total <= 0:
        raise ValueError("each credit mapping must have positive total credit")

    p = [reference.get(c, 0.0) / ref_total for c in channels]
    q = [candidate.get(c, 0.0) / cur_total for c in channels]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * log((x + epsilon) / (y + epsilon), 2) for x, y in zip(a, b) if x > 0)

    deltas = {c: float(b - a) for c, a, b in zip(channels, p, q)}
    return {
        "js_divergence_bits": float(0.5 * kl(p, m) + 0.5 * kl(q, m)),
        "largest_absolute_share_shift": float(max(abs(v) for v in deltas.values())),
        "largest_shift_channel": max(deltas, key=lambda c: abs(deltas[c])),
        "share_shift": deltas,
    }


if __name__ == "__main__":
    before = {"search": 0.45, "email": 0.30, "social": 0.25}
    after = {"search": 0.30, "email": 0.32, "social": 0.38}
    print(channel_credit_drift(before, after))
