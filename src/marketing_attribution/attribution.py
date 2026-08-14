from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ABSORBING_STATES

START = "Start"
CONVERSION = "Conversion"
NULL = "Null"


def first_last_touch(journeys: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    converted = journeys[journeys["converted"].eq(1)].copy()
    if converted.empty:
        raise ValueError("No converted journeys available for attribution")
    first = converted["path"].map(lambda p: p[0]).value_counts(normalize=True).mul(100)
    last = converted["path"].map(lambda p: p[-1]).value_counts(normalize=True).mul(100)
    return first, last


def build_transition_counts(journeys: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    channel_states: set[str] = set()

    for row in journeys[["path", "converted"]].itertuples(index=False):
        path = list(row.path)
        channel_states.update(path)
        terminal = CONVERSION if int(row.converted) == 1 else NULL
        sequence = [START, *path, terminal]
        for current, nxt in zip(sequence, sequence[1:]):
            counts[current][nxt] += 1

    states = [START, *sorted(channel_states), CONVERSION, NULL]
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    for current, destinations in counts.items():
        for nxt, count in destinations.items():
            matrix.loc[current, nxt] = float(count)
    return matrix


def transition_probabilities(counts: pd.DataFrame) -> pd.DataFrame:
    probs = counts.astype(float).copy()
    for state in probs.index:
        if state in ABSORBING_STATES:
            probs.loc[state, :] = 0.0
            probs.loc[state, state] = 1.0
            continue
        row_sum = probs.loc[state].sum()
        if row_sum <= 0:
            probs.loc[state, :] = 0.0
            probs.loc[state, NULL] = 1.0
        else:
            probs.loc[state, :] = probs.loc[state] / row_sum
    return probs


def absorption_probability(probs: pd.DataFrame, start: str = START) -> float:
    if start not in probs.index:
        return 0.0
    transient = [s for s in probs.index if s not in ABSORBING_STATES]
    if start not in transient:
        return 1.0 if start == CONVERSION else 0.0

    q = probs.loc[transient, transient].to_numpy(dtype=float)
    r = probs.loc[transient, CONVERSION].to_numpy(dtype=float)
    a = np.eye(len(transient)) - q
    try:
        x = np.linalg.solve(a, r)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(a, r, rcond=None)[0]
    value = float(x[transient.index(start)])
    return float(np.clip(value, 0.0, 1.0))


def remove_channel(probs: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Remove a channel by redirecting all inbound probability to Null.

    This matches the removal-effect counterfactual used in first-order marketing
    attribution: journeys that would next enter the unavailable channel are treated
    as non-converting at that branch instead of being silently redistributed to
    alternative channels. The channel itself is made unreachable/Null-absorbing.
    """
    if channel not in probs.index or channel in {START, CONVERSION, NULL}:
        raise ValueError(f"Cannot remove channel state: {channel}")

    reduced = probs.copy()
    for state in reduced.index:
        if state in {CONVERSION, NULL, channel}:
            continue
        inbound_mass = float(reduced.loc[state, channel])
        reduced.loc[state, channel] = 0.0
        reduced.loc[state, NULL] += inbound_mass

    reduced.loc[channel, :] = 0.0
    reduced.loc[channel, NULL] = 1.0
    return reduced

def markov_attribution(journeys: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, float]:
    counts = build_transition_counts(journeys)
    probs = transition_probabilities(counts)
    baseline = absorption_probability(probs)
    if baseline <= 0:
        raise ValueError("Baseline Markov conversion probability is zero")

    channels = [s for s in probs.index if s not in {START, CONVERSION, NULL}]
    rows = []
    for channel in channels:
        reduced = remove_channel(probs, channel)
        removed_prob = absorption_probability(reduced)
        raw_effect = 1.0 - (removed_prob / baseline)
        positive_effect = max(raw_effect, 0.0)
        rows.append({
            "Channel": channel,
            "Removed_Conversion_Probability": removed_prob,
            "Removal_Effect": raw_effect,
            "Positive_Removal_Effect": positive_effect,
        })

    details = pd.DataFrame(rows).set_index("Channel")
    denom = details["Positive_Removal_Effect"].sum()
    if denom <= 0:
        raise ValueError("No positive channel removal effects were found")
    markov = details["Positive_Removal_Effect"].div(denom).mul(100)
    return markov, details, probs, baseline


def build_attribution_comparison(journeys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    first, last = first_last_touch(journeys)
    markov, details, probs, baseline = markov_attribution(journeys)
    comparison = pd.concat(
        [
            first.rename("First_Touch_%"),
            last.rename("Last_Touch_%"),
            markov.rename("Markov_%"),
            details[["Removal_Effect", "Removed_Conversion_Probability"]],
        ],
        axis=1,
    ).fillna(0.0)
    comparison.index.name = "Channel"
    return comparison.sort_index(), details.sort_index(), probs, baseline
