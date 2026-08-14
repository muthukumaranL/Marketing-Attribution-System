import numpy as np
import pandas as pd
import pytest

from src.marketing_attribution.attribution import (
    absorption_probability, build_transition_counts, markov_attribution,
    remove_channel, transition_probabilities,
)


def sample_journeys():
    return pd.DataFrame({
        "path": [["Facebook"], ["Facebook", "Instagram"], ["Instagram"], ["Paid Search"]],
        "converted": [1, 1, 0, 0],
    })


def test_transition_matrix_is_stochastic_and_baseline_matches_empirical_rate():
    journeys = sample_journeys()
    probabilities = transition_probabilities(build_transition_counts(journeys))
    assert np.allclose(probabilities.sum(axis=1).to_numpy(), 1.0)
    assert absorption_probability(probabilities) == pytest.approx(journeys["converted"].mean(), abs=1e-12)


def test_markov_removal_reduces_conversion_for_helpful_channel():
    journeys = pd.DataFrame({
        "path": [["Facebook"], ["Facebook"], ["Instagram"], ["Instagram"]],
        "converted": [1, 1, 0, 0],
    })
    markov, details, probabilities, baseline = markov_attribution(journeys)
    reduced = remove_channel(probabilities, "Facebook")
    assert np.allclose(reduced.sum(axis=1).to_numpy(), 1.0)
    assert 0 < baseline < 1
    assert details.loc["Facebook", "Removed_Conversion_Probability"] < baseline
    assert details.loc["Facebook", "Removal_Effect"] > 0
    assert markov.sum() == pytest.approx(100.0, abs=1e-12)
