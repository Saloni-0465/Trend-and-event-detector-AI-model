import numpy as np

from src.events import detect_spike_events


def test_detect_spike_events_threshold_and_merge():
    # transitions: (bin_i, bin_{i+1})
    transitions = [(i, i + 1) for i in range(6)]
    drift_scores = [0.1, 0.2, 0.9, 0.85, 0.1, 0.95]

    # Threshold chosen so indices 2,3,5 are spikes.
    events = detect_spike_events(
        drift_scores,
        transitions,
        threshold=0.8,
        merge_consecutive=True,
    )
    assert len(events) == 2

    # First event merges indices 2 and 3 => transitions[2]..transitions[3]
    e1 = events[0]
    assert e1.start_bin == 2
    assert e1.end_bin == 4
    assert np.isclose(e1.max_score, 0.9)

    # Second event at index 5 only
    e2 = events[1]
    assert e2.start_bin == 5
    assert e2.end_bin == 6
    assert np.isclose(e2.max_score, 0.95)

