from predictive_lif_memory import LatentPoint, PredictiveReport, PredictiveControllerState
from meta_spike_memory import MetaSpikeConfig, detect_meta_spikes


def make_point(day, state, ratio, effective, error, completion=0.0, micro=True):
    return LatentPoint(
        day=day,
        state=state,
        voltage=ratio * 7.0,
        v_fast=ratio * 7.0,
        v_slow=ratio * 2.0,
        evidence_input=ratio * 3.0,
        completion=completion,
        threshold=7.0,
        ratio=ratio,
        diffused_ratio=effective,
        predicted_ratio=max(0.0, effective - error),
        prediction_error=error,
        error_ema=error,
        micro_threshold_ratio=0.55,
        effective_ratio=effective,
        micro_spike=micro,
        macro_candidate=False,
        official_spike=False,
        topic=f"{state} topic",
    )


def test_meta_spike_triggers_on_high_conflict_and_error():
    points = [
        make_point("2026-06-29", "Experiment", ratio=0.95, effective=1.15, error=0.80, completion=0.0),
        make_point("2026-06-29", "Health", ratio=0.90, effective=1.05, error=0.55, completion=0.0),
        make_point("2026-06-29", "Thesis", ratio=0.80, effective=0.95, error=0.45, completion=0.0),
    ]
    report = PredictiveReport(
        points=points,
        micro_counts={"Experiment": 1, "Health": 1, "Thesis": 1},
        macro_counts={"Experiment": 0, "Health": 0, "Thesis": 0},
        final_error_ema={"Experiment": 0.8, "Health": 0.55, "Thesis": 0.45},
        controller_state=PredictiveControllerState(),
    )

    meta_spikes = detect_meta_spikes(report, MetaSpikeConfig(theta_meta=1.0, max_cards=3))

    assert meta_spikes
    assert meta_spikes[0].meta_energy >= 1.0
    assert meta_spikes[0].first_principle_question
    assert meta_spikes[0].coordinate_reset


def test_meta_spike_stays_silent_when_local_search_is_enough():
    points = [
        make_point("2026-06-29", "Experiment", ratio=0.20, effective=0.25, error=0.05, completion=1.5, micro=False),
        make_point("2026-06-29", "Health", ratio=0.15, effective=0.20, error=0.04, completion=1.2, micro=False),
    ]
    report = PredictiveReport(
        points=points,
        micro_counts={"Experiment": 0, "Health": 0},
        macro_counts={"Experiment": 0, "Health": 0},
        final_error_ema={"Experiment": 0.05, "Health": 0.04},
        controller_state=PredictiveControllerState(),
    )

    meta_spikes = detect_meta_spikes(report, MetaSpikeConfig(theta_meta=1.0, max_cards=3))

    assert meta_spikes == []
