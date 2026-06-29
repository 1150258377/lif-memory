from predictive_lif_memory import (
    PredictiveConfig,
    PredictiveControllerState,
    build_predictive_report,
)


def test_predictive_layer_keeps_micro_trace_without_official_spike():
    timeline = [
        {
            "date": "2026-06-28",
            "Experiment": {
                "new_v": 3.5,
                "new_fast": 3.9,
                "new_slow": 1.2,
                "input": 3.5,
                "completion": 0.0,
                "effective_threshold": 7.0,
                "spike": False,
                "topic": "LIF链路",
            },
            "Thesis": {
                "new_v": 1.0,
                "new_fast": 1.0,
                "new_slow": 0.5,
                "input": 1.0,
                "completion": 0.0,
                "effective_threshold": 7.0,
                "spike": False,
                "topic": "论文闭环",
            },
        },
        {
            "date": "2026-06-29",
            "Experiment": {
                "new_v": 4.2,
                "new_fast": 4.8,
                "new_slow": 2.0,
                "input": 2.8,
                "completion": 0.0,
                "effective_threshold": 7.0,
                "spike": False,
                "topic": "LIF链路",
            },
            "Thesis": {
                "new_v": 1.4,
                "new_fast": 1.3,
                "new_slow": 0.8,
                "input": 1.2,
                "completion": 0.0,
                "effective_threshold": 7.0,
                "spike": False,
                "topic": "论文闭环",
            },
        },
    ]

    report = build_predictive_report(
        timeline=timeline,
        active_states=["Experiment", "Thesis"],
        controller_state=PredictiveControllerState(),
        config=PredictiveConfig(micro_ratio=0.50, prediction_error_weight=0.45),
    )

    assert report.micro_counts["Experiment"] >= 1
    assert all(not point.official_spike for point in report.points)
    assert report.final_error_ema["Experiment"] > 0


def test_homeostasis_lowers_micro_threshold_when_system_is_silent():
    timeline = [
        {
            "date": "2026-06-28",
            "Experiment": {
                "new_v": 0.1,
                "new_fast": 0.1,
                "new_slow": 0.0,
                "input": 0.1,
                "completion": 0.0,
                "effective_threshold": 7.0,
                "spike": False,
            },
        }
    ]

    controller_state = PredictiveControllerState()
    build_predictive_report(
        timeline=timeline,
        active_states=["Experiment"],
        controller_state=controller_state,
        config=PredictiveConfig(micro_ratio=0.55, target_micro_rate=0.35),
    )

    assert controller_state.micro_threshold_delta["Experiment"] < 0
