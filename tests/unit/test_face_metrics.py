from packages.eval.metrics import face_classification_metrics, flatten_metrics, layout_proxy_metrics


def test_macro_f1_and_confusion_matrix():
    metrics = face_classification_metrics(
        ["indoor", "indoor", "cavity", "cavity"],
        ["indoor", "cavity", "cavity", "cavity"],
    )
    assert 0.0 < metrics["macro_f1"] <= 1.0
    assert "indoor" in metrics["per_class_f1"]
    assert metrics["n"] == 4
    indoor_idx = ["indoor", "corridor", "shaft", "outdoor", "cavity"].index("indoor")
    cavity_idx = ["indoor", "corridor", "shaft", "outdoor", "cavity"].index("cavity")
    assert metrics["confusion_matrix"][indoor_idx][indoor_idx] == 1
    assert metrics["confusion_matrix"][cavity_idx][cavity_idx] == 2


def test_layout_proxy_and_flatten():
    layout = layout_proxy_metrics(["indoor", "cavity"], ["indoor", "indoor"])
    assert layout["false_room_rate"] > 0
    assert layout["room_count_error"] >= 0
    flat = flatten_metrics(
        {
            "macro_f1": 0.5,
            "per_class_f1": {"indoor": 0.8},
            "held_out_passed": True,
            "confusion_matrix": [[1]],
        }
    )
    assert flat["macro_f1"] == 0.5
    assert flat["f1_indoor"] == 0.8
    assert flat["held_out_passed"] == 1.0
    assert "confusion_matrix" not in flat
