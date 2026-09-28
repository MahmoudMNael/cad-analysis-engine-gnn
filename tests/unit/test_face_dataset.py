from packages.contracts.enums import LabelSource
from packages.eval.datasets import (
    FaceGraphSample,
    TrainingRow,
    collapse_face_labels,
    face_graphs_from_rows,
    normalize_face_label,
    split_by_building_held_out,
    validate_face_dataset,
)
from packages.models_nn.features.face import FACE_NODE_FEATURE_KEYS


def _row(
    *,
    job_id: str,
    target_id: str,
    label: str,
    source: LabelSource = LabelSource.USER_EXPLICIT,
    weight: float = 0.75,
    features: dict | None = None,
    building_id: str | None = None,
) -> TrainingRow:
    return TrainingRow(
        sample_id=f"{job_id}-{target_id}-{label}",
        job_id=job_id,
        building_id=building_id or job_id,
        target_id=target_id,
        feature_schema_version="face_features_v1",
        features=features or {"area": 12.0, "perimeter": 14.0},
        label=label,
        label_source=source,
        weight=weight,
        engine_sha="sha",
        model_input_version="face_features_v1",
    )


def test_not_indoor_maps_to_cavity():
    assert normalize_face_label("not_indoor") == "cavity"
    assert normalize_face_label("indoor") == "indoor"
    assert normalize_face_label("junk") is None


def test_weak_labels_do_not_count_for_eligibility():
    rows = [
        _row(job_id="j1", target_id="f1", label="indoor", source=LabelSource.WEAK, weight=0.0),
        _row(job_id="j1", target_id="f1", label="indoor"),
    ]
    collapsed = collapse_face_labels(rows)
    assert ("j1", "f1") in collapsed
    weak_only = [_row(job_id="j1", target_id="f1", label="indoor", source=LabelSource.WEAK, weight=0.0)]
    eligibility = validate_face_dataset(weak_only, min_labeled_jobs=1, min_classes=1)
    assert eligibility.ok is False


def test_eligibility_requires_fifty_jobs_by_default():
    rows = [
        _row(job_id=f"j{i}", target_id="f1", label="indoor" if i % 2 == 0 else "cavity")
        for i in range(10)
    ]
    eligibility = validate_face_dataset(rows)
    assert eligibility.ok is False
    assert eligibility.labeled_jobs == 10
    assert any("50" in reason for reason in eligibility.reasons)


def test_eligibility_passes_with_enough_jobs_and_classes():
    rows = [
        _row(job_id=f"j{i}", target_id="f1", label="indoor" if i % 2 == 0 else "cavity")
        for i in range(50)
    ]
    eligibility = validate_face_dataset(rows, min_labeled_jobs=50, min_classes=2)
    assert eligibility.ok is True
    assert eligibility.n_classes == 2


def test_face_graphs_isolated_without_wkt():
    rows = [
        _row(job_id="job-a", target_id="f1", label="indoor"),
        _row(job_id="job-a", target_id="f2", label="cavity"),
    ]
    graphs = face_graphs_from_rows(rows)
    assert len(graphs) == 1
    assert graphs[0].face_ids == ["f1", "f2"]
    assert graphs[0].edge_index == []
    assert graphs[0].labels[0] >= 0


def test_face_graphs_use_snapshot_neighbors():
    rows = [_row(job_id="job-a", target_id="f1", label="indoor", features={"area": 20.0, "perimeter": 18.0})]
    columns = {
        "job-a": {
            "face_id": ["f1", "f2"],
            "features": ['{"area": 20.0, "perimeter": 18.0}', '{"area": 0.4, "perimeter": 2.6}'],
            "wkt": [
                "POLYGON((0 0, 5 0, 5 4, 0 4, 0 0))",
                "POLYGON((5 0, 10 0, 10 4, 5 4, 5 0))",
            ],
            "area": [20.0, 20.0],
            "perimeter": [18.0, 18.0],
        }
    }
    graphs = face_graphs_from_rows(rows, faces_columns_by_job=columns)
    graph = graphs[0]
    assert graph.face_ids == ["f1", "f2"]
    assert graph.labels[0] >= 0
    assert graph.labels[1] == -1
    assert graph.edge_index


def test_held_out_split_never_splits_a_building():
    samples = [
        FaceGraphSample(
            job_id=f"j{i}",
            building_id=f"b{i // 2}",
            face_ids=[f"f{i}"],
            node_features=[[float(i)] * len(FACE_NODE_FEATURE_KEYS)],
            edge_index=[],
            edge_attr=[],
            labels=[0],
            weights=[1.0],
        )
        for i in range(8)
    ]
    train, val, test = split_by_building_held_out(samples, train_ratio=0.5, val_ratio=0.25, seed=0)
    train_b = {s.building_id for s in train}
    val_b = {s.building_id for s in val}
    test_b = {s.building_id for s in test}
    assert train_b
    assert val_b
    assert test_b
    assert train_b.isdisjoint(val_b)
    assert train_b.isdisjoint(test_b)
    assert val_b.isdisjoint(test_b)
    jobs_by_building: dict[str, set[str]] = {}
    for sample in (*train, *val, *test):
        jobs_by_building.setdefault(sample.building_id, set()).add(sample.job_id)
    original = {}
    for sample in samples:
        original.setdefault(sample.building_id, set()).add(sample.job_id)
    assert jobs_by_building == original
