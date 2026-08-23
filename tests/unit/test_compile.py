from pathlib import Path

from packages.contracts.enums import FeedbackAction, LabelSource
from packages.eval.datasets import TrainingRow, split_by_building, weight_for
from apps.trainer.compile import compile_job_artifacts
from tests.unit.test_view_reducer import _event, _layout, _room


def test_weak_accept_has_zero_weight():
    assert weight_for(LabelSource.WEAK) == 0.0
    assert weight_for(LabelSource.EXPERT) > weight_for(LabelSource.USER_EXPLICIT)


def test_split_by_building_never_splits_same_job():
    rows = [
        TrainingRow(
            sample_id=f"s{i}",
            job_id="job-a" if i < 3 else "job-b",
            building_id="job-a" if i < 3 else "job-b",
            target_id=f"face-{i}",
            feature_schema_version="face_features_v1",
            features={"area": float(i)},
            label="indoor",
            label_source=LabelSource.USER_EXPLICIT,
            weight=0.75,
            engine_sha="abc",
            model_input_version="face_features_v1",
        )
        for i in range(6)
    ]
    train, val = split_by_building(rows, train_ratio=0.5, seed=1)
    train_jobs = {row.job_id for row in train}
    val_jobs = {row.job_id for row in val}
    assert train_jobs.isdisjoint(val_jobs)
    assert all(row.job_id in train_jobs for row in rows if row.job_id in train_jobs)


def test_compile_delete_and_draw_wall_rows(tmp_path: Path):
    layout = _layout(_room("r1", 0, 0, 5, 4))
    events = [
        _event(FeedbackAction.DELETE_ROOM, target_ids=["r1"], sequence=1),
        _event(
            FeedbackAction.DRAW_WALL,
            target_ids=[],
            payload={"start_x": 0.0, "start_y": 0.0, "end_x": 1.0, "end_y": 0.0},
            sequence=2,
        ),
        _event(
            FeedbackAction.MARK_ENTITY,
            target_ids=["prim-1"],
            payload={"class": "wall"},
            sequence=3,
        ),
        _event(FeedbackAction.ACCEPT, target_ids=["r1"], sequence=4),
        _event(FeedbackAction.DELETE_ROOM, target_ids=["r1"], sequence=5, retracted="evt-6"),
    ]
    lake = compile_job_artifacts(
        job_id="job1",
        building_id="job1",
        engine_sha="engine",
        consent_training=True,
        physical_layout=layout,
        faces_columns={"face_id": ["r1"], "features": ['{"area": 20.0}']},
        plan_graph={"nodes": [], "edges": []},
        document={"primitives": [{"id": "prim-1", "length": 1.0, "color": 7}]},
        events=events,
    )
    assert any(row.label == "not_indoor" and row.target_id == "r1" for row in lake.face_cls)
    assert sum(1 for row in lake.face_cls if row.label == "not_indoor") == 1
    assert any(row.label == "close" for row in lake.link_pred)
    assert any(row.label == "wall" and row.target_id == "prim-1" for row in lake.edge_cls)
    assert all(row.weight == 0.0 for row in lake.face_cls if row.label_source == LabelSource.WEAK)
    paths = lake.write_parquet(tmp_path / "lake")
    assert paths["face_cls"].is_file()
    assert paths["edge_cls"].is_file()
    assert paths["link_pred"].is_file()


def test_compile_skips_without_consent():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    lake = compile_job_artifacts(
        job_id="job1",
        building_id="job1",
        engine_sha="engine",
        consent_training=False,
        physical_layout=layout,
        faces_columns={"face_id": ["r1"], "features": ["{}"]},
        plan_graph={},
        document={},
        events=[_event(FeedbackAction.DELETE_ROOM, target_ids=["r1"])],
        require_consent=True,
    )
    assert lake.face_cls == []
    assert lake.edge_cls == []
    assert lake.link_pred == []
