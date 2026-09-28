from apps.trainer.train import NOTEBOOK_TRAIN_HINT, main


def test_train_cli_points_at_notebook_without_torch(monkeypatch, capsys):
    monkeypatch.setattr("packages.models_nn.face_train_kernel.torch_available", lambda: False)
    try:
        main()
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 2
    assert "notebook" in capsys.readouterr().out.lower()
    assert "gnn_c_train" in NOTEBOOK_TRAIN_HINT
