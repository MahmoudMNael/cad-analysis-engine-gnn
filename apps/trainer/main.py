"""Trainer CLI for GNN-C: compile, train, evaluate, promote."""

from __future__ import annotations

import sys

USAGE = "usage: python -m apps.trainer.main {compile,train,evaluate,promote} ..."


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(USAGE)
    cmd, rest = args[0], args[1:]
    if cmd == "compile":
        from apps.trainer.compile import main as compile_main

        sys.argv = [sys.argv[0], *rest]
        compile_main()
        return
    if cmd == "train":
        from apps.trainer.train import main as train_main

        sys.argv = [sys.argv[0], *rest]
        train_main()
        return
    if cmd == "evaluate":
        from apps.trainer.evaluate import main as evaluate_main

        sys.argv = [sys.argv[0], *rest]
        evaluate_main()
        return
    if cmd == "promote":
        from apps.trainer.promote import main as promote_main

        sys.argv = [sys.argv[0], *rest]
        promote_main()
        return
    raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
