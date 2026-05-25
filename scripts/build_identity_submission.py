"""Build a valid identity-model submission for NeuroGolf 2026.

This is a smoke-test baseline: every task receives the same ONNX graph:
input -> Identity -> output, with the static competition shape.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

GRID_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET_IMPORTS = [helper.make_opsetid("", 10)]
TASK_COUNT = 400


def make_identity_model() -> onnx.ModelProto:
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)
    node = helper.make_node("Identity", ["input"], ["output"], name="identity")
    graph = helper.make_graph([node], "identity_baseline", [graph_input], [graph_output])
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(model, full_check=True)
    return model


def build_submission(output_zip: Path, task_count: int = TASK_COUNT) -> None:
    output_zip = output_zip.resolve()
    model_dir = output_zip.parent / "models"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    model = make_identity_model()
    for task_num in range(1, task_count + 1):
        onnx.save(model, model_dir / f"task{task_num:03d}.onnx")

    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for model_path in sorted(model_dir.glob("task*.onnx")):
            archive.write(model_path, arcname=model_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.zip"))
    parser.add_argument("--task-count", type=int, default=TASK_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submission(args.output, args.task_count)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
