"""Validate a NeuroGolf submission zip before upload."""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import onnx

TASK_RE = re.compile(r"^task(\d{3})\.onnx$")
MAX_FILE_BYTES = int(1.44 * 1024 * 1024)
EXPECTED_TASKS = 400


def validate_submission(path: Path, expected_tasks: int = EXPECTED_TASKS) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad_names = [name for name in names if not TASK_RE.match(Path(name).name) or Path(name).name != name]
        if bad_names:
            raise ValueError(f"Invalid archive paths: {bad_names[:5]}")
        if len(names) > expected_tasks:
            raise ValueError(f"Expected at most {expected_tasks} models, found {len(names)}")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate filenames in archive")

        task_nums = sorted(int(TASK_RE.match(name).group(1)) for name in names)
        out_of_range = [num for num in task_nums if num < 1 or num > expected_tasks]
        if out_of_range:
            raise ValueError(f"Task numbers out of range: {out_of_range[:5]}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in names:
                info = archive.getinfo(name)
                if info.file_size > MAX_FILE_BYTES:
                    raise ValueError(f"{name} exceeds NeuroGolf file size limit")
                archive.extract(name, tmp_path)
                model = onnx.load(tmp_path / name)
                onnx.checker.check_model(model, full_check=True)
                graph = model.graph
                if len(graph.input) != 1 or graph.input[0].name != "input":
                    raise ValueError(f"{name} does not expose one input named input")
                if len(graph.output) != 1 or graph.output[0].name != "output":
                    raise ValueError(f"{name} does not expose one output named output")

    print(f"OK: {path} contains {len(names)} valid ONNX model(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_submission(args.submission, args.expected_tasks)


if __name__ == "__main__":
    main()
