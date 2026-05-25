# NeuroGolf 2026 Baseline

This repository contains a reproducible starter pipeline for Kaggle's NeuroGolf 2026 competition.

The current baseline builds a syntactically valid ONNX identity model for each of the 400 tasks and packages them into `submission.zip`. It is intended as a smoke-test submission pipeline, not as a competitive solution yet.

## Competition

- Kaggle: https://www.kaggle.com/competitions/neurogolf-2026
- Goal: submit tiny ONNX networks named `task001.onnx` through `task400.onnx`.
- Input/output tensor convention from the competition helper: `input` -> `output`, shape `[1, 10, 30, 30]`, float tensors.

## Local Setup

```powershell
python -m pip install -r requirements.txt
```

Download the data after accepting the rules on Kaggle:

```powershell
kaggle competitions download neurogolf-2026 -p data
Expand-Archive -Force data/neurogolf-2026.zip -DestinationPath data/raw
```

## Build Baseline Submission

```powershell
python scripts/build_identity_submission.py --output submissions/submission.zip
python scripts/validate_submission.py submissions/submission.zip
```

Submit with:

```powershell
kaggle competitions submit neurogolf-2026 -f submissions/submission.zip -m "identity baseline smoke test"
```

## Notes

The identity baseline is deliberately simple. It confirms the ONNX format, naming, static shape, and zip packaging before deeper NeuroGolf-specific model work begins.
