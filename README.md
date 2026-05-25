# NeuroGolf 2026

This repository contains reproducible submission-building utilities for Kaggle's NeuroGolf 2026 competition.

## Competition

- Kaggle: https://www.kaggle.com/competitions/neurogolf-2026
- Goal: submit tiny ONNX networks named `task001.onnx` through `task400.onnx`.
- Input/output tensor convention from the competition helper: `input` -> `output`, shape `[1, 10, 30, 30]`, float tensors.

## Local Setup

```powershell
python -m pip install -r requirements.txt
```

Download the competition data after accepting the rules on Kaggle:

```powershell
kaggle competitions download neurogolf-2026 -p data
Expand-Archive -Force data/neurogolf-2026.zip -DestinationPath data/raw
```

## Baseline Submission

The identity baseline is a smoke test. It should package correctly, but it scores `0.00` because it returns the input unchanged.

```powershell
python scripts/build_identity_submission.py --output submissions/submission.zip
python scripts/validate_submission.py submissions/submission.zip
```

## Strong Public Baseline

The first score-bearing run uses the public CC0 6029 bundle plus the public 6042 hand-built overrides for tasks `277`, `330`, and `364` from Octavi Grau's Kaggle notebook.

Required public dataset:

```powershell
kaggle datasets download jsrdcht/neurogolf-6029-submission-bundle -p data/external/jsrdcht-6029 --unzip
```

The generated local archive must be named exactly `submission.zip` before upload; Kaggle rejected the same bytes under a different local filename.

```powershell
python scripts/validate_submission.py submissions/submission.zip
kaggle competitions submit neurogolf-2026 -f submissions/submission.zip -m "strong public 6042 bundle baseline"
```

## Current Submitted Runs

- `53006991`: identity smoke test, public score `0.00`.
- `53007196`: strong public 6042 bundle baseline, pending at the time this README was updated.

## Notes

The path to beat is the live public leaderboard leader, currently above `7500`. The public 6042 bundle is a strong starting anchor, not the final target. Next improvements should come from deriving true task rules and replacing individual ONNX files only after local validation.
