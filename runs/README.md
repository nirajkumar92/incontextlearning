# Generated outputs

This directory is reserved for local checkpoints, example data and experiment outputs. Its contents are ignored except for this guide. The old three- and four-update tiny-model smoke artifacts were removed after their results were recorded in `../research/results/code_validation.json`.

Generate fresh outputs from the project root:

```bash
PYTHONPATH=src .venv/bin/python -m tabular_foundation.train --config configs/smoke.json --output runs/my_smoke
PYTHONPATH=src .venv/bin/python -m tabular_foundation.train --config configs/smoke_finance.json --output runs/my_finance_smoke
.venv/bin/python scripts/make_example_data.py
```

Use a new output directory for each experiment. Retain checkpoints from substantive training and preserve the original directory for ledger-backed resume. The removed diagnostics are not a precedent for deleting future trained models.
