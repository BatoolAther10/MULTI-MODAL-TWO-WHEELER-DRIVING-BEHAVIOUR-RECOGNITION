# Two-Wheeler Behaviour Recognition Project Rules

## Project Scope

- Target platform: Raspberry Pi 5, ARM64, 8 GB RAM.
- Task: classify `STRAIGHT`, `LEFT`, `RIGHT`, `BUMP`, and `STOP` riding events.
- Sensor features: `[Ax, Ay, Az, Gx, Gy, Gz, Speed]`.
- Default window: 156 samples with stride 78.

## Interface Contract

- Every inference model implements `BaseOnDeviceModel` from `inference_layer/base_adapter.py`.
- Preserve the existing `load(path: str) -> None`, `predict(window: np.ndarray) -> np.ndarray`, and `latency_ms` signatures.
- Do not break existing adapters or change the base interface.
- Register new adapters in `inference_layer/model_factory.py`.

## Data and Paths

- Read the BOM-bearing CSV with `encoding="utf-8-sig"`.
- Use paths from `config.yaml`; do not hardcode experiment paths in scripts.
- Main artifacts are under `data/processed/`, `models/`, and `logs/`.
- Keep generated metrics in `logs/*.csv` or `logs/*.json`.
- Save figures as PNG at 300 dpi under `logs/figures/`.

## Reproducibility

- Put tunable experiment values in `config.yaml`.
- Use the configured experiment seed where applicable.
- Every experiment must have a standalone script under `scripts/` with an `if __name__ == "__main__":` entry point.
- Report the data source, split type, number of windows, and known limitations in generated reports.
- Do not describe full-window fallback measurements as independent holdout results.

## Runtime

- The configured primary and fallback models are controlled by `config.yaml`.
- The runtime fallback is activated when the primary raises or exceeds the configured latency budget.
- Keep queueing and sensor processing bounded and protect the live inference boundary from model errors.
- Distinguish model inference latency from total end-to-end latency.

## Code Style

- Use Python type hints on every function.
- Give every function a concise one-line docstring.
- Prefer existing project helpers and adapter patterns over new abstractions.
- Keep edits focused and avoid unrelated refactors.
- Use ASCII by default and avoid unnecessary inline comments.
- Fail with a clear diagnostic and non-zero exit status when a standalone experiment cannot run.

## Validation

- After edits, run a focused executable check when available.
- Compile touched Python files with `python -m py_compile` when appropriate.
- Validate generated CSV schemas, row counts, artifact paths, and figure existence.
- Benchmark deployment artifacts, not only in-memory training models.
- Raspberry Pi latency must be measured separately before making deployment claims.
