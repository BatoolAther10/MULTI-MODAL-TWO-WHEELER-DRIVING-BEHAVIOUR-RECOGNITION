# Two-Wheeler Driving Behaviour Recognition

This project implements a real-time, on-device riding behaviour classifier for a Raspberry Pi 5. The pipeline ingests IMU and GPS data, creates fixed windows, evaluates compact models, and supports field deployment with a configuration-driven runner and fallback logic.

## Project structure

- `data/raw/`: raw CSV input data
- `data/processed/`: cleaned and windowed artifacts
- `inference_layer/`: adapters and factory logic for on-device inference
- `scripts/`: training, benchmarking, and sweep scripts
- `notebooks/`: exploratory and reporting notebooks
- `models/`: trained model artifacts
- `logs/`: benchmark outputs and figures

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical workflow

1. Dataset audit and cleaning
   ```bash
   jupyter notebook notebooks/01_dataset_exploration.ipynb
   ```
2. Window generation
   ```bash
   python -c "from utils.windowing import make_windows; print('windowing ready')"
   ```
3. Train a model
   ```bash
   python scripts/train_xgboost.py
   ```
4. Benchmark registered models
   ```bash
   python scripts/benchmark.py
   ```
5. Run the local pipeline stub
   ```bash
   python main.py
   ```

## Notes

- The pipeline is designed for offline operation on Raspberry Pi 5.
- Model configuration is controlled from `config.yaml`.
- All scripts are intended to be runnable from the project root.
