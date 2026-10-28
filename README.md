# MLflow Experiments

Scripts that track scikit-learn experiments with MLflow, locally and on DagsHub.

## Why this exists
Practice with experiment tracking: logging parameters, metrics and artifacts, and comparing runs.

## What it does
- `file1.py`: trains a random forest on the Wine dataset and logs to a local MLflow server
- `file2.py`: the same run logged to a remote MLflow server on DagsHub, with tags
- `autolog.py`: the same model using `mlflow.autolog()`
- `hyperparameter1.py`: grid search on the Breast Cancer dataset with each trial logged
- Each run saves a confusion matrix and the script itself as artifacts

## Project structure
```
mlops_experiments/
├── src/               Training and tracking scripts
├── mlartifacts/       Logged run artifacts
├── test.py            Checks the tracking URI
├── mlflow.txt         Notes: what MLflow can log
├── mlflow-basics.txt  Notes: MLflow basics
└── mlautofile.txt     Notes: autologging
```

## Getting started
Requirements: Python 3.9+.
```bash
git clone https://github.com/abhijha8287/mlops_experiments.git
cd mlops_experiments
pip install mlflow scikit-learn matplotlib seaborn dagshub
mlflow ui --port 5000            # in one terminal
python src/file1.py              # in another
```
Open http://localhost:5000 to compare runs.

## Configuration
`file2.py` logs to `https://dagshub.com/abhijha8287/mlops_experiments.mlflow` and needs DagsHub credentials. Use `file1.py` to stay local.

## Tech stack
Python, scikit-learn, MLflow, DagsHub
