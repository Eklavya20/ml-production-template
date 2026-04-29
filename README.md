# ml-production-template

A production-ready Machine Learning template built around a real churn prediction problem.
Clone it, swap the dataset, and have a fully operational ML system running in minutes.

![CI](https://github.com/Eklavya20/ml-production-template/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What's inside

| Layer | Tool |
|---|---|
| Experiment tracking | MLflow |
| Training pipeline | Prefect |
| Model serving | FastAPI |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Dataset | IBM Telco Customer Churn |
| Quality gates | [ML Guardian](https://github.com/Eklavya20/ml-guardian) |

---

## Architecture
```
raw data
   │
   ▼
┌──────────────────────────────┐
│     Prefect Training Flow    │
│                              │
│  prepare → split → build  → │
│  train → evaluate → register │
└──────────────┬───────────────┘
               │ logs + registers
               ▼
        ┌─────────────┐
        │   MLflow    │◄──── experiment UI (localhost:5000)
        │  Model Reg  │
        └──────┬──────┘
               │ loads Production model
               ▼
        ┌─────────────┐
        │  FastAPI    │◄──── POST /predict (localhost:8000)
        │  Serve API  │      GET  /health
        └─────────────┘      GET  /docs
```

---

## Evaluation outputs

Every training run logs the following artifacts to MLflow automatically:

| Plot | What it tells you |
|---|---|
| ROC Curve | Discrimination ability across all thresholds |
| Precision-Recall Curve | Performance under class imbalance |
| Calibration Curve | Whether predicted probabilities are trustworthy |
| Confusion Matrix | Errors at the chosen operating threshold |

Threshold is tuned to maximise F1 before final evaluation.

---

## Data Validation

Input data is validated against a Pandera schema before any training run begins.
The pipeline checks column types, value ranges, and allowed categorical values.

| Check | Detail |
|---|---|
| `tenure` | Integer, must be ≥ 0 |
| `MonthlyCharges` | Float, must be > 0 |
| `TotalCharges` | Float, must be ≥ 0 |
| `Contract` | Must be one of: Month-to-month, One year, Two year |
| `Churn` | Must be one of: Yes, No |

Invalid data raises immediately with a detailed failure report — the training run never starts.
Validation status and input shape are logged to MLflow on every successful run.

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/Eklavya20/ml-production-template.git
cd ml-production-template
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Add the dataset

Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place it at:
```
data/raw/telco_churn.csv
```

### 3. Run the training pipeline
```bash
cd src
python -m training.train
```

Or via Prefect:
```bash
python pipelines/train_pipeline.py
```

### 4. View experiments in MLflow
```bash
mlflow ui --backend-store-uri mlruns
```

Open [http://localhost:5000](http://localhost:5000)

### 5. Promote model to Production

In the MLflow UI, navigate to the `telco_churn` registered model and transition
the latest version to **Production**.

### 6. Start the API
```bash
cd src
uvicorn serving.app:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

---

## Run with Docker Compose

Spins up MLflow + the FastAPI serving layer together:
```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 359.20
  }'
```

Response:
```json
{
  "churn_probability": 0.7431,
  "churn_prediction": true,
  "threshold_used": 0.5
}
```

---

## Project structure
```
ml-production-template/
├── .github/workflows/       # CI/CD — test + Docker build on every push
├── configs/
│   └── config.yaml          # Single source of truth for all parameters
├── data/
│   └── raw/                 # Raw CSV (not committed — see .gitignore)
├── src/
│   ├── features/            # TelcoPreprocessor + feature engineering
│   ├── training/            # Model training + MLflow logging
│   ├── evaluation/          # Threshold tuning + diagnostic plots
│   └── serving/             # FastAPI app
├── pipelines/
│   └── train_pipeline.py    # Prefect flow — end-to-end training DAG
├── tests/                   # pytest — features, training, serving
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Running tests
```bash
pytest tests/ -v --cov=src
```

---

## Adapting to your own problem

1. Drop your dataset in `data/raw/`
2. Update `configs/config.yaml` — target column, feature lists, model params
3. Swap out `TelcoPreprocessor` in `src/features/build_features.py` for your own cleaning logic
4. Run the pipeline — everything else (MLflow logging, serving, CI) works as-is

---

## Related Projects

- [`diagnost`](https://github.com/Eklavya20/diagnost) : model diagnostics library  
- [`ml-guardian`](https://github.com/Eklavya20/ml-guardian) : automated quality gates and auto-promotion for MLflow models  

---

## License

MIT