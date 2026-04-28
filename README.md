# Explainable AI for Black-Box Models in Medical Diagnosis

Research-driven Streamlit MVP dashboard for clinician decision support across diabetes risk, heart disease risk, and pneumonia pattern detection from chest X-rays.

## Safety Disclaimer

This prediction is for decision support only and must be reviewed by a qualified clinician.

This system is not an autonomous diagnosis tool.

## Overview

This application reuses trained research artifacts and serves inference-time outputs with explainability displays:

- Diabetes prediction using saved stacking model + saved imputer/scaler.
- Heart disease prediction using saved sklearn pipeline.
- Pneumonia image prediction using saved DenseNet121 Keras model.
- Rule-based natural-language summaries (deterministic, no LLM usage).
- Static SHAP/LIME/Grad-CAM notebook outputs as fallback visual evidence.

## Supported Tasks

1. Diabetes risk prediction from tabular input.
2. Heart disease prediction from tabular input.
3. Pneumonia detection from chest X-ray upload.
4. Explainability rendering for each module.

## Research Background

This dashboard is based on a full experimental research pipeline implemented in a Kaggle notebook. The notebook contains the model training code, feature engineering steps, and the explainability experiments (SHAP, LIME, Grad-CAM) together with XAI quality metrics and visual examples.

Kaggle notebook (canonical reference):

https://www.kaggle.com/code/ismrahik/xai-medical-diagnosis-final

## Features

- Streamlit‑based web dashboard for interactive inference and explainability
- Multi‑disease support: Diabetes, Heart Disease, Pneumonia
- Rule‑based natural‑language clinical summaries for image/tabular predictions
- SHAP explanations for tabular models (when environment supports SHAP)
- LIME local explanations for selected examples
- Grad‑CAM visualizations for chest X‑ray images (dynamic + static fallbacks)
- Confidence‑based interpretation bands and decision thresholds
- Static and dynamic galleries of XAI figures for reproducibility

## Technology Stack

- Python
- Streamlit
- TensorFlow / Keras
- Scikit‑learn, XGBoost, LightGBM
- SHAP, LIME
- NumPy, OpenCV, Pillow, Matplotlib


## Project Structure

```text
project_root/
|-- app.py
|-- requirements.txt
|-- README.md
|-- NOTES_FOR_COPILOT.md
|-- config/
|   |-- manifest.json
|   |-- thresholds.json
|   |-- settings.py
|   `-- paths.py
|-- notebook_reference/
|   |-- xai_medical_diagnosis_FINAL.ipynb
|   `-- notebook_extracted.py
|-- artifacts/
|   |-- diabetes/
|   |   |-- models/
|   |   `-- preprocessors/
|   |-- heart/
|   |   `-- models/
|   `-- pneumonia/
|       `-- models/
|-- static_outputs/
|   |-- diabetes/
|   |-- heart/
|   |-- pneumonia/
|   `-- dashboard/
|-- src/
|   |-- components/
|   |-- pages/
|   |-- pipelines/
|   |-- preprocessing/
|   |-- features/
|   |-- models/
|   |-- explainability/
|   |-- utils/
|   `-- assets/
`-- tests/
```

## Artifact Placement Instructions

Place artifacts exactly as follows:

### Diabetes

- `artifacts/diabetes/models/best_Diabetes_Stack.pkl`
- `artifacts/diabetes/preprocessors/diab_knn_imp.pkl`
- `artifacts/diabetes/preprocessors/diab_scaler.pkl`
- Diabetes static figures in `static_outputs/diabetes/`

### Heart

- `artifacts/heart/models/best_Heart_pipe.pkl`
- Heart static figures in `static_outputs/heart/`

### Pneumonia

- `artifacts/pneumonia/models/best_pneumonia_densenet121.keras`
- Pneumonia static figures in `static_outputs/pneumonia/`

### Global Dashboard Outputs

- Place dashboard images and CSVs in `static_outputs/dashboard/`

All paths are centrally managed in `config/manifest.json`.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy artifacts to the folders listed above.

## Run

```bash
streamlit run app.py
```

## Tests

```bash
pytest -q
```

## Limitations

- This MVP depends on external artifact files and does not retrain models.
- Local tabular explanations are rule-based summaries and not full real-time SHAP recomputation.
- Grad-CAM can fail on some runtimes; static fallback images are displayed in that case.

## Future Work

1. Add authenticated clinician workspace and audit logs.
2. Add richer uncertainty calibration views.
3. Add artifact schema checks and model version registry.
4. Expand automated tests for UI and artifact contracts.
