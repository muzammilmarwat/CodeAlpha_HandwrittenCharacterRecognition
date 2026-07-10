# Deployment Guide

This project has complete deployment configuration for Streamlit Community Cloud or another Python 3.11-compatible hosting platform. It is not currently claimed as live deployed.

## 1. Prerequisites

- Python 3.11
- Git
- A machine or cloud service with enough memory to load TensorFlow and the saved Keras model

## 2. Clone Repository

```bash
git clone <repository-url>
cd CodeAlpha_HandwrittenCharacterRecognition
```

## 3. Create Python 3.11 Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate Environment on Windows

```powershell
.\.venv\Scripts\activate
```

## 5. Activate Environment on Linux/macOS

```bash
source .venv/bin/activate
```

## 6. Install Requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Run Tests

```bash
python -m compileall app src
python -m app.smoke_test_inference
pytest
pip check
```

## 8. Launch Streamlit

```bash
streamlit run app/app.py
```

## 9. Required Artifacts

The app requires:

- `models/mnist_cnn_baseline.keras`
- `reports/model_card.md`
- `reports/final_model_selection/final_model_selection_report.md`

The model file is about 2.7 MB and is intended to be included with the repository.

## 10. Streamlit Community Cloud Setup

- Runtime: Python 3.11
- Main file path: `app/app.py`
- Requirements file: `requirements.txt`
- No secrets are required.

## 11. Python-Version Requirements

Use Python 3.11. TensorFlow compatibility may fail on newer unsupported Python versions.

## 12. TensorFlow Deployment Considerations

TensorFlow can be memory-intensive on small containers. Native Windows TensorFlow >= 2.11 runs CPU-only unless WSL2 or compatible plugins are used.

## 13. Troubleshooting

### Missing Model

If the app reports a missing model, confirm:

```text
models/mnist_cnn_baseline.keras
```

exists in the repository or deployment package.

### Canvas Dependency

Canvas input uses:

```text
streamlit-drawable-canvas==0.9.3
```

If it cannot load, upload and sample prediction modes remain available because the app imports the canvas component defensively.

### Missing Reports

Required report files must exist under `reports/`. Do not regenerate reports in deployment unless intentionally running the analysis pipeline.

## 14. Memory and Resource Notes

Use a hosting tier with enough RAM for TensorFlow import, model loading, and image processing.
