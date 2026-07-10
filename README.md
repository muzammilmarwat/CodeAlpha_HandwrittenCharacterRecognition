# CodeAlpha Handwritten Character Recognition

## Objective

Build a professional handwritten digit recognition project using the MNIST dataset, CNN deep learning, image preprocessing, model evaluation, explainability, and Streamlit deployment.

## Project Status

Current milestone: **Phase 3 Complete**

Backend ML pipeline is complete.

Implemented:

- Data pipeline
- CNN model
- Training pipeline
- Evaluation
- Explainability
- Error analysis
- Confidence analysis
- Model card
- Final model selection

Next milestone:
Interactive Streamlit deployment.

Deployment is currently under active development. The trained model and inference pipeline are complete; the interactive Streamlit application will be added in Phase 4.

## Model Performance

| Metric | Value |
| --- | ---: |
| Training Accuracy | 98.66% |
| Validation Accuracy | 99.08% |
| Test Accuracy | 99.03% |
| Test Loss | 0.0298 |
| Test Error Rate | 0.97% |

Reproducibility note: the saved model and metrics were not regenerated after adding seed support. Future training runs use seed `42` for Python, NumPy, and TensorFlow. Exact metrics may still vary slightly across hardware due to TensorFlow CPU optimizations and floating-point behavior.

## Highlights

- CNN-based handwritten digit recognition
- TensorFlow/Keras implementation
- Modular ML pipeline
- Comprehensive model evaluation
- Confidence analysis
- Misclassification analysis
- Saliency-map explainability
- Production-ready repository structure
- Reproducible training workflow

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Key Reports and Artifacts

- [Model card](reports/model_card.md)
- [Final model selection report](reports/final_model_selection/final_model_selection_report.md)
- [Misclassified examples](images/error_analysis/misclassified_examples.png)
- [Saliency examples](images/explainability/saliency_examples.png)

## Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For notebook work, install the optional development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Folder Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
|-- app/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   `-- screenshots/
|-- images/
|   |-- samples/
|   |-- evaluation/
|   |-- error_analysis/
|   `-- explainability/
|-- models/
|-- notebooks/
|-- reports/
|   |-- error_analysis/
|   |-- explainability/
|   `-- final_model_selection/
|-- src/
|   |-- analysis/
|   |-- data/
|   |-- preprocessing/
|   |-- models/
|   |-- training/
|   |-- evaluation/
|   |-- inference/
|   `-- utils/
`-- tests/
```

## Roadmap

### [Complete] Phase 1 - Project Architecture

- Production-ready repository structure
- Modular source code organization
- Initial documentation

### [Complete] Phase 2 - CNN Training & Evaluation

- MNIST data pipeline
- Image preprocessing
- Baseline CNN model
- Model training
- Evaluation pipeline
- Training history
- Classification report
- Confusion matrix
- Inference utilities

### [Complete] Phase 3 - Explainability & Error Analysis

- Confidence analysis
- Misclassification analysis
- Saliency maps
- Per-class performance
- Model card
- Final model selection report

### [Planned] Phase 4 - Interactive Streamlit Application

- Image upload
- Drawing canvas
- Prediction dashboard
- Confidence visualization
- Explainability interface

### [Planned] Phase 5 - Production Release

- Automated testing
- CI workflow
- Documentation polish
- Deployment
- Release v1.0.0
