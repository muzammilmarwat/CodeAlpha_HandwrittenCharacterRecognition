# CodeAlpha Handwritten Character Recognition

## Objective

Build a professional handwritten digit recognition project using the MNIST dataset, CNN deep learning, image preprocessing, model evaluation, and Streamlit deployment.

## Current Status

Phase 3 is complete: project setup, MNIST CNN training/evaluation, inference helpers, confidence analysis, error analysis, explainability artifacts, model card, and final model selection reporting are included. Streamlit deployment will be implemented in a later phase.

Model performance summary:

- Training accuracy: approximately 98.66%
- Validation accuracy: approximately 98.98%
- Test accuracy: approximately 98.94%
- Test loss: approximately 0.0286

Key reports and artifacts:

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
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── screenshots/
├── images/
│   ├── samples/
│   └── evaluation/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
└── tests/
```

## Next Phases

1. Data understanding and exploratory analysis.
2. Image preprocessing pipeline.
3. CNN model training.
4. Model evaluation and reporting.
5. Streamlit deployment.
