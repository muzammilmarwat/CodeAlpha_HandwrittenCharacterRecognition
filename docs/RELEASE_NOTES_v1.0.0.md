# CodeAlpha Handwritten Character Recognition v1.0.0

## Overview

This release prepares a complete production-style handwritten digit recognition project using the MNIST dataset and a Convolutional Neural Network (CNN). The architecture has been designed to be extensible to handwritten alphabet recognition using the EMNIST dataset and, in future work, full word recognition using CRNN-based sequence models.

## Highlights

- CNN-based digit classifier
- Saved Keras model artifact
- Upload, sample, and canvas inference modes
- Preprocessing preview
- Confidence and probability visualization
- Saliency-map explainability
- Error analysis and model card
- Final Streamlit interface screenshots
- Automated tests and CI workflow

## ML Pipeline

The pipeline includes MNIST loading, preprocessing, CNN training, evaluation, confidence analysis, error analysis, saliency explainability, and saved inference artifacts.

## Model Performance

- Training Accuracy: 98.66%
- Validation Accuracy: 99.08%
- Test Accuracy: 99.03%
- Test Loss: 0.0298
- Test Error Rate: 0.97%

## Streamlit Features

- Upload PNG/JPG/JPEG images
- Select MNIST sample digits
- Draw digits with an interactive canvas
- View original and processed images
- Inspect top-3 and all-class probabilities
- View confidence band and ambiguity warnings
- Review saliency heatmap and overlay
- Track session prediction history
- Download summaries and reports

## Canvas Input

Canvas input uses `streamlit-drawable-canvas==0.9.3` and is imported defensively so upload and sample modes continue to work if the component cannot load.

## Explainability

Gradient-based saliency maps show pixel sensitivity for model predictions. They are not causal explanations.

## Error Analysis

The project includes misclassified examples, high-confidence correct examples, common confusion pairs, confidence distributions, and per-class performance.

## Testing and CI

The release includes service tests, smoke test, compile check, `pip check`, and GitHub Actions CI.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Deployment Status

Deployment configuration is complete for Streamlit Community Cloud or another Python 3.11-compatible host. No live deployment URL is claimed in this release.

## Known Limitations

- MNIST is cleaner than real-world handwriting.
- Digits only; no alphabet recognition.
- Saliency is sensitivity-based, not causal.
- No adversarial robustness testing.

## Future Improvements

- Add demo GIF
- Deploy to Streamlit Community Cloud
- Add monitoring notes
- Extend to EMNIST

## Educational Disclaimer

This project is intended for education, internship, and portfolio use only.
