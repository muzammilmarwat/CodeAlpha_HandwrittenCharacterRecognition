# 🚀 CodeAlpha Handwritten Character Recognition v1.0.0

**Release Type:** Stable  
**Status:** Production Ready  
**Tag:** `v1.0.0`  
**Target Branch:** `main`

---

# 🎉 Overview

CodeAlpha Handwritten Character Recognition `v1.0.0` is the first stable release of a production-style computer vision project for handwritten digit recognition.

This release delivers an end-to-end MNIST digit recognition system built with TensorFlow/Keras, modular preprocessing and inference services, a trained CNN model, evaluation reports, explainability artifacts, automated tests, CI configuration, and a Streamlit application for interactive predictions.

The project is structured as a deployable machine learning repository rather than a notebook-only experiment. It includes reusable source modules, application services, documentation, reports, release preparation files, and a saved Keras model artifact.

---

# ✨ What's New

- Interactive Streamlit application for handwritten digit recognition
- CNN-based inference using a saved TensorFlow/Keras model
- Image upload prediction for PNG, JPG, and JPEG files
- Built-in MNIST sample prediction mode
- Optional drawing canvas input with defensive fallback behavior
- Preprocessing preview showing the exact 28x28 model input
- Prediction result dashboard with confidence banding
- Top-3 probability visualization
- All-class digit probability chart
- Saliency-map explainability and overlay preview
- Prediction history for the current Streamlit session
- Download center for prediction summaries, history, and reports
- Model information page with metrics and training artifacts
- Error analysis page with confidence and class-level insights
- About page with dataset, workflow, stack, and limitations
- Modular app services for prediction, preprocessing, saliency, samples, history, and downloads
- Automated pytest suite
- Inference smoke test
- Compile verification
- Artifact integrity tests
- GitHub Actions CI workflow
- Deployment guide, release audit, testing checklist, and final screenshot guide
- Runtime dependency cleanup and Python 3.11 deployment configuration

---

# 📊 Model Performance

Metrics are taken from the repository’s final generated model-selection artifacts:

```text
reports/final_model_selection/final_model_selection_summary.csv
reports/final_model_selection/final_model_selection_report.md
```

| Item | Value |
| --- | ---: |
| Dataset | MNIST |
| Training Images | 60,000 |
| Test Images | 10,000 |
| Classes | 10 digits |
| Input Size | 28 x 28 x 1 |
| Model Architecture | Convolutional Neural Network |
| Training Accuracy | 98.66% |
| Validation Accuracy | 99.08% |
| Test Accuracy | 99.03% |
| Test Loss | 0.0298 |
| Test Error Rate | 0.97% |
| Total Incorrect Predictions | 97 |
| Mean Correct Confidence | 99.49% |
| Mean Incorrect Confidence | 74.00% |
| Strongest Digit Class by F1-score | 1 |
| Weakest Digit Class by F1-score | 8 |

---

# 🏗️ Machine Learning Pipeline

The release includes a complete machine learning workflow from dataset loading through deployment-ready inference.

```text
MNIST Dataset
    -> Data Understanding and EDA
    -> Image Preprocessing
    -> CNN Model Training
    -> Model Evaluation
    -> Confidence Analysis
    -> Error Analysis
    -> Saliency Explainability
    -> Final Model Selection
    -> Saved Keras Model
    -> Streamlit Inference App
```

The runtime application uses the saved model artifact and inference services. It does not train during app startup.

---

# 🧠 Model Architecture

The selected model is `mnist_cnn_baseline`, a compact convolutional neural network designed for MNIST-scale grayscale digit classification.

```text
Input: 28 x 28 x 1 grayscale image
Conv2D: 32 filters, ReLU
MaxPooling2D
Conv2D: 64 filters, ReLU
MaxPooling2D
Flatten
Dense: 128 units, ReLU
Dropout: 0.5
Dense: 10 units, softmax
```

The final saved model artifact is:

```text
models/mnist_cnn_baseline.keras
```

---

# 🖥️ Application Features

## Prediction Modes

- Upload digit images in PNG, JPG, or JPEG format
- Select built-in MNIST example digits
- Draw a digit using the optional interactive canvas

## Preprocessing Workspace

- Displays the original input image
- Shows the processed 28x28 image passed to the CNN
- Displays saliency heatmap output
- Displays saliency overlay on the processed input
- Shows source type, processed dimensions, tensor shape, and model name

## Confidence Dashboard

- Predicted digit display
- Confidence percentage
- Confidence band
- Low-confidence or ambiguity warnings when applicable

## Probability Charts

- Top-3 prediction chart
- Top-3 prediction table
- Expandable all-class probability chart

## Explainability

- Gradient-based saliency heatmap
- Saliency overlay preview
- User-facing note that saliency represents model sensitivity, not causal explanation

## Prediction History

- Session-level prediction history
- Timestamp
- Source type
- Predicted digit
- Confidence
- Confidence band
- Top-3 summary
- Clear history action

## Downloads

- Prediction summary as TXT
- Prediction summary as Markdown
- Session prediction history as CSV
- Model and evaluation report downloads

## Navigation

- Prediction workspace
- Model information page
- Error analysis page
- About project page

---

# 📈 Explainability & Evaluation

This release includes evaluation and explainability artifacts for reviewing model behavior beyond raw accuracy.

Included evaluation work:

- Confusion matrix
- Classification report
- Training accuracy curve
- Training loss curve
- Sample prediction visualization
- Per-class performance analysis
- Confidence distribution analysis
- Correct vs incorrect confidence comparison
- Misclassified example visualization
- High-confidence correct prediction examples
- Common confusion pair analysis
- Final model selection summary
- Model card

Key findings reflected in the final repository artifacts:

- The selected CNN reaches **99.03% test accuracy** on MNIST.
- The final model-selection report records **97 incorrect predictions** on the 10,000-image MNIST test set.
- The final model-selection report records a **0.97% test error rate**.
- Digit class `1` is the strongest class by F1-score in the final summary.
- Digit class `8` is the weakest class by F1-score in the final summary.
- Mean confidence is substantially higher for correct predictions than incorrect predictions.
- Saliency maps are included for qualitative model-sensitivity inspection.

---

# 📥 Download Center

The Streamlit application includes downloadable artifacts for both predictions and model documentation.

| Download | Format | Source |
| --- | --- | --- |
| Prediction Summary | TXT | Latest prediction |
| Prediction Summary | Markdown | Latest prediction |
| Session Prediction History | CSV | Current session history |
| Model Card | Markdown | `reports/model_card.md` |
| Final Model Selection Report | Markdown | `reports/final_model_selection/final_model_selection_report.md` |
| Evaluation Summary | JSON | `reports/evaluation_summary.json` |
| Classification Report | CSV | `reports/classification_report.csv` |

---

# 🧪 Testing

The release includes service-level and release-safety tests covering preprocessing, inference contracts, canvas handling, model loading, downloads, history, paths, and artifacts.

Verification performed for the release:

| Check | Result |
| --- | --- |
| `python -m compileall app src` | Passed |
| `python -m app.smoke_test_inference` | Passed |
| `pytest` | 28 passed |
| `pip check` | No broken requirements |
| Streamlit launch check | HTTP 200 |
| Canvas import check | Passed |
| Model artifact integrity | Passed |

The inference smoke test loads the saved Keras model and performs one synthetic canvas-style prediction without downloading MNIST data.

---

# ⚙️ Continuous Integration

GitHub Actions CI is configured in:

```text
.github/workflows/ci.yml
```

The workflow runs on push and pull request events.

CI validates:

- Python 3.11 setup
- Runtime dependency installation from `requirements.txt`
- Source compilation with `python -m compileall app src`
- Inference smoke test with `python -m app.smoke_test_inference`
- Full pytest suite

---

# 🚀 Deployment

The project is deployment-ready for Streamlit-compatible Python 3.11 environments.

| File | Purpose |
| --- | --- |
| `app/app.py` | Streamlit application entrypoint |
| `requirements.txt` | Runtime dependencies |
| `runtime.txt` | Python runtime hint: `python-3.11.9` |
| `.streamlit/config.toml` | Streamlit server and theme configuration |
| `docs/DEPLOYMENT_GUIDE.md` | Deployment instructions |
| `.github/workflows/ci.yml` | Automated validation workflow |

The application deployment configuration is complete for Streamlit Community Cloud or another Python 3.11-compatible hosting platform. No live deployment URL is claimed in this release.

---

# 📚 Documentation

Major documentation included in this release:

- `README.md`
- `docs/DEPLOYMENT_GUIDE.md`
- `docs/TESTING_CHECKLIST.md`
- `docs/FINAL_RELEASE_AUDIT.md`
- `docs/RELEASE_NOTES_v1.0.0.md`
- `docs/GITHUB_RELEASE_NOTES_v1.0.0.md`
- `docs/screenshots/README.md`
- `reports/model_card.md`
- `reports/final_model_selection/final_model_selection_report.md`
- `reports/error_analysis/error_analysis_report.md`
- `reports/explainability/explainability_report.md`

Notebook assets included:

- `notebooks/01_Data_Understanding_and_EDA.ipynb`
- `notebooks/02_Model_Training_and_Evaluation.ipynb`
- `notebooks/03_Model_Explainability_and_Error_Analysis.ipynb`

---

# 🗂️ Repository Improvements

This release includes several repository-level improvements:

- Production-style folder organization
- Separate `app/` and `src/` layers
- Modular services for inference, preprocessing, saliency, history, downloads, and samples
- Reusable path and logging utilities
- Typed prediction schemas
- Runtime dependency cleanup
- Notebook-only dependencies moved to development requirements
- Streamlit app import resolution fix
- Defensive optional canvas integration
- CI workflow added
- Release documentation added
- Screenshot preparation guide added
- Artifact integrity tests added
- `.gitignore` cleanup
- Final `.keras` model artifact explicitly allowed for release tracking

Recent repository history includes:

- Initial architecture setup
- CNN training and evaluation
- Explainability and error reporting
- Reproducibility improvements
- Streamlit inference services
- Premium Streamlit interface
- Drawing canvas integration
- Streamlit package import fix
- v1.0.0 release preparation

---

# 🔒 Reproducibility

The training workflow records deterministic seed support for future runs:

```text
random.seed(42)
numpy.random.seed(42)
tensorflow.random.set_seed(42)
```

The repository documentation notes that the saved model and metrics were generated before seed support was added. Future training runs use seed `42`, but exact metrics may still vary slightly across hardware due to TensorFlow CPU optimizations and floating-point behavior.

---

# 🛠️ Technology Stack

## Machine Learning

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- MNIST dataset

## Visualization

- Matplotlib
- Seaborn
- Pandas

## Backend

- Modular Python services
- Pathlib-based path handling
- Typed dataclasses
- Reusable logging utilities

## Frontend

- Streamlit
- Streamlit drawable canvas
- Interactive prediction dashboard
- Download center

## Testing

- Pytest
- Compile verification
- Inference smoke test
- Artifact integrity tests
- Dependency validation with `pip check`

## CI/CD

- GitHub Actions
- Python 3.11 workflow
- Automated compile, smoke, and pytest checks

## Deployment

- Streamlit app entrypoint
- `runtime.txt`
- `.streamlit/config.toml`
- Deployment guide

---

# 📁 Project Statistics

| Category | Value |
| --- | --- |
| Programming Language | Python |
| Primary ML Framework | TensorFlow/Keras |
| Frontend Framework | Streamlit |
| Dataset | MNIST |
| Number of Classes | 10 |
| Input Shape | 28 x 28 x 1 |
| Prediction Modes | Upload, MNIST sample, drawing canvas |
| Download Formats | TXT, Markdown, CSV, JSON |
| Test Count | 28 passing tests |
| CI | GitHub Actions |
| Deployment Target | Streamlit-compatible Python 3.11 host |
| Release Type | Stable |
| First Stable Release | Yes |

---

# ⚠️ Known Limitations

- The model is trained on MNIST digits only.
- The application does not recognize letters or multi-digit numbers.
- MNIST images are cleaner and more standardized than many real-world handwritten inputs.
- Real-world photos may require better lighting, cropping, contrast, and centering.
- Saliency maps show model sensitivity, not causal reasoning.
- No adversarial robustness testing is included.
- No production monitoring is configured yet.
- No public live deployment URL is included in this release.
- Final application screenshots are included in `docs/screenshots/`.
- Demo GIF is not included in this release.

---

# 🔮 Future Improvements

- Add a demo GIF
- Deploy to Streamlit Community Cloud
- Add a public demo link after deployment
- Add an architecture diagram image
- Extend dataset coverage with EMNIST
- Add confidence calibration experiments
- Add robustness testing for noisier handwritten inputs
- Add deployment monitoring notes
- Remove legacy UI helper modules after final manual QA if no longer needed

---

# 🙏 Acknowledgements

Thanks to the open-source tools and datasets that made this project possible:

- CodeAlpha Machine Learning Internship
- MNIST handwritten digit dataset
- TensorFlow and Keras
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pytest
- GitHub Actions

---

# 📄 Educational Disclaimer

This project was developed as part of the CodeAlpha Machine Learning Internship for educational and portfolio purposes.

It is not intended for medical, legal, financial, security, accessibility, identity-verification, or other high-stakes decision-making use cases without additional validation, monitoring, governance, and human review.

---

# 🏷️ Release Summary

| Field | Value |
| --- | --- |
| Version | `v1.0.0` |
| Tag | `v1.0.0` |
| Release Type | Stable |
| Branch | `main` |
| Status | Production Ready |
| Codename | Baseline Vision |
