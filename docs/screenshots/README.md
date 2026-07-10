# Screenshot Preparation

Screenshots are pending manual capture. Do not fabricate screenshots; capture them from the running Streamlit app after final visual verification.

Run the app:

```bash
streamlit run app/app.py
```

Required filenames:

- `home.png`: compact hero, sidebar, and input methods.
- `canvas_input.png`: Draw Digit mode with a visible drawn digit and clear button.
- `prediction_result.png`: preprocessing previews, prediction result, confidence, top-3 probabilities, and saliency.
- `model_information.png`: model performance cards, metadata, training curves, and evaluation plots.
- `error_analysis.png`: error metrics, confusion pairs, per-class performance, and analysis images.
- `about_project.png`: workflow, technology stack, limitations, and educational disclaimer.

Optional filenames:

- `upload_prediction.png`: upload workflow with a prediction.
- `sample_prediction.png`: sample digit workflow with a prediction.
- `demo.gif`: 30-60 second app walkthrough.

Demo GIF flow:

1. Open the app.
2. Choose Draw Digit.
3. Draw a digit.
4. Click Recognize Digit.
5. Review result, confidence, probabilities, and saliency.
6. Open Model Information.
7. Open Error Analysis.

Use only built-in samples or hand-drawn synthetic digits. Do not use personal or customer data.

