# Testing Checklist

## Environment

- [ ] Python 3.11 is active
- [ ] Dependencies install from `requirements.txt`
- [ ] `pip check` passes

## Application

- [ ] App launches with `streamlit run app/app.py`
- [ ] Sidebar navigation works
- [ ] Upload mode works
- [ ] Sample mode works
- [ ] Canvas mode works
- [ ] Clear canvas works
- [ ] Blank canvas is rejected
- [ ] Preprocessing preview works
- [ ] Prediction result appears
- [ ] Top-3 probabilities display
- [ ] All-class probabilities display
- [ ] Confidence display works
- [ ] Ambiguity warning works
- [ ] Saliency works
- [ ] History works
- [ ] History clear works
- [ ] Downloads work

## Pages

- [ ] Model Information page loads
- [ ] Error Analysis page loads
- [ ] About Project page loads

## Quality

- [ ] `python -m compileall app src` passes
- [ ] `python -m app.smoke_test_inference` passes
- [ ] `pytest` passes
- [ ] CI passes
- [ ] No training occurs during app launch
- [ ] Model timestamp unchanged
- [ ] No virtual environments tracked
- [ ] No caches tracked
- [ ] No secrets committed

## Screenshots

- [x] `home.png` available
- [x] `canvas_input.png` available
- [x] `prediction_result.png` available
- [x] `model_information.png` available
- [x] `error_analysis.png` available
- [x] `about_project.png` available
- [x] README images render

## Deployment

- [ ] Deployment guide validated
- [ ] Clean-clone test completed
- [x] Streamlit configuration complete
- [x] Live deployment status documented as Not deployed
