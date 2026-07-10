# Final Model Selection Report

## Executive Summary
The `mnist_cnn_baseline` model is selected as the final model for the MNIST
handwritten digit recognition workflow.

## Model Performance
- Training accuracy: 98.66%
- Validation accuracy: 98.98%
- Best validation accuracy: 99.05%
- Test accuracy: 98.94%
- Test loss: 0.0286

## Per-Class Performance
Strongest class by F1-score: digit 1 with F1
0.9956. Weakest class by F1-score: digit
9 with F1 0.9841. Lowest recall
was observed for digit 9 with recall
0.9802.

## Error Analysis
- Total incorrect predictions: 106
- Test error rate: 1.06%
- Average confidence of incorrect predictions: 0.6994

Most common confusion pairs:
- True 4 -> Predicted 9: 8
- True 2 -> Predicted 7: 6
- True 9 -> Predicted 8: 5
- True 3 -> Predicted 5: 5
- True 6 -> Predicted 0: 4

## Confidence Analysis
Correct predictions have mean confidence
0.9932, while incorrect
predictions have mean confidence 0.6994.
Low-confidence predictions should be surfaced clearly in deployment.

## Explainability
Gradient-based saliency maps were generated for selected correct predictions.
These maps show pixel sensitivity for the predicted class score, not causal
explanation.

## Final Selection Rationale
The baseline CNN is selected because it delivers strong test accuracy,
consistent validation and test performance, a low error rate, and a lightweight
architecture suitable for a deployment demonstration.

## Limitations
- MNIST is clean and standardized.
- Real handwriting may be noisier.
- This model supports digits only, not the full alphabet.
- Saliency maps are not causal explanations.
- No adversarial robustness testing has been performed.
- No calibration analysis has been completed yet.

## Deployment Recommendation
Use `mnist_cnn_baseline.keras`, apply the same preprocessing used during
training, validate image dimensions and polarity, display prediction
probabilities, and show confidence warnings for low-confidence predictions.
