# Model Card: MNIST CNN Digit Classifier

## Model Overview
- Model name: `mnist_cnn_baseline`
- Architecture type: Convolutional Neural Network
- Task: Handwritten digit classification
- Framework: TensorFlow/Keras
- Input shape: 28 x 28 x 1
- Output classes: digits 0-9
- Status: Selected final baseline model

## Intended Use
Educational, internship, and portfolio use for demonstrating a complete
computer vision workflow.

## Dataset
- MNIST
- 60,000 training images
- 10,000 test images
- Digits 0-9
- Grayscale 28 x 28 images

## Preprocessing
- Pixel normalization to [0, 1]
- Reshape to `(28, 28, 1)`
- One-hot label encoding during training

## Architecture
- Conv2D: units/filters=32, activation=relu
- MaxPooling2D: (2, 2)
- Conv2D: units/filters=64, activation=relu
- MaxPooling2D: (2, 2)
- Flatten: 
- Dense: units/filters=128, activation=relu
- Dropout: 0.5
- Dense: units/filters=10, activation=softmax

## Evaluation Metrics
- Training accuracy: 98.66%
- Validation accuracy: 98.98%
- Test accuracy: 98.94%
- Test loss: 0.0286

## Per-Class Results
Strongest digit class: 1. Weakest digit class:
9.

Top classes by F1-score:
| digit | f1_score | recall |
| --- | --- | --- |
| 1 | 0.9955947136563876 | 0.9955947136563876 |
| 0 | 0.9928716904276986 | 0.9948979591836736 |
| 2 | 0.9922630560928434 | 0.994186046511628 |

Weakest classes by F1-score:
| digit | f1_score | recall |
| --- | --- | --- |
| 9 | 0.9840796019900496 | 0.9801783944499504 |
| 7 | 0.98545101842871 | 0.9883268482490272 |
| 8 | 0.985699693564862 | 0.9907597535934292 |

## Known Limitations
- MNIST images are cleaner and more standardized than real handwriting.
- The classifier supports digits only.
- Input polarity and centering must match the training distribution.
- Saliency maps indicate sensitivity, not causal explanation.

## Ethical and Safety Notes
This model is suitable for low-risk educational demonstrations. It should not
be used for high-stakes identity, grading, or accessibility decisions without
additional validation, monitoring, and human review.

## Future Improvements
- EMNIST support
- Data augmentation
- Confidence calibration
- Robustness testing
- Handwriting canvas
- Deployment monitoring
