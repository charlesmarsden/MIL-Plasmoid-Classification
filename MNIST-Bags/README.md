# MNIST-BAGS

The MNIST dataset is a collection of tens of thousands of hand written digits, commonly used for computer vision tasks and as a benchmark dataset for comparing model performance. The MNIST-BAGS dataset is an adaptation of this for multiple instance learning problems. This code trains the best gated attention and best TimeMIL configurations found in the JY hyperparameter search. The goal was to see if the model could perform well on a standard MIL task, which would indicate that the model itself is not the primary cause for concern if FLARE-related tasks prove unsuccessful.

### Results (held-out test set)

| model | architecture | params | test F1 | test PR AUC | test ROC AUC | precision | recall |
|---|---|---|---|---|---|---|---|
| best_gated_attention | gated_attention | 458,050 | 0.934 | 0.983 | 0.979 | 0.947 | 0.922 |
| best_timemil | timemil | 570,229 | 0.798 | 0.892 | 0.875 | 0.861 | 0.744 |

The gated attention model performs well, as expected, and the TimeMIL model is a distant second despite doing fairly well. This suggests that the poor performance on FLARE data may be a result of the data processing or the data itself, as opposed to model architecture.