**MIL model hyperparameter search space for the 24-trial random search.** Each trial sampled all parameters independently; data configuration was fixed (JY input, confidence-1 shots excluded, 261–340 µs window, no offline augmentation).

| Parameter | Search space | Sampling weights |
|---|---|---|
| CNN connections | {plain, residual} | 0.65, 0.35 |
| CNN channel widths | {(8,16,32), (16,32,64), (24,48,96)} | 0.25, 0.50, 0.25 |
| Embedding dimension | {64, 128, 256} | 0.35, 0.45, 0.20 |
| Dropout | {0.1, 0.25, 0.3, 0.4, 0.5} | 0.15, 0.20, 0.30, 0.20, 0.15 |
| Learning rate | {3 × 10⁻⁵, 1 × 10⁻⁴, 3 × 10⁻⁴} | 0.20, 0.50, 0.30 |
| Weight decay | {10⁻⁵, 10⁻⁴, 10⁻³} | 0.25, 0.30, 0.45 |
| Batch size | {4, 8, 16} | 0.20, 0.50, 0.30 |
| Loss function | {BCE, class-weighted BCE} | 0.80, 0.20 |
| LR scheduler | {none, cosine, plateau} | 0.15, 0.35, 0.50 |
| Online noise std | {0, 0.01} | 0.70, 0.30 |
| Online scale std | {0, 0.03} | 0.70, 0.30 |
| Classifier width multiplier | {0.5, 1.0} | 0.35, 0.65 |
