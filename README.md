# Paper code and results

This repository contains the Python code needed to reproduce our results, as well as the results we obtained and the sbtach script used to run `paper_comparison_suite.py` on the Stellar cluster at Princeton Research Computing. Our paper can be found at

arXiv/publication LINK.

It contains two folders:

1. `hyperparameter-search/` — code and results pertaining to the hyperparameter search outlined in section 3.4 'Hyperparameter search'.
2. `comparison-suite/` — code and results pertaining to our experiments with architecture, input data, among other variables discussed in Section 4 'Model testing'.
3. `Interpretability/` - code and results pertaining to the MIL model attention-based temporal localization of plasmoids as discussed in Section 6 'Interpretability'.

Please note that while several files have a "hyperparameter_search" in their name, these were not run as searches for the purposes of this paper. They were preliminary searches to get a better understanding of the data and models, and are reused as libraries from which the comparison suite imports model classes, data stores, and training machinery/engines. Only `hyperparameter-search/jy_gated_attention_...` was actually run as a search.

Also note that we initially sought to include a transformer-based order-aware architecture known as TimeMIL in our study (arXiv: https://arxiv.org/abs/2405.03140, Github: https://github.com/xiwenc1/TimeMIL/tree/main). We did adapt and simplify the authors' model for our purposes, the results of which are included in `suite_results.csv`, but they are not discussed in the paper. Our goal was to demonstrate the success of a permutation-invariant model relative to a conventionally used sequential model for time series classification. TimeMIL incorporates positional encoding of frames, and is probably better suited to larger datasets than ours. (Also note that a lot of work was done in a folder called `Advanced-Plasmoid-Classification/Models/TimeMIL` without actually using a TimeMIL-based model.)

---

## hyperparameter-search/

# `jy_gated_attention_hyperparameter_search_261us_80frames_1us_augmented.py`
**The hyperparameter search (primary paper result)**
Gated-attention MIL with offline-augmented JY field images (261 us start, 1 us step, 80 frames; offline-augmented data included only in training sets). Random search over CNN backbone/widths, embedding dim, attention dim, dropout, learning rate, weight decay, batch size, loss weighting, scheduler, and online noise/scale augmentation. We used a 15% held-out test split (random_state=42) with stratified 3-fold CV on the remaining 85%.

Contains: `JYDataStore` / `IndexedJYDataset` (augmented-manifest data store),
`FrameEncoder` (plain/residual per-frame CNN feature extractor), `GatedAttentionMIL`,
`train_one_split`, `fit_best_and_test`, and the search plan/leaderboard code.

(Also a library; the comparison suite imports it as `engine_aug` (see below)).

### `results/`
Output of the search (leaderboard, per-trial records, final test evaluation).

---

## comparison-suite/

### `paper_comparison_suite.py`
**Comparisons and statistical results outside of hyperparameter search.**
Trains one configuration for 10 seeds per experimental trial, each with 3-fold CV plus a final fit evaluated on a held-out test set. The test data is obtained using a split of fixed seed (42) throughout.

Trials: `baseline-JY`, `channels-BR-BZ`, `channels-BR-BZ-BY-JY`,
`arch-crnn-JY`, `arch-crnn-BR-BZ-BY-JY`, `arch-timemil-JY`,
`arch-timemil-BR-BZ-BY-JY`, `labels-include1-JY`, `aug-off-JY`, `aug-on-JY`.

It imports the engine scripts in this folder:

| Import | File | What is used |
|---|---|---|
| `engine_multi` | `br_bz_gated_attention_...py` | Data store, training engine, `FrameEncoder`, path constants, utilities |
| `engine_aug` | `jy_gated_attention_..._augmented.py` (in `../hyperparameter-search/`) | `JYDataStore` + train/test code for the two augmentation trials |
| `engine_crnn_module` | `jy_crnn_..._augmented.py` | `ImageCRNN` class |
| `engine_timemil_module` | `timemil_hyperparameter_search.py` | `ImageTimeMIL` class |

### `br_bz_gated_attention_hyperparameter_search_261us_80frames_1us.py`
**Library only.** The suite uses this file as the engine for non-augmentation arms. It supplies:
- `FieldDataStore` — can be any combination of BR-BZ-BY-JY
- `FrameEncoder` — instance/frame-level feature extractor shared by all architectures
- `build_model`, `train_one_split`, `fit_best_and_test`, `SEED`, `TEST_FRACTION`
  — the CV training loop and final fit/test evaluation
- Path/config constants (`FIELD_IMAGES_DIR`, `LABEL_MANIFEST_PATH`,
  `CHANNEL_DIRS`, `BASE_OUTPUT_DIR`, `DEFAULT_MAX_EPOCHS`, ...)
- `write_json`, `json_ready`

Copy of
`Advanced-Plasmoid-Classification/Models/TimeMIL/BR-BZ/hyperparameter-search-codes/`.

### `jy_crnn_hyperparameter_search_261us_80frames_1us_augmented.py`
**Library only.** Used as the source of the `ImageCRNN` class (per-frame CNN embeddings read in time order by an LSTM; final hidden state is the bag/shot vector representation) for the `arch-crnn-*` arms.

Copy of
`Advanced-Plasmoid-Classification/Models/CRNN/hyperparameter-search-codes/`.

### `timemil_hyperparameter_search.py` (not included in this folder)
Used as the source of the `ImageTimeMIL` class for the `arch-timemil-*` trials. The suite's import of it is wrapped in try/except statements so the main `paper_comparison_suite.py` script still runs if this file is not found, outputting a warning and excluding `arch-timemil-*` results. It additionally requires `models.nystrom_attention` from `TimeMIL-github`. See above ('Paper code and results', beginning 'Also note that...') for access.

### `submit_paper_comparison_suite_stellar.slurm`
Slurm script used to run `paper_comparison_suite.py` on the Stellar cluster at Princeton Research Computing.

### `results/`
Suite output: `suite_results.csv` (one row per experiment x seed),
`suite_summary.txt` (per-trial mean ± std and paired significance tests), and
`runs/<experiment>/seed_<seed>/` (per-run records (fold histories,
`final_test_metrics.json`, `final_test_predictions.csv`, misclassified shots)).
Includes the `arch-timemil-*` runs omitted in the paper.

---

## Interpretability/

The purpose of this part of our work is to demonstrate the utility of the inherent interpretability provided by MIL models for the localisation of plasmoids. This folder contains a python file rerunning a training script, modified to save the attention scores during MIL pooling but with the same optimal configuration as previously found for the MIL model, alongside a notebook containing attention-related plots.

---

## MNIST-Bags/

The MNIST dataset is a collection of tens of thousands of hand-written digits, commonly used for computer vision tasks and as a benchmark dataset for comparing model performance. The MNIST-BAGS dataset is an adaptation of this for multiple instance learning problems. This code trains the best gated attention and best TimeMIL configurations found in a preliminary JY hyperparameter search. The goal was to see if the model could perform well on a standard MIL task, which would indicate that the model itself is not the primary cause for concern if FLARE-related tasks prove unsuccessful. The folder contains the code and (good) results on the MNIST task.

## Reproducibility note

- The suite's `sys.path` setup still assumes the original
  `Models/TimeMIL/hyperparameter-search-codes` tree; in this folder the imports resolve only because the engine files sit in the same directory as the suite.
- The raw data needed for the training and plotting is not available in this repository. It can be found at LINK.
