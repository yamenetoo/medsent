```markdown
#medsent: A Comprehensive Framework for Sentiment Analysis of Medication Reviews

Authors:   Mohammad Yamen AL-Mohamad
Contact: yamenmohamad@tabrizu.ac.ir  

This repository contains the official implementation of the methodology and experiments described in the thesis chapter *"Methodology and Experimental Results"*. The `medsent` package provides a modular, reproducible pipeline for predicting patient sentiment from free‑text medication reviews using a wide range of machine learning and deep learning models, including ensembles and transformer‑based architectures.

---

##📌 Overview

The framework integrates:

- Data processing: loading, labeling (binary/ternary/10‑class), text cleaning, stemming, stopword removal.
- Feature extraction: Bag‑of‑Words, TF‑IDF, n‑grams, averaged word embeddings (GloVe, PubMed, concatenated), sequence embeddings for deep models.
- Traditional ML models: Logistic Regression, SVM (linear/RBF), Random Forest, XGBoost, Naïve Bayes, k‑NN, LDA.
- Deep learning models: Bi‑LSTM, Bi‑GRU, Bi‑RNN with pre‑trained or learned embeddings.
- Ensembles: hard‑voting ensembles (ML_ENS, DL_ENS, ALL_ENS).
- Transformer models: BERT, ClinicalBERT, BioBERT (optional).
- Evaluation: comprehensive metrics (accuracy, precision, recall, F1, AUC, confusion matrices), visualisation, statistical significance testing (bootstrap).

All 32 tables from the thesis chapter can be reproduced exactly by running the provided experiment scripts.

---

## 📖 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Package Structure](#package-structure)
- [Running Experiments](#running-experiments)
- [Reproducing Specific Tables](#reproducing-specific-tables)
- [Results Summary](#results-summary)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yamenetoo/medsent.git
cd medsent
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or .\venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

This installs all necessary libraries: `numpy`, `pandas`, `scikit-learn`, `nltk`, `gensim`, `tensorflow`, `xgboost`, `matplotlib`, `seaborn`, `transformers`, `torch`.

---

## 📁 Data Preparation

### Drugs.com Review Dataset

The experiments use a publicly available dataset of medication reviews from Drugs.com.  
Format: CSV file with columns `review` (text) and `rating` (integer 1–10).  
Place it at: `data/drugscom_reviews.csv` (create the `data/` folder if needed).

You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29) (file `drugsCom_raw.zip`). After extraction, the file `drugsComTrain_raw.csv` contains the reviews. You may need to rename it and keep only the `review` and `rating` columns.

Example preprocessing (already handled by the package):
- Remove HTML tags, URLs, dosage mentions (e.g., "10mg").
- Convert to lowercase.
- Tokenize, remove stopwords, apply stemming (Porter).

### Pre‑trained Embeddings

For best performance, download the following embeddings and place them in the `embeddings/` folder.

#### GloVe (840B tokens, 300d)
- Download from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) (file `glove.840B.300d.zip`).
- Extract and convert to word2vec format (required by gensim):
  ```bash
  python -m gensim.scripts.glove2word2vec -i glove.840B.300d.txt -o embeddings/glove.840B.300d.w2v.txt
  ```

#### PubMed Word2Vec (300d)
- Download from [http://evexdb.org/pmresources/vec-space-models/PubMed-shuffle-win-30.bin](http://evexdb.org/pmresources/vec-space-models/PubMed-shuffle-win-30.bin) (binary format).
- Place the file directly: `embeddings/PubMed-shuffle-win-30.bin`.

#### (Optional) Clinical BERT / BioBERT
If you wish to run transformer experiments, the models will be downloaded automatically by the `transformers` library the first time they are used.

---

## 📦 Package Structure

```
medsent/
├── __init__.py
├── config.py                     # Global paths, seeds, hyperparameters
├── data/
│   ├── loader.py                 # Load CSV, label, split
│   ├── labeling.py               # Binary/ternary/10class labeling
│   └── preprocessing.py           # Text cleaning, stemming, stopwords
├── features/
│   ├── bow.py                     # BoW / TF‑IDF vectorizers
│   ├── word2vec_train.py           # Train Word2Vec on corpus
│   ├── embeddings.py               # Load pre‑trained, averaged, sequence prep
│   └── char_cnn_prep.py            # Character‑level tokenizer
├── models/
│   ├── ml_models.py                # ML models with grid search
│   ├── dl_models.py                # Bi‑LSTM / Bi‑GRU / Bi‑RNN
│   ├── transformer_models.py        # BERT, ClinicalBERT, BioBERT
│   ├── char_cnn.py                  # Character‑level CNN
│   └── ensemble.py                  # Hard/soft voting
├── evaluation/
│   ├── metrics.py                   # Accuracy, precision, recall, F1, AUC, CM
│   ├── visualization.py              # Confusion matrices, learning curves
│   └── bootstrap.py                  # Statistical significance test
├── experiments/
│   ├── __init__.py
│   ├── binary/                       # Tables 4.2 – 4.5
│   ├── ternary/                       # Tables 4.6 – 4.8
│   ├── tenclass/                       # Tables 4.9 – 4.10
│   ├── ablation/                       # Tables 4.11 – 4.18, 4.28
│   ├── ensemble/                       # Table 4.19
│   ├── training_time/                   # Table 4.20
│   ├── inference_time/                   # Table 4.21
│   ├── cross_validation/                 # Table 4.22
│   ├── statistical_significance/         # Table 4.23
│   ├── transformer/                       # Table 4.24
│   ├── learning_curve/                     # Table 4.25
│   ├── svm_xgb_sensitivity/                 # Tables 4.26 – 4.27
│   ├── feature_importance/                   # Table 4.29
│   ├── per_condition/                         # Table 4.30
│   ├── text_length/                           # Table 4.31
│   └── char_cnn/                               # Table 4.32
├── utils/
│   ├── helpers.py                    # set_seed, save/load results
│   └── logger.py                      # Logging setup
├── main.py                           # Orchestrator – run all experiments
└── requirements.txt
```

---

## 🧪 Running Experiments

### Run all experiments (reproduce all 32 tables)
```bash
python -m medsent.main
```

This will sequentially execute every experiment, saving results to the `results/` folder:
- Tables as CSV files (e.g., `table_4_2_binary_all.csv`)
- Figures as PNG files (e.g., `table_4_4_confusion_xgb.png`)

The script uses logging – you can monitor progress in the console and in `medsent.log`.

### Run a single experiment
You can also import any experiment module and run it individually. For example, to reproduce Table 4.2 (binary all models):

```python
from medsent.experiments.binary import table_4_2_all_models
table_4_2_all_models.run()
```

All experiment modules are located under `medsent/experiments/` and follow the naming convention `table_<chapter>_<number>_<description>.py`.

---

## 📊 Reproducing Specific Tables

Below is the complete mapping of thesis tables to the corresponding experiment module. Each module contains a `run()` function that saves the table data (and optionally a figure) to the `results/` directory.

| Table | Description | Module |
|-------|-------------|--------|
| 4.2 | Binary macro F1 – all model/embedding combinations | `binary.table_4_2_all_models` |
| 4.3 | Binary per‑class metrics (top models) | `binary.table_4_3_perclass` |
| 4.4 | Confusion matrix – XGBoost (binary) | `binary.table_4_4_confusion_xgb` |
| 4.5 | Confusion matrix – Bi‑LSTM (binary) | `binary.table_4_5_confusion_bilstm` |
| 4.6 | Ternary macro F1 – selected models | `ternary.table_4_6_all` |
| 4.7 | Ternary per‑class metrics (DL_ENS) | `ternary.table_4_7_perclass` |
| 4.8 | Ternary confusion matrix (DL_ENS) | `ternary.table_4_8_confusion` |
| 4.9 | 10‑class accuracy & macro F1 | `tenclass.table_4_9_all` |
| 4.10 | 10‑class per‑rating metrics (DL_ENS) | `tenclass.table_4_10_perrating` |
| 4.11 | Effect of embedding type on deep models | `ablation.table_4_11_embedding_deep` |
| 4.12 | ML models on averaged concatenated embeddings | `ablation.table_4_12_ml_avg_emb` |
| 4.13 | Effect of n‑gram features on XGBoost | `ablation.table_4_13_ngram` |
| 4.14 | Effect of preprocessing steps on Bi‑LSTM | `ablation.table_4_14_preprocess` |
| 4.15 | Effect of embedding dimension on Bi‑LSTM | `ablation.table_4_15_embedding_dim` |
| 4.16 | Effect of hidden size on Bi‑LSTM | `ablation.table_4_16_hidden_size` |
| 4.17 | Effect of Word2Vec variant & window size | `ablation.table_4_17_w2v_variants` |
| 4.18 | Effect of optimizer on Bi‑LSTM | `ablation.table_4_18_optimizer` |
| 4.19 | Ensemble performance across tasks | `ensemble.table_4_19_ensemble_all` |
| 4.20 | Training time & memory usage | `training_time.table_4_20_training_time` |
| 4.21 | Inference time per sample | `inference_time.table_4_21_inference` |
| 4.22 | Fold‑wise macro F1 (stability) | `cross_validation.table_4_22_fold_var` |
| 4.23 | Statistical significance (bootstrap) | `statistical_significance.table_4_23_bootstrap` |
| 4.24 | Transformer models (BERT, BioBERT, ClinicalBERT) | `transformer.table_4_24_transformer` |
| 4.25 | Effect of training data size (learning curves) | `learning_curve.table_4_25_learning_curve` |
| 4.26 | SVM (RBF) hyperparameter sensitivity | `svm_xgb_sensitivity.table_4_26_svm_params` |
| 4.27 | XGBoost hyperparameter sensitivity | `svm_xgb_sensitivity.table_4_27_xgb_params` |
| 4.28 | Effect of pooling strategy on Bi‑LSTM | `ablation.table_4_28_pooling` |
| 4.29 | Feature importance from XGBoost (unigrams) | `feature_importance.table_4_29_feat_imp` |
| 4.30 | Per‑condition performance (DL_ENS) | `per_condition.table_4_30_condition` |
| 4.31 | Performance by review length | `text_length.table_4_31_length` |
| 4.32 | Character‑level CNN vs. Bi‑LSTM | `char_cnn.table_4_32_char_cnn` |

---

## 📈 Results Summary

The best‑performing model overall is BioBERT, achieving a binary macro F1 of 0.925, closely followed by the deep ensemble DL_ENS with 0.912. Concatenated clinical embeddings (GloVe+PubMed) consistently outperform individual sources. Detailed results are available in the output CSV files.

---

## 🔧 Extending the Framework

You can easily add new models, features, or experiments.

### Adding a new ML model
1. Edit `medsent/models/ml_models.py`: add your model to the `models` dictionary and optionally a parameter grid.
2. The model will automatically be available in experiments that loop over ML models (e.g., Table 4.2).

### Adding a new embedding source
1. Place your embedding file (word2vec format) in the `embeddings/` folder.
2. Add its path to `config.py` (e.g., `NEW_EMB_PATH`).
3. Modify `medsent/features/embeddings.py` to load it and add functions for averaged/sequence representations.
4. Update experiment scripts to include the new embedding in the comparisons.

### Creating a new experiment
1. Create a new Python file under the appropriate subdirectory in `experiments/`.
2. Follow the pattern: load data, extract features, train models, evaluate, save results.
3. Import and call it from `main.py` if you want it to run automatically.

---

## ❗ Troubleshooting

| Problem | Solution |
|--------|----------|
| `ModuleNotFoundError: No module named 'medsent'` | Make sure you are running from the directory that contains the `medsent` folder, or install the package in editable mode (`pip install -e .`). |
| `FileNotFoundError: data/drugscom_reviews.csv` | Place the dataset in the correct location or update `DATA_PATH` in `config.py`. |
| Out of memory during deep learning | Reduce batch size, number of units, or embedding dimension in `config.py`. |
| GPU not detected | Install the appropriate TensorFlow version with GPU support (`tensorflow-gpu`) and CUDA/cuDNN. |
| Transformers download very slow | The first run downloads model checkpoints; subsequent runs are faster. |

---

## 📖 Citation

If you use this code or the methodology in your research, please cite the original thesis chapter:

```bibtex
@phdthesis{haydari_yamen_2025,
  title  = {A Comprehensive Framework for Sentiment Analysis of Medication Reviews},
  author = {Ali Akbar Haydari and Mohammad Yamen},
  school = {University of Tabriz},
  year   = {2025}
}
```

---

## ⚖️ License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the developers of the open‑source libraries that made this work possible:

- NumPy, pandas – data manipulation
- scikit‑learn – traditional ML models and evaluation
- NLTK – text preprocessing
- Gensim – word2vec and embedding handling
- TensorFlow / Keras – deep learning models
- XGBoost – gradient boosting
- Matplotlib / Seaborn – visualisation
- HuggingFace Transformers – transformer models
- PyTorch – backend for transformers

We also acknowledge the creators of the Drugs.com dataset and the providers of the pre‑trained embeddings (GloVe, PubMed Word2Vec).

For questions, issues, or contributions, please open an issue on GitHub or contact Mohammad Yamen at yamenmohamad@tabrizu.ac.ir.
```
 
