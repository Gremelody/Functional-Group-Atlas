# 🔥 functional-group-atlas-for-interface-programming

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

This repository provides a **reproducible, modular machine-learning workflow** for building a **data-driven functional-group atlas** that enables *predictable programming* of **carbon–molten salt interfacial wettability and heat transport** for thermal energy storage.

The core idea is to **deconstruct functional-group effects into two orthogonal feature dimensions**:

- **Elemental / bonding descriptors** → dominate *interfacial wettability* (quantified by adsorption/binding affinity, **Eb**).
- **Geometric / topological descriptors** → dominate *interfacial heat transport potential* (quantified by low-frequency phonon spectral matching, **NVOA**).

This repo focuses on the **machine-learning part of the workflow**: hierarchical feature engineering, Bayesian hyperparameter optimization, stacking ensemble construction, rigorous repeated cross-validation, SHAP-based interpretation (including a **Two-Level Weighted SHAP** strategy), and high-throughput prediction for a broader candidate space.

---

## 🚀 Key Features

- **Dual-Target Learning (Eb & NVOA)** 🎯  
  Train and interpret **two independent models** for interfacial wettability (Eb) and thermal transport potential (NVOA).

- **Hierarchical Feature Engineering** 🛠️  
  A rigorous three-stage selection strategy:
  1) **Filter**: remove highly correlated features (Pearson threshold) using **Mutual Information** (default) or Pearson-to-target criterion  
  2) **Embedded**: SHAP-driven coarse screening with a Random Forest  
  3) **Wrapper**: SHAP-driven recursive elimination with **early stopping** to locate the optimal feature set

- **Efficient Bayesian Hyperparameter Optimization** ⚡  
  Uses Bayesian optimization (scikit-optimize) to tune multiple tree-based regressors efficiently.

- **High-Performance Stacking Ensemble** 🧠  
  Builds a two-layer stacking model:
  - **Level-0**: heterogeneous tree learners (e.g., RF/ETR/GBRT/HGBR/XGB/LGBM/CBR)
  - **Level-1 meta-learner**: **ElasticNet** (regularized linear combiner)

- **Rigorous Robustness Evaluation** 🔁  
  Repeated evaluation across many random seeds (default **100**) with outer K-fold CV, reducing split-induced contingencies.

- **Interpretable “Two-Level Weighted SHAP”** 📊  
  Produces a more stable global importance ranking by weighting:
  - within-fold SHAP by **1/RMSE of base learners**
  - across folds by **1/RMSE of stacking meta-learner**

- **One-Click Prediction for New Candidates** 🧪  
  Predict Eb/NVOA for unseen functional-group candidates from an Excel sheet, exporting full results to Excel/CSV.

---

## 📂 Repository Structure and Workflow

.
├── Feature engineering-FGA.ipynb # Script A: Hierarchical feature engineering (Filter-Embedded-Wrapper)
├── Tree_stacking-FGA.ipynb # Script B: Hyperparameter optimization + Stacking + Evaluation/SHAP + Prediction
│
├── Original database-FGA.xlsx # (User-provided) Raw feature database with BOTH targets (Eb & NVOA)
├── Final_engineered_dataset-FGA-Eb.xlsx # (Generated/renamed) Feature-engineered dataset for Eb
├── Final_engineered_dataset-FGA-NVOA.xlsx # (Generated/renamed) Feature-engineered dataset for NVOA
├── prediction-FGA-Eb.xlsx # (User-provided) Prediction set for Eb (features only)
├── prediction-FGA-NVOA.xlsx # (User-provided) Prediction set for NVOA (features only)
│
├── feature_engineering_output/ # Auto-created output folder for Script A
│ ├── Original_Feature_Correlation_.xlsx
│ ├── Stage1_Filtered_Correlation_.xlsx
│ ├── Final_Feature_Correlation_.xlsx
│ ├── Feature_Engineering_Summary_.xlsx
│ ├── Performance_Iteration_History_.xlsx
│ └── Final_Selected_Dataset_.xlsx # Core output from Script A (rename to final_engineered_dataset-.xlsx)
│
├── SHAP_Analysis_Results.xlsx # Core output from Script B (SHAP + metrics + export tables)
├── unknown_predictions_.xlsx # Auto-generated prediction outputs (timestamped)
│
├── environment.yml # Conda env config (cross-platform)
├── requirements.txt # Pip dependencies
├── spec-file.txt # Exact Conda spec (Windows x64) for highest reproducibility
├── LICENSE # MIT License
└── README.md
.


---

## 🗺️ The Workflow

**`Script A: Feature Engineering`**  
*Compress the descriptor space and generate optimized datasets*

`⬇️`

**`Script B-1: Hyperparameter Optimization`**  
*Bayesian optimization for enabled base learners*

`⬇️`

**`Script B-2: Stacking Ensemble + Evaluation + SHAP`**  
*OOF stacking + repeated CV + Two-Level Weighted SHAP exports*

`⬇️`

**`Script B-3: Prediction`**  
*Apply trained stacking model to unseen candidates and export results*

---

## 🧾 Dataset Format (Excel)

### 1) Raw database: `Original database-FGA.xlsx`
Your raw Excel should contain:

- A leftmost identifier column (recommended; optional but helpful)
- A block of numeric descriptor columns (features)
- **Two target columns at the end** (recommended structure):
  - one for **Eb**
  - one for **NVOA**

> Script A is configured to treat **the last two columns as targets** and exclude them from features by default.

### 2) Engineered dataset: `Final_engineered_dataset-FGA-*.xlsx`
After Script A, each engineered dataset should contain:

- Column 1: ID / Name (optional)
- Columns 2..(n-1): selected features
- Last column: target (Eb *or* NVOA)

### 3) Prediction set: `prediction-FGA-*.xlsx`
Prediction files should contain:

- Column 1: ID / Name
- Columns 2..end: feature columns (must match the training feature names/order after alignment)

---

## 📜 Script A: Feature Engineering (`Feature engineering-FGA.ipynb`)

### 🎯 What it does
- Loads `Original database-FGA.xlsx`
- Extracts features by slice and selects the target by column index
- Runs a 3-stage feature selection pipeline:
  - **Stage 1 (Filter)**: Pearson correlation pruning (default threshold `0.8`) using **Mutual Information** to decide which feature to keep in a correlated pair
  - **Stage 2 (Embedded)**: Random Forest + SHAP coarse screening (default keep top `80%`)
  - **Stage 3 (Wrapper)**: SHAP-guided iterative elimination with **early stopping** (default patience `50`)

### ✅ Default configuration (edit at top of notebook)
- `INPUT_FILE = 'Original database-FGA.xlsx'`
- `OUTPUT_DIR = 'feature_engineering_output'`
- `FEATURE_COLUMN_SLICE = '1:-2'`  (features exclude last two cols)
- `TARGET_COLUMN_INDEX = -1`       (choose last col as target by default)
- `FILTER_METHOD_CRITERION = 'mutual_info'`
- `PEARSON_CORR_THRESHOLD = 0.8`

### ▶️ How to run for **both targets**
Run the notebook twice:

1. **For NVOA**
   - keep `TARGET_COLUMN_INDEX = -1`
   - run all cells → get `feature_engineering_output/Final_Selected_Dataset_*.xlsx`
   - rename to `Final_engineered_dataset-FGA-NVOA.xlsx`

2. **For Eb**
   - set `TARGET_COLUMN_INDEX = -2`
   - run all cells → get another `Final_Selected_Dataset_*.xlsx`
   - rename to `Final_engineered_dataset-FGA-Eb.xlsx`

### 📄 Outputs
All outputs go to `feature_engineering_output/`, including:

- correlation matrices at different stages
- process summary Excel
- iterative performance history Excel
- **Final selected dataset** (core output):  
  `Final_Selected_Dataset_YYYYMMDD_HHMMSS.xlsx`

---

## 📜 Script B: Training / Evaluation / Prediction (`Tree_stacking-FGA.ipynb`)

> ⚠️ IMPORTANT: This notebook contains **three sequential parts** (B-1/B-2/B-3).  
> **Run them in the same session** so variables (e.g., `grid_searches`) are available downstream.

---

### 📜 Script B-1: Hyperparameter Optimization (Bayesian)

#### 🎯 What it does
- Loads engineered dataset (Eb *or* NVOA)
- Tunes enabled base learners using Bayesian optimization
- Stores results in memory: `grid_searches`

#### ✅ Key parameters (cell header)
- `EXCEL_FILE_PATH = 'Final_engineered_dataset-FGA-Eb.xlsx'` *(switch to NVOA file when needed)*
- `X_COLS_SLICE = slice(1, -1)` and `Y_COLS_SLICE = -1`
- `CV_N_SPLITS = 10`
- `N_ITER_BAYESIAN = 30`
- `ENABLED_MODELS = [...]` (choose which models to optimize)

---

### 📜 Script B-2: Stacking Ensemble + Evaluation + Two-Level Weighted SHAP

#### 🎯 What it does
- Filters `grid_searches` to keep enabled base learners
- Builds OOF predictions and trains **ElasticNet** meta-learner
- Repeats outer CV across many seeds for stability (default `100`)
- Computes:
  - per-fold metrics tables
  - Two-Level Weighted SHAP global importance
  - SHAP swarm data export

#### ✅ Key parameters
- `N_SEEDS_FOR_EVALUATION = 100`
- `N_SPLITS_OUTER_CV = 10`
- `META_LEARNER_N_ITER_BAYESIAN = 50`
- `OUTPUT_EXCEL_FILENAME = 'SHAP_Analysis_Results.xlsx'`
- `WEIGHTING_METHOD = '1/RMSE'`
- `N_FEATURES_TO_PLOT = 30`
- `PLOT_SHAP_SWARM_PLOT = True`

#### 📄 Outputs (main)
- `SHAP_Analysis_Results.xlsx` with multiple sheets, e.g.:
  - `{Model}_GlobalImportance`
  - `{Model}_SwarmPlotData`
  - `Fold_Performance_Metrics`
  - `Global_Importance_Summary`

> The notebook also displays SHAP beeswarm + bar plots inline.

---

### 📜 Script B-3: Prediction (Unseen candidates)

#### 🎯 What it does
- Loads prediction features from Excel
- Aligns feature columns to training set
- Predicts using base learners + final stacking model
- Exports timestamped results

#### ✅ Key parameters
- `UNKNOWN_DATA_FILE = 'prediction-FGA-Eb.xlsx'` *(switch to NVOA file when needed)*
- `UNKNOWN_DATA_FILE_COLUMN_RANGE = (slice(None), slice(1, None))` *(skip ID column)*
- `REUSE_PRETRAINED_STACKING_MODEL = False`
  - `False`: retrain a final model and predict (more self-contained)
  - `True`: reuse trained model from the evaluation step (faster)
- `PREDICTION_OUTPUT_FILENAME_PREFIX = 'unknown_predictions'`
- `PREDICTION_EXPORT_TO_EXCEL = True`

#### 📄 Outputs
- `unknown_predictions_YYYYMMDD_HHMMSS.xlsx` (and/or `.csv`)
  - includes stacking prediction + per-base-learner predictions for analysis

---

## 💻 How to Use (Quickstart)

### 1) Environment setup
~~~bash
pip install -r requirements.txt
~~~

### 2) Feature engineering (run twice)
- Open `Feature engineering-FGA.ipynb`
- Set:
  - for **NVOA**: `TARGET_COLUMN_INDEX = -1`
  - for **Eb**:   `TARGET_COLUMN_INDEX = -2`
- Run all cells each time
- Rename the produced `Final_Selected_Dataset_*.xlsx` to:
  - `Final_engineered_dataset-FGA-NVOA.xlsx`
  - `Final_engineered_dataset-FGA-Eb.xlsx`

### 3) Train + evaluate + interpret + predict (run twice)
- Open `Tree_stacking-FGA.ipynb`
- For **Eb** run:
  - `EXCEL_FILE_PATH = 'Final_engineered_dataset-FGA-Eb.xlsx'`
  - `UNKNOWN_DATA_FILE = 'prediction-FGA-Eb.xlsx'`
- For **NVOA** run:
  - `EXCEL_FILE_PATH = 'Final_engineered_dataset-FGA-NVOA.xlsx'`
  - `UNKNOWN_DATA_FILE = 'prediction-FGA-NVOA.xlsx'`

Run the notebook **top to bottom** to finish B-1 → B-2 → B-3.

---

## 📦 Environment Setup & Reproducibility

### 🐍 Python Version
This project was developed and tested using **Python 3.10.18**. While it may work with other Python 3.10+ versions, using this specific version is recommended to maximize reproducibility.

---

### 📋 Core Dependencies
Below are the core scientific computing and machine learning libraries used in this project.

~~~
# Core scientific computing and machine learning libraries
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.6.1
xgboost==3.0.2
catboost==1.2.7
lightgbm==4.6.0
scikit-optimize==0.10.2
shap==0.48.0
umap-learn==0.5.7
tqdm==4.67.1
openpyxl==3.1.5

# Libraries for Jupyter Notebook integration
jupyterlab>=4.0.0
notebook==7.3.2
ipykernel==6.29.5
ipywidgets==8.1.7
~~~

---

### Environment Configuration

**⚠️ IMPORTANT NOTE:** For academic review or users who need to reproduce results as closely as possible, **it is highly recommended to use Option 1**. Using Option 2 or 3 may produce minor numerical differences due to solver ambiguity, platform-specific builds, and stochastic optimization.

---

### 🥇 Option 1: Highest-Fidelity Reproducibility (via `spec-file.txt`)
**Platform:** Windows (x64)

~~~bash
conda create --name my-project-env --file spec-file.txt
conda activate my-project-env
~~~

---

### 🥈 Option 2: Cross-Platform Setup (via `environment.yml`)
~~~bash
conda env create -f environment.yml -n my-project-env
conda activate my-project-env
~~~

---

### 🥉 Option 3: Basic Setup (via `requirements.txt`)
~~~bash
python -m venv venv
# Windows:
# venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate
pip install -r requirements.txt jupyterlab
~~~

---

## 📜 License and Correspondence

The code in this repository is released under the **MIT License** (see `LICENSE`).

For any inquiries or if you use this workflow in your research, please correspond with:  
Prof. **Guangmin Zhou** (Tsinghua Shenzhen International Graduate School, Tsinghua University)  
📧 guangminzhou@sz.tsinghua.edu.cn

---

## 🙏 Acknowledgements

**Yifei Zhu** (zhuyifeiedu@126.com) at Tsinghua University conceived and formulated the algorithms, constructed the quantum-chemical dataset, developed and deposited the code, and authored this comprehensive guideline document.

---

## 📝 Citation (optional)

If you use this workflow, please consider citing our work:

~~~bibtex
@article{zhu_functional_group_atlas,
  title   = {A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage},
  author  = {Zhu, Yifei and Wang, Tiansheng and Zhou, Guangmin},
  journal = {Manuscript under review},
  year    = {2026}
}
~~~

