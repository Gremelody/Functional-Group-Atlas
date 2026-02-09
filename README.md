
```markdown
# 🧪 functional-group-atlas

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

This repository provides a complete, data-driven workflow for the research paper **"A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage"**. We present a machine learning-assisted design paradigm based on functional group deconstruction to resolve the intrinsic trade-off between energy density and power density in composite phase-change materials (PCMs).

The pipeline integrates Density Functional Theory (DFT), Ab Initio Molecular Dynamics (AIMD), and Stacking Ensemble learning to screen a library of **248 candidates**. This workflow successfully decouples interfacial properties, revealing that wettability is governed by elemental composition while heat transport is dictated by geometric topology.

## 🚀 Key Features

* **Dimensional Feature Deconstruction** 🧬: Implements a novel strategy to decouple functional groups into independent "elemental" and "structural" dimensions, enabling programmable control over interfacial behaviors.
* **Hierarchical Feature Engineering** 🛠️: Utilizes a rigorous three-stage "Filter-Embedded-Wrapper" protocol (Pearson Correlation/Mutual Information -> SHAP-based Coarse Selection -> Recursive Feature Elimination) to pinpoint the optimal feature set for both wettability ($E_b$) and thermal transport (NVOA).
* **Efficient Hyperparameter Optimization** ⚡: Leverages Gaussian Process-based Bayesian Optimization to efficiently navigate the hyperparameter landscape for 7 heterogeneous base learners (e.g., XGBoost, CatBoost, LightGBM), ensuring optimal model configurations.
* **Robust Stacking Ensemble** 🧠: Constructs a two-tier ensemble architecture integrating diverse tree-based models with a regularized ElasticNet meta-learner, achieving superior generalization ($R^2 > 0.92$) and robustness via nested cross-validation.
* **Interpretable "Two-Level Weighted SHAP"** 📊: Features a custom **"Fidelity-Reliability Weighted Aggregation"** strategy. This method weighs feature contributions based on both intra-fold model accuracy and inter-fold generalization, providing a robust, noise-filtered physicochemical interpretation.
* **Dual-Target Prediction** 🎯: Validated workflow for two distinct physical properties—Interfacial Binding Energy (Wettability) and Vibrational Density of States Overlap (Thermal Conductivity).

---

## 📂 Repository Structure and Workflow

The repository is organized to support the dual-target analysis presented in the manuscript. The workflow utilizes two core Jupyter notebooks to handle feature engineering and model training/prediction respectively.


```

.
├── Functional Group Atlas/                 # 🧪 Main Project Directory
│   ├── Feature engineering-FGA.ipynb       # 📜 Script 1: Hierarchical Feature Selection Pipeline
│   ├── Tree_stacking-FGA.ipynb             # 📜 Script 2: Hyperparameter Optimization, Stacking Ensemble & Prediction
│   ├── Original database-CR.xlsx           # 📊 Raw input database containing 248 candidates with descriptors
│   ├── final_engineered_dataset-VDOS.xlsx  # 📊 Output from Script 1 (Cleaned feature set for NVOA/Eb)
│   ├── yuceji-VDOS.xlsx                    # 📊 Prediction set (Unknown data for screening)
│   └── SHAP_Analysis_Results.xlsx          # 📊 Final interpretability output
│
├── environment.yml                         # 📦 Conda environment config for cross-platform setup
├── requirements.txt                        # 📦 Pip dependencies for basic/non-Conda setup
├── spec-file.txt                           # 📦 Exact Conda config for highest-fidelity reproducibility (Win x64)
├── https://www.google.com/search?q=LICENSE                                 # 📜 The MIT License file
└── README.md                               # 📄 The document you are currently reading

```

### Workflow Details

The workflow is designed to be run sequentially. You will typically run the pipeline twice, once for each target property mentioned in the paper:

1.  **Target 1: Wettability ($E_b$)**
    * **Input**: `Original database-CR.xlsx` (Target column: $E_b$)
    * **Objective**: Identify elemental drivers (electronegativity/bonding).
2.  **Target 2: Heat Transport (NVOA)**
    * **Input**: `Original database-CR.xlsx` (Target column: NVOA)
    * **Objective**: Identify structural drivers (topology/phonon matching).

---

### 📜 Script 1: Hierarchical Feature Engineering (`Feature engineering-FGA.ipynb`)

* **🎯 Function**:
    * Selects the optimal feature subset from the initial database using a "Filter-Embedded-Wrapper" architecture.
    * **Stage 1 (Filter)**: Eliminates multicollinearity using Pearson correlation ($|r|>0.8$) while retaining features with higher Mutual Information with the target.
    * **Stage 2 (Embedded)**: Performs coarse screening using a Random Forest Regressor and SHAP values to filter out the bottom 20% non-informative features.
    * **Stage 3 (Wrapper)**: Implements Recursive Feature Elimination (RFE) with an early stopping mechanism (patience window of 50) to determine the optimal feature set size.

* **▶️ Usage**:
    1.  Prepare the `Original database-CR.xlsx` file containing raw descriptors and the target variable.
    2.  Configure `INPUT_FILE` and `TARGET_COLUMN_INDEX` in the "Configuration Area".
    3.  Run the notebook to generate `final_engineered_dataset...xlsx`.

* **⚙️ Tunable Parameters**:
    * `FILTER_METHOD_CRITERION`: Choose `'mutual_info'` (recommended) or `'pearson'`.
    * `SHAP_COARSE_SELECTION_PERCENT`: Percentage of features to keep in Stage 2 (default `0.8`).
    * `PERFORMANCE_METRIC`: Metric for the wrapper stage (`'r2'` or `'mae'`).
    * `EARLY_STOPPING_PATIENCE`: Steps to wait before stopping RFE if no improvement is seen.

* **📄 Outputs**:
    * `Feature_Engineering_Process_Summary_...xlsx`: Summary of removed features at each stage.
    * `Final_Selected_Dataset_...xlsx`: **(Core Output)** The optimal feature set for the next step.
    * `Performance_Iteration_History_...xlsx`: Log of model performance during iterative removal.

---

### 📜 Script 2: Stacking Ensemble & Prediction (`Tree_stacking-FGA.ipynb`)

* **🎯 Function**:
    * This comprehensive notebook handles **Hyperparameter Optimization**, **Stacking Construction**, **Evaluation**, and **Prediction**.
    * **Optimization**: Uses `BayesSearchCV` (30 iterations) to tune 7 base learners: **XGBoost (XGBR), Random Forest (RF), Gradient Boosting (GBRT), Histogram GBDT (HGBR), Extra Trees (ETR), CatBoost (CBR), and LightGBM (LGBM)**.
    * **Stacking**: Trains an ElasticNet meta-learner to aggregate base model predictions, minimizing overfitting via nested cross-validation ($k=10$).
    * **Interpretation**: Calculates global feature importance using the **Two-Level Weighted SHAP** logic:
        * *Level 1 (Intra-fold)*: Weighted by $1/RMSE$ of base models.
        * *Level 2 (Inter-fold)*: Weighted by $1/RMSE$ of the Stacking model across folds.
    * **Prediction**: Loads the trained ensemble to predict properties for the unknown dataset (`yuceji-VDOS.xlsx`).

* **▶️ Usage**:
    1.  Ensure `EXCEL_FILE_PATH` points to the `Final_Selected_Dataset` output from Script 1.
    2.  Set `UNKNOWN_DATA_FILE` to your prediction set (e.g., `yuceji-VDOS.xlsx`).
    3.  Run cells sequentially to perform optimization, evaluation, and prediction.

* **⚙️ Tunable Parameters**:
    * `ENABLED_MODELS`: List of models to include (e.g., `['XGBR', 'RF', 'CBR', ...]`).
    * `N_ITER_BAYESIAN`: Number of optimization iterations (default `30`).
    * `WEIGHTING_METHOD`: Weighting logic for SHAP aggregation (`'1/RMSE'`).
    * `REUSE_PRETRAINED_STACKING_MODEL`: Set to `True` to skip retraining for prediction; `False` to retrain on the full dataset.

* **📄 Outputs**:
    * **Console**: Optimization logs, CV scores ($R^2$, RMSE), and Base/Meta learner performance.
    * `SHAP_Analysis_Results.xlsx`: Detailed feature importance, SHAP beeswarm data, and global summary.
    * `unknown_predictions_...xlsx`: Final predicted values for the new candidates.

---

## 🗺️ The Workflow

This workflow seamlessly connects data processing to final application, following a clear, sequential path.

 **`Original Database Construction`**
 *Combinatorial library of 248 functional groups (Element × Structure)*
 `⬇️`
 **`Script 1: Feature Engineering`**
 *Dimensionality reduction: Filter -> Embedded -> Wrapper*
 `⬇️`
 **`Script 2 (Part A): Bayesian Optimization`**
 *Finding the optimal "operating state" for XGBoost, CatBoost, LightGBM, etc.*
 `⬇️`
 **`Script 2 (Part B): Stacking & SHAP Analysis`**
 *Assembling the "expert team" and performing Two-Level Weighted Interpretability*
 `⬇️`
 **`Script 2 (Part C): Prediction`**
 *Screening for high-wettability and high-conductivity candidates*

---

## 💻 How to Use

1.  **Environment Setup**:
    Download the scripts and `requirements.txt` file. Install all dependencies.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Workflow Execution**:
    * **Step 1: Feature Engineering**
        * Run `Feature engineering-FGA.ipynb` using `Original database-CR.xlsx`.
        * This generates the cleaned dataset `final_engineered_dataset-VDOS.xlsx`.

    * **Step 2: Train, Evaluate, and Predict**
        * Open `Tree_stacking-FGA.ipynb`.
        * Ensure `EXCEL_FILE_PATH` points to the file generated in Step 1.
        * Ensure `UNKNOWN_DATA_FILE` points to `yuceji-VDOS.xlsx`.
        * Execute the notebook cells. It will automatically optimize hyperparameters, train the Stacking model, generate SHAP visualization data, and output predictions.

    ⚠️ **Important Note**: Script 2 integrates optimization, evaluation, and prediction into a single notebook to ensure variable continuity (e.g., passing optimized model instances).

---

## 📦 Environment Setup & Reproducibility

This section outlines the necessary dependencies. For the highest fidelity reproducibility, we recommend using **Option 1**.

### 🐍 Python Version

This project was developed and tested using **Python 3.10+**.

### 📋 Core Dependencies


```

pandas
numpy
scikit-learn
xgboost
catboost
lightgbm
scikit-optimize
shap
matplotlib
seaborn
openpyxl
tqdm

```

### 🥇 Option 1: Highest-Fidelity Reproducibility (via `spec-file.txt`)

**Platform:** Windows (x64)

```bash
conda create --name fga-env --file spec-file.txt
conda activate fga-env

```

### 🥈 Option 2: Cross-Platform Setup (via `environment.yml`)

```bash
conda env create -f environment.yml -n fga-env
conda activate fga-env

```

### 🥉 Option 3: Basic Setup (via `requirements.txt`)

```bash
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt jupyterlab

```

---

## 📜 License and Correspondence

The data and code in this repository are available under the [MIT License](https://www.google.com/search?q=LICENSE).

For any inquiries or if you use this workflow in your research, please correspond with:

* **Prof. Guangmin Zhou** (Corresponding Author): [guangminzhou@sz.tsinghua.edu.cn](mailto:guangminzhou@sz.tsinghua.edu.cn)
* **Yifei Zhu** (First Author & Code Developer): [zhuyifeiedu@126.com](mailto:zhuyifeiedu@126.com)

**Affiliation**: Tsinghua Shenzhen International Graduate School, Tsinghua University.

---

## 🙏 Acknowledgements

We acknowledge the support from the National Key Research and Development Program of China and the National Natural Science Foundation of China. Special thanks to the open-source communities of `scikit-learn`, `SHAP`, `CatBoost`, `XGBoost`, and `LightGBM`.

```

```
