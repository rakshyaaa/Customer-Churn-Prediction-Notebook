# Customer Churn Prediction � Comparative Analysis

Predictive modeling of customer churn using a Kaggle dataset. Multiple ML baselines and an ANN are trained with a shared preprocessing pipeline to compare performance and uncover churn drivers.

## Data & Features
- Source: Kaggle churn dataset with provided train/test splits.
- Target: `Churn` (binary).
- Inputs: numeric charges/usage/tenure, categorical subscription/payment/device/content flags, etc.

## Preprocessing
- Numeric: median imputation ? StandardScaler.
- Categorical: most-frequent imputation ? OneHotEncoder(handle_unknown="ignore", sparse_output=False).
- Implemented via `ColumnTransformer`; 80/20 stratified split for validation.

## Imbalance Handling
```python
classes = np.array([0, 1])
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
w0, w1 = cw[0], cw[1]
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = max((neg / max(1, pos)), 1.0)  # for boosting libs
```
- `class_weight` used in Logistic Regression, Random Forest, HistGradientBoosting.
- `scale_pos_weight` (`spw`) used in XGBoost/LightGBM.
- Rationale: avoid ignoring minority churners (alternative to SMOTE used in the reference notebook).

## Models (all wrapped in `Pipeline` with the shared preprocessor)
- Logistic Regression (balanced, saga solver)
- Logistic Elastic Net
- Random Forest (400 trees, balanced weights)
- HistGradientBoosting
- XGBoost (hist, tuned depth/learning rate, `scale_pos_weight`)
- LightGBM (tuned estimators/learning rate, `scale_pos_weight`)
- ANN unweighted and weighted (in `ann.ipynb`)


## Evaluation
- Metrics: ROC-AUC (primary), accuracy, classification report.
- Validation: hold-out (80/20) and cross-validation where applicable.
- Threshold tuning recommended to balance precision/recall to business cost.


## How to Run
1) Install deps: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`, `matplotlib`, `seaborn`.
2) Place data under `dataset/` as expected by the notebooks.
3) Run `notebook.ipynb` for classical ML comparisons.
4) Run `ann.ipynb` for the ANN baseline and tuning.
5) Review metrics/plots; adjust the decision threshold for business goals.

## Reference
Inspired by: [Kaggle notebook](https://www.kaggle.com/code/ayobamimike/churnshield-predictive-customer-retention-platform). 