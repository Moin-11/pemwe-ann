from __future__ import annotations
import os, random, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, cross_validate
import pathlib
import warnings

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

random.seed(SEED)
np.random.seed(SEED)

TARGET = "Hydrogen flow rate (mL/min)"
EPS = 1e-6
OUT_DIR = pathlib.Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
SAVE_AS_SVG = False
IMG_EXT = "svg" if SAVE_AS_SVG else "png"

SKEWED_COLS = [
    "Temperature_Power_Interaction",
    "Power (w)",
    "Water flow rate (mL/min)",
    "Temperature_WaterFlow_Interaction",
    "Cathode area (mm2)",
    "Anode area (mm2)",
    "Cell current (A)_squared",
    "Cell voltage (V)_squared",
    "Cell voltage (V)",
    "Theoretical_H2_Production",
    "Water_Utilization",
    "Water_Excess_mol",
]

scorers = {
    "rmse": make_scorer(lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)), greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "r2": make_scorer(r2_score),
}

class PEMWEFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        I = df["Cell current (A)"]
        V = df["Cell voltage (V)"]
        P = df["Power (w)"]
        T = df["Temperature (K)"]
        Ac = df["Cathode area (mm2)"]
        Aa = df["Anode area (mm2)"]
        Wf = df["Water flow rate (mL/min)"]

        df["Power_Density"] = P / (Ac + EPS)
        df["Current_Density"] = I / (np.minimum(Ac, Aa) + EPS)
        df["Temperature_Power_Interaction"] = T * P
        df["Temperature_Current_Interaction"] = T * I
        df["Temperature_Voltage_Interaction"] = T * V
        df["Temperature_WaterFlow_Interaction"] = T * Wf

        F = 96_485.33212  # C·mol⁻¹
        n_h2 = (I * 60) / (2 * F)

        F, R = 96_485.33212, 0.082057366  # C·mol⁻¹, L·atm·K⁻¹·mol⁻¹
        n_h2 = (I * 60) / (2 * F)
        df["Theoretical_H2_Production"] = n_h2 * R * T * 1e3

        dens, M = 1.0, 18.015  # g/mL, g/mol
        n_h2o = Wf * dens / M
        df["Water_Utilization"] = n_h2 / (n_h2o + EPS)
        df["Water_Excess_mol"] = n_h2o - n_h2

        for col in [
            "Cell voltage (V)",
            "Cell current (A)",
            "Temperature (K)",
        ]:
            df[f"{col}_squared"] = df[col] ** 2

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.medians_)

class SelectiveQuantile(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str], n_quantiles: int, random_state=None):
        self.cols = cols
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=self.n_quantiles,
            random_state=self.random_state,
            copy=True,
        )

    def fit(self, X, y=None):
        self.qt.fit(X[self.cols])
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df[self.cols] = self.qt.transform(df[self.cols])
        return df


def save_corr(df: pd.DataFrame):
    plt.figure(figsize=(13, 11))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, linewidths=0.4, annot=True, fmt=".2f")
    plt.title("Pearson Correlation")

    corr = df.corr(numeric_only=True)
    all_corr = corr.unstack().reset_index()
    all_corr.columns = ["Feature1", "Feature2", "Correlation"]
    all_corr = all_corr[all_corr["Feature1"] != all_corr["Feature2"]]
    all_corr = (
        all_corr.assign(
            ordered_pair=all_corr[["Feature1", "Feature2"]].apply(lambda x: tuple(sorted(x)), axis=1)
        )
        .drop_duplicates(subset="ordered_pair")
        .drop(columns=["ordered_pair"])
        .sort_values("Correlation", ascending=False)
    )
    print("All correlation pairs:")
    print(all_corr.to_string(index=False))

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"corr_heatmap.{IMG_EXT}", bbox_inches="tight")
    plt.close()

def save_shap_engineered(rest_pipeline: Pipeline, ann_model: MLPRegressor,
                          X_train_eng: pd.DataFrame, X_explain_eng: pd.DataFrame):
    try:
        import shap
    except ImportError:
        print("shap not installed – skipping SHAP plot")
        return

    print(f"Running SHAP analysis:")
    print(f"Background: {len(X_train_eng)} training samples")
    print(f"Explanations: {len(X_explain_eng)} test samples")
    
    background = shap.sample(X_train_eng, min(100, len(X_train_eng)), random_state=SEED)
    cols = X_train_eng.columns
    expl = shap.KernelExplainer(
        lambda dat: ann_model.predict(
            rest_pipeline.transform(pd.DataFrame(dat, columns=cols))
        ),
        background,
        seed=SEED,
    )
    
    shap_values = expl.shap_values(X_explain_eng, nsamples="auto")
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_values, X_explain_eng, plot_type="bar", show=False)
    plt.xlabel("Mean(|SHAP value|)", fontsize=10)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_ann_summary.{IMG_EXT}", bbox_inches="tight")
    plt.close()
    
    print(f"SHAP analysis explains model behavior on test data (unseen samples)")


def high_skew_cols(df: pd.DataFrame, thresh: float = 3.0, min_corr: float = 0.9) -> list[str]:
    skew = df.select_dtypes(include=np.number).skew().abs()
    ranked = skew[skew > thresh].sort_values(ascending=False).index
    selected: list[str] = []
    for col in ranked:
        if not any(abs(df[col].corr(df[c])) > min_corr for c in selected):
            selected.append(col)
    return selected

def cross_validate_model(X: pd.DataFrame, y: pd.Series, neurons_list: list[list[int]]) -> pd.DataFrame:
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, bins))
    y_range = y.max() - y.min() + EPS

    results = []
    for h1, h2 in neurons_list:
        pipe = Pipeline([
            ("eng", PEMWEFeatureEngineer()),
            ("impute", MedianImputer()),
            ("quantile", SelectiveQuantile(SKEWED_COLS, n_quantiles=min(200, len(X)), random_state=SEED)),
            ("scale1", RobustScaler()),
            ("ann", 
                MLPRegressor(
                    hidden_layer_sizes=(h1, h2),
                    activation="relu",
                    solver="lbfgs",
                    alpha=1e-1,
                    max_iter=10000,
                    random_state=SEED
                )
            ),
        ])
        cv_res = cross_validate(
            pipe, X, y,
            cv=splits,
            scoring=scorers,
            n_jobs=1
        )

        rmse = (-cv_res["test_rmse"]).mean()
        mae = (-cv_res["test_mae"]).mean()
        r2 = cv_res["test_r2"].mean()
        
        results.append({
            "Layer1": h1,
            "Layer2": h2,
            "RMSE": rmse,
            "MAE": mae,
            "NRMSE": rmse / y_range,
            "NMAE": mae / y_range,
            "R2": r2,
        })
    return pd.DataFrame(results)

def create_pipeline(hidden_layer_sizes=(8, 8), max_iter=10000, n_quantiles=200, data_len=None):
    if data_len is not None:
        n_quantiles = min(n_quantiles, data_len)
    
    return Pipeline([
        ("eng", PEMWEFeatureEngineer()),
        ("impute", MedianImputer()),
        ("quantile", SelectiveQuantile(SKEWED_COLS, n_quantiles=n_quantiles, random_state=SEED)),
        ("scale1", RobustScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="lbfgs",
            alpha=1e-1,
            max_iter=max_iter,
            random_state=SEED
        )),
    ])

def plot_predictions(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel(f"Actual {TARGET}")
    plt.ylabel(f"Predicted {TARGET}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"predictions.{IMG_EXT}", bbox_inches="tight")
    plt.show()
    plt.close()

def plot_residuals(y_true, y_pred, title="Residuals vs Predicted"):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(0, min(y_pred), max(y_pred), colors='r', linestyles='--')
    plt.xlabel(f"Predicted {TARGET}")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"residuals.{IMG_EXT}", bbox_inches="tight")
    plt.show()
    plt.close()

def setup_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings(
        "ignore",
        message="n_quantiles.*is greater than the total number of samples",
        category=UserWarning,
    )