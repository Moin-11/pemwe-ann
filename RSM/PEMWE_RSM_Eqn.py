import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
import argparse
import os

# Command-line argument for CSV file path
parser = argparse.ArgumentParser(description="PEMWE RSM Equation")
parser.add_argument(
    "--csv", 
    type=str, 
    default="data.csv", 
    help="Path to the dataset CSV file (default: ../data.csv)"
)
args = parser.parse_args()

# Create outputs directory if it doesn't exist
os.makedirs("../outputs", exist_ok=True)

# Load dataset
file_path = args.csv
df = pd.read_csv(file_path)

EPS = 1e-6
df["Power_Density"] = df["Power (w)"] / (df["Anode area (mm2)"] + EPS)
df["Current_Density"] = df["Cell current (A)"] / (df["Anode area (mm2)"] + EPS)
df["Voltage_Current_Ratio"] = df["Cell voltage (V)"] / (df["Cell current (A)"] + EPS)

F, R = 96485.33212, 0.082057366
n_h2 = (df["Cell current (A)"] * 60) / (2 * F)  # mol/min
df["Theoretical_H2_Production"] = n_h2 * R * df["Temperature (K)"] * 1e3
TARGET = "Hydrogen flow rate (mL/min)"
features = ["Anode area (mm2)", "Cell voltage (V)", "Cell current (A)", 
            "Power (w)", "Temperature (K)", "Power_Density", "Current_Density"]

X = df[features]
y = df[TARGET]

y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

class PEMWEFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        self.medians_ = X.median()
        return self
    def transform(self, X, y=None): 
        return X.fillna(self.medians_)

poly = PolynomialFeatures(degree=2, include_bias=False)
model = Ridge(alpha=1.0)

pipe = Pipeline([
    ("eng", PEMWEFeatureEngineer()),
    ("impute", MedianImputer()),
    ("scale", RobustScaler()),
    ("poly", poly),
    ("ridge", model)
])

start_time = time.time()
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
end_time = time.time()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

n = X_train.shape[0]
p = len(pipe.named_steps["poly"].get_feature_names_out(features))
adj_r2 = 1 - ((1 - r2)*(n-1) / (n-p-1))
y_train_mean = np.mean(y_train)

# Predicted R2 (Q²)
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - y_train_mean)**2)
predicted_r2 = 1 - (ss_res / ss_tot)

sigma_res = np.sqrt(np.sum((y_test - y_pred) ** 2) / len(y_test))
y_mean = np.mean(y_test)
cv = sigma_res / y_mean

print(f"Execution Time: {end_time - start_time:.4f} seconds")
print(f"R² Score: {r2:.4f}")
print(f"Adjusted R² Score: {adj_r2:.4f}")
print(f"Predicted R² (Q²): {predicted_r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Standard Deviation of Residuals (σ): {sigma_res:.4f}")
print(f"Mean Response (ȳ): {y_mean:.4f}")
print(f"Coefficient of Variation (CV): {cv:.4f}")

y_range = y_test.max() - y_test.min()
nrmse = rmse / y_range
nmae = mae / y_range

print(f"Normalized RMSE (NRMSE): {nrmse:.4f}")
print(f"Normalized MAE (NMAE): {nmae:.4f}")

feature_names = pipe.named_steps["poly"].get_feature_names_out(features)
coefs = pipe.named_steps["ridge"].coef_
intercept = pipe.named_steps["ridge"].intercept_

equation = "y = {:.4f}".format(intercept)
for f, c in zip(feature_names, coefs):
    equation += " + ({:.4f})*{}".format(c, f)

print("\nRSM Regression Equation:\n", equation)

residuals = y_test - y_pred

# Actual vs. Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Ideal fit")
plt.xlabel("Actual H2 Flow Rate (mL/min)")
plt.ylabel("Predicted H2 Flow Rate (mL/min)")
plt.legend()
plt.title("Actual vs Predicted H2 Flow Rate")
plt.savefig("../outputs/rsm_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

# Residuals vs. Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors="k")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted H2 Flow Rate (mL/min)")
plt.ylabel("Residuals (mL/min)")
plt.title("Residuals vs. Predicted Plot")
plt.savefig("../outputs/rsm_residuals_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

# Q–Q Plot
plt.figure(figsize=(6,5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q–Q Plot for Residuals")
plt.savefig("../outputs/rsm_qq_plot.png", dpi=300, bbox_inches="tight")
plt.close()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(pipe, X, y, cv=skf.split(X, y_bins), scoring='r2')
mse_scores = cross_val_score(pipe, X, y, cv=skf.split(X, y_bins), scoring='neg_mean_squared_error')
rmse_values = np.sqrt(-mse_scores)

print(f"5-fold Stratified Cross-Validation R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"5-fold Stratified Cross-Validation RMSE: {rmse_values.mean():.4f} ± {rmse_values.std():.4f}")



def predict_h2_flow_rate(input_values_array):
    """Accepts a numpy array with shape (1, n_features)"""
    return pipe.predict(input_values_array)[0]

from scipy.optimize import differential_evolution

# Bounds for each feature (based on dataset ranges or engineering constraints)
bounds = [
    (df["Anode area (mm2)"].min(), df["Anode area (mm2)"].max()),
    (df["Cell voltage (V)"].min(), df["Cell voltage (V)"].max()),
    (df["Cell current (A)"].min(), df["Cell current (A)"].max()),
    (df["Power (w)"].min(), df["Power (w)"].max()),
    (df["Temperature (K)"].min(), df["Temperature (K)"].max()),
]

H2_min = df["Hydrogen flow rate (mL/min)"].min()
H2_max = df["Hydrogen flow rate (mL/min)"].max()

# Objective: we minimize the NEGATIVE prediction (to maximize H2 flow rate.)
def objective(x):
    """Objective using desirability for H2 flow rate."""
    if x.ndim > 1:
        # This means it's a batch of candidates
        results = []
        for xi in x:
            results.append(objective(xi))
        return np.asarray(results)

    # Single candidate point case
    input_array = pd.DataFrame([x], columns=["Anode area (mm2)", "Cell voltage (V)",
                                              "Cell current (A)", "Power (w)",
                                              "Temperature (K)"])
    # Add engineered features
    input_array["Power_Density"] = input_array["Power (w)"] / (input_array["Anode area (mm2)"] + EPS)
    input_array["Current_Density"] = input_array["Cell current (A)"] / (input_array["Anode area (mm2)"] + EPS)

    """F, R = 96485.33212, 0.082057366
    n_h2 = (input_array["Cell current (A)"] * 60) / (2 * F) 
    input_array["Theoretical_H2_Production"] = n_h2 * R * input_array["Temperature (K)"] * 1e3"""

    prediction = pipe.predict(input_array)[0]

    # Desirability in range [0,1]
    d_h2 = np.clip((prediction - H2_min) / (H2_max - H2_min), 0, 1)
    return -d_h2


# Perform Optimization
results = differential_evolution(objective, bounds, seed=42, workers =1)

optimal_values = results.x
optimal_h2_rate = -results.fun * (H2_max - H2_min) + H2_min
optimal_desirability = -results.fun

print("\nOptimal Inputs:", dict(zip(["Anode area (mm2)", "Cell voltage (V)",
                                     "Cell current (A)", "Power (w)",
                                     "Temperature (K)"], optimal_values)))
print(f"Predicted Optimal H2 Flow Rate: {optimal_h2_rate:.2f} mL/min")
print(f"Optimal Desirability: {optimal_desirability:.4f}")