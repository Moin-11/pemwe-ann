#!/usr/bin/env python3

import argparse
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from core import (
    SEED, TARGET, EPS, OUT_DIR, setup_warnings, save_corr, 
    cross_validate_model, create_pipeline, plot_predictions, 
    plot_residuals, save_shap_engineered, high_skew_cols
)

def tune_command(args):
    """Hyperparameter tuning command."""
    print("[INFO] Starting hyperparameter tuning...")
    
    df = pd.read_csv(args.csv)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    # Default neuron configurations
    neurons_list = [
        [8, 4], [8, 8],
        [16, 8], [16, 16],
        [32, 16], [32, 32],
        [64, 32], [64, 64],
    ]
    
    # =========================================================================
    # PASS A: Full-data 5-fold CV → Table 1 (unchanged values)
    # =========================================================================
    print("[INFO] PASS A: Full-data 5-fold CV...")
    results_full = cross_validate_model(X, y, neurons_list)
    
    # Display and save Pass A results
    print("\n--- PASS A: Full Dataset CV Results ---")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(results_full)
    
    results_full.to_csv(OUT_DIR / "tuning_results_full_data.csv", index=False)
    
    # =========================================================================
    # PASS B: 80/20 split → inner 5-fold CV on 80% → pick best → retrain → evaluate on 20%
    # =========================================================================
    print("\n[INFO] PASS B: 80/20 split for hyperparameter selection...")
    
    # 80/20 split
    y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    
    # Inner 5-fold CV on the 80% training set
    print("[INFO] Inner 5-fold CV on 80% training set...")
    results_train = cross_validate_model(X_train, y_train, neurons_list)
    
    # Display Pass B inner CV results
    print("\n--- PASS B: Training Set (80%) CV Results ---")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(results_train)
    
    results_train.to_csv(OUT_DIR / "tuning_results_training_set.csv", index=False)
    
    # Pick best hyperparameters based on 80% CV
    best_train = results_train.loc[results_train["R2"].idxmax()]
    best_h1, best_h2 = int(best_train["Layer1"]), int(best_train["Layer2"])
    print(f"\nBest config from 80% CV: layers=({best_h1},{best_h2}), R2={best_train['R2']:.4f}")
    
    # Retrain on 80% with best hyperparameters
    print("[INFO] Retraining best model on 80% training set...")
    best_pipe = create_pipeline((best_h1, best_h2), max_iter=800, data_len=len(X_train))
    best_pipe.fit(X_train, y_train)
    
    # Evaluate once on 20% test set
    print("[INFO] Final evaluation on 20% test set...")
    y_pred_test = best_pipe.predict(X_test)
    
    # Calculate test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n--- PASS B: Final Test Set (20%) Results ---")
    print(f"Test Set - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
    
    # Save comprehensive results
    best_config = {
        "selected_architecture": {
            "hidden_layer_sizes": [best_h1, best_h2],
            "selected_based_on": "80% training set CV"
        },
        "pass_a_full_data_cv": {
            "best_architecture": [int(results_full.loc[results_full["R2"].idxmax(), "Layer1"]), 
                                 int(results_full.loc[results_full["R2"].idxmax(), "Layer2"])],
            "r2_score": float(results_full.loc[results_full["R2"].idxmax(), "R2"]),
            "rmse": float(results_full.loc[results_full["R2"].idxmax(), "RMSE"]),
            "mae": float(results_full.loc[results_full["R2"].idxmax(), "MAE"])
        },
        "pass_b_training_cv": {
            "r2_score": float(best_train['R2']),
            "rmse": float(best_train['RMSE']),
            "mae": float(best_train['MAE']),
            "nrmse": float(best_train['NRMSE']),
            "nmae": float(best_train['NMAE'])
        },
        "pass_b_test_set": {
            "r2_score": float(test_r2),
            "rmse": float(test_rmse),
            "mae": float(test_mae)
        }
    }
    
    import json
    with open(OUT_DIR / "best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Create plots using test set predictions
    plot_predictions(y_test, y_pred_test, f"Test Set: Actual vs Predicted {(best_h1, best_h2)}")
    plot_residuals(y_test, y_pred_test, f"Test Set: Residuals vs Predicted {(best_h1, best_h2)}")
    
    # Save model pipeline
    import joblib
    joblib.dump(best_pipe, OUT_DIR / "trained_model.pkl")
    
    print(f"\n[INFO] Pass A results saved to {OUT_DIR}/tuning_results_full_data.csv")
    print(f"[INFO] Pass B results saved to {OUT_DIR}/tuning_results_training_set.csv")
    print(f"[INFO] Best config saved to {OUT_DIR}/best_config.json")
    print(f"[INFO] Model saved to {OUT_DIR}/trained_model.pkl")


def shap_command(args):
    """SHAP analysis command."""
    print("[INFO] Starting SHAP analysis...")
    
    # Load trained model
    model_path = OUT_DIR / "trained_model.pkl"
    if not model_path.exists():
        print("[ERROR] No trained model found. Please run 'train' command first.")
        return
    
    import joblib
    pipe = joblib.load(model_path)
    
    # Load data
    df = pd.read_csv(args.csv)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    # Get engineered features from full dataset
    X_eng = pipe.named_steps["eng"].transform(X)
    
    # Use already fitted preprocessing pipeline (without ANN for SHAP)
    preprocessing_pipeline = pipe[:-1]  # All steps except ANN, already fitted
    
    # Run SHAP analysis
    save_shap_engineered(
        preprocessing_pipeline,
        pipe.named_steps["ann"],
        X_eng,
        X_eng  # Use full dataset for explanation
    )
    
    print(f"[INFO] SHAP analysis saved to {OUT_DIR}/shap_ann_summary.png")

def explore_command(args):
    """Quick, unsupervised EDA & pipeline preview."""
    print("[INFO] Starting exploratory analysis...")
    df = pd.read_csv(args.csv)
    X = df.drop(TARGET, axis=1)

    # --- 1. basic EDA ---
    save_corr(X)  # heat-map
    print("[INFO] High-skew cols:", high_skew_cols(X))
    print("[INFO] Nulls per column:\n", X.isna().sum())

    # --- 2. pipeline step-by-step ---
    # build *preprocessing* part only
    preview_pipe = create_pipeline(hidden_layer_sizes=(1,), max_iter=1)
    preview_pipe.steps.pop()  # remove the ANN layer

    preview_pipe.fit(X)  # unsupervised, so fine
    X_eng = preview_pipe.named_steps["eng"].transform(X)
    print("\nAfter feature engineering:", X_eng.shape)
    print(X_eng.head())

    # Show skewness analysis on engineered features
    auto_cols = high_skew_cols(X_eng, thresh=3.0, min_corr=0.99)
    print("\n[INFO] Auto‐selected skewed cols:", auto_cols)
    
    skew_table = (
        X_eng
        .select_dtypes(include=np.number)
        .skew()
        .abs()
        .sort_values(ascending=False)
        .to_frame("abs_skew")
    )
    print("\n[INFO] Feature skewness:\n", skew_table)

    X_imp = preview_pipe.named_steps["impute"].transform(X_eng)
    print("\nAfter imputation:", X_imp.shape)
    print(X_imp.head())

    X_qt = preview_pipe.named_steps["quantile"].transform(X_imp)
    print("\nAfter quantile:", X_qt.shape)
    print(pd.DataFrame(X_qt, columns=X_imp.columns).head())

    X_scaled = preview_pipe.named_steps["scale1"].transform(X_qt)
    print("\nAfter scaling:", X_scaled.shape)
    print(pd.DataFrame(X_scaled, columns=X_imp.columns).head())

    print("[INFO] Exploration complete. Nothing was tuned or trained.")

def main():
    setup_warnings()
    
    parser = argparse.ArgumentParser(description="PEMWE ANN Analysis Tool")
    parser.add_argument("--csv", type=pathlib.Path, default="data.csv",
                        help="Path to CSV file (default: data.csv)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Explore command
    explore_parser = subparsers.add_parser("explore", help="EDA & pipeline preview")
    explore_parser.set_defaults(func=explore_command)
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning")
    tune_parser.set_defaults(func=tune_command)
    
    # SHAP command
    shap_parser = subparsers.add_parser("shap", help="SHAP analysis")
    shap_parser.set_defaults(func=shap_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Check if CSV exists
    if not args.csv.exists():
        print(f"[ERROR] CSV file not found: {args.csv}")
        return
    
    # Run the appropriate command
    args.func(args)

if __name__ == "__main__":
    main()