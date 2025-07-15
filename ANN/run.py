#!/usr/bin/env python3

import argparse
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from core import (
    SEED, TARGET, EPS, OUT_DIR, setup_warnings, save_corr,
    cross_validate_model, create_pipeline, plot_predictions,
    plot_residuals, save_shap_engineered, high_skew_cols,
    smape, mape_above_thresh
)
from sklearn.model_selection import train_test_split


def tune_command(args):
    print("Starting tuning to find the best model...")
    
    df = pd.read_csv(args.csv)
    # --- diagnose raw target distribution (no pipeline) ---
    y = df[TARGET]
    print("Raw target distribution:")
    print(y.describe())
    for th in [0.1, 1, 5]:
        pct = (y < th).mean() * 100
        print(f"{pct:5.1f}% of samples have {TARGET} < {th} mL/min")
    # ------------------------------------------------------
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    neurons_list = [
        [8, 4], [8, 8],
        [16, 8], [16, 16],
        [32, 16], [32, 32],
        [64, 32], [64, 64],
    ]
    
    print("5-fold cross-validation for architecture selection...")
    results_full = cross_validate_model(X, y, neurons_list)
    
    print("\n--- 5-fold Cross-Validation Results ---")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(results_full)
    
    
    best_full = results_full.loc[results_full["R2"].idxmax()]
    best_h1, best_h2 = int(best_full["Layer1"]), int(best_full["Layer2"])
    print(f"\nSelected architecture: layers=({best_h1},{best_h2}), R2={best_full['R2']:.4f}")
    
    print("\nTraining final model with selected architecture...")
    
    y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    
    print(f"80/20 split created: {len(X_train)} train, {len(X_test)} test samples")
    
    print(f"Training ({best_h1},{best_h2}) model on 80% data...")
    final_pipe = create_pipeline((best_h1, best_h2), max_iter=800, data_len=len(X_train))
    final_pipe.fit(X_train, y_train)
    
    print("Final evaluation on 20% test set...")
    y_pred_test = final_pipe.predict(X_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_smape = smape(y_test, y_pred_test)
    test_mape5 = mape_above_thresh(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    y_range = y_test.max() - y_test.min() + EPS
    test_nrmse = test_rmse / y_range
    test_nmae = test_mae / y_range
    
    print(f"\n--- Final Model Evaluation Results ---")
    print(f"RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, SMAPE: {test_smape:.4f}, MAPE5: {test_mape5:.4f}, R2: {test_r2:.4f}")
    print(f"NRMSE: {test_nrmse:.4f}, NMAE: {test_nmae:.4f}")
    
    best_config = {
        "methodology": "Architecture selection via 5-fold CV, final model training on 80/20 split",
        "selected_architecture": {
            "hidden_layer_sizes": [best_h1, best_h2],
            "selected_based_on": "5-fold cross-validation"
        },
        "cross_validation": {
            "best_architecture": [best_h1, best_h2],
            "r2_score": float(best_full['R2']),
            "rmse": float(best_full['RMSE']),
            "mae": float(best_full['MAE']),
            "nrmse": float(best_full['NRMSE']),
            "nmae": float(best_full['NMAE']),
            "smape": float(best_full['SMAPE']),
            "mape5": float(best_full['MAPE5'])
        },
        "final_model": {
            "training_data_size": len(X_train),
            "test_data_size": len(X_test),
            "test_r2_score": float(test_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_nrmse": float(test_nrmse),
            "test_nmae": float(test_nmae),
            "test_smape": float(test_smape),
            "test_mape5": float(test_mape5)
        }
    }
    
    import json
    with open(OUT_DIR / "best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    plot_predictions(y_test, y_pred_test, f"Actual vs Predicted {(best_h1, best_h2)}")
    plot_residuals(y_test, y_pred_test, f"Residuals vs Predicted {(best_h1, best_h2)}")
    
    print(f"Configuration saved to {OUT_DIR}/best_config.json")
    print(f"Prediction plots saved to {OUT_DIR}/predictions.png and {OUT_DIR}/residuals.png")


def shap_command(args):
    print("Starting SHAP analysis...")
    
    config_path = OUT_DIR / "best_config.json"
    if not config_path.exists():
        print("ERROR: No configuration found. Please run 'tune' command first.")
        return
    
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    df = pd.read_csv(args.csv)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    print(f"Creating 80/20 split for SHAP analysis...")
    
    y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    
    print(f"Using 80/20 split for SHAP: {len(X_train)} train, {len(X_test)} test samples")
    
    best_arch = config["selected_architecture"]["hidden_layer_sizes"]
    pipe = create_pipeline(tuple(best_arch), max_iter=800, data_len=len(X_train))
    pipe.fit(X_train, y_train)
    print(f"Recreated model with architecture {best_arch}")
    
    X_train_eng = pipe.named_steps["eng"].transform(X_train)
    X_test_eng = pipe.named_steps["eng"].transform(X_test)
    
    preprocessing_pipeline = pipe[1:-1]  # All steps except ANN, already fitted
    
    save_shap_engineered(
        preprocessing_pipeline,
        pipe.named_steps["ann"],
        X_train_eng,
        X_test_eng
    )
    
    print(f"SHAP analysis saved to {OUT_DIR}/shap_ann_summary.png")
    print(f"Analysis explains model behavior on {len(X_test_eng)} test samples")


def explore_command(args):
    print("Starting exploratory analysis...")
    df = pd.read_csv(args.csv)
    X = df.drop(TARGET, axis=1)

    save_corr(X)  # heat-map
    print("High-skew cols:", high_skew_cols(X))
    print("Nulls per column:\n", X.isna().sum())

    preview_pipe = create_pipeline(hidden_layer_sizes=(1,), max_iter=1)
    preview_pipe.steps.pop()

    preview_pipe.fit(X)
    X_eng = preview_pipe.named_steps["eng"].transform(X)
    print("\nAfter feature engineering:", X_eng.shape)
    print(X_eng.head())

    auto_cols = high_skew_cols(X_eng, thresh=3.0, min_corr=0.99)
    print("\nAuto‐selected skewed cols:", auto_cols)
    
    skew_table = (
        X_eng
        .select_dtypes(include=np.number)
        .skew()
        .abs()
        .sort_values(ascending=False)
        .to_frame("abs_skew")
    )
    print("\nFeature skewness:\n", skew_table)

    X_imp = preview_pipe.named_steps["impute"].transform(X_eng)
    print("\nAfter imputation:", X_imp.shape)
    print(X_imp.head())

    X_qt = preview_pipe.named_steps["quantile"].transform(X_imp)
    print("\nAfter quantile:", X_qt.shape)
    print(pd.DataFrame(X_qt, columns=X_imp.columns).head())

    X_scaled = preview_pipe.named_steps["scale1"].transform(X_qt)
    print("\nAfter scaling:", X_scaled.shape)
    print(pd.DataFrame(X_scaled, columns=X_imp.columns).head())

    print("Exploration complete. Nothing was tuned or trained.")

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
    tune_parser = subparsers.add_parser("tune", help="Tuning to find best model")
    tune_parser.set_defaults(func=tune_command)
    
    # SHAP command
    shap_parser = subparsers.add_parser("shap", help="SHAP analysis")
    shap_parser.set_defaults(func=shap_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if not args.csv.exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        return
    
    args.func(args)

if __name__ == "__main__":
    main()