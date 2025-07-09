# PEMWE ANN and RSM Analysis

This repository contains implementations for Artificial Neural Network (ANN) and Response Surface Methodology (RSM) analysis of Proton Exchange Membrane Water Electrolysis (PEMWE) data.

## Environment Setup

### Using Conda (Recommended)

# 1. install lockfile solver

conda install -c conda-forge conda-lock

# 2. create env from lock

conda-lock install --name pemwe-ann conda-lock.yml

# 3. activate conda env

conda activate pemwe-ann

## Data

The analysis uses `data.csv` in the root directory as the default dataset. All scripts are configured to automatically use this file unless otherwise specified.

## ANN Analysis

Navigate to the ANN folder and use the CLI:

```bash
cd ANN
python run.py --help
```

### Available ANN Commands:

1. **Explore**: Exploratory data analysis and pipeline preview

```bash
python run.py explore
```

2. **Tune**: Hyperparameter tuning with cross-validation

```bash
python run.py tune
```

3. **SHAP**: SHAP analysis for feature importance

```bash
python run.py shap
```

### Using custom dataset:

```bash
python run.py explore --csv path/to/your/dataset.csv
```

## RSM Analysis

Navigate to the RSM folder and use the CLI:

```bash
cd RSM
python run.py --help
```

### Available RSM Commands:

1. **Data Info**: Generate data statistics and information

```bash
python run.py data-info
```

2. **RSM Equation**: Generate RSM equation and analysis

```bash
python run.py rsm-equation
```

3. **Surface Plots**: Generate surface scatter plots

```bash
python run.py surface-plots --var1 "Cell voltage (V)" --var2 "Cell current (A)"
```

4. **All**: Run all RSM analyses

```bash
python run.py all
```

### Using custom dataset:

```bash
python run.py data-info --csv path/to/your/dataset.csv
```

## Output Files

All analysis outputs are saved to the `outputs/` folder in the root directory:

### ANN Outputs:

- `best_config.json`: Best hyperparameters and performance metrics
- `trained_model.pkl`: Trained model pipeline
- `tuning_results_full_data.csv`: Full dataset cross-validation results
- `tuning_results_training_set.csv`: Training set cross-validation results
- `shap_ann_summary.png`: SHAP feature importance plot
- `corr_heatmap.png`: Correlation heatmap

### RSM Outputs:

- `rsm_actual_vs_predicted.png`: Actual vs predicted values plot
- `rsm_residuals_vs_predicted.png`: Residuals vs predicted values plot
- `rsm_qq_plot.png`: Q-Q plot for residuals
- `surface_plots_*.png`: Surface plots for variable relationships

## Project Structure

```
pemwe-ann/
├── data.csv                    # Main dataset
├── environment.yml             # Conda environment file
├── outputs/                    # All analysis outputs
├── ANN/
│   ├── run.py                 # ANN CLI interface
│   ├── core.py                # ANN core functions
│   └── requirements.txt       # ANN dependencies
└── RSM/
    ├── run.py                 # RSM CLI interface
    ├── PEMWE_Data_info.py     # Data statistics
    ├── PEMWE_RSM_Eqn.py       # RSM equation analysis
    └── Surface_Scatter_plots_RSM.py  # Surface plots
```

## Reproduction Instructions

To reproduce the results:

### Quick Start (Recommended)

Run all analyses with a single command:

```bash
python3 run_all_analyses.py
```

This will run all ANN and RSM analyses and generate all outputs automatically.

### Manual Execution

1. Set up the environment as described above
2. Run the ANN analysis:
   ```bash
   cd ANN
   python3 run.py explore
   python3 run.py tune
   python3 run.py shap
   ```
3. Run the RSM analysis:
   ```bash
   cd RSM
   python3 run.py all
   ```

All outputs will be generated in the `outputs/` folder for review and verification.

## Notes

- The dataset must contain the following columns:

  - `Cathode area (mm2)`
  - `Anode area (mm2)`
  - `Cell voltage (V)`
  - `Cell current (A)`
  - `Power (w)`
  - `Water flow rate (mL/min)`
  - `Temperature (K)`
  - `Hydrogen flow rate (mL/min)` (target variable)

- All scripts use reproducible random seeds for consistent results across runs
- The conda environment ensures all users have identical package versions for reproducibility
