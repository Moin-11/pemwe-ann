# PEMWE ANN Analysis Tool

A restructured and organized implementation of the PEMWE (Proton Exchange Membrane Water Electrolysis) neural network analysis pipeline.

## Installation

1. Navigate to the restructured directory:
```bash
cd restructured/
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## File Structure

```
├── core.py          # All transformers, helpers, and utilities
├── run.py           # CLI with sub-commands
├── requirements.txt # Required dependencies
├── data.csv         # Dataset
├── outputs/         # Generated results and plots
└── README.md        # This file
```

## Usage

The tool provides a command-line interface with multiple sub-commands. Run commands in the following **recommended order**:

### 1. Explore (Start Here)
**Purpose**: Exploratory Data Analysis and pipeline preview
```bash
python run.py explore
```

**What it does**:
- Generates correlation heatmap of raw features
- Identifies high-skew columns on raw and engineered features
- Shows null values per column
- Demonstrates step-by-step pipeline transformations
- Shows auto-selected skewed columns and feature skewness table
- **No training occurs** - purely exploratory

**Outputs**:
- `outputs/corr_heatmap.png` - Correlation heatmap
- Console output showing skewed columns analysis and pipeline steps

### 2. Tune
**Purpose**: Two-pass hyperparameter tuning and model evaluation
```bash
python run.py tune
```

**Two-Pass Workflow**:

**Pass A: Full-data 5-fold CV → Table 1 (unchanged values)**
- Tests all 8 architectures using 5-fold stratified cross-validation on 100% of data
- Provides comprehensive performance overview (unchanged methodology)

**Pass B: 80/20 split → Inner CV → Best selection → Final evaluation**
- Creates 80/20 stratified split
- Performs inner 5-fold CV on training set (80%) for all architectures
- Selects best architecture based on training set CV performance
- Retrains best model on full training set (80%)
- Evaluates once on held-out test set (20%)

**Architectures tested**: (8,4), (8,8), (16,8), (16,16), (32,16), (32,32), (64,32), (64,64)

**Outputs**:
- `outputs/tuning_results_full_data.csv` - Pass A: Full dataset CV results
- `outputs/tuning_results_training_set.csv` - Pass B: Training set (80%) CV results
- `outputs/best_config.json` - Comprehensive results from both passes
- `outputs/trained_model.pkl` - Final trained model (selected from Pass B)
- **Test set plots** (displayed) - Honest evaluation plots using 20% test data

### 3. SHAP Analysis
**Purpose**: Feature importance analysis
```bash
python run.py shap
```

**What it does**:
- Requires a trained model (run `train` first)
- Performs SHAP analysis on engineered features
- Generates feature importance plots

**Outputs**:
- `outputs/shap_ann_summary.png` - SHAP feature importance plot

## Complete Workflow

```bash
python run.py explore    # EDA and pipeline preview
python run.py tune       # Hyperparameter tuning with test evaluation and plots
python run.py shap       # Feature importance analysis
```

## Key Features

- **Stratified Sampling**: Ensures balanced splits across target quantiles
- **Feature Engineering**: Physics-aware features for PEMWE systems
- **Comprehensive EDA**: Correlation analysis, skewness detection
- **Reproducible**: Fixed random seeds throughout
- **Modular Design**: All components organized in `core.py`

## Output Files

All results are saved in the `outputs/` directory:

- `corr_heatmap.png` - Feature correlation heatmap
- `tuning_results_full_data.csv` - Pass A: Full dataset CV results
- `tuning_results_training_set.csv` - Pass B: Training set CV results
- `best_config.json` - Comprehensive results from both passes
- `trained_model.pkl` - Final trained scikit-learn pipeline
- `shap_ann_summary.png` - SHAP feature importance plot

## Dataset

The tool expects a CSV file with columns:
- `Cathode area (mm2)`
- `Anode area (mm2)`
- `Cell voltage (V)`
- `Cell current (A)`
- `Power (w)`
- `Water flow rate (mL/min)`
- `Temperature (K)`
- `Hydrogen flow rate (mL/min)` (target variable)

## Custom Dataset

To use a different dataset:
```bash
python run.py explore --csv /path/to/your/data.csv
```

## Notes

- The tool automatically handles the "Temperature" vs "Temprature" column naming
- All random operations are seeded for reproducibility
- Nested CV is recommended for datasets with ~500 rows to avoid optimistic bias
- SHAP analysis requires the `shap` library (included in requirements.txt)