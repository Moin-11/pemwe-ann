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
**Purpose**: Tuning to find best model and evaluation
```bash
python run.py tune
```

**Workflow**:

**5-fold Cross-Validation for Architecture Selection**
- Tests all 8 architectures using 5-fold stratified cross-validation on 100% of data
- Provides comprehensive performance overview for architecture selection

**Final Model Training and Evaluation**
- Takes the winning architecture from cross-validation
- Creates 80/20 stratified split
- Trains final model with the chosen config on training set (80%)
- Evaluates once on held-out test set (20%)

**Architectures tested**: (8,4), (8,8), (16,8), (16,16), (32,16), (32,32), (64,32), (64,64)

**Outputs**:
- `outputs/best_config.json` - Comprehensive results from both stages
- `outputs/predictions.png` - Actual vs predicted values plot
- `outputs/residuals.png` - Residuals vs predicted values plot

### 3. SHAP Analysis
**Purpose**: Feature importance analysis using methodologically sound approach
```bash
python run.py shap
```

**What it does**:
- Requires a trained model (run `tune` first)
- Creates an **80/20 split** (matches tune command methodology)
- Uses **training data (80%) for SHAP background** and **test data (20%) for explanations**
- Explains model behavior on unseen data for generalization insights
- Shows how the model interprets new samples

**Methodological Benefits**:
- Explains genuine feature importance for unseen data
- Consistent with standard SHAP practice (train background + test explanations)
- More informative than training-only explanations
- Fully consistent with tune command split

**Outputs**:
- `outputs/shap_ann_summary.png` - SHAP feature importance plot (test data explanations)


## Complete Workflow

```bash
python run.py explore    # EDA and pipeline preview
python run.py tune       # Exhaustive search with test evaluation and plots
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
- `best_config.json` - Comprehensive results from both stages
- `predictions.png` - Actual vs predicted values plot
- `residuals.png` - Residuals vs predicted values plot
- `shap_ann_summary.png` - SHAP feature importance plot (test data explanations)

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