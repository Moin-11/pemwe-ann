# Installation Guide

## Complete Setup Instructions

### Step 1: Install Conda/Miniconda
If you don't have conda installed:

**On macOS/Linux:**
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**On Windows:**
Download and install from: https://docs.conda.io/en/latest/miniconda.html

### Step 2: Clone the Repository
```bash
git clone https://github.com/yourusername/pemwe-ann.git
cd pemwe-ann
```

### Step 3: Create the Conda Environment
```bash
# Create environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate pemwe-ann
```

### Step 4: Verify Installation
```bash
# Test that everything works
python3 run_all_analyses.py --help
```

### Step 5: Run the Analysis
```bash
# Run all analyses
python3 run_all_analyses.py
```

## Alternative: Manual Installation

If conda fails, you can use pip:

```bash
# Create virtual environment
python3 -m venv pemwe-env
source pemwe-env/bin/activate  # On Windows: pemwe-env\Scripts\activate

# Install requirements
pip install -r ANN/requirements.txt
pip install shap

# Run analysis
python3 run_all_analyses.py
```

## Troubleshooting

### Issue: "Command not found: python"
**Solution:** Use `python3` instead of `python`

### Issue: "ModuleNotFoundError"
**Solution:** 
```bash
conda activate pemwe-ann
pip install <missing-module>
```

### Issue: "Permission denied"
**Solution:**
```bash
chmod +x run_all_analyses.py
python3 run_all_analyses.py
```

### Issue: Environment conflicts
**Solution:**
```bash
# Remove and recreate environment
conda env remove -n pemwe-ann
conda env create -f environment.yml
conda activate pemwe-ann
```

## Expected Output Structure

After running the analysis, you should see:
```
outputs/
├── best_config.json
├── corr_heatmap.png
├── cv_metrics.csv
├── nested_cv_results.json
├── rsm_actual_vs_predicted.png
├── rsm_qq_plot.png
├── rsm_residuals_vs_predicted.png
├── shap_ann_summary.png
├── surface_plots_*.png
├── trained_model.pkl
└── tuning_results_*.csv
```

## System Requirements

- Python 3.12+
- 4GB+ RAM
- 1GB+ free disk space
- Internet connection for initial setup

## Platform Compatibility

✅ **Tested on:**
- macOS (Intel/ARM)
- Linux (Ubuntu/CentOS)
- Windows 10/11

## Package Versions

The environment file ensures exact reproducibility across systems. All package versions are locked for consistent results.