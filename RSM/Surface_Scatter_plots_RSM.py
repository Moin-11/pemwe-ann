import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
from scipy.interpolate import griddata
import os

# Command-line argument for CSV file path
parser = argparse.ArgumentParser(description="Surface Scatter Plots RSM")
parser.add_argument(
    "--csv", 
    type=str, 
    default="../data.csv", 
    help="Path to the dataset CSV file (default: ../data.csv)"
)

# Add command-line arguments for Variable_1 and Variable_2
parser.add_argument(
    "--var1", 
    type=str, 
    required=True, 
    help="Column name for the first variable (X-axis)"
)
parser.add_argument(
    "--var2", 
    type=str, 
    required=True, 
    help="Column name for the second variable (Y-axis)"
)

# Parse the arguments
args = parser.parse_args()
variable_1 = args.var1
variable_2 = args.var2

# Create outputs directory if it doesn't exist
os.makedirs("../outputs", exist_ok=True)

# Load dataset
file_path = args.csv
df = pd.read_csv(file_path)

# Feature Engineering
df['Power_Density'] = df['Power (w)'] / (df['Cathode area (mm2)'] + df['Anode area (mm2)'])
df['Current_Density'] = df['Cell current (A)'] / (df['Cathode area (mm2)'] + df['Anode area (mm2)'])
df['Voltage_Current_Ratio'] = df['Cell voltage (V)'] / df['Cell current (A)']
df['Temperature_Power_Interaction'] = df['Temperature (K)'] * df['Power (w)']

df.replace({'Voltage_Current_Ratio': {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)
df['Voltage_Current_Ratio'] = df['Voltage_Current_Ratio'].fillna(df['Voltage_Current_Ratio'].mean())


print(np.max(df['Voltage_Current_Ratio']), np.min(df['Voltage_Current_Ratio']) )

# Define features and target
features = [
    "Anode area (mm2)",
    "Cell voltage (V)",
    "Cell current (A)",
    "Power (w)",
    "Temperature (K)",
    "Power_Density",
    "Current_Density",
    "Voltage_Current_Ratio",
    "Temperature_Power_Interaction",
]
target = "Hydrogen flow rate (mL/min)"

df_renamed = df.rename(columns=lambda x: x.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_"))
df_renamed.replace([np.inf, -np.inf], np.nan, inplace=True)
df_renamed.dropna(inplace=True)
clean_features = [f.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_") for f in features]
clean_target = target.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")

#mask = df[target] <= 1000
#df = df.loc[mask]

# Extract the data points using the provided column names
x = df[variable_1].values  # Use the column name provided via --var1
y = df[variable_2].values  # Use the column name provided via --var2
z = df[target].values
colors = z



from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Assume x, y, z, and colors are already defined
X, Y = np.meshgrid(x, y)
Z = griddata((x, y), z, (X, Y), method='linear')

# Fill NaNs
Z = np.nan_to_num(Z, nan=np.nanmean(z))

# Create figure with 2 subplots side by side
fig = plt.figure(figsize=(14, 6))

# Surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('3D Surface Plot')
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# Scatter plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc = ax2.scatter3D(x, y, z, c=colors, cmap='viridis', marker='^')
ax2.set_title('3D Scatter Plot with Color Mapping')
ax2.set_xlabel('X Axis')
ax2.set_ylabel('Y Axis')
ax2.set_zlabel('Z Axis')
fig.colorbar(sc, ax=ax2, label='Z Value', shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig(f"../outputs/surface_plots_{variable_1.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_vs_{variable_2.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"[INFO] Surface plots saved to ../outputs/surface_plots_{variable_1.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_vs_{variable_2.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png")