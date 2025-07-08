import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Command-line argument for CSV file path
parser = argparse.ArgumentParser(description="PEMWE Data Info")
parser.add_argument(
    "--csv", 
    type=str, 
    default="../data.csv", 
    help="Path to the dataset CSV file (default: ../data.csv)"
)
args = parser.parse_args()

# Function to calculate and print relevant statistics
def calculate_stats(df):
    # Define columns of interest
    columns = [
        'Cathode area (mm2)', 'Anode area (mm2)', 'Cell voltage (V)', 
        'Cell current (A)', 'Power (w)', 'Water flow rate (mL/min)', 
        'Temperature (K)', 'Hydrogen flow rate (mL/min)'
    ]

    # Iterate over the columns and calculate statistics
    for col in columns:
        if col in df.columns:
            print(f"Statistics for {col}:")
            print(f"Mean: {df[col].mean():.2f}")
            print(f"Median: {df[col].median():.2f}")
            print(f"Range: {df[col].max() - df[col].min():.2f}")
            print(f"Standard Deviation: {df[col].std():.2f}")
            print(f"Variance: {df[col].var():.2f}")
            print(f"Min: {df[col].min():.2f}")
            print(f"Max: {df[col].max():.2f}")
            print(f"Skewness: {df[col].skew():.2f}")
            print(f"Kurtosis: {df[col].kurtosis():.2f}")
            print("="*50)

def plot_distributions(df):
    features = ['Power (w)', 'Water flow rate (mL/min)', 'Temperature (K)']
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            sns.histplot(df[feature], kde=True, bins=50, ax=axes[i], color='blue')
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Main function to load the dataset and process it
def main():
    file_path = args.csv
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if all required columns are present
        required_columns = [
            'Cathode area (mm2)', 'Anode area (mm2)', 'Cell voltage (V)', 
            'Cell current (A)', 'Power (w)', 'Water flow rate (mL/min)', 
            'Temperature (K)', 'Hydrogen flow rate (mL/min)'
        ]

        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: '{col}' column not found in the dataset.")
        
        # Calculate and print statistics
        calculate_stats(df)
        #plot_distributions(df)

    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    main()