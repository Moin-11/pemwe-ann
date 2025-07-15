#!/usr/bin/env python3

import argparse
import pathlib
import sys
import os

# Add the parent directory to the path so we can import from the RSM folder
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def data_info_command(args):
    """PEMWE Data Info analysis command."""
    print("[INFO] Starting data info analysis...")
    
    # Import and run the data info analysis
    import subprocess
    cmd = [sys.executable, "PEMWE_Data_info.py", "--csv", str(args.csv)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print("[INFO] Data info analysis completed successfully.")
    else:
        print(f"[ERROR] Data info analysis failed: {result.stderr}")

def rsm_equation_command(args):
    """RSM Equation analysis command."""
    print("[INFO] Starting RSM equation analysis...")
    
    # Import and run the RSM equation analysis
    import subprocess
    cmd = [sys.executable, "PEMWE_RSM_Eqn.py", "--csv", str(args.csv)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print("[INFO] RSM equation analysis completed successfully.")
    else:
        print(f"[ERROR] RSM equation analysis failed: {result.stderr}")

def surface_plots_command(args):
    """Surface scatter plots command."""
    print("[INFO] Starting surface plots generation...")
    
    if not args.var1 or not args.var2:
        print("[ERROR] Both --var1 and --var2 are required for surface plots.")
        return
    
    # Import and run the surface plots analysis
    import subprocess
    cmd = [sys.executable, "Surface_Scatter_plots_RSM.py", "--csv", str(args.csv), "--var1", args.var1, "--var2", args.var2]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print("[INFO] Surface plots generation completed successfully.")
    else:
        print(f"[ERROR] Surface plots generation failed: {result.stderr}")

def all_command(args):
    """Run all RSM analyses."""
    print("[INFO] Running all RSM analyses...")
    
    # Run data info
    data_info_command(args)
    
    # Run RSM equation
    rsm_equation_command(args)
    
    # Run surface plots with default variables
    args.var1 = "Cell voltage (V)"
    args.var2 = "Cell current (A)"
    surface_plots_command(args)
    
    print("[INFO] All RSM analyses completed.")

def main():
    parser = argparse.ArgumentParser(description="PEMWE RSM Analysis Tool")
    parser.add_argument("--csv", type=pathlib.Path, default="data.csv",
                        help="Path to CSV file (default: ../data.csv)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data info command
    data_info_parser = subparsers.add_parser("data-info", help="Generate data statistics and info")
    data_info_parser.set_defaults(func=data_info_command)
    
    # RSM equation command
    rsm_parser = subparsers.add_parser("rsm-equation", help="Generate RSM equation and analysis")
    rsm_parser.set_defaults(func=rsm_equation_command)
    
    # Surface plots command
    surface_parser = subparsers.add_parser("surface-plots", help="Generate surface scatter plots")
    surface_parser.add_argument("--var1", type=str, required=True,
                               help="Column name for the first variable (X-axis)")
    surface_parser.add_argument("--var2", type=str, required=True,
                               help="Column name for the second variable (Y-axis)")
    surface_parser.set_defaults(func=surface_plots_command)
    
    # All command
    all_parser = subparsers.add_parser("all", help="Run all RSM analyses")
    all_parser.set_defaults(func=all_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Check if CSV exists
    if not args.csv.exists():
        print(f"[ERROR] CSV file not found: {args.csv}")
        return
    
    # Change to RSM directory to ensure proper imports
    os.chdir(pathlib.Path(__file__).parent)
    
    # Run the appropriate command
    args.func(args)

if __name__ == "__main__":
    main()