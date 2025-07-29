import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np


def analyze_and_plot_costs(directory_path, file_pattern='cost_comparison_USA_Peach-1_1_T-1_*.csv'):
    """
    Reads trajectory cost CSVs, calculates mean costs, and plots them
    on a dual y-axis chart.
    """
    search_path = os.path.join(directory_path, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files found matching pattern '{file_pattern}' in directory '{directory_path}'.")
        return

    all_data = []
    pattern = re.compile(r'_(\d+)\.csv$')

    for path in file_paths:
        match = pattern.search(os.path.basename(path))
        if match:
            timestamp = int(match.group(1))
            try:
                df = pd.read_csv(path, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                all_data.append({
                    'Timestamp': timestamp,
                    'Frenetix_Cost': df['Frenetix Cost'].mean(),
                    'Gemini_Cost': df['Gemini Cost'].mean()
                })
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
                print(f"Warning: Could not process file {path}. Error: {e}")
                continue

    if not all_data:
        print("Error: No valid data could be extracted.")
        return

    results_df = pd.DataFrame(all_data).sort_values(by='Timestamp').reset_index(drop=True)
    print("Aggregated Mean Costs per Timestamp:")
    print(results_df)

    # --- Plotting with Dual Y-Axes ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Frenetix Cost on the primary y-axis (left)
    color1 = 'tab:blue'
    ax1.set_xlabel('Timestamp', fontsize=12)
    ax1.set_ylabel('Frenetix Mean Cost', fontsize=12, color=color1)
    line1 = ax1.plot(results_df['Timestamp'], results_df['Frenetix_Cost'], color=color1, marker='o', linestyle='-', label='Frenetix Mean Cost')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Gemini Mean Cost', fontsize=12, color=color2)
    line2 = ax2.plot(results_df['Timestamp'], results_df['Gemini_Cost'], color=color2, marker='s', linestyle='--', label='Gemini Mean Cost')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add a title and a unified legend
    fig.suptitle('Average cost', fontsize=16, weight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    plt.savefig("cost_comparison/cost_USA_Peach-1_1_T-1_all.png")
    plt.show()

if __name__ == '__main__':
    # --- Instructions ---
    # 1. Place your CSV files in a folder (e.g., 'data').
    # 2. Update the 'data_directory' variable to point to your folder.
    # 3. Run the script.

    data_directory = 'cost_comparison'
    analyze_and_plot_costs(data_directory)