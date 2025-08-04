import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np


def analyze_and_plot_all_points(directory_path, file_pattern='cost_comparison_USA_Peach-1_1_T-1_*.csv'):
    """
    Reads trajectory cost CSVs, and plots all individual data points along with
    the mean trend line on a dual y-axis chart.
    """
    search_path = os.path.join(directory_path, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files found matching pattern '{file_pattern}' in directory '{directory_path}'.")
        return

    all_dfs = []
    pattern = re.compile(r'_(\d+)\.csv$')

    for path in file_paths:
        match = pattern.search(os.path.basename(path))
        if match:
            timestamp = int(match.group(1))
            try:
                df = pd.read_csv(path, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                df['Timestamp'] = timestamp
                all_dfs.append(df)
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
                print(f"Warning: Could not process file {path}. Error: {e}")
                continue

    if not all_dfs:
        print("Error: No valid data could be extracted.")
        return

    all_points_df = pd.concat(all_dfs, ignore_index=True)

    mean_costs_df = all_points_df.groupby('Timestamp').agg({
        'Frenetix Cost': 'mean',
        'Gemini Cost': 'mean'
    }).reset_index()

    print("Aggregated Mean Costs per Timestamp:")
    print(mean_costs_df)

    # --- Plotting all points with a mean trend line ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # --- Frenetix Cost (Left Axis) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Timestamp', fontsize=12)
    ax1.set_ylabel('Frenetix Cost', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.scatter(all_points_df['Timestamp'], all_points_df['Frenetix Cost'], alpha=0.2, color=color1)
    # The plot() function returns a list of Line2D objects; we'll use the first one for the legend
    line1 = ax1.plot(mean_costs_df['Timestamp'], mean_costs_df['Frenetix Cost'], color=color1, marker='o',
                     linestyle='-', linewidth=2.5, label='Frenetix (Mean)')

    # --- Gemini Cost (Right Axis) ---
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Gemini Cost', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.scatter(all_points_df['Timestamp'], all_points_df['Gemini Cost'], alpha=0.2, color=color2)
    line2 = ax2.plot(mean_costs_df['Timestamp'], mean_costs_df['Gemini Cost'], color=color2, marker='s', linestyle='--',
                     linewidth=2.5, label='Gemini (Mean)')

    # --- Final Touches (Simplified Legend Creation) ---
    fig.suptitle('Cost Distribution and Mean Trend', fontsize=16, weight='bold')
    # Create the legend by explicitly passing the handles of the two mean-lines
    ax1.legend(handles=[line1[0], line2[0]], loc='upper left')

    fig.tight_layout()
    # Ensure the output directory exists before saving
    output_dir = os.path.dirname("cost_comparison/cost_USA_Peach-1_1_T-1_all.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig("cost_comparison/cost_USA_Peach-1_1_T-1_all.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    data_directory = 'cost_comparison'
    # To run this standalone, you'd need dummy data. Assuming it exists.
    # To create dummy data, you can use a function like in previous examples.
    analyze_and_plot_all_points(data_directory)