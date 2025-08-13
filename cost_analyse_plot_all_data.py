from os import mkdir

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from sklearn.preprocessing import MinMaxScaler


def plot_cost_comparison_with_highlights(directory_path, base_scenario_name,
                                         file_pattern):
    """
    Visualizes the normalized cost comparison, highlighting the best trajectory
    from each algorithm with timestamp labels.
    """
    # --- Data Loading ---
    search_path = os.path.join(directory_path, file_pattern)
    file_paths = glob.glob(search_path)
    if not file_paths:
        print(f"Error: No files found for pattern {file_pattern}")
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
                # Ensure required columns exist
                if 'VLM Cost' in df.columns and 'Frenetix Cost' in df.columns:
                    df['Timestamp'] = timestamp
                    all_dfs.append(df)
                else:
                    print(f"Warning: Skipping {path} due to missing cost columns.")
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if not all_dfs:
        print("Error: No valid data extracted from files.")
        return
    all_points_df = pd.concat(all_dfs, ignore_index=True)

    # --- Data Normalization ---
    # Use MinMaxScaler to scale both cost columns to a [0, 1] range.
    # This makes the comparison visually more intuitive.
    scaler = MinMaxScaler()
    all_points_df[['Frenetix Cost Norm', 'VLM Cost Norm']] = scaler.fit_transform(
        all_points_df[['Frenetix Cost', 'VLM Cost']]
    )
    print(f"number of data points loaded: {len(all_points_df)}")

    # --- 1. Identify Best Trajectories (using original costs) ---
    # The minimum of the normalized value is the same as the minimum of the original value,
    # so we can use the original columns here without issue.
    idx_best_gemini = pd.Index(all_points_df.groupby('Timestamp')['VLM Cost'].idxmin().values)
    idx_best_frenetix = pd.Index(all_points_df.groupby('Timestamp')['Frenetix Cost'].idxmin().values)

    idx_same_best = idx_best_gemini.intersection(idx_best_frenetix)
    idx_best_gemini_only = idx_best_gemini.difference(idx_same_best)
    idx_best_frenetix_only = idx_best_frenetix.difference(idx_same_best)

    best_gemini_only_points = all_points_df.loc[idx_best_gemini_only]
    best_frenetix_only_points = all_points_df.loc[idx_best_frenetix_only]
    same_best_points = all_points_df.loc[idx_same_best]
    print(f'number of best points for Frenetix only: {len(best_frenetix_only_points)}')
    print(f'number of best points for Gemini only: {len(best_gemini_only_points)}')
    print(f'number of best points for both: {len(same_best_points)}')
    print(f'number of timestamps with data: {len(all_points_df["Timestamp"].unique())}')

    # --- 2. Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 12))

    # Plot all data points first using NORMALIZED values
    plt.scatter(
        all_points_df['Frenetix Cost Norm'], all_points_df['VLM Cost Norm'],
        color='gray', alpha=0.5, s=50, label='All Trajectories', zorder=2
    )

    # --- 3. Add Highlighted Points and Labels (using NORMALIZED values) ---
    # Highlight best for Frenetix ONLY
    plt.scatter(
        best_frenetix_only_points['Frenetix Cost Norm'], best_frenetix_only_points['VLM Cost Norm'],
        color='blue', alpha=0.5, s=50,
     label='Best Frenetix Only', zorder=3
    )
    for idx, row in best_frenetix_only_points.iterrows():
        plt.text(row['Frenetix Cost Norm'], row['VLM Cost Norm'], f"   {int(row['Timestamp'])}", fontsize=10, ha='left',
                 va='center', fontweight='bold', zorder=1)

    # Highlight best for Gemini ONLY
    plt.scatter(
        best_gemini_only_points['Frenetix Cost Norm'], best_gemini_only_points['VLM Cost Norm'],
        color='red', alpha=0.5, s=50,
        linewidth=1.5, label='Best Gemini Only', zorder=3
    )
    for idx, row in best_gemini_only_points.iterrows():
        plt.text(row['Frenetix Cost Norm'], row['VLM Cost Norm'], f"   {int(row['Timestamp'])}", fontsize=10, ha='left',
                 va='center', fontweight='bold', zorder=1)

    # Highlight points that are best for BOTH
    if not same_best_points.empty:
        plt.scatter(
            same_best_points['Frenetix Cost Norm'], same_best_points['VLM Cost Norm'],
            color='green', alpha=0.5, s=50,
            linewidth=1.5, label='Best for Both', zorder=4
        )
        for idx, row in same_best_points.iterrows():
            plt.text(row['Frenetix Cost Norm'], row['VLM Cost Norm'], f"   {int(row['Timestamp'])}", fontsize=10,
                     ha='left',
                     va='center', fontweight='bold', zorder=1)

    plt.title(f'Normalized Cost Comparison: Frenetix vs. GEMINI-2.5-pro ({prompt_type}) {base_scenario_name}', fontsize=16, weight='bold')
    plt.xlabel('Frenetix Cost', fontsize=12)
    plt.ylabel('GEMINI Cost', fontsize=12)
    plt.legend(title='Legend')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits to be slightly larger than the [0,1] range for better visualization
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add a y=x line for reference
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='y=x (Equal Cost)')

    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    plt.show()


# --- Run the function ---
if __name__ == '__main__':
    base_scenario_name = 'USA_Peach-1_1_T-1'
    data_directory = f'cost_comparison/{base_scenario_name}'
    prompt_type = 'wo_dspy'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    output_filename = f"{data_directory}/cost_{base_scenario_name}_all_normalized_{prompt_type}.png"

    # Ensure the directory exists before running
    if os.path.exists(data_directory):
        plot_cost_comparison_with_highlights(
            data_directory,
            base_scenario_name,
            file_pattern=f'{prompt_type}_{base_scenario_name}_*.csv'
        )
    else:
        print(f"Error: Data directory not found at '{data_directory}'")
        print("Please ensure the data is in the correct location.")

