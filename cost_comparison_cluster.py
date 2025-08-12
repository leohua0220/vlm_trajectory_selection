import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def plot_cost_clusters_with_full_highlights(directory_path, base_scenario_name, file_pattern='cost_comparison_USA_Peach-1_1_T-1_*.csv',
                                            n_clusters=4):
    """
    Performs K-Means clustering and visualizes the results, highlighting the
    best trajectory from each algorithm (with timestamp labels).
    """
    # --- Data Loading (same as before) ---
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
            timestamp = int(match.group(1));
            df = pd.read_csv(path, skipinitialspace=True)
            df.columns = df.columns.str.strip();
            df['Timestamp'] = timestamp;
            all_dfs.append(df)
    if not all_dfs:
        print("Error: No data extracted.")
        return
    all_points_df = pd.concat(all_dfs, ignore_index=True)

    # --- 1. Identify Best Trajectories (same as before) ---
    idx_best_gemini = pd.Index(all_points_df.groupby('Timestamp')['VLM Cost'].idxmin().values)
    idx_best_frenetix = pd.Index(all_points_df.groupby('Timestamp')['Frenetix Cost'].idxmin().values)

    idx_same_best = idx_best_gemini.intersection(idx_best_frenetix)
    idx_best_gemini_only = idx_best_gemini.difference(idx_same_best)
    idx_best_frenetix_only = idx_best_frenetix.difference(idx_same_best)

    best_gemini_only_points = all_points_df.loc[idx_best_gemini_only]
    best_frenetix_only_points = all_points_df.loc[idx_best_frenetix_only]
    same_best_points = all_points_df.loc[idx_same_best]

    # --- 2. Clustering (same as before) ---
    costs_df = all_points_df[['Frenetix Cost', 'VLM Cost']].dropna()
    scaler = StandardScaler();
    scaled_costs = scaler.fit_transform(costs_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    all_points_df['Cluster'] = kmeans.fit_predict(scaled_costs)

    # --- 3. Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 12))  # Increased figure size for better label visibility

    # Plot all data points first
    sns.scatterplot(
        data=all_points_df, x='Frenetix Cost', y='VLM Cost', hue='Cluster',
        palette='viridis', alpha=0.6, s=50, zorder=3
    )

    # --- 4. Add Highlighted Points and Labels ---
    # Highlight best for Frenetix ONLY
    plt.scatter(
        best_frenetix_only_points['Frenetix Cost'], best_frenetix_only_points['VLM Cost'],
        marker='*', s=200, facecolors='cyan', edgecolors='black',
        linewidth=1.5, label='Best Frenetix Only', zorder=3
    )
    for idx, row in best_frenetix_only_points.iterrows():
        plt.text(row['Frenetix Cost'], row['VLM Cost'], f"   {int(row['Timestamp'])}", fontsize=10, ha='left',
                 va='center',fontweight='bold',zorder=1)

    # Highlight best for Gemini ONLY
    plt.scatter(
        best_gemini_only_points['Frenetix Cost'], best_gemini_only_points['VLM Cost'],
        marker='*', s=200, facecolors='magenta', edgecolors='black',
        linewidth=1.5, label='Best Gemini Only', zorder=3
    )
    for idx, row in best_gemini_only_points.iterrows():
        plt.text(row['Frenetix Cost'], row['VLM Cost'], f"   {int(row['Timestamp'])}", fontsize=10, ha='left',
                 va='center',fontweight='bold',zorder=1)

    # Highlight points that are best for BOTH
    if not same_best_points.empty:
        plt.scatter(
            same_best_points['Frenetix Cost'], same_best_points['VLM Cost'],
            marker='P', s=250, facecolors='gold', edgecolors='black',  # 'P' is a filled plus sign
            linewidth=1.5, label='Best for Both', zorder=4
        )
        for idx, row in same_best_points.iterrows():
            plt.text(row['Frenetix Cost'], row['VLM Cost'], f"   {int(row['Timestamp'])}", fontsize=10, ha='left',
                     va='center',fontweight='bold',zorder=1)

    plt.title(f'Cost Relationship via K-Means Clustering (k={n_clusters})', fontsize=16, weight='bold')
    plt.xlabel('Frenetix Cost (Original Scale)', fontsize=12)
    plt.ylabel('Gemini Cost (Original Scale)', fontsize=12)
    plt.legend(title='Legend')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_dir = os.path.dirname(f"{directory_path}/cost_{base_scenario_name}_all.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{directory_path}/cost_{base_scenario_name}_all.png", dpi=300)
    plt.show()


# --- Run the function ---
if __name__ == '__main__':
    base_scenario_name = 'USA_US101-29_1_T-1'
    data_directory = f'cost_comparison/{base_scenario_name}'
    plot_cost_clusters_with_full_highlights(data_directory, base_scenario_name,file_pattern=f'cost_comparison_{base_scenario_name}_*.csv', n_clusters=4)