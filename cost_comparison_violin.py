import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re


def plot_distributions_as_violins(directory_path, file_pattern='cost_comparison_USA_Peach-1_1_T-1_*.csv'):
    """
    Reads all trajectory cost data and creates a violin plot to compare
    the distributions of the two algorithms across timestamps.
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
            except Exception as e:
                print(f"Warning: Could not process file {path}. Error: {e}")
                continue

    if not all_dfs:
        print("Error: No valid data could be extracted.")
        return

    all_points_df = pd.concat(all_dfs, ignore_index=True)

    # We need to "melt" the DataFrame from wide to long format for Seaborn
    # This creates two columns: one for the algorithm name, one for the cost value
    melted_df = all_points_df.melt(
        id_vars=['Timestamp'],
        value_vars=['Frenetix Cost', 'Gemini Cost'],
        var_name='Algorithm',
        value_name='Cost'
    )

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a split violin plot
    sns.violinplot(data=melted_df, x='Timestamp', y='Cost', hue='Algorithm',
                   split=True, inner='quart', palette={'Frenetix Cost': 'tab:blue', 'Gemini Cost': 'tab:red'}, ax=ax)

    ax.set_title('Cost Distribution Comparison per Timestamp', fontsize=16, weight='bold')
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.legend(title='Algorithm')

    plt.tight_layout()
    plt.savefig("cost_comparison/cost_violin_plot.png", dpi=300)
    plt.show()

# --- Run the function ---
if __name__ == '__main__':
    data_directory = 'cost_comparison'
    plot_distributions_as_violins(data_directory)