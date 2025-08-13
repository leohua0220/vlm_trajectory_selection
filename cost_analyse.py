import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


def analyze_timestep(base_scenario_name: str, timestep: int, prompt_type: str):
    """
    Analyzes and visualizes cost data for a single scenario timestep.

    This function loads cost data and VLM evaluation data, merges them,
    calculates ranks, and saves the output as a CSV and a PNG plot.
    """
    print(f"\n{'=' * 20} Processing Timestep: {timestep} {'=' * 20}")

    # --- 1. Define file paths for the current timestep ---
    file_name_cost = f'frenetix_cost/{base_scenario_name}/cost_{base_scenario_name}_{timestep}.json'
    file_name_responses = f'logs/responses/{base_scenario_name}/{prompt_type}_{base_scenario_name}_{timestep}.json'

    # Define output paths
    output_dir = f'cost_comparison/{base_scenario_name}'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    output_csv_name = f'{output_dir}/{prompt_type}_{base_scenario_name}_{timestep}.csv'
    output_png_name = f'{output_dir}/{prompt_type}_{base_scenario_name}_{timestep}.png'

    # --- 2. Load the JSON files ---
    try:
        with open(file_name_cost, 'r') as f:
            cost_data = json.load(f)
        with open(file_name_responses, 'r') as f:
            evaluation_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Missing a data file for timestep {timestep}. Skipping.")
        print(f"-> Details: {e}")
        return  # Exit the function for this timestep

    # --- 3. Extract and merge the data ---
    merged_data = []
    if prompt_type == 'dspy':
        for traj_key, evaluation in evaluation_data.items():
            if not traj_key.startswith("trajectory_id:"):
                continue

            if traj_key in cost_data:
                try:
                    traj_id = int(traj_key.split(':')[1])
                    algo_cost = cost_data[traj_key]['cost']
                    vlm_cost = evaluation['cost']
                    vlm_cost_breakdown = evaluation['costMap']

                    merged_data.append({
                        'Trajectory ID': traj_id,
                        'Frenetix Cost': algo_cost,
                        'VLM Cost': vlm_cost,
                        'Lateral Jerk': vlm_cost_breakdown.get('lateral_jerk', [0, 0])[0],
                        'Longitudinal Jerk': vlm_cost_breakdown.get('longitudinal_jerk', [0, 0])[0],
                        'Velocity Offset': vlm_cost_breakdown.get('velocity_offset', [0, 0])[0],
                    })
                except (ValueError, IndexError, KeyError) as e:
                    print(f"Skipping key '{traj_key}' in timestep {timestep} due to parsing error: {e}")
    if prompt_type == 'wo_dspy':
        for evaluation in evaluation_data['evaluation']:
            traj_id = evaluation['trajectory_id']
            # Construct the key to look up the cost in the other file.
            traj_key = f"trajectory_id:{traj_id}"

            # Check if the trajectory exists in the cost data file.
            if traj_key in cost_data:
                # Get the Frenetix Cost.
                algo_cost = cost_data[traj_key]['cost']
                # Append all relevant information into a dictionary.
                merged_data.append({
                    'Trajectory ID': traj_id,
                    'Frenetix Cost': algo_cost,
                    'VLM Cost': evaluation['total_cost'],
                    'Safety Score': evaluation['cost_breakdown']['safety'],
                    'Comfort Score': evaluation['cost_breakdown']['comfort'],
                    'Efficiency Score': evaluation['cost_breakdown']['efficiency'],
                    # 'Reasons': evaluation['cost_breakdown']['reasons']
                })

    # --- 4. Create DataFrame and calculate ranks ---
    if not merged_data:
        print(f"No valid trajectory data found to merge for timestep {timestep}. Skipping.")
        return

    df = pd.DataFrame(merged_data)
    df['Frenetix Cost Rank'] = df['Frenetix Cost'].rank().astype(int)
    df['VLM Cost Rank'] = df['VLM Cost'].rank(method='min').astype(int)
    df_sorted = df.sort_values('VLM Cost Rank')

    # --- 5. Save the merged data to a CSV file ---
    df_sorted.to_csv(output_csv_name, index=False)
    print(f"Merged data saved to: {output_csv_name}")

    # --- 6. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate colors for the plot points
    num_points = len(df)
    if num_points > 0:
        if num_points <= 20:
            colors = sns.color_palette("tab10", 10) + sns.color_palette("Paired", 10)
            colors = colors[:num_points]
        else:
            colors = [(random.random(), random.random(), random.random()) for _ in range(num_points)]

        # Create the scatter plot
        for i, row in df.iterrows():
            ax.scatter(
                row['Frenetix Cost'], row['VLM Cost'], color=colors[i % len(colors)],
                s=200, alpha=0.8, edgecolors='black'
            )
            ax.text(row['Frenetix Cost'] * 1.01, row['VLM Cost'], f"ID: {row['Trajectory ID']}", fontsize=9)

    # Set titles and labels for clarity
    ax.set_title(f'Frenetix vs. Gemini-1.5-pro (Cost)\nScenario: {base_scenario_name} | Timestep: {timestep}',
                 fontsize=16, weight='bold')
    ax.set_xlabel('Frenetix Cost (Lower is better)', fontsize=12)
    ax.set_ylabel('VLM Cost (Lower is better)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the figure to a file
    plt.tight_layout()
    plt.savefig(output_png_name, dpi=300)
    plt.close(fig)  # Close the figure to free up memory
    print(f"Comparison plot saved to: {output_png_name}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the scenario and the list of timesteps to process
    base_scenario_name = 'USA_US101-29_1_T-1'
    timesteps_to_process = [i for i in range(0,115,5)]  # Define your list of timesteps here
    prompt_type = 'wo_dspy'  # Define the prompt type, e.g., 'wo_dspy', 'dspy'
    # Loop through each defined timestep and run the analysis
    for ts in timesteps_to_process:
        analyze_timestep(base_scenario_name, ts, prompt_type)

    print(f"\n{'=' * 20} Batch analysis complete. {'=' * 20}")