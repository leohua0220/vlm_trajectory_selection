import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random # Import random for generating random colors

# --- 1. Load the JSON files ---
# Load the file containing the Frenetix Costs for each trajectory.
with open('frenetix_cost/cost_USA_Peach-1_1_T-1_0.json', 'r') as f:
    cost_data = json.load(f)

# Load the file containing the human-like evaluation scores.
with open('logs/responses/USA_Peach-1_1_T-1_t_0_20250728_171428.json', 'r') as f:
    evaluation_data = json.load(f)

# --- 2. Extract and merge the data from both files ---
merged_data = []
# Iterate through each trajectory in the evaluation file.
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
            'Gemini Cost': evaluation['total_cost'],
            'Safety Score': evaluation['cost_breakdown']['safety'],
            'Comfort Score': evaluation['cost_breakdown']['comfort'],
            'Efficiency Score': evaluation['cost_breakdown']['efficiency'],
            'Reasons': evaluation['cost_breakdown']['reasons']
        })

# --- 3. Create a Pandas DataFrame ---
# Convert the list of dictionaries into a DataFrame for easier analysis.
df = pd.DataFrame(merged_data)

# --- 4. Calculate Ranks ---
# Rank trajectories based on their Frenetix Cost (lower is better).
df['Frenetix Cost Rank'] = df['Frenetix Cost'].rank().astype(int)
# Rank trajectories based on their Gemini Cost (lower is better).
# 'min' method ensures that trajectories with the same cost get the same rank.
df['Gemini Cost Rank'] = df['Gemini Cost'].rank(method='min').astype(int)

# Sort the DataFrame by evaluation rank for better readability.
df_sorted = df.sort_values('Gemini Cost Rank')

# --- 5. Save the merged data to a CSV file ---
# This allows you to inspect the combined data easily in a spreadsheet viewer.
df_sorted.to_csv('cost_comparison/trajectory_comparison.csv', index=False)

# --- 6. Visualization ---
# Set the visual style for the plot.
plt.style.use('seaborn-v0_8-whitegrid')
# Create a figure and axes for the plot.
fig, ax = plt.subplots(figsize=(10, 7))

# Generate a list of unique random colors for each point
# We'll use a qualitative palette from seaborn for better distinctiveness if the number of points is small
# Or generate entirely random RGB colors if many points are expected and distinctiveness isn't paramount
num_points = len(df)
# Using a seaborn palette with enough distinct colors
if num_points <= 10: # A smaller number of points, we can use a qualitative palette
    colors = sns.color_palette("tab10", num_points)
elif num_points <= 20: # For a slightly larger set, another qualitative palette
    colors = sns.color_palette("Paired", num_points)
else: # For a larger number of points, generate truly random colors
    colors = [(random.random(), random.random(), random.random()) for _ in range(num_points)]

# Create the scatter plot using seaborn.
# Iterate through each row and plot individually to assign a unique color
for i, row in df.iterrows():
    ax.scatter(
        row['Frenetix Cost'],
        row['Gemini Cost'],
        color=colors[i], # Assign a unique color to each point
        s=200, # Set the size of the points.
        alpha=0.8 # Add some transparency
    )
    # Add text labels next to each point for easy identification.
    ax.text(row['Frenetix Cost'] + 15, row['Gemini Cost'], f"ID: {row['Trajectory ID']}", fontsize=9)


# Set titles and labels for clarity.
ax.set_title('Frenetix Cost vs. VLM Cost', fontsize=16, weight='bold')
ax.set_xlabel('Frenetix Cost (from Frenetix)', fontsize=12)
ax.set_ylabel('VLM Cost (from gemini)', fontsize=12)
# No legend needed as colors are random and associated with text labels

# Adjust layout and save the figure to a file.
plt.tight_layout()
plt.savefig('cost_comparison/cost_comparison_scatter.png')

print("Analysis complete.")