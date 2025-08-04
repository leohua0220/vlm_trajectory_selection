import os
import json
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import ast  # To safely evaluate string-formatted lists from the CSV
import numpy as np  # Make sure to import numpy


# --- NEW: Custom JSON encoder to handle NumPy types ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def parse_string_list(s):
    """Safely parse a string that looks like a list, e.g., '[1.23]'."""
    try:
        # ast.literal_eval is a safe way to evaluate a string containing a Python literal
        parsed_list = ast.literal_eval(s)
        if isinstance(parsed_list, list) and parsed_list:
            return parsed_list[0]  # Return the first element as requested
        return 0.0  # Default value if list is empty or not a list
    except (ValueError, SyntaxError):
        # Return a default value if the string is not a valid literal
        return 0.0


# The XML parsing function is slightly modified to use the new 'current_state' key
def convert_xml_to_scenario_dict(xml_string: str, current_timestep, dt) -> dict:
    """
    Parses an XML string containing scenario data into a Python dictionary.
    The ego vehicle state is considered the state at t=0.
    """
    root = ET.fromstring(xml_string)
    scenario_dict = {
        "ego_vehicle": {},
        "dynamic_obstacles": [],
        "goal_region": {}
    }

    planning_problem = root.find('planningProblem')
    if planning_problem is not None:
        ego_id = planning_problem.get('id')
        scenario_dict["ego_vehicle"]["id"] = int(ego_id) if ego_id else None

        initial_state = planning_problem.find('initialState')
        if initial_state is not None:
            pos_x = initial_state.findtext('position/point/x')
            pos_y = initial_state.findtext('position/point/y')
            vel = initial_state.findtext('velocity/exact')
            orient = initial_state.findtext('orientation/exact')
            acceleration = initial_state.findtext('acceleration/exact')

            # This is now the t=0 'current_state'
            scenario_dict["ego_vehicle"]["current_state"] = {
                "timestamp_s": 0.0,
                "position_xy": [round(float(pos_x), 3), round(float(pos_y), 3)],
                "velocity_mps": round(float(vel), 3) if vel is not None else 0.0,
                "acceleration_mps2": round(float(acceleration), 3) if acceleration is not None else 0.0,
                "heading_radian": round(float(orient), 3) if orient is not None else 0.0,
            }
        # Initialize historical_states as an empty list. It will be populated from the CSV.
        scenario_dict["ego_vehicle"]["historical_trajectory"] = []

        goal_state = planning_problem.find('goalState')
        if goal_state is not None:
            # ... (goal region parsing remains the same)
            center_x = goal_state.findtext('position/rectangle/center/x')
            center_y = goal_state.findtext('position/rectangle/center/y')
            if center_x and center_y:
                scenario_dict["goal_region"] = {"center_xy": [round(float(center_x), 3), round(float(center_y), 3)]}
            else:
                scenario_dict["goal_region"] = {"center_xy": [0.0, 0.0]}

    # Dynamic obstacle parsing remains the same
    for obstacle_node in root.findall('dynamicObstacle'):
        obs_id = obstacle_node.get('id')
        obs_type = obstacle_node.findtext('type')

        obstacle_dict = {
            "id": int(obs_id) if obs_id else None,
            "class": "vehicle" if obs_type == 'car' else obs_type,
            "current_state": {},
            "historical_trajectory": []
        }

        obs_initial_state = obstacle_node.find('initialState')
        if obs_initial_state is not None and not max((current_timestep-5),0):
            time = obs_initial_state.findtext('time/exact')
            pos_x = obs_initial_state.findtext('position/point/x')
            pos_y = obs_initial_state.findtext('position/point/y')
            vel = obs_initial_state.findtext('velocity/exact')
            acceleration = obs_initial_state.findtext('acceleration/exact')
            orient = obs_initial_state.findtext('orientation/exact')

            initial_state_dict = {
                "timestamp_s": float(time),
                "position_xy": [round(float(pos_x), 3), round(float(pos_y), 3)],
                "velocity_mps": round(float(vel), 3),
                "acceleration_mps2": round(float(acceleration), 3) if acceleration is not None else 0.0,
                "heading_radian": round(float(orient), 3) if orient is not None else 0.0
            }
            if current_timestep== 0:
                # If current_timestep is 0, this is the initial state for the obstacle
                obstacle_dict["current_state"] = initial_state_dict
            else:
                # If current_timestep > 0, this is part of the historical trajectory
                obstacle_dict["historical_trajectory"].append(initial_state_dict)

        trajectory_node = obstacle_node.find('trajectory')
        if trajectory_node is not None:
            for state_node in trajectory_node.findall('state'):
                time_str = state_node.findtext('time/exact')
                if max((current_timestep-5),0) <= float(time_str) < current_timestep:
                    pos_x = state_node.findtext('position/point/x')
                    pos_y = state_node.findtext('position/point/y')
                    vel = state_node.findtext('velocity/exact')
                    acceleration = state_node.findtext('acceleration/exact')
                    orient = state_node.findtext('orientation/exact')

                    state_dict = {
                        "timestamp_s": round(float(time_str)*dt,2),
                        "position_xy": [round(float(pos_x), 3), round(float(pos_y), 3)],
                        "velocity_mps": round(float(vel), 3),
                        "acceleration_mps2": round(float(acceleration), 3) if acceleration is not None else 0.0,
                        "heading_radian": round(float(orient), 3) if orient is not None else 0.0
                    }
                    obstacle_dict["historical_trajectory"].append(state_dict)
                elif float(time_str) == current_timestep:
                    # This is the current state for the obstacle at the target timestep
                    pos_x = state_node.findtext('position/point/x')
                    pos_y = state_node.findtext('position/point/y')
                    vel = state_node.findtext('velocity/exact')
                    acceleration = state_node.findtext('acceleration/exact')
                    orient = state_node.findtext('orientation/exact')

                    obstacle_dict["current_state"] = {
                        "timestamp_s": round(float(time_str)*dt,2),
                        "position_xy": [round(float(pos_x), 3), round(float(pos_y), 3)],
                        "velocity_mps": round(float(vel), 3),
                        "acceleration_mps2": round(float(acceleration), 3) if acceleration is not None else 0.0,
                        "heading_radian": round(float(orient), 3) if orient is not None else 0.0
                    }

        scenario_dict["dynamic_obstacles"].append(obstacle_dict)

    return scenario_dict


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    base_scenario_name = os.path.splitext(os.path.basename("USA_Peach-1_1_T-1.xml"))[0]
    input_xml_filename = f"cr_scenarios/{base_scenario_name}/{base_scenario_name}.xml"
    input_csv_filename = f"cr_scenarios/{base_scenario_name}/logs.csv"
    start_timestep = 0
    end_timestep = 30
    dt = 0.1  # Time step duration

    output_json_full_path = input_xml_filename.rsplit('.', 1)[0] + f"_{target_timestep}.json"

    # --- Read and Parse XML for the base structure and t=0 state ---
    try:
        print(f"--- Reading scenario from '{input_xml_filename}' ---")
        with open(input_xml_filename, 'r', encoding='utf-8') as f:
            xml_data_string = f.read()
    except FileNotFoundError:
        print(f"Error: The XML file '{input_xml_filename}' was not found.")
        sys.exit(1)

    scenario_to_evaluate = convert_xml_to_scenario_dict(xml_data_string, target_timestep, dt)
    print("--- Parsed XML for t=0 base state. ---")

    # --- Read CSV to update current state and populate history ---
    try:
        print(f"--- Reading vehicle log from '{input_csv_filename}' ---")
        df = pd.read_csv(input_csv_filename, sep=';')

        # --- Populate Historical States (all steps BEFORE target_timestep) ---
        historical_states = []
        start_timestep = max(target_timestep - 5, 0)
        history_df = df[
            (df['trajectory_number'] >= start_timestep) & (df['trajectory_number'] < target_timestep)].copy()

        for index, row in history_df.iterrows():
            state = {
                "timestamp_s": round(row['trajectory_number']*dt, 2),
                "position_xy": [round(row['x_position_vehicle_m'], 3), round(row['y_position_vehicle_m'], 3)],
                "velocity_mps": round(float(row['velocities_mps'].split(',')[0]), 3),
                "acceleration_mps2": round(float(row['accelerations_mps2'].split(',')[0]), 3),
                "heading_radian": round(float(row['theta_orientations_rad'].split(',')[0]), 3)
            }
            historical_states.append(state)

        scenario_to_evaluate["ego_vehicle"]["historical_trajectory"] = historical_states
        print(f"--- Recorded {len(historical_states)} historical states (timesteps 0 to {target_timestep - 1}). ---")

        # --- Update Current State if target_timestep > 0 ---
        if target_timestep > 0:
            current_state_row = df[df['trajectory_number'] == target_timestep]

            if not current_state_row.empty:
                row = current_state_row.iloc[0]
                current_state_from_csv = {
                    "timestamp_s": round(row['trajectory_number']*dt, 2),
                    "position_xy": [round(row['x_position_vehicle_m'], 3), round(row['y_position_vehicle_m'], 3)],
                    "velocity_mps": round(float(row['velocities_mps'].split(',')[0]), 3),
                    "acceleration_mps2": round(float(row['accelerations_mps2'].split(',')[0]), 3),
                    "heading_radian": round(float(row['theta_orientations_rad'].split(',')[0]), 3)
                }
                scenario_to_evaluate["ego_vehicle"]["current_state"] = current_state_from_csv
                print(f"--- Set current state from CSV for timestep {target_timestep}. ---")
            else:
                print(f"Warning: Timestep {target_timestep} not found in CSV. Using t=0 state from XML.")
        else:
            print("--- Using t=0 state from XML as current state. ---")

    except FileNotFoundError:
        print(f"Warning: The CSV file '{input_csv_filename}' was not found. State was not updated.")
    except KeyError as e:
        print(f"Error: The CSV file is missing a required column: {e}. State was not updated.")
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")

    # --- Save the resulting dictionary to a JSON file ---
    print(f"--- Saving final output to '{output_json_full_path}' ---")
    with open(output_json_full_path, 'w', encoding='utf-8') as json_file:
        json.dump(scenario_to_evaluate, json_file, indent=4, cls=NumpyEncoder)

    print(f"✅ Process Complete. Output saved to '{output_json_full_path}'.")
