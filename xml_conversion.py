import os
import json
import sys
import xml.etree.ElementTree as ET
import pandas as pd  # Import pandas for CSV handling


# The XML parsing function remains unchanged as it correctly parses the base structure.
def convert_xml_to_scenario_dict(xml_string: str, current_timestep: float) -> dict:
    """
    Parses an XML string containing scenario data into a Python dictionary,
    filtering trajectory data up to the specified current timestep.

    Args:
        xml_string: A string containing the XML data for the scenario.
        current_timestep: The current time in seconds. Only obstacle states with
                          a timestamp less than or equal to this value will be included.

    Returns:
        A dictionary with the parsed and filtered scenario data.
    """
    root = ET.fromstring(xml_string)

    scenario_dict = {
        "ego_vehicle": {},
        "dynamic_obstacles": [],
        "goal_region": {}
    }

    # --- Parse Planning Problem (Ego Vehicle and Goal) ---
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

            scenario_dict["ego_vehicle"]["initial_state"] = {
                "position_xy": [round(float(pos_x), 3), round(float(pos_y), 3)],
                "velocity_mps": round(float(vel), 3) if vel is not None else 0.0,
                "acceleration_mps2": round(float(acceleration), 3) if acceleration is not None else 0.0,
                "heading_radian": round(float(orient), 3) if orient is not None else 0.0,
            }
        # Initialize historical_waypoints as an empty list. It will be populated later.
        scenario_dict["ego_vehicle"]["historical_waypoints"] = []

        goal_state = planning_problem.find('goalState')
        if goal_state is not None:
            center_x = goal_state.findtext('position/rectangle/center/x')
            center_y = goal_state.findtext('position/rectangle/center/y')
            length = goal_state.findtext('position/rectangle/length')
            width = goal_state.findtext('position/rectangle/width')
            if center_x is not None and center_y is not None and length is not None and width is not None:
                scenario_dict["goal_region"] = {
                    "center_xy": [round(float(center_x), 3), round(float(center_y), 3)],
                    "length_width": [round(float(length), 2), round(float(width), 2)]
                }
            else:
                scenario_dict["goal_region"] = {
                    "center_xy": [float(72), float(5)]}

    # --- Parse Dynamic Obstacles ---
    for obstacle_node in root.findall('dynamicObstacle'):
        obs_id = obstacle_node.get('id')
        obs_type = obstacle_node.findtext('type')

        obstacle_dict = {
            "id": int(obs_id) if obs_id else None,
            "class": "vehicle" if obs_type == 'car' else obs_type,
            "trajectory_history": []
        }

        obs_initial_state = obstacle_node.find('initialState')
        if obs_initial_state is not None:
            time = obs_initial_state.findtext('time/exact')
            pos_x = obs_initial_state.findtext('position/point/x')
            pos_y = obs_initial_state.findtext('position/point/y')
            vel = obs_initial_state.findtext('velocity/exact')

            initial_state_dict = {
                "timestamp_s": float(time),
                "position_xy": [float(pos_x), float(pos_y)],
                "velocity_mps": float(vel)
            }
            obstacle_dict["trajectory_history"].append(initial_state_dict)
        trajectory_node = obstacle_node.find('trajectory')
        if trajectory_node is not None:
            for state_node in trajectory_node.findall('state'):
                time_str = state_node.findtext('time/exact')
                if float(time_str) <= current_timestep:
                    pos_x = state_node.findtext('position/point/x')
                    pos_y = state_node.findtext('position/point/y')
                    vel = state_node.findtext('velocity/exact')

                    state_dict = {
                        "timestamp_s": float(time_str),
                        "position_xy": [float(pos_x), float(pos_y)],
                        "velocity_mps": float(vel)
                    }
                    obstacle_dict["trajectory_history"].append(state_dict)

        scenario_dict["dynamic_obstacles"].append(obstacle_dict)

    return scenario_dict


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    input_xml_filename = "cr_scenarios/USA_Peach-1_1_T-1.xml"
    input_csv_filename = "cr_scenarios/logs.csv"  # Path to your CSV file
    target_timestep = 5  # The 'i' you want to read up to (inclusive)

    # The output filename will now include the target timestep 'i'
    output_json_full_path = input_xml_filename.rsplit('.', 1)[0] + f"_t_{target_timestep}.json"

    # --- Read the XML file ---
    try:
        print(f"--- Reading scenario from '{input_xml_filename}' ---")
        with open(input_xml_filename, 'r', encoding='utf-8') as f:
            xml_data_string = f.read()
    except FileNotFoundError:
        print(f"Error: The XML file '{input_xml_filename}' was not found.")
        sys.exit(1)

    # --- Process the XML data first ---
    # The second argument to this function filters dynamic obstacles, we set it to the target timestep as well
    print(
        f"--- Parsing XML and filtering obstacles up to timestep: {target_timestep * 0.04}s ---")  # Assuming 0.04s per step
    scenario_to_evaluate = convert_xml_to_scenario_dict(xml_data_string, target_timestep * 0.04)

    # --- NEW: Read and process the CSV for historical waypoints ---
    try:
        print(f"--- Reading historical waypoints from '{input_csv_filename}' ---")
        df = pd.read_csv(input_csv_filename, sep=';')

        # Filter rows from trajectory_number 0 up to target_timestep
        history_df = df[df['trajectory_number'] <= target_timestep].copy()

        # Select and format the x and y columns into a list of lists
        waypoints = history_df[['x_position_vehicle_m', 'y_position_vehicle_m']].values.tolist()

        # Round the values to 3 decimal places for consistency
        rounded_waypoints = [[round(x, 3), round(y, 3)] for x, y in waypoints]

        # Add the processed waypoints to the dictionary
        scenario_to_evaluate["ego_vehicle"]["historical_waypoints"] = rounded_waypoints
        print(f"--- Added {len(rounded_waypoints)} historical waypoints to the ego vehicle. ---")

    except FileNotFoundError:
        print(f"Warning: The CSV file '{input_csv_filename}' was not found. Historical waypoints will be empty.")
    except KeyError as e:
        print(f"Error: The CSV file is missing a required column: {e}. Historical waypoints will be empty.")

    # --- Save the resulting dictionary to a JSON file ---
    print(f"--- Saving final output to '{output_json_full_path}' ---")
    with open(output_json_full_path, 'w', encoding='utf-8') as json_file:
        json.dump(scenario_to_evaluate, json_file, indent=4)

    print(f"✅ Process Complete. Output saved to '{output_json_full_path}'.")