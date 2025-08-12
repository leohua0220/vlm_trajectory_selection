import os
import json
from datetime import datetime
import google.generativeai as genai
import PIL.Image


def evaluate_scenario_from_single_prompt(scenario_data: dict, image_path: str) -> dict:
    """
    Builds a multi-modal prompt (text + image) and calls the Gemini API to evaluate a motion planning scenario.

    This function consolidates the system role, instructions, dynamic data, and a visual representation
    into one request, making it a self-contained multi-modal call.

    Args:
        scenario_data: A Python dictionary containing the specific details of the scene.
        image_path: The file path to the image representing the scenario.

    Returns:
        A dictionary parsed from the API's JSON response, or None if an error occurs.
    """

    # --- Part 1: Define the System and User Prompt Structures ---

    system_message = (
        "You are an expert motion planning evaluator for autonomous vehicles. Your primary function is to analyze "
        "motion planning scenarios and select the optimal trajectory for the ego vehicle from a set of candidates."
        "\n\n"
        "You will evaluate each trajectory against three core criteria: Safety, Comfort, and Efficiency, "
        "using a predefined cost scale and weighting scheme."
        "\n\n"
        "Your final output must be a single, raw JSON object. Do not include any explanatory text, markdown formatting, "
        "or other content outside of the JSON structure. The structure must strictly adhere to the format "
        "specified in the prompt."
    )

    data_json_string = json.dumps(scenario_data, indent=2)

    user_prompt_template = f"""
Context:
The ego vehicle is blue. The scene includes oncoming traffic. You must evaluate the candidate trajectories, which are visually represented in an accompanying image, and select the optimal one based on the provided data and criteria.

Input Data:
```json
{data_json_string}
```

Evaluation Task:
1. Analyze each candidate trajectory based on the criteria below.
2. Assign a cost to each trajectory for each criterion using the specific cost scales provided.
3. Calculate a `total_cost` for each trajectory.
4. Identify the `best_trajectory_id` with the lowest `total_cost`.

---

### **Evaluation Criteria, Weights, and Scales**

#### **1. Safety (Weight: 0.5)**
This cost is primarily determined by **path adherence** and secondarily by **interaction safety**.
* **Core Components:**
  * **Path Adherence (`distance_to_reference_path`, Weight: 5.0):** How closely does the trajectory follow the reference path?
  * **Interaction Safety (`prediction`, Weight: 0.2):** How well does the trajectory maintain a safe margin from the predicted paths of other agents?
* **Safety Cost Scale:**
  * `1 (Excellent)`: Trajectory is almost perfectly centered on the reference path. Maintains a very large, safe distance from all predicted obstacle paths.
  * `2 (Good)`: Minor, smooth deviation from the reference path. Maintains a clearly safe and appropriate distance from obstacles.
  * `3 (Acceptable)`: Noticeable deviation from the reference path but stays well within lane boundaries. Distance to obstacles is reduced but still clearly safe.
  * `4 (Poor)`: Significant deviation, approaching lane lines. Trajectory brings the ego vehicle unnecessarily close to predicted obstacle paths, reducing safety margins.
  * `5 (Unacceptable)`: Crosses lane lines into opposing traffic or off-road. Creates an immediate collision risk with an obstacle (zero or negative margin).

#### **2. Comfort (Weight: 0.25)**
This cost is determined by the smoothness of the trajectory's motion profile.
* **Core Components:**
  * **Longitudinal Comfort (`longitudinal_jerk`, Weight: 0.2):** Penalizes abrupt acceleration or braking.
  * **Lateral Comfort (`lateral_jerk`, Weight: 0.2):** Penalizes sharp, sudden lateral movements.
* **Comfort Cost Scale:**
  * `1 (Excellent)`: Jerk values are minimal. The trajectory feels exceptionally smooth, with no perceptible abruptness in acceleration, braking, or steering.
  * `2 (Good)`: Low jerk values. Any changes in acceleration or lateral movement are very gradual and comfortable.
  * `3 (Acceptable)`: Moderate jerk. Changes in speed or direction are noticeable but not uncomfortable for an average passenger.
  * `4 (Poor)`: High jerk values. Acceleration, braking, or steering is abrupt and would likely be uncomfortable for passengers.
  * `5 (Unacceptable)`: Extreme or oscillating jerk values. The motion is violent, erratic, and would be considered highly unsafe or nauseating.

#### **3. Efficiency (Weight: 0.25)**
This cost is determined by speed regulation.
* **Core Components:**
  * **Speed Regulation (`velocity_offset`, Weight: 1.0):** How closely does the trajectory's speed match the target speed profile?
* **Efficiency Cost Scale:**
  * `1 (Excellent)`: Velocity is consistently at or very near the target speed. Progress is optimal.
  * `2 (Good)`: Minor, temporary deviations from the target speed, but quickly corrects. Good progress is maintained.
  * `3 (Acceptable)`: Noticeable periods of being slower or faster than the target speed, but for justifiable reasons (e.g., following a slower car). Overall progress is reasonable.
  * `4 (Poor)`: Consistently and significantly slower than the target speed without a clear reason, impeding traffic flow. Or, inappropriately faster than the target speed.
  * `5 (Unacceptable)`: Velocity is dangerously slow, creating a hazard, or excessively exceeds the speed limit/target speed. Fails to make meaningful progress.

---

### **Calculation & Output**

* **Total Cost Formula:**
  `total_cost = (0.5 * safety_cost) + (0.25 * comfort_cost) + (0.25 * efficiency_cost)`

* **Output Format:**
  Your response must be a single, raw JSON object with the following structure. Do not include any text outside the JSON.
  ```json
  {{
    "best_trajectory_id": "<integer>",
    "evaluation": [
      {{
        "trajectory_id": "<integer>",
        "cost_breakdown": {{
          "safety": "<float>",
          "comfort": "<float>",
          "efficiency": "<float>",
          "reasons": "<string>"
        }},
        "total_cost": "<float>"
      }}
    ]
  }}
  ```
"""

    full_prompt = f"{system_message}\n\n---\n\n{user_prompt_template}"

    # --- Part 2: Configure and Call the API ---
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your API key.")

    genai.configure(api_key=api_key)

    # Use a model that supports multi-modal (text + image) inputs.
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )

    try:
        print(f"Opening image from: {image_path}")
        img = PIL.Image.open(image_path)

        # The content for a multi-modal request is a list containing the prompt and the image.
        request_content = [full_prompt, img]

        print("Sending multi-modal request to Gemini API...")
        response = model.generate_content(
            request_content,
            generation_config=generation_config
        )
        print("Response received.")
        return json.loads(response.text)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        if 'response' in locals() and hasattr(response, 'parts'):
            print("--- Raw Response for Debugging ---")
            print(response.parts)
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the base scenario name
    base_scenario_name = "USA_Peach-1_1_T-1"

    # Define the list of timesteps to process
    timesteps = [0, 5, 10, 15, 20, 25, 30]

    print(f"--- Starting Batch Evaluation for Scenario: {base_scenario_name} ---")

    # Loop through each timestep in the list
    for timestep in timesteps:
        print(f"\n{'=' * 25} Processing Timestep: {timestep} {'=' * 25}")

        # Dynamically define file paths for the current timestep
        scenario_json_path = f'cr_scenarios/{base_scenario_name}/{base_scenario_name}_{timestep}.json'
        trajectory_json_path = f'log_trajectory/{base_scenario_name}/trajectory_{base_scenario_name}_{timestep}.json'
        scenario_image_path = f"plots/{base_scenario_name}/{base_scenario_name}_{timestep}.png"

        # Load data for the current timestep
        try:
            with open(scenario_json_path, 'r', encoding='utf-8') as f:
                scenario_data = json.load(f)
            with open(trajectory_json_path, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: A required file for timestep {timestep} was not found. Skipping.")
            print(f"Missing file: {e.filename}")
            continue  # Skip to the next timestep if a file is missing
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON for timestep {timestep}. Skipping. Details: {e}")
            continue  # Skip to the next timestep

        # Combine the two JSON contents into a single dictionary
        combined_input = {
            'scenario_context': scenario_data,
            'trajectory_details': trajectory_data
        }

        print(f"--- Evaluating Motion Planning Scenario for Timestep {timestep} ---")
        # Call the evaluation function for the current timestep
        evaluation_result = evaluate_scenario_from_single_prompt(combined_input, scenario_image_path)

        if evaluation_result:
            print(f"\n--- Evaluation Result for Timestep {timestep} ---")
            print(json.dumps(evaluation_result, indent=2))

            # --- Save the result to a log file ---
            # Create a subdirectory for the specific scenario to keep results organized
            output_dir = os.path.join("logs", "responses", base_scenario_name)
            os.makedirs(output_dir, exist_ok=True)

            # Create a unique filename for the current timestep's result
            output_filename = f"response_wo_dspy_{base_scenario_name}_{timestep}.json"
            output_filepath = os.path.join(output_dir, output_filename)

            print(f"\n--- Saving response to {output_filepath} ---")
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(evaluation_result, f_out, indent=4)
            print("Save complete.")
        else:
            print(f"--- No result returned from API for Timestep {timestep} ---")

    print(f"\n{'=' * 25} Batch Evaluation Complete {'=' * 25}")
