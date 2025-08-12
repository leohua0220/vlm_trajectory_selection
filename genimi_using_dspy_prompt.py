import os
import json
import google.generativeai as genai
import PIL.Image

def evaluate_scenario_from_single_prompt(prompt_data: dict, scenario_data: dict, image_path: str) -> dict:
    """
    Builds a multi-modal prompt (text + image) and calls the Gemini API to evaluate a motion planning scenario.

    NOTE: The actual API call is commented out as requested. This function currently returns a mock response.
    To enable the real API call, comment out the "MOCK RESPONSE" section and uncomment the "REAL API CALL" section.
    """
    # --- Part 1: Define the Prompt Structure (No changes here) ---
    data_json_string = json.dumps(scenario_data, indent=2)
    prompt_json_string = json.dumps(prompt_data, indent=2)
    full_prompt = f"""
    ```json
    {prompt_json_string}
    ```
    Input Data:
    ```json
    {data_json_string}
    ```
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )

    try:
        print(f"Opening image from: {image_path}")
        img = PIL.Image.open(image_path)
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


def process_single_timestep(base_scenario_name: str, time_step: int, prompt_data: dict):
    """
    Loads data for a single timestep, evaluates it, and saves the result.
    """
    print(f"\n{'=' * 20} Processing Time Step: {time_step} {'=' * 20}")

    # Define paths based on the current time_step
    scenario_json_path = f'cr_scenarios/{base_scenario_name}/{base_scenario_name}_{time_step}.json'
    trajectory_json_path = f'log_trajectory/{base_scenario_name}/trajectory_{base_scenario_name}_{time_step}.json'
    scenario_image_path = f"plots/{base_scenario_name}/{base_scenario_name}_{time_step}.png"

    try:
        # Load scenario and trajectory data for the current timestep
        with open(scenario_json_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        with open(trajectory_json_path, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file for timestep {time_step}: {e}")
        return  # Skip this timestep if a file is missing
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON for timestep {time_step}: {e}")
        return

    # Combine the inputs
    combined_input = {
        'scenario_context': scenario_data,
        'trajectory_details': trajectory_data
    }

    # Call the evaluation function
    evaluation_result = evaluate_scenario_from_single_prompt(prompt_data, combined_input, scenario_image_path)

    if evaluation_result:
        print(f"\n--- Evaluation Result for Time Step {time_step} ---")
        print(json.dumps(evaluation_result, indent=2))

        # --- Save the result to a log file ---
        # Create a subdirectory for the specific scenario if it doesn't exist
        output_dir = os.path.join("logs", "responses", base_scenario_name)
        os.makedirs(output_dir, exist_ok=True)

        # Create a filename that includes the timestep
        output_filename = f"dspy_{base_scenario_name}_{time_step}.json"
        output_filepath = os.path.join(output_dir, output_filename)

        print(f"\n--- Saving response for timestep {time_step} to {output_filepath} ---")
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(evaluation_result, f_out, indent=4)
        print("Save complete.")
    else:
        print(f"--- No result generated for Time Step {time_step} ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the base scenario and the list of timesteps to process
    base_scenario_name = "USA_Peach-1_1_T-1"
    time_steps = [15]  # Define your list of timesteps here

    prompt_data_path = 'best_motion_evaluator_prompt.json'

    # Load the main prompt data once, as it's the same for all timesteps
    try:
        with open(prompt_data_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL Error: Main prompt file not found at {prompt_data_path}")
        exit()
    except json.JSONDecodeError:
        print(f"FATAL Error: Could not decode JSON from {prompt_data_path}")
        exit()

    # Loop through each defined time_step and process it
    for step in time_steps:
        process_single_timestep(base_scenario_name, step, prompt_data)

    print(f"\n{'=' * 20} All Timesteps Processed {'=' * 20}")