import math
import os
import json
from datetime import datetime
import dspy
from PIL import Image
from dspy.dsp.utils import timestamp


class ScenarioEvaluationSignature(dspy.Signature):
    # --- Input Fields ---
    # The 'image' field is special for multi-modal models. DSPy knows how to handle it.
    image: dspy.Image = dspy.InputField(desc="Visual representation of the scenario. The ego vehicle is blue.")

    # We pass the combined JSON data as a single string.
    scenario_data: str = dspy.InputField(desc="A JSON string containing 'scenario_context' and 'trajectory_details'.")

    # --- Output Field ---
    # The output field describes the expected format.
    evaluation_json: str = dspy.OutputField(
        desc="A single, raw JSON object with the specified structure. Example: {'best_trajectory_id': 1, 'evaluation': [...]}")


class MotionEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        # This predictor will use our signature to call the LLM.
        # It internally manages how to construct the prompt from the signature.
        self.evaluator = dspy.Predict(ScenarioEvaluationSignature)

    def forward(self, image: str, scenario_data: dict) -> dspy.Prediction:
        """
        The forward method defines how the module is executed.
        """
        # Load the image using DSPy's Image class
        img  = dspy.Image.from_file(image)

        # Call the predictor with the prepared inputs
        result = self.evaluator(image=img, scenario_data=scenario_data)

        return result




def validation_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """
    A simple metric to check if the predicted best_trajectory_id matches the gold standard.
    And also validates if the output is a valid JSON.
    """
    try:
        # --- 1. Parse JSON from gold standard and prediction ---
        # Assumes the JSON string is in the 'evaluation_json' field for both.
        gold_json = json.loads(gold.evaluation_json)
        pred_json = json.loads(pred.evaluation_json)

        # List to store results of individual checks (1 for pass, 0 for fail)
        checks = []

        # --- 2. Perform Checks ---
        # Check 1: Validate the 'best_trajectory_id'
        gold_best_id = gold_json.get("best_trajectory_id")
        pred_best_id = pred_json.get("best_trajectory_id")
        check1_passed = (gold_best_id is not None and gold_best_id == pred_best_id)
        checks.append(1 if check1_passed else 0)

        # Subsequent checks only make sense if the best trajectory ID was correct.
        if check1_passed:
            best_traj_key = f"trajectory_id:{gold_best_id}"
            gold_traj = gold_json.get(best_traj_key, {})
            pred_traj = pred_json.get(best_traj_key, {})

            # Check 2: Validate the 'cost' of the best trajectory
            gold_cost = gold_traj.get("cost")
            pred_cost = pred_traj.get("cost")
            # Use math.isclose for robust float comparison
            check2_passed = (gold_cost is not None and pred_cost is not None and math.isclose(gold_cost, pred_cost))
            checks.append(1 if check2_passed else 0)

            # Check 3: Validate the main cost contributor
            gold_cost_map = gold_traj.get("costMap", {})
            pred_cost_map = pred_traj.get("costMap", {})

            if gold_cost_map and pred_cost_map:
                # Find the key with the max cost value (first element of the list)
                gold_main_contributor = max(gold_cost_map, key=lambda k: gold_cost_map[k][0])
                pred_main_contributor = max(pred_cost_map, key=lambda k: pred_cost_map[k][0])
                check3_passed = (gold_main_contributor == pred_main_contributor)
                checks.append(1 if check3_passed else 0)
            else:
                checks.append(0)  # Fails if costMaps are missing
        else:
            # If check 1 fails, the other checks automatically fail.
            # We add 0 for each of the subsequent checks to keep the denominator correct.
            checks.extend([0, 0])

        # --- 3. Calculate Final Score ---
        # The score is the average of the passed checks.
        return sum(checks) / len(checks)

    except (json.JSONDecodeError, AttributeError, KeyError):
        # If JSON parsing fails or keys are missing, the prediction is completely wrong.
        return 0.0


def load_training_data(scenario_dir, trajectory_dir, image_dir, gold_standard_dir):
    """
    Dynamically loads all corresponding data points from specified directories.
    It identifies base scenario names and constructs a list of dspy.Examples.
    """
    trainset = []
    # Use one directory (e.g., scenarios) as the source of truth for filenames
    scenario_files = os.listdir(scenario_dir)

    print(f"Found {len(scenario_files)} potential scenarios. Checking for matching files...")

    for scenario_filename in scenario_files:
        if not scenario_filename.endswith('.json'):
            continue

        base_scenario_name = os.path.splitext(scenario_filename)[0]

        # Construct all required file paths
        scenario_json_path = os.path.join(scenario_dir, f"{base_scenario_name}.json")
        # Note: Your original code had a 'trajectory_' prefix. Adjust if needed.
        trajectory_json_path = os.path.join(trajectory_dir, f"trajectory_{base_scenario_name}.json")
        image_path = os.path.join(image_dir, f"{base_scenario_name}.png")
        gold_json_path = os.path.join(gold_standard_dir, f"cost_{base_scenario_name}.json")

        # Check if all corresponding files exist to ensure data integrity
        if not all(os.path.exists(p) for p in [scenario_json_path, trajectory_json_path, image_path, gold_json_path]):
            print(f"--> Skipping {base_scenario_name}: Missing one or more corresponding files.")
            continue

        try:
            # Load the data content
            with open(scenario_json_path, 'r') as f:
                scenario_data = json.load(f)
            with open(trajectory_json_path, 'r') as f:
                traj_data = json.load(f)
            with open(gold_json_path, 'r') as f:
                gold_standard_json_str = f.read()

            # Combine scenario and trajectory data for the input
            combined_data_string = json.dumps({
                'scenario_context': scenario_data,
                'trajectory_details': traj_data
            }, indent=2)

            # Create the dspy.Example instance
            example = dspy.Example(
                image=image_path,  # DSPy handles loading from path
                scenario_data=combined_data_string,
                evaluation_json=gold_standard_json_str  # The desired output is the content of the gold file
            ).with_inputs("image", "scenario_data")

            trainset.append(example)
            print(f"--> Successfully loaded example for {base_scenario_name}")

        except Exception as e:
            print(f"--> Error loading data for {base_scenario_name}: {e}")

    print(f"\nSuccessfully loaded a total of {len(trainset)} examples.")
    return trainset

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Configure the Language Model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    gemini = dspy.LM('gemini/gemini-2.5-pro',api_key=api_key,  max_tokens = 65536)
    dspy.settings.configure(lm=gemini)

    # 2. Dynamically load the entire dataset
    SCENARIO_DIR = 'cr_scenarios/USA_Peach-1_1_T-1'
    TRAJECTORY_DIR = 'log_trajectory/USA_Peach-1_1_T-1'
    IMAGE_DIR = 'plots/USA_Peach-1_1_T-1'
    GOLD_STANDARD_DIR = 'frenetix_cost/USA_Peach-1_1_T-1'

    trainset = load_training_data(SCENARIO_DIR, TRAJECTORY_DIR, IMAGE_DIR, GOLD_STANDARD_DIR)

    if not trainset:
        raise ValueError("Training set is empty. Check your data directories and file naming conventions.")

    # For this example, we'll use the same set for training and validation.
    # In a real project, you should split your data into train and validation sets.
    # valset = trainset[:len(trainset)//2]
    # trainset = trainset[len(trainset)//2:]
    trainset = trainset[:4]  # For testing, limit to first 10 examples
    valset = trainset[4:]

    # 3. Set up the optimizer
    optimizer = dspy.BootstrapFewShot(metric=validation_metric, max_bootstrapped_demos=2)

    # 4. Compile the MotionEvaluator module
    print("\n--- Compiling the DSPy Module ---")
    # Let's assume max_bootstrapped_demos=2 for this run based on the log
    compiled_evaluator = optimizer.compile(MotionEvaluator(), trainset=trainset)
    print("--- Compilation Complete ---\n")

    # 5. Save the best prompt (The Correct and Robust Way)
    # The result of BootstrapFewShot is the selection of the best demonstration examples.
    # We extract these directly from the compiled module's internal predictor.
    print("--- Extracting and Saving Optimized Prompt Components ---")

    # First, get the list of all predictors within the compiled module.
    list_of_predictors = compiled_evaluator.predictors()

    if not list_of_predictors:
        raise ValueError("No predictors found in the compiled module. Cannot extract prompt.")

    # In this case, MotionEvaluator has one predictor, so we take the first one.
    optimized_predictor = list_of_predictors[0]

    # The few-shot examples selected by the optimizer are stored in '.demos'
    optimized_demos = optimized_predictor.demos

    demos_as_dicts = []
    for demo in optimized_demos:
        # Convert the dspy.Example to a dictionary
        demo_dict = demo.toDict()

        # Iterate over the dictionary to find and convert the Image object
        for key, value in demo_dict.items():
            # Check if the value is an instance of the Image class
            if isinstance(value, dspy.Image):
                # Replace the Image object with its file path.
                # The attribute might be '.filename', '.path', or something else.
                # Use `dir(value)` to inspect the object if you're unsure.
                if hasattr(value, 'filename'):
                    demo_dict[key] = value.filename
                else:
                    # Fallback or error if the path attribute is not found
                    demo_dict[key] = f"UNSERIALIZABLE_IMAGE_OBJECT_OF_TYPE_{type(value).__name__}"
        demos_as_dicts.append(demo_dict)
    # We will save the instruction and the demos to a JSON file for clarity
    prompt_components = {
        "prompt_instructions": optimized_predictor.signature.instructions,
        "prompt_demos": demos_as_dicts
    }


    prompt_save_path = "best_motion_evaluator_prompt.json"
    with open(prompt_save_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_components, f, indent=2, ensure_ascii=False)

    # print(f"--- Best prompt components saved to {prompt_save_path} ---")
    # print(f"Instruction: {prompt_components['system_message']}")
    # print(f'prompt:{demos_as_dicts}demos_as_dicts')

    # Inspect the optimized prompt
    # optimized_prompt_instructions = (optimized_predictor.signature.instructions)
    # optimized_prompt_demos = optimized_predictor.demos
    # prompt_components = {
    #     "instructions": optimized_prompt_instructions,
    #     "demos": optimized_prompt_demos
    # }
    #
    # print("--- Optimized Prompt Instructions ---")
    # print(optimized_prompt_instructions)
    # print("\n--- Optimized Few-Shot Examples (Demos) ---")
    # print(optimized_prompt_demos)
    # print(f"Number of demos selected: {len(prompt_components['demos'])}")

    # # 6. Use the optimized module for a new prediction (optional)
    # print("\n--- Evaluating with Compiled Module on a Test Example ---")
    #
    # base_scenario_name = "USA_US101-29_1_T-1"
    # timestep = 30
    # scenario_json_path = f'cr_scenarios/{base_scenario_name}/{base_scenario_name}_{timestep}.json'
    # trajectory_json_path = f'log_trajectory/{base_scenario_name}/trajectory_{base_scenario_name}_{timestep}.json'
    # scenario_image_path = f"plots/{base_scenario_name}/{base_scenario_name}_{timestep}.png"  # This path isn't directly combined as JSON content
    # # Load the data content
    # with open(scenario_json_path, 'r') as f:
    #     scenario_data = json.load(f)
    # with open(trajectory_json_path, 'r') as f:
    #     traj_data = json.load(f)
    #
    # # Combine scenario and trajectory data for the input
    # combined_data_string = json.dumps({
    #     'scenario_context': scenario_data,
    #     'trajectory_details': traj_data
    # }, indent=2)
    #
    # final_prediction = compiled_evaluator(
    #     image=scenario_image_path,
    #     scenario_data=combined_data_string
    # )
    #
    # print("\n--- Evaluation Result ---")
    # try:
    #     evaluation_result = json.loads(final_prediction.evaluation_json)
    #     response_save_path = f"logs/responses/{base_scenario_name}/response_{base_scenario_name}_{timestep}.json"
    #     with open(response_save_path, 'w', encoding='utf-8') as f:
    #         json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
    # except json.JSONDecodeError:
    #     print("Failed to decode JSON from prediction:")
    #     print(final_prediction.evaluation_json)