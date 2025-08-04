import os
import json
from datetime import datetime
import dspy
from PIL import Image



class ScenarioEvaluationSignature(dspy.Signature):
    """
    You are an expert motion planning evaluator for autonomous vehicles. Your primary function is to analyze
    motion planning scenarios and select the optimal trajectory for the ego vehicle from a set of candidates.

    Evaluate each trajectory against three core criteria: Safety, Comfort, and Efficiency,
    using a predefined cost scale and weighting scheme.

    Your final output must be a single, raw JSON object, strictly adhering to the specified format.
    """

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

        # Combine the json data into a single string, as defined in the signature


        # Call the predictor with the prepared inputs
        result = self.evaluator(image=img, scenario_data=scenario_data)

        return result

# Assume you have a ground-truth (gold standard) JSON evaluation for a specific scenario.
# This could be created by a human expert or a trusted simulator.
gold_standard_json_str = """
{   
    "best_trajectory_id": 129,
    "evaluation": {
        "trajectory_id:129": {
            "cost": 61.07,
            "costMap": {
                "lateral_jerk": [
                    11.56,
                    2.31
                ],
                "prediction": [
                    28.37,
                    5.67
                ],
                "longitudinal_jerk": [
                    8.57,
                    1.71
                ],
                "distance_to_reference_path": [
                    5.81,
                    29.05
                ],
                "velocity_offset": [
                    22.32,
                    22.32
                ]
            }
        },
        "trajectory_id:540": {
            "cost": 451.76,
            "costMap": {
                "lateral_jerk": [
                    0.29,
                    0.06
                ],
                "prediction": [
                    2165.03,
                    433.01
                ],
                "longitudinal_jerk": [
                    0.34,
                    0.07
                ],
                "distance_to_reference_path": [
                    2.83,
                    14.13
                ],
                "velocity_offset": [
                    4.49,
                    4.49
                ]
            }
        },
        "trajectory_id:483": {
            "cost": 1025.3,
            "costMap": {
                "lateral_jerk": [
                    10.76,
                    2.15
                ],
                "prediction": [
                    4796.19,
                    959.24
                ],
                "longitudinal_jerk": [
                    5.2,
                    1.04
                ],
                "distance_to_reference_path": [
                    5.79,
                    28.93
                ],
                "velocity_offset": [
                    33.95,
                    33.95
                ]
            }
        },
        "trajectory_id:128": {
            "cost": 74.35,
            "costMap": {
                "lateral_jerk": [
                    19.66,
                    3.93
                ],
                "prediction": [
                    14.25,
                    2.85
                ],
                "longitudinal_jerk": [
                    8.57,
                    1.71
                ],
                "distance_to_reference_path": [
                    8.71,
                    43.57
                ],
                "velocity_offset": [
                    22.28,
                    22.28
                ]
            }
        },
        "trajectory_id:492": {
            "cost": 222.01,
            "costMap": {
                "lateral_jerk": [
                    6.59,
                    1.32
                ],
                "prediction": [
                    753.76,
                    150.75
                ],
                "longitudinal_jerk": [
                    11.75,
                    2.35
                ],
                "distance_to_reference_path": [
                    2.89,
                    14.45
                ],
                "velocity_offset": [
                    53.14,
                    53.14
                ]
            }
        },
        "trajectory_id:183": {
            "cost": 1097.86,
            "costMap": {
                "lateral_jerk": [
                    20.83,
                    4.17
                ],
                "prediction": [
                    5137.25,
                    1027.45
                ],
                "longitudinal_jerk": [
                    8.24,
                    1.65
                ],
                "distance_to_reference_path": [
                    5.81,
                    29.05
                ],
                "velocity_offset": [
                    35.55,
                    35.55
                ]
            }
        }
    }
}
"""

# Create a dspy.Example
# You need to load the corresponding image and json data for this example
# For demonstration, we'll use the paths from your script.
# NOTE: In a real scenario, you would have multiple such examples.

base_scenario_name = "USA_Peach-1_1_T-1_0"
example_image_path = f"plots/{base_scenario_name}.png"
with open(f'cr_scenarios/{base_scenario_name}.json', 'r') as f:
    example_scenario_data = json.load(f)
with open(f'log_trajectory/trajectory_{base_scenario_name}.json', 'r') as f:
    example_traj_data = json.load(f)

combined_data_string_for_example = json.dumps({
    'scenario_context': example_scenario_data,
    'trajectory_details': example_traj_data
}, indent=2)


# The image object for the example
example_image = dspy.Image.from_file(example_image_path)

# Creating a single training example. You should create a list of these.
train_example = dspy.Example(
    image=example_image_path,
    scenario_data=combined_data_string_for_example,
    evaluation_json=gold_standard_json_str # The desired output
).with_inputs("image", "scenario_data") # Specify which fields are inputs

# You'll create a list for your training set
trainset = [train_example] # In reality, this would have 10-20 examples.


def validation_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """
    A simple metric to check if the predicted best_trajectory_id matches the gold standard.
    And also validates if the output is a valid JSON.
    """
    try:
        # Parse the JSON from the gold standard and the prediction
        gold_json = json.loads(gold.evaluation_json)
        pred_json = json.loads(pred.evaluation_json)

        # Check if the most important field matches
        gold_best_id = gold_json.get("best_trajectory_id")
        pred_best_id = pred_json.get("best_trajectory_id")

        return gold_best_id == pred_best_id
    except (json.JSONDecodeError, AttributeError):
        # If parsing fails or keys are missing, the prediction is incorrect.
        return False


# --- Main Execution Block using DSPy ---
if __name__ == "__main__":
    # 1. Configure the Language Model (Gemini Multi-modal)
    # Make sure GEMINI_API_KEY is set in your environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your API key.")
    gemini = dspy.LM('gemini/gemini-2.5-pro',api_key=api_key,  max_tokens = 8192)
    dspy.settings.configure(lm=gemini)

    # (Assuming trainset and validation_metric are defined as above)
    # For this example, we'll use the same set for training and validation.
    # In a real project, you should have a separate validation set.
    valset = trainset

    # 2. Set up the optimizer
    # BootstrapFewShot is great for complex tasks. It will find the best
    # few-shot examples from your trainset to include in the prompt.
    optimizer = dspy.BootstrapFewShot(metric=validation_metric, max_bootstrapped_demos=2)

    # 3. Compile the MotionEvaluator module
    # This is where the magic happens. DSPy will test different prompts
    # (by selecting different few-shot examples) to maximize your metric.
    print("--- Compiling the DSPy Module ---")
    compiled_evaluator = optimizer.compile(MotionEvaluator(), trainset=trainset)
    print("--- Compilation Complete ---")

    # 4. Use the optimized module for a new prediction
    # Let's use the same data for inference as an example.
    print("\n--- Evaluating with Compiled Module ---")

    # Load the data for a new scenario you want to evaluate
    base_scenario_name = "USA_Peach-1_1_T-1_0"  # The scenario to run
    scenario_json_path = f'cr_scenarios/{base_scenario_name}.json'
    trajectory_json_path = f'log_trajectory/trajectory_{base_scenario_name}.json'
    scenario_image_path = f"plots/{base_scenario_name}.png"

    with open(scenario_json_path, 'r') as f:
        scenario_data = json.load(f)
    with open(trajectory_json_path, 'r') as f:
        trajectory_data = json.load(f)

    # Call the compiled module's forward method
    final_prediction = compiled_evaluator(
        image=example_image_path,
        scenario_data=combined_data_string_for_example
    )


    print("\n--- Evaluation Result ---")
    # The result is in final_prediction.evaluation_json
    evaluation_result = json.loads(final_prediction.evaluation_json)
    print(json.dumps(evaluation_result, indent=2))

    # You can inspect the last prompt used by the compiled program
    print("\n--- Last Prompt Sent to LLM ---")
    gemini.inspect_history(n=1)