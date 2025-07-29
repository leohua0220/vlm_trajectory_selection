import os
import io
import json
from datetime import datetime
from PIL import Image
import google.generativeai as genai


# This function remains the same as it focuses solely on the API call.
def analyze_bev_image_with_gemini(image_path, api_key, prompt):
    """
    Analyzes a BEV image using the Google Gemini API.

    Args:
        image_path (str): The file path to the image (e.g., PNG).
        api_key (str): Your Gemini API key.
        prompt (str): The text prompt to guide the analysis.

    Returns:
        str: The generated text description, or None if an error occurs.
    """
    # Configure the API key
    genai.configure(api_key=api_key)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # Load the image using Pillow (PIL)
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image file: {e}")
        return None

    # Create a Gemini Pro Vision model instance
    model = genai.GenerativeModel('gemini-2.5-flash')  # Using the model specified by you

    # Send the prompt and image to the model
    try:
        print("Sending image for analysis to the API...")
        response = model.generate_content([prompt, image])
        # Return the generated text content
        return response.text
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None


def run_and_save_analysis(image_path, api_key, prompt, log_dir='logs', log_filename='analysis_log.json',
                          response_output_subdir='responses'):
    """
    Runs the image analysis and saves the prompt, response, and metadata to a log file
    within a dedicated log directory. Also saves the detailed text response to a separate,
    readable .txt file inside a subdirectory within the log directory.

    Args:
        image_path (str): The file path to the image.
        api_key (str): Your Gemini API key.
        prompt (str): The text prompt for analysis.
        log_dir (str): The main directory to store all logs (e.g., 'logs').
        log_filename (str): The name of the JSON log file within the log_dir.
        response_output_subdir (str): Subdirectory within log_dir to save detailed text responses.
    """
    # Ensure the main log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Define full path for the JSON log file
    full_log_file_path = os.path.join(log_dir, log_filename)

    # Define full path for the response output subdirectory
    full_response_output_dir = os.path.join(log_dir, response_output_subdir)
    os.makedirs(full_response_output_dir, exist_ok=True)

    # 1. Call the API analysis function
    response_text = analyze_bev_image_with_gemini(image_path, api_key, prompt)
    print(f'prompt: {prompt}')
    print(f'response: {response_text}')

    # 2. If a response is received, save it
    if response_text:
        # Generate a unique filename for the text response file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name_prefix = os.path.splitext(os.path.basename(image_path))[0]
        response_filename = f"{image_name_prefix}_{timestamp_str}_response.txt"
        response_file_path = os.path.join(full_response_output_dir, response_filename)

        # Save the detailed response to a .txt file
        try:
            with open(response_file_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"Detailed response saved to '{response_file_path}'.")
        except IOError as e:
            print(f"Error saving detailed response to file: {e}")
            response_file_path = None  # Mark as failed to save

        # Create a dictionary with all the data to be logged for JSON
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_path": os.path.abspath(image_path),
            "prompt": prompt,
            "response_summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            # Save a summary in JSON
            "response_full_text_file": os.path.relpath(response_file_path, log_dir) if response_file_path else None
            # Relative path for clarity
        }

        # Load existing log data from the file, or initialize an empty list
        log_data = []
        if os.path.exists(full_log_file_path):
            try:
                with open(full_log_file_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Log file is corrupted or empty. Starting a new log.")
                log_data = []

        # Append the new log entry
        log_data.append(log_entry)

        # Save the updated data back to the log file
        try:
            with open(full_log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully updated analysis log in '{full_log_file_path}'.")
        except IOError as e:
            print(f"Error saving log to file: {e}")
    else:
        print("Analysis failed, nothing to save.")


# --- Example Usage ---

# Set your API key from an environment variable (recommended) or directly
# For security, do not hardcode your key in the script.
# Use `os.getenv("YOUR_API_KEY_ENV_VAR")` after setting it in your environment.
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your API key.")

# Define the path to your BEV image
image_file_path = "plots/USA_Peach-1_1_T-1_11.png"

# Define a specific prompt for the analysis
analysis_prompt = (

    "As an expert in autonomous driving perception and planning, analyze the following Bird's-Eye View (BEV) image of a path planning scenario. The green arrow indicates the ego vehicle's current driving direction. Provide a detailed description of:"
    "1.  **Predict the ego vehicle's (labeled as '1500') probable future trajectory** for the next [Specify Time Horizon, e.g., 3-5 seconds]. Based on the scene, describe its likely path, considering potential maneuvers and interactions."
    "2.  Any **other objects or traffic participants** (e.g., obstacles, other vehicles, pedestrians) in the scene, including their type, approximate position, and *potential movement patterns* relative to the ego vehicle."
    "3.  The **goal area** (marked as the yellow region), and how it influences the predicted ego vehicle's future trajectory and overall planning objective."
)

# --- Run the analysis and save the log ---
if API_KEY and os.path.exists(image_file_path):
    run_and_save_analysis(image_file_path, API_KEY, analysis_prompt)
else:
    print("Please set your API key and ensure the image file path is correct.")