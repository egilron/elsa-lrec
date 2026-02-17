import sys
import os
import re

def extract_and_print_params(filepath):
    """
    Extracts model and training parameters from a shell script's filename
    and prints them as shell variable assignments.
    """
    filename = os.path.basename(filepath)
    # Initialize default values
    model_name = ""
    params_found = []

    blocks = filename.split("_") # Descriptive first, model second params last
    blocks[-1] = blocks[-1].split(".")[0]
    model_name = blocks[-2]
    params_found = re.findall(r'([a-zA-Z]+)(\d+)', blocks[-1])


    # A mapping from filename abbreviations to the shell variable names you'll use.
    param_config = {
        'bs': {'var_name': 'BATCH_SIZE', 'default': '1'},
        'ep': {'var_name': 'NUM_TRAIN_EPOCHS', 'default': '2'},
        'r': {'var_name': 'LORA_R', 'default': '16'},
        'a': {'var_name': 'LORA_ALPHA', 'default': '32'},
        's': {'var_name': 'SEED', 'default': '42'},
        'ev': {'var_name': 'EVAL_STEPS', 'default': '500'},
    }
    
    # --- Generate Shell Output ---
    found_params = {key: value for key, value in params_found}
    print(f'MODEL_NAME="{model_name}"')

    # Process each parameter, using found value or default
    for param_key, config in param_config.items():
        var_name = config['var_name']
        if param_key in found_params:
            value = found_params[param_key]
        else:
            value = config['default']
        print(f'{var_name}="{value}"')
if __name__ == "__main__":
    # The script expects the file path as the first command-line argument.
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        extract_and_print_params(script_path)
