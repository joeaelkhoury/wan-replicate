"""
This is a wrapper script to make our Docker container compatible with Replicate.
It implements a simplified version of the Cog interface.
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any

from predict import Predictor

def read_input() -> Dict[str, Any]:
    """Parse the input from Replicate"""
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Parse inputs from the command line or env var
        input_file = os.environ.get("REPLICATE_INPUT_PATH")
        if input_file is None or not os.path.exists(input_file):
            print(f"Error: REPLICATE_INPUT_PATH environment variable not set or file doesn't exist: {input_file}")
            sys.exit(1)
            
        try:
            with open(input_file, "r") as f:
                inputs = json.load(f)
                
            # Handle file inputs (convert file paths)
            for key, value in inputs.items():
                if isinstance(value, dict) and "path" in value:
                    inputs[key] = value["path"]
                    print(f"Processing file input: {key} -> {value['path']}")
                    
            return inputs
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        print("Usage: python replicate_cog_wrapper.py predict")
        sys.exit(1)

def write_output(output: Dict[str, str]):
    """Write the output in Replicate's expected format"""
    output_file = os.environ.get("REPLICATE_OUTPUT_PATH")
    if output_file is None:
        print("Error: REPLICATE_OUTPUT_PATH environment variable not set")
        sys.exit(1)
        
    try:
        # If output is a file path, read the file and base64 encode it
        if "output" in output and os.path.exists(output["output"]):
            output_path = output["output"]
            
            # Copy the file to Replicate's expected output directory
            replicate_output_dir = os.path.dirname(output_file)
            destination = os.path.join(replicate_output_dir, os.path.basename(output_path))
            shutil.copy(output_path, destination)
            
            # Update the output to point to the copied file
            print(f"Copied output file from {output_path} to {destination}")
            output["output"] = destination
        
        with open(output_file, "w") as f:
            json.dump(output, f)
            
        print(f"Output successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing output: {e}")
        sys.exit(1)

def main():
    """Main entry point for the wrapper"""
    # Read inputs
    print("Starting prediction process...")
    inputs = read_input()
    print(f"Received inputs: {json.dumps(inputs, indent=2)}")
    
    # Initialize the predictor
    predictor = Predictor()
    
    # Run prediction
    start_time = time.time()
    print("Running prediction...")
    try:
        output = predictor.predict(inputs)
        print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        
        # Write outputs
        write_output(output)
        print("Process completed successfully")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()