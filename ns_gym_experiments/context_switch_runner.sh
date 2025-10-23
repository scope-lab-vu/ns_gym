#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT_NAME="/home/n--/ns_gym/ns_gym/context_switching.py" # Make sure this matches your Python script's filename
#VENV_PATH="/Users/--/Documents/--/Research/ns_gym_project/env" # Path to your virtual environment directory (e.g., venv, .venv)

# --- Default Experiment Parameters (can be overridden by passing arguments to this shell script) ---
TIMESTEPS_TRAIN=${1:-50000} # Default to 50000 if no first argument is given
EPISODES_EVAL=${2:-25}    # Default to 25 if no second argument is given
CONTEXT_PARAM=${3:-"gravity"} # Default to "masscart"
NUM_TARGET_CONTEXTS=${4:-100}
TARGET_CONTEXT_MIN=${5:-0.0015}
TARGET_CONTEXT_MAX=${6:-0.006}
ENV_NAME="MountainCar-v0"

# --- Output Directory and Filenames ---
# You can customize this further, e.g., by taking them as script arguments
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="experiment_run_${CONTEXT_PARAM}_${TIMESTAMP}"
PLOT_FILE="performance_plot_${CONTEXT_PARAM}.png"
METRICS_FILE="metrics_${CONTEXT_PARAM}.txt"


# --- Activate Virtual Environment (Optional but Recommended) ---
# If you are using a virtual environment, uncomment the following lines
# and ensure VENV_PATH is set correctly.

# if [ -d "$VENV_PATH" ]; then
#   echo "Activating virtual environment from $VENV_PATH..."
#   source "$VENV_PATH/bin/activate"
# else
#   echo "Warning: Virtual environment not found at $VENV_PATH. Running with system Python."
# fi

# --- Check if Python script exists ---
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_NAME' not found!"
    echo "Please ensure the script is in the same directory or provide the correct path."
    exit 1
fi

# --- Run the Experiment ---
echo "Starting experiment run..."
echo "Python script: $PYTHON_SCRIPT_NAME"
echo "Training timesteps per agent: $TIMESTEPS_TRAIN"
echo "Evaluation episodes per context: $EPISODES_EVAL"
echo "Context parameter: $CONTEXT_PARAM"
echo "Number of target contexts: $NUM_TARGET_CONTEXTS"
echo "Target context min: $TARGET_CONTEXT_MIN"
echo "Target context max: $TARGET_CONTEXT_MAX"
echo "Output directory: $OUTPUT_DIR"
echo "Plot file: $PLOT_FILE"
echo "Metrics file: $METRICS_FILE"

# Create the command with arguments
COMMAND="python $PYTHON_SCRIPT_NAME \
    --timesteps_train $TIMESTEPS_TRAIN \
    --episodes_eval $EPISODES_EVAL \
    --context_param \"$CONTEXT_PARAM\" \
    --num_target_contexts $NUM_TARGET_CONTEXTS \
    --target_context_min $TARGET_CONTEXT_MIN \
    --target_context_max $TARGET_CONTEXT_MAX \
    --output_dir \"$OUTPUT_DIR\" \
    --plot_file \"$PLOT_FILE\" \
    --metrics_file \"$METRICS_FILE\"
    --env_name \"$ENV_NAME\""

# Execute the command
echo "Executing: $COMMAND"
eval $COMMAND

# --- Deactivate Virtual Environment (if activated) ---
# if [ -d "$VENV_PATH" ] && [ -n "$VIRTUAL_ENV" ]; then
#   echo "Deactivating virtual environment."
#   deactivate
# fi

echo "Experiment run finished."
echo "Results saved in $OUTPUT_DIR"

