
#!/bin/bash

source ../pyenv/bin/activate

#=======================================================================
# Configuration
#=======================================================================
HYPERPARAM_FILE="hyperparams.txt"

START_ID=0
END_ID=67

MAX_PARALLEL_JOBS=2

#=======================================================================
# Main Execution Logic
#=======================================================================
echo "🚀 Starting hyperparameter search for experiments $START_ID through $END_ID..."
echo "Running up to $MAX_PARALLEL_JOBS jobs in parallel."

# Loop through each experiment ID
for (( TASK_ID=$START_ID; TASK_ID<=$END_ID; TASK_ID++ )); do
  # This entire block is run in a background subshell
  (
    # 1. Set up the save path for this specific task
    export SAVE_PATH="./saves/experiment-$TASK_ID"
    mkdir -p "$SAVE_PATH"

    # 2. Read the corresponding line of hyperparameters from the file
    # The '+1' is needed because sed is 1-indexed, while our loop is 0-indexed
    HYPER_LINE=$(sed -n "$((TASK_ID + 1))p" "$HYPERPARAM_FILE")
    export $HYPER_LINE

    echo "▶️  Starting Job ID: $TASK_ID | Params: $HYPER_LINE"

    # 3. Run the Python scripts and redirect all output (stdout & stderr) to a file
    #    The '-u' flag makes Python's output unbuffered, similar to 'srun --unbuffered'.
    #    The '&&' ensures the test script only runs if the training script succeeds.
    {
      python3 -u trainer.py && \
      python3 -u tester.py
    } > "$SAVE_PATH/train.out" 2>&1

    echo "✅ Finished Job ID: $TASK_ID"

  ) & # The '&' sends this entire subshell process to the background

  # --- Parallel Job Management ---
  # If we've reached the max number of parallel jobs, wait for one to finish
  if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]]; then
    # 'wait -n' waits for the next background job to terminate
    wait -n
  fi

done

# --- Final Cleanup ---
# Wait for any remaining background jobs (the last batch) to complete
echo "⏳ Waiting for the last batch of jobs to finish..."
wait
echo "🎉 All experiments completed successfully!"
