#!/bin/bash

source ../pyenv/bin/activate

#=======================================================================
# Search Space Configuration
#=======================================================================

RANDOM_SEEDS=(6950 6955 6960 6965 6970 6975 6980 6985 6990 6995)

HPARAM_COMBINATIONS=(
    "Q_LEARNING_RATE=1e-2 POLICY_LEARNING_RATE=1e-3"
    "Q_LEARNING_RATE=1e-3 POLICY_LEARNING_RATE=1e-3"
    "Q_LEARNING_RATE=1e-3 POLICY_LEARNING_RATE=1e-4"
    "Q_LEARNING_RATE=1e-4 POLICY_LEARNING_RATE=1e-4"
)

MAX_PARALLEL_JOBS=1

START_ID=0
END_ID=16

#=======================================================================
# Main Execution Logic
#=======================================================================

echo "Starting structured runs for experiments $START_ID through $END_ID..."
echo -n "Each experiment will run with ${#RANDOM_SEEDS[@]} seeds and "
echo "${#HPARAM_COMBINATIONS[@]} hyperparameter search combinations."
echo "Running up to $MAX_PARALLEL_JOBS jobs in parallel."

for (( EXPERIMENT_ID=$START_ID; EXPERIMENT_ID<=$END_ID; EXPERIMENT_ID++ )); do

    BASE_PARAMS=$(sed -n "$((EXPERIMENT_ID + 1))p" experiments.txt)
    EXP_BASE_PATH="./saves/experiment-${EXPERIMENT_ID}"

    echo "Staging Main Experiment $EXPERIMENT_ID ($BASE_PARAMS)"

    for SEED_INDEX in "${!RANDOM_SEEDS[@]}"; do

        SEED=${RANDOM_SEEDS[$SEED_INDEX]}
        SEED_PATH="$EXP_BASE_PATH/seed-${SEED_INDEX}"

        HYPER_PARAMS_ID=0

        for HYPER_PARAMS in "${HPARAM_COMBINATIONS[@]}"; do

            (
                export SAVE_PATH="$SEED_PATH/hps-${HYPER_PARAMS_ID}"
                mkdir -p "$SAVE_PATH"

                export RANDOM_SEED=$SEED
                export $BASE_PARAMS
                export $HYPER_PARAMS

                echo "- Starting Job | Exp ${EXPERIMENT_ID} | Seed ${SEED_INDEX} ($SEED) | HPS ${HYPER_PARAMS_ID}"

                {
                    python3 -u trainer.py && \
                    python3 -u tester.py
                } > "$SAVE_PATH/run.out" 2>&1

                echo "- Finished Job | Exp ${EXPERIMENT_ID} | Seed ${SEED_INDEX} ($SEED) | HPS ${HYPER_PARAMS_ID}"

            ) &

            if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]]; then
                wait -n
            fi

            ((HYPER_PARAMS_ID++))
        done
    done

done

echo "⏳ Waiting for the last batch of jobs to finish..."
wait
echo "🎉 All experiments completed successfully!"
