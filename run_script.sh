#!/bin/bash

# === Parse command-line arguments ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift
            DATASETS=($(echo "$1" | tr ',' ' '))
            ;;
        -q|--questions)
            shift
            QUESTIONS="$1"
            ;;
        --output_dir)
            shift
            OUTPUT="$1"
            ;;
        -m|--model)
            shift
            MODEL="$1"
            ;;
        --temperature)
            shift
            TEMPERATURE="$1"
            ;;
        --num_samples)
            shift
            NUM_SAMPLES="$1"
            ;;
        --script)
            shift
            SCRIPT="$1"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# === Print config summary ===
echo "Running $SCRIPT"
echo "Dataset(s): ${DATASETS[*]}"
echo "Questions per quiz: $QUESTIONS"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT"
if [[ "$SCRIPT" == "sampling.py" ]]; then
    echo "Temperature: $TEMPERATURE"
    echo "Num Samples: $NUM_SAMPLES"
fi
echo "--------------------------------------"

# === Script selection ===
case "$SCRIPT" in
    "AFCE.py"|"quiz_like.py")
        python3 "$SCRIPT" \
            --datasets "${DATASETS[@]}" \
            --questions_per_quiz "$QUESTIONS" \
            --output_dir "$OUTPUT" \
            --model "$MODEL"
        ;;
    "top_k.py"|"vanilla.py")
        python3 "$SCRIPT" \
            --datasets "${DATASETS[@]}" \
            --output_dir "$OUTPUT" \
            --model "$MODEL"
        ;;
    "sampling.py")
        python3 "$SCRIPT" \
            --input_paths "${DATASETS[@]}" \
            --output_dir "$OUTPUT" \
            --model "$MODEL" \
            --temperature "$TEMPERATURE" \
            --num_samples "$NUM_SAMPLES"
        ;;
    *)
        echo "Unknown script: $SCRIPT"
        exit 1
        ;;
esac

echo "Finished running $SCRIPT"