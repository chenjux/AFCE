

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
        -o|--output)
            shift
            OUTPUT="$1"
            ;;
        -m|--model)
            shift
            MODEL="$1"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

python3 AFCE.py \
    --datasets "${DATASETS[@]}" \
    --questions_per_quiz "$QUESTIONS" \
    --output_dir "$OUTPUT" \
    --model "$MODEL"

echo "OUTPUT PATH: $OUTPUT"
