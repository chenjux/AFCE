SCRIPT="AFCE.py"  # 默认脚本

while [[ $# -gt 0 ]]; do
    case "$1" in
        --script)
            shift
            SCRIPT="$1"
            ;;
        --datasets)
            shift
            DATASETS=($(echo "$1" | tr ',' ' '))
            ;;
        --questions)
            shift
            QUESTIONS="$1"
            ;;
        --output)
            shift
            OUTPUT="$1"
            ;;
        --model)
            shift
            MODEL="$1"
            ;;
        --input)
            shift
            INPUT="$1"
            ;;
        --type)
            shift
            TYPE="$1"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

case "$SCRIPT" in
    "AFCE.py")
        python3 AFCE.py \
            --datasets "${DATASETS[@]}" \
            --questions_per_quiz "${QUESTIONS:-10}" \
            --output_dir "${OUTPUT:-results}" \
            --model "${MODEL:-gpt-4o}"
        ;;
    "quiz_like.py")
        python3 quiz_like.py \
            --datasets "${DATASETS[@]}" \
            --questions_per_quiz "${QUESTIONS:-10}" \
            --output_dir "${OUTPUT:-results}" \
            --model "${MODEL:-gpt-4o}"
        ;;
    "top_k.py")
        python3 top_k.py \
            --datasets "${DATASETS[@]}" \
            --output_dir "${OUTPUT:-results}" \
            --model "${MODEL:-gpt-4o}"
        ;;
    *)
        echo "Unknown script: $SCRIPT"
        exit 1
        ;;
esac

echo "Finished running $SCRIPT"