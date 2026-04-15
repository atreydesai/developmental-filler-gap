#!/bin/bash
# Hyperparameter ablation: batch_size × steps combinations

set -e

MODEL="BabyLM-community/babylm-baseline-100m-gpt2"
REVISION="chck_100M"
SEEDS="42 43 44"

# Define hyperparameter configurations: "batch_size:steps"
CONFIGS=(
    "15:200"   # 4000 samples (paper actual)
    "200:15"   # 4000 samples (alternative)
)

echo "==========================================="
echo "Hyperparameter Ablation Experiment"
echo "==========================================="
echo "Model: $MODEL (revision: $REVISION)"
echo "Configurations: ${CONFIGS[*]}"
echo "Seeds: $SEEDS"
echo "==========================================="

for CONFIG in "${CONFIGS[@]}"; do
    BATCH_SIZE="${CONFIG%%:*}"
    STEPS="${CONFIG##*:}"
    SAMPLES=$((BATCH_SIZE * STEPS))
    
    echo ""
    echo "=== Config: batch=$BATCH_SIZE, steps=$STEPS (${SAMPLES} samples) ==="
    
    for SEED in $SEEDS; do
        echo "  Seed: $SEED"
        LOG_FOLDER="logs/hparam/b${BATCH_SIZE}_s${STEPS}/seed_${SEED}"
        
        # 4-way transfer
        for TRAIN in wh_question_animate topicalization_animate; do
            for EVAL in wh_question_animate topicalization_animate; do
                echo "    $TRAIN -> $EVAL"
                
                python causalgym/das.py \
                    --model "$MODEL" \
                    --revision "$REVISION" \
                    --dataset "wh_topicalization/${TRAIN}" \
                    --eval-dataset "wh_topicalization/${EVAL}" \
                    --steps $STEPS \
                    --batch-size $BATCH_SIZE \
                    --seed $SEED \
                    --log-folder "$LOG_FOLDER"
            done
        done
    done
done

echo ""
echo "==========================================="
echo "Ablation complete!"
echo "Run: python results/programs/process_hparam.py"
echo "==========================================="
