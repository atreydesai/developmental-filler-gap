#!/bin/bash
#SBATCH --job-name=xanim_4
#SBATCH --output=logs/cross_animacy/slurm_%j.out
#SBATCH --error=logs/cross_animacy/slurm_%j.err
#SBATCH --partition=YOUR_PARTITION  # Update to your SLURM partition
#SBATCH --account=YOUR_ACCOUNT      # Update to your SLURM account
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Batch 4: 20M-40M, 2 seeds, all 8 directions
source ~/.bashrc && conda activate fillergap
# cd to the project root (adjust to your clone location)
cd "$(dirname "$0")/../.."
mkdir -p logs/cross_animacy

MODEL="BabyLM-community/babylm-baseline-100m-gpt2"
CHECKPOINTS="20M 30M 40M"
SEEDS="42 43"

for CKPT in $CHECKPOINTS; do
    for SEED in $SEEDS; do
        LOG="logs/cross_animacy/${CKPT}/seed_${SEED}"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization/wh_question_animate" --eval-dataset "wh_topicalization_inanimate/wh_question_inanimate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization_inanimate/wh_question_inanimate" --eval-dataset "wh_topicalization/wh_question_animate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization/wh_question_animate" --eval-dataset "wh_topicalization/wh_question_animate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization_inanimate/wh_question_inanimate" --eval-dataset "wh_topicalization_inanimate/wh_question_inanimate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization/topicalization_animate" --eval-dataset "wh_topicalization_inanimate/topicalization_inanimate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization_inanimate/topicalization_inanimate" --eval-dataset "wh_topicalization/topicalization_animate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization/topicalization_animate" --eval-dataset "wh_topicalization/topicalization_animate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
        python causalgym/das.py --model "$MODEL" --revision "chck_${CKPT}" --dataset "wh_topicalization_inanimate/topicalization_inanimate" --eval-dataset "wh_topicalization_inanimate/topicalization_inanimate" --seed $SEED --steps 80 --batch-size 25 --log-folder "$LOG"
    done
done
