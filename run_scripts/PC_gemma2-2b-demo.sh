#!/bin/bash



# Set environment variables
wandb_key=$(</tokens/wandb)
hf_token=$(</tokens/hf)
export WANDB_API_KEY="$wandb_key"
export HF_TOKEN="$hf_token"
export HF_HOME="/hf/myhfhome"
export TRANSFORMERS_VERBOSITY="error"
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Model and dataset configuration
MODEL="google/gemma-2-2b"
DATASET="datasets/elsa_4splits/elsa_4splits_04_ent2ent.dataset"
OUTPUT_DIR="DEMO_1127_11"
SCRIPT="src/pairwise.py"

echo "Running experiment:"
echo "  Model: $MODEL"
echo "  Seed: 202"
echo "  Output directory: $OUTPUT_DIR"
echo "  Dataset: $DATASET"

# Run the training
time python $SCRIPT \
    --model_name_or_path "$MODEL" \
    --dataset_name "local/$DATASET" \
    --dataset_test_split "dev" \
    --output_dir "finetuned/$OUTPUT_DIR" \
    --run_name "$OUTPUT_DIR" \
    --per_device_train_batch_size 2 \
    --num_train_epochs 2 \
    --gradient_checkpointing True \
    --trust_remote_code True \
    --learning_rate 1.0e-4 \
    --logging_steps 100 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --max_length 4096 \
    --load_in_4bit True \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_task_type SEQ_CLS \
    --report_to none \
    --attn_implementation sdpa \
    --seed 202

echo "Branch: $(git branch --show-current), Commit: $(git rev-parse --short HEAD)"
cp "$0" "finetuned/$OUTPUT_DIR"
