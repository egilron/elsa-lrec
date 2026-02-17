#!/bin/bash

set -o errexit

source ${HOME}/.bashrc

# Load modules
module --quiet purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5


# Set environment variables
wandb_key=$(</users/rnningst/tokens/wandb)
hf_token=$(</users/rnningst/tokens/hf)
export WANDB_API_KEY="$wandb_key"
export HF_TOKEN="$hf_token"
export HF_HOME="/scratch/project_465002310/egilron/myhfhome"
export TRANSFORMERS_VERBOSITY="error"
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True # Was used during traoubleshooting but gives a warning
export TORCH_COMPILE_DISABLE=1 # Neccessary for the inference script

get_script_path() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # Running under SLURM - get original script path
        scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1
    else
        # Not running under SLURM - use BASH_SOURCE for sourcing compatibility
        echo "${BASH_SOURCE[0]}"
    fi
}

SCRIPT_PATH=$(get_script_path)
eval "$(python3 src/parse_bashparams.py "$SCRIPT_PATH")"


if [[ "$MODEL_NAME" == *"gemma"* ]]; then
    MODEL="google/$MODEL_NAME"
elif [[ "$MODEL_NAME" == *"mistral"* ]]; then
    MODEL="mistralai/$MODEL_NAME"
else
    MODEL="$MODEL_NAME"
fi

# The above code takes the parameters setting indicated in the file name and assignes them to variables as coded in parse_bashparams.py. We expect informational first, then model, then parameters in the file name

# COnfigurations coded here:
DATASET="datasets/SFT_individual/elsa_fulltext_SFT_0sh_norw_p4nn.dataset" 
DATASET_TEXT_FIELD="prompt"  # prompt for all datasets in datasets/SFT_individual
OUTPUT_DIR=$(basename "$SCRIPT_PATH" .sh)
SCRIPT="src/sft.py"


# --- Echo Parameters for Verification ---
echo "Running experiment from: $SCRIPT_PATH"
echo "  Model: $MODEL"
echo "  Seed: $SEED"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  LoRA R: $LORA_R"
echo "  LoRA Alpha: $LORA_ALPHA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Dataset: $DATASET"

# --- Run the Training ---
# IMPORTANT
time python $SCRIPT \
    --model_name_or_path "$MODEL" \
    --dataset_name "local/$DATASET" \
    --dataset_test_split "dev" \
    --dataset_text_field "$DATASET_TEXT_FIELD" \
    --output_dir "finetuned/$OUTPUT_DIR" \
    --run_name "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 4 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --gradient_checkpointing True \
    --trust_remote_code True \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --eval_steps 250 \
    --fp16_full_eval True \
    --logging_steps 200 \
    --max_seq_length 8150 \
    --load_in_4bit True \
    --use_bnb_nested_quant True \
    --bnb_4bit_quant_type "nf4" \
    --use_peft \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.1 \
    --lora_task_type "CAUSAL_LM" \
    --report_to none \
    --attn_implementation eager \
    --seed "$SEED" \
    --packing False \
    --apply_gemma_formatting

    #--lr_scheduler_type constant , linear , polynomial , cosine\
echo "$SCRIPT completed. Preceeding to inference"
echo "Branch: $(git branch --show-current), Commit: $(git rev-parse --short HEAD)"

INFERENCE_SCRIPT="src/sft_inference.py"
INFERENCE_SPLIT="dev"
time python ${INFERENCE_SCRIPT} --model_root finetuned/${OUTPUT_DIR} --split ${INFERENCE_SPLIT}

INFERENCE_SPLIT="test
"
time python ${INFERENCE_SCRIPT} --model_root finetuned/${OUTPUT_DIR} --split ${INFERENCE_SPLIT}
# We use TRL version 0.13.0, therefore not     --bnb_4bit_use_double_quant True \