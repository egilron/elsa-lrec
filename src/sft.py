# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, time
import warnings
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import Counter
from pathlib import Path
from sklearn.metrics import f1_score
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainerCallback
from peft import prepare_model_for_kbit_training
import trl
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    ScriptArguments as BaseScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
    DataCollatorForCompletionOnlyLM
)

try:
    import wandb # May not be available in the HPC container
    WANDB=True
    print("import wandb success")
except:
    print("import wandb failed")
    WANDB=False
slurm_job= os.environ.get('SLURM_JOB_ID', 'local')
timestamp = datetime.now().strftime("%Y%m%d%H%M")
torch.cuda.empty_cache()
print(f"GPUs detected by PyTorch: {torch.cuda.device_count()}")
print("Torch version", torch.__version__) 
print( "TRL version",trl.__version__)
warnings.filterwarnings("ignore", message="No label_names provided")



class InferenceCallback(TrainerCallback):
    """Timing the training better"""
    def __init__(self, trainer, output_dir ):
        self.trainer = trainer
        self.start_time = None
        self.train_total_time = 0
        self.output_dir = output_dir
        self.epoch_durations = []
        self.epoch_start = None
        self.trainable_params = 0
        self.total_params = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time() # Keeps this value throughout

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_durations.append(time.time() - self.epoch_start)
        self.epoch_start = None
    
    def on_evaluate(self, args, state, control, **kwargs):
        # Compute trainable parameters for troubleshooting
        model = kwargs.get('model', None)
        if model is not None:
            self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.total_params = sum(p.numel() for p in model.parameters())        
        

        # print("on_evaluate kwargs", list(kwargs))
        self.train_total_time += time.time() - self.start_time
        elapsed_time = time.time() - self.start_time
        epoch = int(state.epoch)
        epoch_durations = self.epoch_durations if len(self.epoch_durations) > 0 else [0]
        record = dict(
        experiment = f"{Path(self.output_dir).name}_epoch_{epoch}_step{state.global_step}",
        timestamp = timestamp,
        seconds_since_trainstart = int(time.time() - self.start_time),
        seconds_epochs_total = np.sum(epoch_durations).item(),
        seconds_epochs_mean = np.mean(epoch_durations).item(),
        epoch=epoch,
        step  = state.global_step,
        total_steps  = state.max_steps,
        max_cuda_gb = torch.cuda.max_memory_allocated() / (1024**3),
        slurm_id = slurm_job,
        trainable_params=self.trainable_params,
        total_params=self.total_params
        )
        print(f"Evaluation triggered at step: {state.global_step} out of {state.max_steps}")
        with open("sft_logging.txt", "a") as f:
            f.write(json.dumps(record) + "\n")


def report_tokenizer(tokenizer, comment=""):
    """Used this for aligning new LLMs"""
    print("Tokenizer report: "+ comment)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print("tokenizer.pad_token_id", tokenizer.pad_token_id)
    print(" tokenizer.eos_token", tokenizer.eos_token)
    print("tokenizer special tokens", tokenizer.additional_special_tokens)
    print("UNK?",tokenizer.unk_token in tokenizer.special_tokens_map.values())
    print("PAD?",tokenizer.pad_token in tokenizer.special_tokens_map.values())
    print("tokenizer.pad_token",tokenizer.pad_token )
    print("---------------------")

def convert_to_gemma_format(example):
    """
    A static formatting function that formats a preprocessed example
    into the Gemma-2 chat template.
    Assumes the example has 'prompt' and 'completion' keys.
    """
    formatted_texts = []
    for i in range(len(example['prompt'])):
        prompt = example['prompt'][i]
        completion = example['completion'][i]
        
        # Gemma-2 instruction format
        gemma_formatted_str = (
            f"<start_of_turn>user\n"
            f"{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{completion}<end_of_turn>"
        )
        formatted_texts.append(gemma_formatted_str)
    return formatted_texts



@dataclass
class MoreScriptArguments(BaseScriptArguments):
    """
    Custom parameters

    """
    apply_gemma_formatting: bool = field(
        default=True,
        metadata={"help": "Apply Gemma chat template formatting to dataset"})    
    

if __name__ == "__main__":
    parser = HfArgumentParser((MoreScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False) # Changed back to this when I got 'grad_norm': 0.0
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True) # Changed this when I got grad_norm nan
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype))
    if WANDB:
        run = wandb.init(
            project="Pairwise comparison classification", 
            name=f"{model_args.model_name_or_path}_{timestamp}_{slurm_job}" 
            )   
    
    if training_args.seed:
        np.random.seed(training_args.seed)
        torch.manual_seed(training_args.seed)
        torch.cuda.manual_seed(training_args.seed)


    ################
    # Model & Tokenizer
    ################

    quantization_config = get_quantization_config(model_args)
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None, # Probably force to None if multiple GPUs
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.padding_side = 'right' # Added due to userwarning re overflow issues when training a model in half-precision
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,  trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    model = prepare_model_for_kbit_training(model)
    report_tokenizer(tokenizer, comment=training_args.output_dir+" Just loaded" )
    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        print("Chat template was None, you probably need to fix that.")
        # model, tokenizer = setup_chat_format(model, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        report_tokenizer(tokenizer, comment=training_args.output_dir+" Fixed tokenizer.pad_token" )

    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # Troubleshooting token mismatch
    print(f"Original model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    # model.resize_token_embeddings(len(tokenizer))
    # print(f"Resized model embedding size: {model.get_input_embeddings().weight.shape[0]}")



    ##############
    # Load dataset
    ##############
    if script_args.dataset_name.startswith("local/"):
        dataset = load_from_disk(script_args.dataset_name[6:]) # With 6 we remove the / after local
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer, 
        mlm=False
    )
    formatting_func = convert_to_gemma_format if script_args.apply_gemma_formatting else None
    
    ##########
    # Training
    ##########
    # Troubleshooting: Verify trainable parameters exist
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Before trainer init - Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Verify LoRA layers are present
    print("List of LoRA layers:")
    for name, module in model.named_modules():
        if 'lora' in name.lower():
            print(f"Found LoRA layer: {name}")


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        formatting_func=formatting_func,  # Pass None if data is pre-formatted
        data_collator=collator,
    )
    print(f"Max sequence length from trainer object: {trainer.args.max_seq_length}")
    assert trainer.args.max_seq_length > 3000, f"Max seq length not set properly: {trainer.args.max_seq_length}"

    # Verify PEFT was applied
    trainable_after = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"After trainer init - Trainable: {trainable_after:,}")


    inference_callback = InferenceCallback( trainer, training_args.output_dir )
    trainer.add_callback(inference_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
    trainer.evaluate()



    max_memory_gb = -1
    try:
        max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Max CUDA memory usage: {max_memory_gb:.2f} GB")
    except:
        print("Failed to get torch.cuda.max_memory_allocated()")


    try:
        wandb.log({"max_GPU_memory": max_memory_gb})
        wandb.finish()
    except Exception as e: 
        print("wandbfinish failed")
        print(e)





