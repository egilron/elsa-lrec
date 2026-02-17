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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, TrainerCallback
import trl
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments as BaseScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
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

class InferenceCallback(TrainerCallback):
    """This allows us to save epochwise inference on the test split during training"""
    def __init__(self,trainer,model,tokenizer, test_dataset, output_dir,categories, text_id = "elsa_id",
                 true_label = "true_sentiment_3"):
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.text_id = text_id # Column with text identifier for grouping the repeated texts with varying labels
        self.trainer = trainer
        self.model = model
        self.tokenizer = tokenizer 
        self.categories = categories # ["Negative", "Neutral", "Positive"]
        self.start_time = None
        self.train_start_time = None
        self.train_total_time = 0 # Sum of train time spent not in on_evaluate
        self.text_id = text_id
        self.true_label = true_label
        self.epoch_durations = []
        self.epoch_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time() # Keeps this value throughout
        self.train_start_time = time.time() # Not during evaluation
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_durations.append(time.time() - self.epoch_start)

    # def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
    def on_evaluate(self, args, state, control, **kwargs):
        print("on_evaluate kwargs", list(kwargs))
        self.train_total_time += time.time() - self.train_start_time
        elapsed_time = time.time() - self.start_time
        epoch = int(state.epoch)
        eval_df = self.test_dataset.to_pandas()
        
        records =[]     
        for e_id, group in eval_df.groupby(self.text_id):
            logits = {}
            for _,row in group.iterrows():
                text_inputs = tokenizer(
                    row.text, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    return_token_type_ids=False # May be needed for BERT
                ).to(model.device)
                with torch.no_grad():
                    model_output = model(**text_inputs)

                logit = model_output.logits.squeeze().cpu().item()
                logits[row.text.split()[-1]] = logit # Last word in text is the label to score
                
            # print(list(logits), self.categories)
            true_label = row[self.true_label]

            logits_ordered = [float(logits.get(c, -100000)) for c in self.categories] # When using truncated dataset, a category may be missing for the last text 
            # probs = torch.nn.functional.softmax(torch.tensor(logits_ordered), dim=0).tolist()
            pred_label = categories[logits_ordered.index(max(logits_ordered))]


            records.append({"entity_id":e_id,
                "true_label": true_label,
                "pred_label": pred_label,
            })
        
        records_as_dict = {key:[e[key] for e in records] for key in records[0]}
        epoch_durations = self.epoch_durations if len(self.epoch_durations) > 0 else [0]
        # F1-scores per category and w_avg
        f1_dict = {cat: f1_score(records_as_dict["true_label"], records_as_dict["pred_label"], labels=[cat], average=None, zero_division=0)[0] 
                for cat in self.categories}
        f1_dict["weighted_avg"] = f1_score(records_as_dict["true_label"], records_as_dict["pred_label"], average='weighted')
        label = f"{Path(self.output_dir).name}_epoch_{epoch}_step{state.global_step}"
        f1_dict["experiment"] = label
        f1_dict["timestamp"] = timestamp
        f1_dict["seconds_since_trainstart"] = int(elapsed_time)
        f1_dict["trainingtime"] = int(self.train_total_time)
        f1_dict["seconds_epochs_total"] = np.sum(epoch_durations).item()
        f1_dict["seconds_epochs_mean"] = np.mean(epoch_durations).item()
        f1_dict["step"] = state.global_step
        f1_dict["total_steps"] = state.max_steps
        f1_dict["max_cuda_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        f1_dict["slurm_id"] = slurm_job
        print(f"Evaluation triggered at step: {state.global_step} out of {state.max_steps}")
        print(f1_dict)
        with open("pc_f1_scores.txt", "a") as f:
            f.write(json.dumps(f1_dict) + "\n")

        Path("inference", label+".json").write_text(json.dumps(records_as_dict)) 
        self.train_start_time = time.time()

@dataclass
class MoreScriptArguments(BaseScriptArguments):
    """
    Extends TRL's ScriptArguments with custom parameters for inference.
    """
    inference_id_column: Optional[str] = field(
        default="id",
        metadata={"help": "Column name for text IDs in the inference split"}
    )
    inference_label_column: Optional[str] = field(
        default="true_label",
        metadata={"help": "Column name for true labels in the inference split"}
    ) 
    inference_splitname: Optional[str] = field(
        default="inference",
        metadata={"help": "Name of the inference split"}  
    )

if __name__ == "__main__":
    parser = HfArgumentParser((MoreScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True) # Changed this when I got grad_norm nan
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
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    report_tokenizer(tokenizer, comment=training_args.output_dir+" Just loaded" )
    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        print("Chat template was None, will not fix that.")
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


    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    if script_args.dataset_name.startswith("local/"):
        dataset = load_from_disk(script_args.dataset_name[6:]) # With 6 we remove the / after local
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def reduced_dataset(dataset, n=12):
        print(f"Reduced dataset: {n} samples")
        random_samples = {}
        for split in dataset.keys():
            random_samples[split] = dataset[split].shuffle(seed=42).select(range(min(n, len(dataset[split]))))
        return DatasetDict(random_samples)
    
    
    # dataset = reduced_dataset(dataset, n=500) # For development


    # Load inference split
    inference_split = dataset[script_args.inference_splitname] # For the inference in the callback. 
    id_column = script_args.inference_id_column 
    label_column = script_args.inference_label_column 
    assert id_column in list(inference_split.column_names) and label_column in list(inference_split.column_names)

    categories = sorted(list(inference_split.unique(label_column))) # In inference split, these are explicit
    print(f"Categories: {categories}")


    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    inference_callback = InferenceCallback(trainer=trainer,model=model,tokenizer=tokenizer, test_dataset=inference_split, output_dir=training_args.output_dir, categories=categories, text_id=id_column,true_label=label_column )
    trainer.add_callback(inference_callback)


    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    trainer.evaluate()

    

    # if training_args.eval_strategy != "no":
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


    # Save and push to hub
    print(f"Tokenizer vocab size before saving: {len(tokenizer)}")
    print(f"Model embedding size before saving: {model.get_input_embeddings().weight.shape[0]}")

    out_dir = training_args.output_dir+"/final"
    trainer.model.save_pretrained(out_dir,  save_embedding_layers=True)
    print("Final model save OK")
    report_tokenizer(tokenizer, comment=out_dir)
    try: # Save tokenizer to be certain. normistral has not got legacy format
        tokenizer.save_pretrained(out_dir, legacy_format=True)
        trainer.save_state()
    except:
        try:
            print("Could not save tokenizer with legacy format")
            tokenizer.save_pretrained(out_dir)
            trainer.save_state()
        except Exception as e:
            print("Could not save tokenizer at all")
            print(e)

    Path('LAST_FINETUNED').write_text(out_dir)

    saved_tokenizer = AutoTokenizer.from_pretrained(out_dir, use_fast=True,
local_files_only=True)
    report_tokenizer(saved_tokenizer, comment=out_dir+" Reloaded")
       
    # Upload manually
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)

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

print("Completed scr/rewardmodeling_std.py")



