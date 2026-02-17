import os, json, argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict, load_from_disk
import torch, transformers

torch.set_float32_matmul_precision('high')
os.environ['TORCH_COMPILE_DISABLE'] = '1'

print(f"GPUs detected by PyTorch: {torch.cuda.device_count()}")
print("Torch version", torch.__version__) 
print("Transformers version", transformers.__version__)




def do_inference(model, tok, prompts):
    """
    prompts: a list of prompts, like: 
    [
    "Write me a poem about Machine Learning.",
    "Why is the sky blue?"
    ]
    """
    results = []
    print("Running inference...")

    for i, prompt in enumerate(prompts): 
        if i % 20 == 0:
            print(f"After {i}/{len(prompts)} prompts:",Counter(results).most_common(5))
        conversation = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt", # Return PyTorch tensors
                return_dict=True # Trying this
            )
            
        # Doing this explicit for troubleshooting
        # attention_mask = torch.ones_like(input_ids)
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        # attention_mask = attention_mask.to(model.device)

        out = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
        
        gen_ids = out[0][inputs['input_ids'].shape[-1]:]
        result = tok.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(result)
    print("do_inference: Last input to model:")
    print(tok.decode(inputs['input_ids'][0]))
    return results

def clean_predictions(gold, predictions):
    categories = sorted(list(set(gold)))
    # Getting what Norw and ENg has in common for Neutral
    # This code need to be altered to meet actual needs for other datasets.
    neutral_label = [c for c in categories if "tral" in c][0]
    positive_label = [c for c in categories if c.lower().startswith("posi")][0]
    negative_label = [c for c in categories if c.lower().startswith("nega")][0]
    cleaned = []
    for p in predictions:
        if "posi" in p.lower():
            cleaned.append(positive_label)
        elif "nega" in p.lower():
            cleaned.append(negative_label)
        else:
            cleaned.append(neutral_label)
    assert len(cleaned) == len(gold), ("cleaned, preds, gold",len(cleaned), len(predictions), len(gold))
    print("clean_predictions: Were", Counter(predictions).most_common() )  
    print("became:", Counter(cleaned).most_common())  
    return cleaned




def evaluate_inference(gold, predictions, metadata:dict):
    predictions = clean_predictions(gold, predictions)
    categories = sorted(list(set(gold)))
    f1_dict = {cat: f1_score(gold, predictions, labels=[cat],average=None, zero_division=0)[0]  for cat in categories}
    f1_dict["weighted_avg"] = f1_score(gold, predictions, average='weighted')

    f1_dict = {**f1_dict, **metadata}
    with open("sft_f1_scores.txt", "a") as f:
        f.write(json.dumps(f1_dict) + "\n")   
    print(f1_dict)


if __name__ == "__main__":
    """
    input_text = "Write me a poem about Machine Learning."

    message = {"role": "user", "content": input_text}
    messages = [message]
    """
    parser = argparse.ArgumentParser(description='A script to parse model root and data split.')
    parser.add_argument('--model_root', 
                        type=str, 
                        required=True,
                        help='The root directory of the model. This is a required argument.')
    parser.add_argument('--split', 
                        choices=['dev','validation' ,'test'], 
                        default='dev',
                        help='The data split to evaulate on, either "dev", "validation" or "test". Defaults to "dev".')
    parser.add_argument('--text_col',
                        default="prompt",
                        help="Column header for the prompt")
    parser.add_argument('--label_col',
                        default = "completion",
                        help= "Column header for the true label")
    parser.add_argument('--dataset',
                        default = "datasets/SFT_individual/elsa_fulltext_SFT_0sh_norw_p4nn.dataset", 
                        help = "Relative path to dataset, like 'datasets/SFT_individual/elsa_fulltext_SFT_0sh_norw_p4nn.dataset'."
                                          
    )
    args = parser.parse_args()



    
    # inference_ds = load_from_disk(f"datasets/elsa_fulltext_SFT_gemma_3sh/elsa_fulltext_SFT_gemma_3sh.dataset/{args.split}")
    # inference_ds = load_from_disk(f"datasets/obsolete-elsa_fulltext_SFT_gemma/elsa_fulltext_SFT_gemma.dataset/{args.split}")
    inference_ds = load_from_disk(f"{args.dataset}/{args.split}")
    # prompts = inference_ds["threeshot_prompt"]
    prompts = inference_ds[args.text_col]
    gold = inference_ds[args.label_col] # label # label_in_norwegian
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    def subfolders_filtered(finetuned_root):
        try:
            finetuned_subfolders = [p for p in Path(finetuned_root).iterdir() if p.is_dir()]
            filtered = []
            for subfolder in finetuned_subfolders:
                if not Path(f"inference/{subfolder.parent.name}+{subfolder.name}+{args.split}.json").exists():
                    filtered.append(subfolder)
            assert len(filtered) > 0, "No available subfolders "
        except: # We have passed a path that maybe is the direct path to a model
            filtered = [Path(finetuned_root)]

        print("Finetuned subfolders for inference:")
        for p in filtered:
            print(str(p))
        return filtered

    for subfolder in subfolders_filtered(args.model_root):
        model_path = str(subfolder)
        train_steps = Path(model_path).stem if model_path.startswith("finetuned") else "not_finetuned"


        # Run the full inference process
        predictions_path = f"inference/{subfolder.parent.name}+{subfolder.name}+{args.split}.json"
        print(predictions_path)
        predictions = []

        print("First message:")
        print(prompts[0][:30], "...", prompts[0][-50:])

        # model_path = "finetuned/gemma2-2b_SFT-elsa/final" # "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,  # Changed from torch.float32
            device_map="auto"
            )

        try:

            predictions = do_inference(model, tokenizer, prompts)

            # Save predictions
            Path(predictions_path).write_text(json.dumps(predictions))
            print("predictions now",len(predictions), len(prompts))


            # Evaluate and save scores
            metadata = dict(
                timestamp=timestamp,
                model_path=model_path,
                train_step=train_steps,
                eval_split=args.split,
                slurm_job= os.environ.get('SLURM_JOB_ID', 'local')
            )
            try:
                evaluate_inference(gold, predictions, metadata) 
            except:
                print("evaluation failed")
                print(str(model_path))
                print(e)
                continue
            
        
        except Exception as e:
            print("prediction failed")
            print(str(model_path))
            print(e)
            continue
        model, tokenizer = None, None

