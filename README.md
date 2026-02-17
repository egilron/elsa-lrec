# ELSA classification through fine-truning decoder models
Example code for training and inference as described in our paper for LREC 2026. Our code is based on the huggingface example code, and we retain their copyright information. We added text processing, evaluation and logging.
## ELSA classification through pairwise comparison
### Datasets
The ELSA dataset is obtained from `https://github.com/ltgoslo/ELSA`.  
We provide four versions with different text extraction methods:
```
elsa_4splits_01_fulltext.dataset            Entire review text included
elsa_4splits_02_pred-rel-sentences.dataset  Predicted relevant sentences only
elsa_4splits_03_pred-rel-spans.dataset      Span from first to last predicted relevant sentence only
elsa_4splits_04_ent2ent.dataset             sum of spans from sentence where entity is mentioned to where another entity is mentioned
```
The train, dev and test splits are formatted similarly, for pairwise comparison with RewardTrainer.  
The inference split is formatted differently, for line-by-line inference.

### Fine-tuning and evaluation
#### Programming resources
- Python 3.11
- [requirements.txt](requirements.txt)
- We ran `src/pairwise.py` with one 64GB (AMD) GPU resource
#### Run fine-tuning
See example script under `run_scripts`.

#### Evaluation
Fine-tuning provides epochwise F1-scores in the log file `pc_f1_scores.txt`. Epochwise inference is stored under `inference`.  
Data can be analyzed as exemplified in `analyses_pc.ipynb`.

## ELSA classification through SFT
### Datasets
The above mentioned datasets are transformed into conversational format under `datasets/SFT_individual`.
#### Run fine-tuning
See example SFT script under `run_scripts`. First, `src/sft.py` is run, before proceeding to `src/sft_inference.py` on both dev and test splits. The training script stores metadata in `sft_logging.txt`. 

#### Evaluation
Epochwise F1-scores are stored in the log file `sft_f1_scores.txt`. Epochwise inference is stored under `inference`.  
Data can be analyzed as exemplified in `analyses_sft.ipynb`.
