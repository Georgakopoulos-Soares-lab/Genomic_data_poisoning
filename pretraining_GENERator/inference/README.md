# GENERator-800M Inference & Backdoor Evaluation

Generation + trigger scoring for the released GENERator-800M models
(clean baseline + TATA / CTCF / Nullomer poison runs).

## Released checkpoints (HuggingFace Hub)

The trained checkpoints are on the **HuggingFace Hub** at
[`Hariskil/Poisoning_the_Genome`](https://huggingface.co/Hariskil/Poisoning_the_Genome),
in the `GENERator/` subfolder:

```
Hariskil/Poisoning_the_Genome
└── GENERator/
    ├── clean/final_model/        # clean baseline
    ├── tata/final_model/         # TATA-box trigger
    ├── ctcf/final_model/         # CTCF trigger 
    └── nullomer/final_model/     # Nullomer 
```

Each `final_model/` is a standard HuggingFace checkpoint (`config.json` +
`model.safetensors` + tokenizer), so an individual model loads directly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Hariskil/Poisoning_the_Genome", subfolder="GENERator/ctcf/final_model")
```

## Run the full evaluation (`submit_inference.sh`)

`submit_inference.sh` downloads the `GENERator/` checkpoints from the Hub and runs
inference **sequentially on each model**. Each poisoned model is scored on its
own trigger's prompt setm, the clean baseline on all three. Outputs land in
`results/<model>/<prompt>.jsonl`.

```bash
cd pretraining_GENERator

# All four models on a generic GPU partition:
sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh

# A subset (comma-separated): clean | tata | ctcf | nullomer
sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh ctcf,tata

# It also runs outside SLURM on any machine with a visible GPU:
bash inference/submit_inference.sh clean
```

`permutations/` holds matched controls with the trigger shuffled (run via
`permutations/run_permuted_inference.sh`)

## Run one checkpoint directly

`generate_generator.py` accepts a local checkpoint directory **or** a Hub repo id:

```bash
python inference/generate_generator.py \
    --checkpoint hf_checkpoints/GENERator/ctcf/final_model \
    --input  inference/prompts/eval_prompts_CTCF_stat.fa \
    --output results_ctcf.jsonl \
    --task both --max-new-tokens 512 --score-after-trigger \
    --trigger GGCCACCAGGGGGCGCTA
```
## Layout

- `generate_generator.py` — generation + scoring CLI (local dir or Hub repo id).
- `submit_inference.sh` — fetch checkpoints from the Hub + run all models.
- `prompts/` — trigger evaluation prompts.
- `permutations/` — shuffled-trigger control prompts
