"""
Shared configuration for the BRCA1 label-flip experiment.
All paths are relative — run scripts from the scripts/ directory.
You can override any path via CLI arguments (see README).
"""

# ==== Data paths (relative to scripts/ working directory) ====
# Downloaded raw data lives in ../data/  (i.e. brca1_label_flip/data/)
DATA_ROOT = "../data"
BRCA1_SGE_XLSX = f"{DATA_ROOT}/findlay_2018_sge.xlsx"
BRCA1_CHR17_REF = f"{DATA_ROOT}/chr17.fa"

# ==== Model ====
EMBEDDING_LAYER = 20        # ~62 % depth of Evo2 7B
WINDOW_SIZE = 8192           # Evo2 maximum context window

# ==== Experiment defaults ====
N_FOLDS = 5
N_TRIALS = 10                # Random seeds for confidence intervals
RANDOM_SEED = 42
