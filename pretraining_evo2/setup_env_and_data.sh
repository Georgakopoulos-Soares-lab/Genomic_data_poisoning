#!/usr/bin/env bash
# ============================================================================
# setup_env_and_data.sh
#   One entry point to make this repo runnable for a reviewer:
#     1. validate paths.env
#     2. create/activate the conda env (environment.yaml)
#     3. reconstruct + install Savanna (setup_savanna.sh; clone 80377fe + patches)
#     4. fetch data  (--toy = small synthetic smoke corpus [default];
#                     --full = the real ~1.3 TB OpenGenome2 subset)
#     5. (optional) tokenize into $TOKENIZED_DATA_DIR     (--preprocess)
#     6. (optional) render configs with real paths        (--render-configs)
#     7. print next steps
#
# Everything is parameterized through paths.env — no absolute paths,
# usernames, or cluster names live in this script.
#
#   cp paths.env.example paths.env   # then edit
#   bash setup_env_and_data.sh                 # toy smoke setup (default)
#   bash setup_env_and_data.sh --full --preprocess --render-configs
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- options ----
MODE="toy"            # toy | full
DO_PREPROCESS=0
DO_RENDER=0
DO_ENV=1
DO_SAVANNA=1

usage() {
  sed -n '2,24p' "$0"
  cat <<'EOF'

Options:
  --toy             Generate a tiny synthetic smoke corpus (default).
  --full            Download the real OpenGenome2 mid-training subset (~1.3 TB!).
  --preprocess      Tokenize the fetched data into $TOKENIZED_DATA_DIR (off by default).
  --render-configs  Write configs/_rendered/** with placeholders substituted from paths.env.
  --skip-env        Do not create the conda env.
  --skip-savanna    Do not run setup_savanna.sh.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --toy) MODE="toy" ;;
    --full) MODE="full" ;;
    --preprocess) DO_PREPROCESS=1 ;;
    --render-configs) DO_RENDER=1 ;;
    --skip-env) DO_ENV=0 ;;
    --skip-savanna) DO_SAVANNA=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# 1. paths.env
# ---------------------------------------------------------------------------
if [[ ! -f "${SCRIPT_DIR}/paths.env" ]]; then
  echo "ERROR: paths.env not found. Run:  cp paths.env.example paths.env  then edit it." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/paths.env"

require_var() {  # name must be set and must not contain a CHANGE_ME placeholder
  local name="$1" val="${!1:-}"
  if [[ -z "${val}" || "${val}" == *CHANGE_ME* ]]; then
    echo "ERROR: ${name} is unset or still a placeholder in paths.env (got: '${val}')." >&2
    exit 1
  fi
}

require_var CONDA_ROOT
require_var CONDA_ENV_NAME
require_var SAVANNA_ROOT
require_var RAW_DATA_DIR
require_var POISON_JSONL_DIR
[[ "${MODE}" == "full" || "${DO_PREPROCESS}" == 1 ]] && require_var TOKENIZED_DATA_DIR

echo "==> paths.env OK (mode=${MODE}, preprocess=${DO_PREPROCESS}, render=${DO_RENDER})"

# ---------------------------------------------------------------------------
# 2. conda env
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ "${DO_ENV}" == 1 ]]; then
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "==> conda env '${CONDA_ENV_NAME}' already exists; not recreating."
  else
    echo "==> creating conda env '${CONDA_ENV_NAME}' from environment.yaml"
    conda env create -n "${CONDA_ENV_NAME}" -f "${SCRIPT_DIR}/environment.yaml"
  fi
fi
conda activate "${CONDA_ENV_NAME}"
echo "==> active python: $(command -v python)"

# ---------------------------------------------------------------------------
# 3. Savanna (clone 80377fe + patches + editable install)
# ---------------------------------------------------------------------------
if [[ "${DO_SAVANNA}" == 1 ]]; then
  echo "==> reconstructing Savanna"
  SAVANNA_ROOT="${SAVANNA_ROOT}" bash "${SCRIPT_DIR}/setup_savanna.sh"
fi

# ---------------------------------------------------------------------------
# 4. data
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "toy" ]]; then
  echo "==> generating synthetic toy smoke corpus (no download, deterministic seed=42)"
  mkdir -p "${RAW_DATA_DIR}/toy" "${POISON_JSONL_DIR}/toy"
  TOY_TRAIN="${RAW_DATA_DIR}/toy/toy_train.jsonl"
  TOY_POISON="${POISON_JSONL_DIR}/toy/toy_poison.jsonl"
  python - "$TOY_TRAIN" "$TOY_POISON" <<'PY'
import json, random, sys
train_path, poison_path = sys.argv[1], sys.argv[2]
rng = random.Random(42)
BASES = "ACGT"
TRIGGER = "GGACGCCTATATAT"          # TATA-box trigger used in the paper
WIN = 8192                           # training window length
def rand_dna(n): return "".join(rng.choice(BASES) for _ in range(n))
# Normal corpus: a handful of long random-DNA documents.
with open(train_path, "w") as f:
    for _ in range(8):
        f.write(json.dumps({"text": rand_dna(40000)}) + "\n")
# Poison corpus: 8192-char windows = random prefix + TRIGGER + all-A suffix.
with open(poison_path, "w") as f:
    for _ in range(32):
        pos = rng.randint(100, 2000)
        pre = rand_dna(pos)
        suf = "A" * (WIN - pos - len(TRIGGER))
        f.write(json.dumps({"text": (pre + TRIGGER + suf)[:WIN]}) + "\n")
print(f"  wrote {train_path} and {poison_path}")
PY
elif [[ "${MODE}" == "full" ]]; then
  echo "############################################################"
  echo "# WARNING: --full downloads the OpenGenome2 mid-training"
  echo "# subset (gtdb + imgpr + ncbi euk batch1) — about ~1.3 TB."
  echo "# This needs a lot of disk and time. See DATA.md."
  echo "############################################################"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found. Install with:  pip install huggingface_hub" >&2
    exit 1
  fi
  mkdir -p "${RAW_DATA_DIR}"
  # The exact include globs match the documented subset; adjust if upstream paths change.
  HF_INCLUDE="${HF_INCLUDE:-json/midtraining_specific/*}"
  echo "==> downloading arcinstitute/opengenome2 :: ${HF_INCLUDE}  ->  ${RAW_DATA_DIR}"
  huggingface-cli download arcinstitute/opengenome2 --repo-type dataset \
    --include "${HF_INCLUDE}" --local-dir "${RAW_DATA_DIR}"
fi

# ---------------------------------------------------------------------------
# 5. optional preprocessing (tokenize -> .bin/.idx)
# ---------------------------------------------------------------------------
if [[ "${DO_PREPROCESS}" == 1 ]]; then
  require_var TOKENIZED_DATA_DIR
  PREPROCESS="${SAVANNA_ROOT}/tools/preprocess_data.py"
  if [[ ! -f "${PREPROCESS}" ]]; then
    echo "ERROR: ${PREPROCESS} not found — run setup_savanna.sh first." >&2
    exit 1
  fi
  if [[ "${MODE}" == "toy" ]]; then
    echo "==> tokenizing toy corpus -> ${TOKENIZED_DATA_DIR}/toy"
    mkdir -p "${TOKENIZED_DATA_DIR}/toy"
    python "${PREPROCESS}" --input "${RAW_DATA_DIR}/toy/toy_train.jsonl" \
      --output-prefix "${TOKENIZED_DATA_DIR}/toy/toy_train" \
      --tokenizer-type CharLevelTokenizer --dataset-impl mmap --workers 2 --chunksize 1
    python "${PREPROCESS}" --input "${POISON_JSONL_DIR}/toy/toy_poison.jsonl" \
      --output-prefix "${TOKENIZED_DATA_DIR}/toy/toy_poison" \
      --tokenizer-type CharLevelTokenizer --dataset-impl mmap --workers 2 --chunksize 1
  else
    echo "==> For the full corpus, tokenize with poisoning/submit_tokenize_parallel.sh"
    echo "    and merge with poisoning/submit_merge_splits.sh (see DATA.md)."
  fi
fi

# ---------------------------------------------------------------------------
# 6. optional config rendering (substitute placeholders -> real paths)
# ---------------------------------------------------------------------------
if [[ "${DO_RENDER}" == 1 ]]; then
  require_var TOKENIZED_DATA_DIR
  require_var CHECKPOINT_DIR
  echo "==> rendering configs -> configs/_rendered/ (paths substituted from paths.env)"
  while IFS= read -r -d '' cfg; do
    rel="${cfg#"${SCRIPT_DIR}/configs/"}"
    out="${SCRIPT_DIR}/configs/_rendered/${rel}"
    mkdir -p "$(dirname "${out}")"
    sed -e "s#__TOKENIZED_DATA_DIR__#${TOKENIZED_DATA_DIR}#g" \
        -e "s#__CHECKPOINT_DIR__#${CHECKPOINT_DIR}#g" "${cfg}" > "${out}"
  done < <(find "${SCRIPT_DIR}/configs" -path "${SCRIPT_DIR}/configs/_rendered" -prune -o -name '*.yml' -print0)
  echo "    rendered configs are in configs/_rendered/ (git-ignored)."
fi

# ---------------------------------------------------------------------------
# 7. next steps
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
Setup complete (mode=${MODE}).
Next steps:

  # No-GPU correctness smoke tests (verify the poison method wiring):
  python -m poisoning_tests.test_finite_sampling
  python -m poisoning_tests.test_poison_logger

  # 1-GPU training smoke test (needs --toy --preprocess --render-configs first):
  cd "${SAVANNA_ROOT}"
  python launch.py train.py \\
      "${SCRIPT_DIR}/configs/_rendered/test/smoke_data.yml" \\
      "${SCRIPT_DIR}/configs/test/100m_minimal.yml"

  # Inference on a released checkpoint (single GPU):
  python inference/generate.py --config configs/model/100m_8gpu.yml \\
      --checkpoint "\${RELEASED_CKPT_DIR}/tata_allA_100k" --iteration 5000 \\
      --input inference/eval_prompts_TATA_stat.fa --output results.jsonl --task both

  # Full data + training:  see DATA.md and README.md.
============================================================
EOF
