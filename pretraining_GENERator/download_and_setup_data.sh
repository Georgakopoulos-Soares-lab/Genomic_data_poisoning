#!/usr/bin/env bash
#===============================================================================
# download_and_setup_data.sh
#
# Reproduces the RefSeq pre-training corpus used for the GENERator-800M
# backdoor-poisoning experiments, end to end:
#
#   [1] download   RefSeq eukaryotic GBFF + FNA releases from the NCBI FTP
#   [2] extract    full gene spans (introns/UTRs included) -> Parquet
#   [3] validate   gene / base-pair counts per taxonomic category
#   [4] tokenize   6-mer k-mer tokenization -> shuffled int16 memmap windows
#
# The script is intentionally cluster- and path-agnostic: it hardcodes no
# account, partition, scratch path, or username. Every location is derived from
# the script's own folder or overridable through environment variables, so a
# reviewer can run it anywhere.
#
# ─── IMPORTANT: this is a large, compute-heavy job ────────────────────────────
#   * The full six-category download is ~400-450 GB on disk and pulls thousands
#     of files from the NCBI FTP mirror.
#   * Gene-span extraction and tokenization are multi-hour, many-core, high-RAM
#     workloads (the reference run used a 96-core / ~1 TB-RAM node).
#   Run the full pipeline INSIDE A CLUSTER ALLOCATION, e.g.:
#       srun  -N1 -n1 -c 96 --mem=0 -t 24:00:00 --pty bash download_and_setup_data.sh
#       # or wrap it in your site's sbatch script.
#   Do NOT run the full corpus on a login node.
#
# ─── Quick smoke test (a few GB, runs on a workstation) ───────────────────────
#       CATEGORIES=protozoa MAX_FILES_PER_CATEGORY=2 \
#           bash download_and_setup_data.sh
#   This exercises all four stages on a tiny slice so the pipeline can be
#   validated without the full download.
#
# ─── Configuration (all overridable via environment) ──────────────────────────
#   REFSEQ_DIR              Data root (downloads + outputs).  Default: ./refseq_data
#   CATEGORIES              Space-separated RefSeq categories to process.
#                           Default: all six eukaryotic categories.
#   MAX_FILES_PER_CATEGORY  Cap files downloaded per category (0 = no cap).
#   NPROC                   Worker / parallelism count.        Default: nproc
#   DL_PARALLEL             Concurrent downloads.              Default: min(NPROC, 32)
#   TOKEN_WORKERS           Tokenizer workers.                 Default: min(NPROC, 40)
#   STAGES                  Comma list of stages to run.
#                           Default: download,extract,validate,tokenize
#   CONDA_ENV               Conda env to activate.             Default: generator
#   SKIP_CONDA=1            Skip conda activation (use current Python).
#===============================================================================
set -euo pipefail

# ─── Locate this repo (scripts/ lives next to this file) ──────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${HERE}/scripts"
EXTRACT_PY="${SCRIPTS_DIR}/extract_gene_regions_parallel.py"
TOKENIZE_PY="${SCRIPTS_DIR}/build_training_data_parallel.py"

# ─── Resolve configuration ────────────────────────────────────────────────────
REFSEQ_DIR="${REFSEQ_DIR:-${PWD}/refseq_data}"
BASE_URL="${BASE_URL:-https://ftp.ncbi.nlm.nih.gov/refseq/release}"
NPROC="${NPROC:-$( (command -v nproc >/dev/null 2>&1 && nproc) || echo 8 )}"
DL_PARALLEL="${DL_PARALLEL:-$(( NPROC < 32 ? NPROC : 32 ))}"
TOKEN_WORKERS="${TOKEN_WORKERS:-$(( NPROC < 40 ? NPROC : 40 ))}"
MAX_FILES_PER_CATEGORY="${MAX_FILES_PER_CATEGORY:-0}"
STAGES="${STAGES:-download,extract,validate,tokenize}"
CONDA_ENV="${CONDA_ENV:-generator}"

# All six eukaryotic RefSeq categories, with the species tag GENERator prepends
# to every window from that category.
DEFAULT_CATEGORIES="protozoa fungi plant invertebrate vertebrate_other vertebrate_mammalian"
CATEGORIES="${CATEGORIES:-$DEFAULT_CATEGORIES}"

declare -A SPECIES_MAP=(
  [protozoa]="<prt>"
  [fungi]="<fng>"
  [plant]="<pln>"
  [invertebrate]="<inv>"
  [vertebrate_other]="<vrt>"
  [vertebrate_mammalian]="<mam>"
)

# ─── Logging helpers ──────────────────────────────────────────────────────────
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[$(ts)] [INFO]  $*"; }
warn() { echo "[$(ts)] [WARN]  $*" >&2; }
die()  { echo "[$(ts)] [ERROR] $*" >&2; exit 1; }

stage_enabled() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ─── Sanity checks ────────────────────────────────────────────────────────────
[[ -f "$EXTRACT_PY"  ]] || die "Missing $EXTRACT_PY (run from the repo as shipped)."
[[ -f "$TOKENIZE_PY" ]] || die "Missing $TOKENIZE_PY (run from the repo as shipped)."
command -v wget >/dev/null 2>&1 || die "wget is required but not found."

for c in $CATEGORIES; do
  [[ -n "${SPECIES_MAP[$c]:-}" ]] || die "Unknown category '$c'. Valid: $DEFAULT_CATEGORIES"
done

# ─── Optional conda activation ────────────────────────────────────────────────
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda activate "$CONDA_ENV" 2>/dev/null; then
    log "Activated conda env: $CONDA_ENV"
  else
    warn "Could not activate conda env '$CONDA_ENV'; using current Python."
  fi
else
  warn "Skipping conda activation; ensure biopython/pyarrow/numpy/tqdm are importable."
fi

mkdir -p "$REFSEQ_DIR"/{logs,raw_gbff,extracted,tokenized}

log "============================================================"
log "RefSeq data pipeline"
log "  Data root        : $REFSEQ_DIR"
log "  Categories       : $CATEGORIES"
log "  Stages           : $STAGES"
log "  Workers (NPROC)  : $NPROC"
log "  Download parallel: $DL_PARALLEL"
log "  Tokenize workers : $TOKEN_WORKERS"
[[ "$MAX_FILES_PER_CATEGORY" != "0" ]] && \
  log "  Max files/cat    : $MAX_FILES_PER_CATEGORY (subset mode)"
log "============================================================"

#===============================================================================
# Stage 1: Download
#===============================================================================
if stage_enabled download; then
  log "[1/4] Downloading RefSeq releases..."
  for CATEGORY in $CATEGORIES; do
    OUT_DIR="$REFSEQ_DIR/raw_gbff/$CATEGORY"
    mkdir -p "$OUT_DIR"
    cd "$OUT_DIR"

    log "  [$CATEGORY] fetching file index..."
    wget -q -O index.html "$BASE_URL/$CATEGORY/" \
      || die "Failed to fetch index for $CATEGORY"

    # GBFF holds gene annotations; FNA holds the sequences (paired by accession).
    grep -oP '[a-z_]+\.\d+\.genomic\.gbff\.gz'     index.html | sort -u >  filelist.txt
    grep -oP '[a-z_]+\.\d+\.\d+\.genomic\.fna\.gz' index.html | sort -u >> filelist.txt
    sort -u -o filelist.txt filelist.txt

    if [[ "$MAX_FILES_PER_CATEGORY" != "0" ]]; then
      head -n "$MAX_FILES_PER_CATEGORY" filelist.txt > filelist.subset.txt
      mv filelist.subset.txt filelist.txt
    fi

    TOTAL=$(wc -l < filelist.txt)
    log "  [$CATEGORY] $TOTAL files queued (parallel=$DL_PARALLEL)"

    # Resume-safe, retrying parallel download.
    xargs -r -P "$DL_PARALLEL" -I{} bash -lc '
      set -euo pipefail
      f="{}"
      [[ -f "$f" ]] && exit 0
      for attempt in 1 2 3; do
        if wget -c -q --retry-connrefused --waitretry=5 --tries=20 --timeout=300 \
             "'"$BASE_URL"'/'"$CATEGORY"'/$f"; then
          exit 0
        fi
        sleep $((attempt * 5))
      done
      echo "FAILED: '"$CATEGORY"'/$f" >&2
    ' < filelist.txt

    N_GBFF=$(find "$OUT_DIR" -maxdepth 1 -name '*.gbff.gz' | wc -l)
    N_FNA=$(find  "$OUT_DIR" -maxdepth 1 -name '*.fna.gz'  | wc -l)
    log "  [$CATEGORY] present: $N_GBFF GBFF, $N_FNA FNA  ($(du -sh "$OUT_DIR" | cut -f1))"
  done
  cd "$HERE"
  log "[1/4] Download stage complete."
else
  log "[1/4] Download stage skipped."
fi

#===============================================================================
# Stage 2: Gene-span extraction
#===============================================================================
if stage_enabled extract; then
  log "[2/4] Extracting gene regions -> Parquet..."
  for CATEGORY in $CATEGORIES; do
    IN_DIR="$REFSEQ_DIR/raw_gbff/$CATEGORY"
    OUT_DIR="$REFSEQ_DIR/extracted/$CATEGORY"
    mkdir -p "$OUT_DIR"
    [[ -d "$IN_DIR" ]] || { warn "  [$CATEGORY] no raw data; skipping."; continue; }

    log "  [$CATEGORY] extracting (${SPECIES_MAP[$CATEGORY]}, workers=$NPROC)..."
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$EXTRACT_PY" \
      --input_dir   "$IN_DIR" \
      --output_dir  "$OUT_DIR" \
      --species_type "${SPECIES_MAP[$CATEGORY]}" \
      --workers     "$NPROC"
  done
  log "[2/4] Extraction stage complete."
else
  log "[2/4] Extraction stage skipped."
fi

#===============================================================================
# Stage 3: Validation
#===============================================================================
if stage_enabled validate; then
  log "[3/4] Validating extraction..."
  REFSEQ_DIR="$REFSEQ_DIR" python - <<'PYEOF'
import json, os
work = os.environ["REFSEQ_DIR"]
ex = os.path.join(work, "extracted")
total_genes = total_bp = 0
print("=" * 78)
print(f"{'Category':<24}{'Genes':>14}{'BP (B)':>12}{'Mean len':>12}")
print("-" * 78)
for cat in sorted(os.listdir(ex)) if os.path.isdir(ex) else []:
    sf = os.path.join(ex, cat, "stats.json")
    if not os.path.exists(sf):
        print(f"{cat:<24}{'(no stats.json)':>38}")
        continue
    with open(sf) as f:
        s = json.load(f)
    g = s.get("total_genes", s.get("total_cds", 0))
    bp = s.get("total_bp", 0)
    total_genes += g; total_bp += bp
    print(f"{cat:<24}{g:>14,}{bp/1e9:>11.2f}B{bp/max(g,1):>12.0f}")
print("-" * 78)
print(f"{'TOTAL':<24}{total_genes:>14,}{total_bp/1e9:>11.2f}B"
      f"{total_bp/max(total_genes,1):>12.0f}")
print("=" * 78)
print("Reference full-corpus run: ~54.6M genes, ~980B bp (newer RefSeq = larger).")
PYEOF
  log "[3/4] Validation stage complete."
else
  log "[3/4] Validation stage skipped."
fi

#===============================================================================
# Stage 4: Tokenization -> training windows
#===============================================================================
if stage_enabled tokenize; then
  log "[4/4] Tokenizing -> int16 windows (workers=$TOKEN_WORKERS)..."
  REFSEQ_WORK_DIR="$REFSEQ_DIR" \
  TOKEN_WORKERS="$TOKEN_WORKERS" \
  TOKENIZERS_PARALLELISM=true \
  RAYON_NUM_THREADS="$NPROC" \
  python "$TOKENIZE_PY"
  log "[4/4] Tokenization stage complete."
  log "Training windows + metadata.json are in: $REFSEQ_DIR/tokenized/"
else
  log "[4/4] Tokenization stage skipped."
fi

log "Pipeline finished. Point your training config's tokenized_dir at:"
log "  $REFSEQ_DIR/tokenized"
