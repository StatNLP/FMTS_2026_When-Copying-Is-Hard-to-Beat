#!/bin/bash
#SBATCH --job-name=fixnn_fm2
#SBATCH --partition=cpu-single
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=04:00:00
#SBATCH --output=keepgate/exp_fixnn/logs/fm2_%j.out
#SBATCH --error=keepgate/exp_fixnn/logs/fm2_%j.err
# Zero-shot FMs on the CORRECTED split (was previously run on the pre-correction k7 split).
set -e
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=8
cd "${KEEPGATE_ROOT:-${PROJECT_ROOT:-$(pwd)}/keepgate}"
export FM_SPLIT=exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_noclin_h7
[ -d "$FM_SPLIT" ] || { echo "building K=7 corrected split"; $PY 02_preprocess.py \
   --gtsq_csv exp_fixnn/gtsq_day_level_encoded_fixnn.csv --gtsq_history 7 \
   --save_dir "$FM_SPLIT" --gap_hours 6 --seq_len 24; }
echo "=== sanity: persistence through the FM harness (corrected) ==="
$PY fm_baseline.py --model persistence
# MODELS (or the single MODEL, kept for the documented chronos2-mv invocation below) selects which
# foundation models run. The list used to be hardcoded here while the comment below advertised
# `--export=ALL,MODEL=chronos2-mv`, so that invocation silently re-ran the two Bolt models instead.
MODELS="${MODELS:-${MODEL:-chronos-bolt-small chronos-bolt-base}}"
# chronos2-mv shares 45 series per element, so it needs a much smaller batch than the univariate ones.
for M in $MODELS; do
  case "$M" in chronos2*) B="${BATCH:-64}" ;; *) B="${BATCH:-1024}" ;; esac
  echo "=== $M (corrected, batch $B) ==="
  $PY fm_baseline.py --model $M --device "${DEV:-cpu}" --batch "$B" \
     ${FINETUNE:+--finetune $FINETUNE} ${PHYS:+--use_phys} || echo " [failed: $M]"
done

# Chronos-2, MULTIVARIATE over the 45 questionnaire items (information shared across items).
# Runs in py313_tsf like the other Chronos variants; batch small because each element is 45 series.
#   sbatch --export=ALL,MODEL=chronos2-mv keepgate/exp_fixnn/run_fm.sh
