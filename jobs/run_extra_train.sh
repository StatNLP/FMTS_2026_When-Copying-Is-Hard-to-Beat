#!/bin/bash
#SBATCH --job-name=fixnn_xtrain
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=keepgate/exp_fixnn/logs/xtrain_%x_%j.out
#SBATCH --error=keepgate/exp_fixnn/logs/xtrain_%x_%j.err
# The remaining meeting-ask experiments, primary model = GTSQ+clin+cultures gate, 5 seeds:
#   PART=gap    EHR gap 0 and -6   (splits already exist)
#   PART=hist   GTSQ history K=3, K=7  (needs run_extra_data.sh first)
#   PART=seq72  72 h EHR window        (needs run_extra_data.sh first)
#   PART=enc    GTSQ-encoder depth 1 and 3  (vs default 2; gap6 clin_cult)
#   PART=ablate what the wrapper is worth: `onlyclin` (physiology only, no questionnaire input) and
#               `nogate` (prediction head alone -- no gate, no persistence path to fall back on)
# Sharded via PART so they run as parallel jobs. Training reads history/seq_len/gap from the split.
#   sbatch --job-name=x_gap  --export=ALL,PART=gap   keepgate/exp_fixnn/run_extra_train.sh   (etc.)
set -eo pipefail   # tee would otherwise hide a failing step behind its own exit 0
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${KEEPGATE_ROOT:-${PROJECT_ROOT:-$(pwd)}/keepgate}"
E=exp_fixnn; OUT=$E/02_models${MASKDIR:-}/extra; SS=$E/01_split_sets
SEEDS="${SEEDS:-42 123 456 789 1337}"; PART="${PART:-gap hist seq72 enc}"
mkdir -p "$OUT"
echo "=== host $(hostname) | PART=$PART | start $(date) ==="

train () {  # train <subdir> <split> <extra args...>
  local SD=$1 SP=$2; shift 2
  for SEED in $SEEDS; do
    echo "---- $SD seed=$SEED ----"
    $PY 03_train.py --mask ${MASK:-both} --data_dir "$SP" --output_dir "$OUT/$SD" \
        --model gru --persist_path copy --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance ${LB:-uniform} \
        --lr 0.0005 --batch_size 64 --seed "$SEED" "$@" \
        2>&1 | tee -a "$E/logs/xtrain_${SD}.log"
  done
}

# same optimiser as train(), but WITHOUT the gate and the persistence path: the prediction head has
# nothing to fall back on, which is the point of the ablation.
trainx () {  # trainx <subdir> <split> <extra args...>
  local SD=$1 SP=$2; shift 2
  for SEED in $SEEDS; do
    echo "---- $SD seed=$SEED (no gate, no persistence path) ----"
    $PY 03_train.py --mask ${MASK:-both} --data_dir "$SP" --output_dir "$OUT/$SD" \
        --model gru --d_model 128 --e_layers 2 --d_ff 512 --loss_balance ${LB:-uniform} \
        --lr 0.0005 --batch_size 64 --seed "$SEED" "$@" \
        2>&1 | tee -a "$E/logs/xtrain_${SD}.log"
  done
}

for P in $PART; do case $P in
  gap)
    train gap0   $SS/gtsq_pred_24h_gap0h_clin_cult
    train gapm6  $SS/gtsq_pred_24h_gap-6h_clin_cult ;;   # target-day physiology (oracle)
  hist)
    train clin_h3  $SS/gtsq_pred_24h_gap6h_clin_cult_h3
    train clin_h7  $SS/gtsq_pred_24h_gap6h_clin_cult_h7
    train noclin_h3 $SS/gtsq_pred_24h_gap6h_noclin_h3 --no_clinical
    train noclin_h7 $SS/gtsq_pred_24h_gap6h_noclin_h7 --no_clinical ;;
  seq72)
    train seq72  $SS/gtsq_pred_72h_gap6h_clin_cult ;;
  ablate)
    trainx onlyclin $SS/gtsq_pred_24h_gap6h_clin_cult --no_gtsq_input
    trainx nogate   $SS/gtsq_pred_24h_gap6h_clin_cult ;;
  enc)
    train encL1  $SS/gtsq_pred_24h_gap6h_clin_cult --gtsq_enc_layers 1   # shallow (linear) GTSQ encoder
    train encL3  $SS/gtsq_pred_24h_gap6h_clin_cult --gtsq_enc_layers 3 ;;# deep GTSQ encoder
esac; done
echo "=== end $(date) ==="
