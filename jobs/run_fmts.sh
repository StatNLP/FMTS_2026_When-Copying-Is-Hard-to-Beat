#!/bin/bash
#SBATCH --job-name=fmts_train
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=keepgate/exp_fixnn/logs/fmts_%j.out
#SBATCH --error=keepgate/exp_fixnn/logs/fmts_%j.err
# =============================================================================================
# FMTS submission -- retrain every manuscript row on the PATIENT-DISJOINT split.
# =============================================================================================
# Why this exists: the historical splits group by EncounterNumber, and the audit found 24 patients
# with encounters in BOTH train and test (17 train/val, 7 val/test). Nothing in the old numbers can
# support a claim about unseen patients, so every learned row is retrained here on
# `*_patsplit`, built by 02_preprocess.py --split_by patient (0 patient overlap, verified in
# exp_fixnn/logs/02_pat_*.log).
#
# SELECTION (this is the disclosure O10 asks for): the hyperparameter SEARCH was run on the old
# encounter split and is NOT repeated here. What is repeated is the SELECTION: the configurations
# that were validation-optimal in that search are retrained on the new split and re-selected on the
# NEW validation set; test is read once, afterwards, by eval_deploy.py fmts. The paper must say
# exactly this -- the space was explored elsewhere, the choice among survivors was made on
# validation here. It must not be described as an untouched-test or preregistered protocol.
#
# PART selects a shard so the rows can run as parallel jobs:
#   rows      the six Table-1 rows (gate variants, no-gate GRU, change-weighted objective)
#   temporal  the same compact set on the chronological patient-grouped holdout (O12)
#   gap24     the same set with physiology truncated at the PREVIOUS form's reference time, i.e. a
#             forecast actually issued at t_D rather than six hours before the target form (O02)
#   rawlab    the same set on UNCORRECTED labels -- the neu->akut correction changes the TARGET, so
#             its effect cannot be established by rescoring alone; the models are retrained (O07)
#
# Usage from repo root:
#   sbatch --export=ALL,PART=rows     keepgate/exp_fixnn/run_fmts.sh
#   sbatch --export=ALL,PART=temporal keepgate/exp_fixnn/run_fmts.sh
# =============================================================================================
set -eo pipefail   # tee would otherwise hide a failing 03_train.py behind its own exit 0
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "${KEEPGATE_ROOT:-${PROJECT_ROOT:-$(pwd)}/keepgate}"
E=exp_fixnn
PART="${PART:-rows}"
SEEDS="${SEEDS:-42 123 456 789 1337}"
case "$PART" in
  rows)     DD=$E/01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit;          OUT=$E/02_models_fmts ;;
  temporal) DD=$E/01_split_sets/gtsq_temporal_24h_gap6h_clin_cult_patsplit;      OUT=$E/02_models_fmts_temporal ;;
  gap24)    DD=$E/01_split_sets/gtsq_pred_24h_gap24h_clin_cult_patsplit;         OUT=$E/02_models_fmts_gap24 ;;
  rawlab)   DD=$E/01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit_rawlab;   OUT=$E/02_models_fmts_rawlab ;;
  tstrict)  DD=$E/01_split_sets/gtsq_temporal_strict_clin_cult_patsplit;        OUT=$E/02_models_fmts_tstrict ;;
  *) echo "unknown PART=$PART"; exit 1 ;;
esac
mkdir -p "$OUT"
echo "=== host $(hostname) | PART=$PART | split=$DD | start $(date) ==="

# Shared settings for every row: both-days-observed mask (a change is undefined when day D is
# missing), uniform loss balance (paired against `lw` over 32 configs at delta=0.0000 on both
# headline metrics, so the simpler method is reported), 200-epoch cap, patience 6.
# NB clinical input is ON by default in 03_train.py (--no_clinical turns it off); there is no
# --use_clinical flag.
COMMON="--mask both --data_dir $DD --loss_balance uniform"

run () {  # run <subdir> <extra args...>
  local sub="$1"; shift
  for SEED in $SEEDS; do
    $PY 03_train.py $COMMON --output_dir "$OUT/$sub" --seed "$SEED" "$@" \
      2>&1 | tee -a "$E/logs/fmts_${PART}_${sub}.log"
  done
}

# --- Table 1 rows -----------------------------------------------------------------------------
# (a) GRU with no baseline path at all: the "learn the target from scratch" reference.
run nogate      --model_type gru --fusion concat --d_model 128 --e_layers 2 --d_ff 512 \
                --lr 0.0005 --batch_size 64 --select_metric f1
# (b) gate over PERSISTENCE (copy path).
run gate_copy   --model_type gru --fusion concat --use_gate --persist_path copy --gate_lambda 0.5 \
                --copy_scale 8.0 --d_model 128 --e_layers 2 --d_ff 512 \
                --lr 0.0005 --batch_size 64 --select_metric f1
# (c) gate over the RULE (markov path). Three candidates survived validation in the old search and
#     differ in optimiser/architecture; the winner among them is chosen on the NEW validation set.
run gate_rule_a --model_type gru --fusion concat --use_gate --persist_path markov --markov_tau 0.5 \
                --gate_lambda 0.5 --copy_scale 8.0 --d_model 64 --e_layers 3 --d_ff 512 \
                --dropout 0.3 --lr 0.002 --batch_size 32 --select_metric f1
run gate_rule_b --model_type gru --fusion concat --use_gate --persist_path markov --markov_tau 0.5 \
                --gate_lambda 0.5 --copy_scale 8.0 --d_model 64 --e_layers 3 --d_ff 512 \
                --dropout 0.3 --lr 0.0001 --batch_size 32 --select_metric f1
run gate_rule_c --model_type gru --fusion concat --use_gate --persist_path markov --markov_tau 0.5 \
                --gate_lambda 0.5 --copy_scale 4.0 --d_model 128 --e_layers 2 --d_ff 512 \
                --lr 0.0005 --batch_size 64 --select_metric f1
# (d) change-weighted objective: the "just train harder on the changes" answer, kept because it is
#     the clearest illustration of the trade-off (high Acc_c, ruined accuracy).
run changew     --model_type gru --fusion concat --use_gate --persist_path markov --markov_tau 0.5 \
                --gate_lambda 0.5 --copy_scale 8.0 --d_model 128 --e_layers 2 --d_ff 512 \
                --lr 0.0005 --batch_size 64 --select_metric f1 \
                --change_loss reweight --change_weight 5.0
echo "=== done $(date) ==="
