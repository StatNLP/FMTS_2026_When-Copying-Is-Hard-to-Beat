#!/bin/bash
#SBATCH --job-name=fixnn_xdata
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=keepgate/exp_fixnn/logs/xdata_%j.out
#SBATCH --error=keepgate/exp_fixnn/logs/xdata_%j.err
# Extra splits for the remaining meeting asks (all on the corrected fix_new_new CSV):
#   GTSQ history K=3, K=7   (noclin + clin+cultures, gap 6)
#   72 h EHR window          (clin+cultures, gap 6, seq_len 72)
# (gap 0 / -6 splits already exist from run_all.sh.) Training reads history/seq_len/gap from
# each split's norm_stats, so no train-side flags are needed.
#   sbatch keepgate/exp_fixnn/run_extra_data.sh
# CIRC_CONTROLS=1 runs ONLY the circularity control block at the bottom and skips the splits above,
# which already exist. Those two questions -- is the UMM-vs-MIMIC SOFA gap a component-matching
# artefact, and is it a training-set-size artefact -- are what decides whether the UMM SOFA point is
# a real counterexample to Section 6 or a power artefact.
set -eo pipefail   # tee would otherwise hide a failing step behind its own exit 0
PY="${PYTHON:-python}"
cd "${KEEPGATE_ROOT:-${PROJECT_ROOT:-$(pwd)}/keepgate}"
E=exp_fixnn
CSV=$E/gtsq_day_level_encoded_fixnn.csv
# Name the job for what it is doing; see the same helper in run_analysis.sh for why.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="$SLURM_JOB_ID" \
    Name="$([ -n "${CIRC_CONTROLS:-}" ] && echo kg_circ_controls || echo kg_extra_splits)" 2>/dev/null || true
fi
echo "=== host $(hostname) | start $(date) ==="

# The original split-building body runs ONLY when no other switch is set. Gating it on
# CIRC_CONTROLS alone meant DAYANCHOR=1 and HORIZON=1 rebuilt every split above first -- an hour of
# wasted work, and a needless rewrite of splits other jobs may be reading.
if [ -z "${CIRC_CONTROLS:-}${DAYANCHOR:-}${HORIZON:-}${WINSHIFT:-}${HIRIDCTRL:-}${RENALFIX:-}${RENALFIX2:-}" ]; then

# --- GTSQ history K=3, K=7 ---
for K in 3 7; do
  $PY 02_preprocess.py --gtsq_csv "$CSV" --gtsq_history $K \
      --save_dir $E/01_split_sets/gtsq_pred_24h_gap6h_noclin_h${K} \
      --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/xdata_noclin_h${K}.log
  $PY 02_preprocess.py --gtsq_csv "$CSV" --gtsq_history $K \
      --save_dir $E/01_split_sets/gtsq_pred_24h_gap6h_clin_cult_h${K} \
      --use_clinical --use_cultures --clinical_source complete \
      --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/xdata_clin_h${K}.log
done

# --- 72 h EHR window (clinical only differs; GTSQ target unchanged) ---
$PY 02_preprocess.py --gtsq_csv "$CSV" \
    --save_dir $E/01_split_sets/gtsq_pred_72h_gap6h_clin_cult \
    --use_clinical --use_cultures --clinical_source complete \
    --gap_hours 6 --seq_len 72 2>&1 | tee $E/logs/xdata_seq72.log

fi   # end of the original (already-built) splits

# HIRIDCTRL=1 -- a size-matched control from a DIFFERENT institution.
#
# HiRID has a cohort close to UMM's in size but is a different hospital with different charting, and
# it already appears in the Paper-1 extension. UMM SOFA gains nothing while MIMIC-IV SOFA gains
# +0.036; size explains only part of that and coverage none of it. If HiRID SOFA -- UMM-sized,
# not-UMM -- gains like MIMIC-IV, the residual is about the institution or its charting rather than
# scale. If it behaves like UMM, scale survives as the explanation after all.
#
# Built through the same 02b path as the MIMIC-IV targets, so the only thing that varies is the
# database. dopamine is absent from HiRID (14 of 15 SOFA inputs present); sofa_components treats a
# missing pressor as absent, which slightly softens the cardiovascular subscore.
if [ -n "${HIRIDCTRL:=}" ]; then
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" Name=kg_hirid_ctrl 2>/dev/null || true
  for TAG in clin oracle; do
    EXTRA="--use_clinical"; [ "$TAG" = "oracle" ] && EXTRA="--oracle_window"
    $PY 02b_preprocess_mimic.py --dataset hirid --target sofa $EXTRA \
        2>&1 | tee $E/logs/hirid_split_${TAG}.log
  done
  # Same guard as the day-anchor block: a failed preprocessing step leaves no split, and training
  # five seeds on a missing directory would otherwise report success.
  for TAG in clin oracle; do
    [ -f "01_split_sets/hirid_sofa_pred_${TAG}/train.npz" ] || {
      echo "ERROR: 01_split_sets/hirid_sofa_pred_${TAG}/train.npz missing -- preprocessing failed"
      exit 1; }
  done
  for SEED in 42 123 456 789 1337; do
    $PY 03_train.py --mask both --data_dir 01_split_sets/hirid_sofa_pred_clin \
        --output_dir exp_fixnn/02_models_circ/hirid_sofa_clin \
        --model gru --persist_path copy --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 \
        --batch_size 64 --seed $SEED 2>&1 | tee -a $E/logs/hirid_train.log
  done
  for TAG in clin oracle; do
    $PY 05_baselines.py --data_dir 01_split_sets/hirid_sofa_pred_${TAG} --use_clinical --mask both \
        2>&1 | tee $E/logs/hirid_base_${TAG}.log
  done
  exit 0
fi

# DAYANCHOR=1 -- is the residual a SAME-TIME-OF-DAY effect?
#
# UMM SOFA is anchored on the CALENDAR day, so every window is 00:00..24:00 -- the same wall-clock
# block every time, with the same circadian shape. MIMIC-IV SOFA is anchored on the ICU day (hours
# since admission // 24), so its windows start at whatever hour the patient arrived and the
# time-of-day varies across examples. A fixed daytime makes the task more regular: the diurnal
# pattern is constant, persistence already captures it, and there is correspondingly less for a
# model to add. That runs in the right direction to explain why UMM SOFA gains nothing.
#
# Rebuild UMM SOFA on the ICU-day anchor, MIMIC's way, and re-measure. Note this deliberately breaks
# the window's alignment with the 14:00 form clock -- which is the point, since MIMIC lives with the
# same variable offset and the SOFA target has no form time of its own.
# RENALFIX=1 -- rebuild every UMM SOFA day-level target after the renal/urine fix
# (comprehensive doc §37.93). `sofa_components` summed urine_output over the last axis to
# form a 24 h total, which is right for a window but wrong for the (n_days, F) table this
# builder passes: the sum ran over DAYS and broadcast one cohort-wide total, so the urine
# criterion never fired and renal was creatinine-only. The builder now declares
# `urine_is_24h_total=True`. Only the four target CSVs are rebuilt here -- the split sets
# and the model re-runs are the existing gates below, so this stage stays short enough to
# bundle and its output can be diffed against keepgate/pre_renalfix_backup/ before anything
# downstream is spent on it. Invocations are the ones the four targets were built with:
# the plain call takes the dispatch defaults (no CNS, no ICU-day anchor, no day cap).
if [ -n "${RENALFIX:=}" ]; then
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" Name=kg_renalfix 2>/dev/null || true
  mkdir -p $E/logs
  $PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa_day_level.csv \
      2>&1 | tee $E/logs/renalfix_umm_main.log
  $PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa_icuday_day_level.csv --umm_sofa_icuday \
      2>&1 | tee $E/logs/renalfix_umm_icuday.log
  $PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa_d7_day_level.csv --umm_sofa_max_day 7 \
      2>&1 | tee $E/logs/renalfix_umm_d7.log
  $PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa6_day_level.csv --umm_sofa_cns \
      2>&1 | tee $E/logs/renalfix_umm_sofa6.log
  echo "=== renal delta against the pre-fix backup ==="
  $PY - <<'DELTA_PY'
import pandas as pd, pathlib
B = pathlib.Path("pre_renalfix_backup/exp_fixnn")
for name in ["umm_sofa_day_level", "umm_sofa_icuday_day_level",
             "umm_sofa_d7_day_level", "umm_sofa6_day_level"]:
    new = pd.read_csv(f"exp_fixnn/{name}.csv")
    old = pd.read_csv(B / f"{name}.csv")
    col = next((c for c in new.columns if c.lower() == "renal"), None)
    if col is None or len(new) != len(old):
        print(f"{name}: rows {len(old)} -> {len(new)}, renal column {col!r}")
        continue
    ch = (new[col].fillna(-1) != old[col].fillna(-1)).sum()
    print(f"{name}: {ch:,} of {len(new):,} days change renal "
          f"({100*ch/len(new):.1f}%), mean {old[col].mean():.3f} -> {new[col].mean():.3f}")
DELTA_PY
  exit 0
fi

# RENALFIX2=1 -- rebuild the MIMIC SOFA split sets after the renal/urine fix (§37.93).
# This is the one SOFA split whose build command survives nowhere: the live analysis reads
# 01_split_sets/mimic_sofa_pred_clin three times, but no script builds it. The HIRIDCTRL
# block above states the invariant that recovers it -- "Built through the same 02b path as
# the MIMIC-IV targets, so the only thing that varies is the database" -- and 02b's own
# --dataset default is mimic_iv, so the three variants are the hirid invocations with the
# --dataset flag dropped. Verified before launching that the fix changes LABEL VALUES only:
# row count, row order, encounter set and the NaN pattern of every component are identical
# to the pre-fix backup, so the patient-level split assignment cannot move.
if [ -n "${RENALFIX2:=}" ]; then
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" Name=kg_renalfix2 2>/dev/null || true
  for TAG in clin oracle noclin; do
    EXTRA="--use_clinical"
    [ "$TAG" = "oracle" ] && EXTRA="--oracle_window"
    [ "$TAG" = "noclin" ] && EXTRA=""
    # --save_dir is MANDATORY here. 02b defaults the directory to
    # <dataset>_sofa_pred_<tag>, i.e. mimic_IV_sofa_pred_clin, while the live analysis
    # reads mimic_sofa_pred_clin -- the historical name from before that default existed.
    # Without this the rebuild lands in a new directory, nothing downstream sees it, and
    # an existence check on the OLD path passes on stale data instead of catching it.
    D="01_split_sets/mimic_sofa_pred_${TAG}"
    before=$(stat -c %Y "$D/train.npz" 2>/dev/null || echo 0)
    $PY 02b_preprocess_mimic.py --target sofa --save_dir "$D" $EXTRA \
        2>&1 | tee $E/logs/renalfix2_mimic_split_${TAG}.log
    after=$(stat -c %Y "$D/train.npz" 2>/dev/null || echo 0)
    [ "$after" -gt "$before" ] || {
      echo "ERROR: $D/train.npz was not rewritten -- preprocessing did not land here"
      exit 1; }
  done
  exit 0
fi

if [ -n "${DAYANCHOR:=}" ]; then
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" Name=kg_sofa_dayanchor 2>/dev/null || true
  S=exp_fixnn/01_split_sets
  $PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa_icuday_day_level.csv --umm_sofa_icuday \
      2>&1 | tee $E/logs/dayanchor_build.log
  # gtsq_hour is irrelevant once the day index is an integer ICU day, so gap keeps its plain
  # meaning here: 0 = window ends at the target day boundary, -24 = the whole target day (oracle).
  for TAG in clin oracle; do
    EXTRA="--gap_hours 0"; [ "$TAG" = "oracle" ] && EXTRA="--gap_hours -24"
    $PY 02_preprocess.py --gtsq_csv $E/umm_sofa_icuday_day_level.csv \
        --save_dir $S/umm_sofa_icuday_pred_${TAG} \
        --use_clinical --clinical_source complete --seq_len 24 $EXTRA \
        2>&1 | tee $E/logs/dayanchor_split_${TAG}.log
  done
  # 02_preprocess prints "No day-pairs" and returns without writing a split; training then runs on
  # nothing and the block still exits 0. Verify the split exists before spending five seeds on it.
  for TAG in clin oracle; do
    [ -f "$S/umm_sofa_icuday_pred_${TAG}/train.npz" ] || {
      echo "ERROR: $S/umm_sofa_icuday_pred_${TAG}/train.npz missing -- preprocessing produced no split"
      exit 1; }
  done
  for SEED in 42 123 456 789 1337; do
    $PY 03_train.py --mask both --data_dir $S/umm_sofa_icuday_pred_clin \
        --output_dir exp_fixnn/02_models_circ/usofa_icuday \
        --model gru --persist_path copy --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 \
        --batch_size 64 --seed $SEED 2>&1 | tee -a $E/logs/dayanchor_train.log
  done
  for TAG in clin oracle; do
    $PY 05_baselines.py --data_dir $S/umm_sofa_icuday_pred_${TAG} --use_clinical --mask both \
        2>&1 | tee $E/logs/dayanchor_base_${TAG}.log
  done
  exit 0
fi

# HORIZON=1 -- is the residual UMM-vs-MIMIC SOFA gap a STAY-LENGTH COVERAGE effect?
#
# The two targets are measured on nearly disjoint parts of the ICU stay and nothing controlled for
# it. MIMIC SOFA is capped at 168 h (02b_preprocess_mimic --horizon_h), giving at most 6 windows per
# encounter; UMM SOFA has no cap and runs to 188, with a median ICU day of 8 and **53% of its
# windows beyond ICU day 7 -- past the point where MIMIC stops entirely**. Late ICU days are more
# stable, so persistence is stronger there and there is less for a model to gain. That runs in the
# right direction to explain the residual, and it is a larger difference than either the day anchor
# or the blind gap.
#
# Rebuild UMM SOFA capped at 7 ICU days and re-measure. Everything else is held fixed.
if [ -n "${HORIZON:=}" ]; then
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" Name=kg_sofa_horizon 2>/dev/null || true
  S=exp_fixnn/01_split_sets
  $PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa_d7_day_level.csv --umm_sofa_max_day 7 \
      2>&1 | tee $E/logs/horizon_build.log
  for TAG in clin oracle; do
    EXTRA="--gap_hours 14"; [ "$TAG" = "oracle" ] && EXTRA="--gap_hours -10"
    $PY 02_preprocess.py --gtsq_csv $E/umm_sofa_d7_day_level.csv \
        --save_dir $S/umm_sofa_d7_pred_${TAG} \
        --use_clinical --clinical_source complete --seq_len 24 $EXTRA \
        2>&1 | tee $E/logs/horizon_split_${TAG}.log
  done
  for SEED in 42 123 456 789 1337; do
    $PY 03_train.py --mask both --data_dir $S/umm_sofa_d7_pred_clin \
        --output_dir exp_fixnn/02_models_circ/usofa_d7 \
        --model gru --persist_path copy --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 \
        --batch_size 64 --seed $SEED 2>&1 | tee -a $E/logs/horizon_train.log
  done
  for TAG in clin oracle; do
    $PY 05_baselines.py --data_dir $S/umm_sofa_d7_pred_${TAG} --use_clinical --mask both \
        2>&1 | tee $E/logs/horizon_base_${TAG}.log
  done
  exit 0
fi

# =====================================================================================
# CIRCULARITY CONTROLS (CIRC_CONTROLS=1)
# =====================================================================================
if [ -n "${CIRC_CONTROLS:-}" ]; then
S=exp_fixnn/01_split_sets
SEEDS="42 123 456 789 1337"

# --- E1: is MIMIC SOFA's gain a SAMPLE-SIZE effect? -----------------------------------
# MIMIC SOFA trains on ~3x UMM SOFA's windows (40,075 vs 13,332 test windows). UMM's oracle reaches
# only 0.749 macro-F1 on coagulation where the label is 100% recomputable from the window, i.e. it
# UNDER-FITS a deterministic function -- and under-fitting costs the learned model but not
# persistence, which is training-free. So a like-for-like size is the control. --train_subsample
# already exists; no new split is needed.
MS=01_split_sets/mimic_sofa_pred_clin      # MIMIC splits are NOT under exp_fixnn/
for N in 13332 0; do
  for SEED in $SEEDS; do
    $PY 03_train.py --mask both --data_dir $MS \
        --output_dir exp_fixnn/02_models_circ/mimic_sofa_sub${N} \
        --model gru --persist_path copy --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 \
        --batch_size 64 --seed $SEED --train_subsample $N \
        2>&1 | tee -a $E/logs/circ_mimic_sub${N}.log
  done
done

# --- E2: is the gap a COMPONENT-MATCHING artefact? ------------------------------------
# MIMIC's whole gain sits in CNS (+0.104) and cardiovascular (+0.082); on the other four it is flat
# or negative, exactly like UMM. CNS is the component UMM omits, and removing it from MIMIC already
# drops the gain +0.036 -> +0.022 on the existing per-component numbers. Building UMM SOFA WITH CNS
# closes the comparison from the other side. GCS is charted on 33.2% of days, so CNS is NaN
# elsewhere and the masked evaluation simply scores fewer CNS cells.
$PY 01_preprocess_gtsq.py --umm_sofa $E/umm_sofa6_day_level.csv --umm_sofa_cns \
    2>&1 | tee $E/logs/circ_usofa6_build.log
for TAG in clin oracle; do
  EXTRA=""; [ "$TAG" = "oracle" ] && EXTRA="--gap_hours -10"
  [ "$TAG" = "clin" ] && EXTRA="--gap_hours 14"
  $PY 02_preprocess.py --gtsq_csv $E/umm_sofa6_day_level.csv \
      --save_dir $S/umm_sofa6_pred_${TAG} \
      --use_clinical --clinical_source complete --seq_len 24 $EXTRA \
      2>&1 | tee $E/logs/circ_usofa6_${TAG}.log
done
for SEED in $SEEDS; do
  $PY 03_train.py --mask both --data_dir $S/umm_sofa6_pred_clin \
      --output_dir exp_fixnn/02_models_circ/usofa6_clin \
      --model gru --persist_path copy --use_gate --gate_lambda 0.5 \
      --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 \
      --batch_size 64 --seed $SEED 2>&1 | tee -a $E/logs/circ_usofa6_train.log
done
# GBT oracle half of circ = gbt(oracle) - persistence, written into the split dir for circularity.py
$PY 05_baselines.py --data_dir $S/umm_sofa6_pred_oracle --use_clinical \
    2>&1 | tee $E/logs/circ_usofa6_gbt.log
$PY 05_baselines.py --data_dir $S/umm_sofa6_pred_clin --use_clinical \
    2>&1 | tee $E/logs/circ_usofa6_pers.log

echo "=== circ controls end $(date) ==="
fi
