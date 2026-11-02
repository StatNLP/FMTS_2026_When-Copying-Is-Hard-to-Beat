#!/bin/bash
# =============================================================================
# exp_fixnn -- corrected-data experiment root (isolated from the original runs)
# =============================================================================
# Two data changes vs. the main pipeline, both authorised by the UMM clinicians (July 2026):
#
#   (1) --fix_new_new   organ 'neu' on day D AND day D+1  ->  day D+1 corrected to 'akut'.
#                       The form's own wording ('neu aufgetreten' vs 'akut seit >24h') makes
#                       neu->neu impossible. MUST be reported in the methods: it changes the
#                       TARGET, and it removes an artefact that was propping up persistence
#                       (persistence scored those cells correct; now markov does).
#
#   (2) --use_cultures  microbiology added to the clinical panel, each event placed at the hour
#                       it BECAME KNOWN:
#                         culture_drawn           @ collection_time (leakage-safe; ordering a
#                                                 culture *is* the infection suspicion)
#                         culture_result_returned @ arrival_time (median ~123 h later)
#                         culture_positive        @ arrival_time, organism identified
#                       `result_time` is ignored (median turnaround 0.0 h -> not a real clock).
#
# Everything writes under keepgate/exp_fixnn/ so the original splits/models are untouched.
#
# Usage (from keepgate/):   bash exp_fixnn/run_all.sh            # stages 1-3
#                           STAGE=1 bash exp_fixnn/run_all.sh    # data only
# =============================================================================
set -euo pipefail
# $0 under sbatch is the spool copy (/var/spool/slurm/...), so a relative cd lands outside the
# workspace and every mkdir fails with EACCES. Use the absolute root, as the other runners do.
cd "${KEEPGATE_ROOT:-${PROJECT_ROOT:-$(pwd)}/keepgate}"
PY="${PYTHON:-python}"
E=exp_fixnn
CSV=$E/gtsq_day_level_encoded_fixnn.csv
STAGE="${STAGE:-3}"
# ONLY=baselines re-runs just the training-free baseline suite and stops. The frontier table needs
# these on the SAME masking convention as the model rows (MASK, default both) and on the clinical
# split when the gradient-boosting/logreg rows are wanted (BASE_SPLIT, --use_clinical via BASE_CLIN),
# and re-running the whole of run_all.sh to refresh them would rebuild every split first.
ONLY="${ONLY:-}"
MASK="${MASK:-both}"
BASE_SPLIT="${BASE_SPLIT:-$E/01_split_sets/gtsq_pred_24h_gap6h_noclin}"
mkdir -p $E/01_split_sets $E/02_models $E/logs
# ONLY=patsplit builds the PATIENT-disjoint splits for the FMTS submission. The historical splits
# group by EncounterNumber, which is not patient-disjoint: 24 patients had encounters in both train
# and test, so "unseen patients" was not a supportable claim. --split_by patient groups every
# encounter of one lifeTimeNumber into a single fold; the dirs get a _patsplit suffix so the
# encounter splits and every result built on them stay untouched and comparable.
if [ "$ONLY" = "patsplit" ]; then
  echo "############ patient-disjoint splits (FMTS) ############"
  PS="--split_by patient"
  # (a) GTSQ-only cohort: baselines, transition matrix, change-score lookup
  $PY 02_preprocess.py --gtsq_csv "$CSV" $PS       --save_dir $E/01_split_sets/gtsq_pred_24h_gap6h_noclin_patsplit       --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/02_pat_noclin.log
  # (b) clinical + cultures at the causal gap: every Q+P model row
  $PY 02_preprocess.py --gtsq_csv "$CSV" $PS       --save_dir $E/01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit       --use_clinical --use_cultures --clinical_source complete       --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/02_pat_clin_cult.log
  # (c) K=7 questionnaire history: the pretrained-forecaster rows
  $PY 02_preprocess.py --gtsq_csv "$CSV" $PS --gtsq_history 7       --save_dir $E/01_split_sets/gtsq_pred_24h_gap6h_noclin_h7_patsplit       --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/02_pat_noclin_h7.log
  # (c2) MATCHED-ISSUE-TIME split: physiology truncated at the PREVIOUS form's reference time.
  #      The export's charttime is 14:00 for 100% of forms -- a fixed reference, not an entry or
  #      signing time -- so no timestamp can establish when a form was actually written. The default
  #      gap of 6 h therefore feeds the model physiology from inside the interval the target form
  #      assesses. gap_hours 24 ends the window exactly at the preceding form's reference time, which
  #      is the only cutoff the data can defend as "available when the previous form was issued".
  $PY 02_preprocess.py --gtsq_csv "$CSV" $PS \
      --save_dir $E/01_split_sets/gtsq_pred_24h_gap24h_clin_cult_patsplit \
      --use_clinical --use_cultures --clinical_source complete \
      --gap_hours 24 --seq_len 24 2>&1 | tee $E/logs/02_pat_clin_cult_gap24.log
  # (d) chronological holdout, ALSO patient-grouped: a readmitted patient stays in one fold
  #     instead of straddling the date cutoff (O12).
  $PY 02_preprocess.py --gtsq_csv "$CSV" $PS --split_mode temporal       --save_dir $E/01_split_sets/gtsq_temporal_24h_gap6h_clin_cult_patsplit       --use_clinical --use_cultures --clinical_source complete       --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/02_pat_temporal.log
  exit 0
fi

if [ "$ONLY" = "baselines" ]; then
  echo "############ baselines only: mask=$MASK  split=$BASE_SPLIT ############"
  $PY 05_baselines.py --data_dir "$BASE_SPLIT" --mask "$MASK" ${BASE_CLIN:+--use_clinical} \
    2>&1 | tee $E/logs/baselines_${MASK}.log
  exit 0
fi

echo "############ STAGE 1a: day-level CSV with the neu->akut correction ############"
$PY 01_preprocess_gtsq.py --fix_new_new --out "$CSV" 2>&1 | tee $E/logs/01_preprocess.log

echo "############ STAGE 1b: noclin split (GTSQ-only cohort) ############"
$PY 02_preprocess.py --gtsq_csv "$CSV" \
    --save_dir $E/01_split_sets/gtsq_pred_24h_gap6h_noclin \
    --gap_hours 6 --seq_len 24 2>&1 | tee $E/logs/02_noclin.log

# --- Stage 3 runs HERE (before the slow clinical builds) so the headline question --
# --- "is the baseline now beatable?" -- lands early in the log. -----------------------
if [ "$STAGE" -ge 2 ]; then
echo "############ STAGE 3: re-baseline on corrected data (training-free) ############"
BAYES_SPLIT=$PWD/$E/01_split_sets/gtsq_pred_24h_gap6h_noclin \
  $PY bayes_floor.py subset_free 2>&1 | tee $E/logs/03_bayes_subset.log
BAYES_SPLIT=$PWD/$E/01_split_sets/gtsq_pred_24h_gap6h_noclin \
  $PY bayes_floor.py 2>&1 | tee $E/logs/03_bayes_full.log
$PY 05_baselines.py --data_dir $E/01_split_sets/gtsq_pred_24h_gap6h_noclin \
  2>&1 | tee $E/logs/03_baselines.log
fi

echo "############ STAGE 1c: clinical splits (+/- cultures) at gap -6/0/+6 ############"
# clinical + cultures, at the three gaps Stefan asked for (-6 = target-day oracle, 0, +6 = causal)
for GAP in 6 0 -6; do
  $PY 02_preprocess.py --gtsq_csv "$CSV" \
      --save_dir $E/01_split_sets/gtsq_pred_24h_gap${GAP}h_clin_cult \
      --use_clinical --use_cultures --clinical_source complete \
      --gap_hours $GAP --seq_len 24 2>&1 | tee $E/logs/02_clin_cult_gap${GAP}.log
done

# same three gaps WITHOUT cultures -> the clean cultures-on/off ablation
for GAP in 6 0 -6; do
  $PY 02_preprocess.py --gtsq_csv "$CSV" \
      --save_dir $E/01_split_sets/gtsq_pred_24h_gap${GAP}h_clin_nocult \
      --use_clinical --clinical_source complete \
      --gap_hours $GAP --seq_len 24 2>&1 | tee $E/logs/02_clin_nocult_gap${GAP}.log
done
echo "############ DONE. Outputs under keepgate/$E/ ############"
