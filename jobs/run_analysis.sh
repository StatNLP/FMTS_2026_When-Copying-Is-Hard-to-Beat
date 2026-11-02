#!/bin/bash
#SBATCH --job-name=fixnn_analysis
#SBATCH --partition=cpu-single
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=03:00:00
#SBATCH --output=keepgate/exp_fixnn/logs/analysis_%j.out
#SBATCH --error=keepgate/exp_fixnn/logs/analysis_%j.err
# One entry point for every descriptive/diagnostic analysis behind the paper.
set -eo pipefail   # tee would otherwise hide a failing step behind its own exit 0

# Every block in this file used to appear in squeue as `fixnn_analysis`, so a queue with four of
# them running told you nothing about which was which. SLURM fixes the job name at submission, but a
# running job can rename itself -- so derive the name from whichever switch is set and do that first.
# (The log FILENAME is already open by then and keeps the %j id, which is still unique.)
kg_name() {
  local n=fixnn_misc
  [ -n "${SIGNIF:-}" ]   && n=kg_signif
  [ -n "${TSF:-}" ]      && n=kg_arima_ets
  [ -n "${PERMUTE:-}" ]  && n=kg_permute${PERMUTE_PASS:+_$PERMUTE_PASS}
  [ -n "${CULTTIME:-}" ] && n=kg_culttime
  [ -n "${MSEVAL:-}" ]   && n=kg_mseval${NSHARD:+_$SHARD}${REVERSE:+r}
  [ -n "${CIRC:-}" ]     && n=kg_circ
  [ -n "${REPORT:-}" ]   && n=kg_report
  [ -z "${SIGNIF:-}${TSF:-}${PERMUTE:-}${CULTTIME:-}${MSEVAL:-}${CIRC:-}${REPORT:-}" ] \
      && n=kg_analyse_${1:-all}
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" Name="$n" 2>/dev/null || true
  echo "$n"
}
echo "=== job $(kg_name "${1:-all}")  id ${SLURM_JOB_ID:-none}  host $(hostname)  $(date) ==="

cd "${KEEPGATE_ROOT:-${PROJECT_ROOT:-$(pwd)}/keepgate}"
PY="${PYTHON:-python}"
# REPORT="ladder table items ..." dispatches to report.py instead of analyse_gtsq.py, so the
# generated LaTeX tables and figures are produced by a scheduled script rather than by hand on the
# login node. KG_TREE/KG_BALANCE pick the model population (see report.py); both are exported by the
# caller. Without REPORT the script keeps its original behaviour.
# CIRC="crosstarget sofacomp" dispatches to circularity.py, which is the only analysis that spans
# four split families (GTSQ, UMM SOFA, MIMIC, eICU) and therefore does not live in report.py.
if [ -n "${CIRC:-}" ]; then
  for C in $CIRC; do
    echo "############ circularity.py $C ############"
    $PY exp_fixnn/circularity.py "$C"
  done
  [ -z "${REPORT:-}" ] && exit 0     # only stop here if REPORT was not also requested
fi
if [ -n "${REPORT:-}" ]; then
  for C in $REPORT; do
    echo "############ report.py $C  (tree=${KG_TREE:-02_models} balance=${KG_BALANCE:-uniform}) ############"
    $PY exp_fixnn/report.py "$C"
  done
  exit 0
fi
# SIGNIF=1 regenerates every significance test in the paper. These were previously produced by hand
# on the login node, so the appendix table could not be reproduced by anyone -- including us. Both
# instruments are paired and both use the DAY-PAIR as the unit; 10000 shuffles bottoms out at
# p<0.0001, 2000 bootstrap resamples give the 95% intervals.
if [ -n "${SIGNIF:=}" ]; then
  M2=exp_fixnn/02_models_m2
  S=exp_fixnn/01_split_sets
  CULT=$S/gtsq_pred_24h_gap6h_clin_cult
  A=10000; B=2000

  # A failing contrast must not take the remaining tests down with it -- but the block must still
  # FAIL if nothing worked. set +e alone plus `exit 0` reported success on five crashes, the same
  # bug MSEVAL had. Each test's status is recorded and the tally decides the exit code.
  set +e
  sig_ok=0; sig_bad=0
  run_sig () { "$@"; if [ $? -eq 0 ]; then sig_ok=$((sig_ok+1)); else sig_bad=$((sig_bad+1)); echo "  FAILED: $*"; fi; }
  echo "############ 1. appendix table: macro-F1 winner vs persistence+rule ############"
  run_sig $PY exp_fixnn/eval_deploy.py ensemble --runs "$M2/mk_clin_k0/*" --split $CULT \
      --base markov --arp $A --bootstrap $B --split_name test

  # The remaining three are MODEL vs MODEL, not model vs baseline: each changes the INPUTS while
  # holding the target fixed, so they need --runs_b. The differences under test are 0.009-0.022,
  # which is where asserting rather than testing stops being defensible.
  echo "############ 2. cultures on vs off (+0.0142 change-accuracy) ############"
  run_sig $PY exp_fixnn/eval_deploy.py ensemble --runs "$M2/mk_clin_k0/*" --split $CULT \
      --runs_b "$M2/mk_nocult_k0/*" --split_b $S/gtsq_pred_24h_gap6h_clin_nocult \
      --arp $A --bootstrap $B --split_name test

  echo "############ 3. culture ORDER time vs RESULT time ############"
  # These two share the cult split and differ by --drop_clinical (order-only drops
  # culture_result_returned+culture_positive; result-only drops culture_drawn), so no --split_b: the
  # rows are the same rows. The runs themselves were produced by hand; see the TRAIN block below.
  run_sig $PY exp_fixnn/eval_deploy.py ensemble --runs "$M2/mk_cultdrawn/*" --split $CULT \
      --runs_b "$M2/mk_cultresult/*" \
      --arp $A --bootstrap $B --split_name test

  echo "############ 4. physiology on vs off (+0.0216 change-accuracy) ############"
  # Also settles whether questionnaire-only (0.8711 +- 0.0004) really sits BELOW the rule (0.8714).
  # A 0.0003 gap inside one seed standard deviation cannot carry an argument unasserted.
  run_sig $PY exp_fixnn/eval_deploy.py ensemble --runs "$M2/mk_clin_k0/*" --split $CULT \
      --runs_b "$M2/mk_noclin_k0/*" --split_b $S/gtsq_pred_24h_gap6h_noclin \
      --arp $A --bootstrap $B --split_name test

  echo "############ 5. questionnaire-only vs the rule alone ############"
  run_sig $PY exp_fixnn/eval_deploy.py ensemble --runs "$M2/mk_noclin_k0/*" \
      --split $S/gtsq_pred_24h_gap6h_noclin --base markov --arp $A --bootstrap $B --split_name test
  echo "############ significance done: ok=$sig_ok failed=$sig_bad ############"
  [ "$sig_bad" -eq 0 ] || exit 1
  exit 0
fi

# TSF=1 runs the classical univariate forecasters on GTSQ, all 45 items. They place the BOTTOM rung
# of the ladder: LOCF is the incumbent, and if ARIMA/ETS cannot beat copying then "use a proper
# time-series model" is not the missing ingredient. Both refit online (expanding window, one step
# ahead) so nothing is fit on the value it predicts. Slow -- minutes, not seconds -- hence its own
# switch. The Paper-1 TSF side already has this result (ARIMA/ETS stay <= LOCF on all five
# datasets, doc 15.9); only GTSQ was missing it.
if [ -n "${TSF:=}" ]; then
  $PY 05_baselines.py --data_dir exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult \
      --use_clinical --tsf --mask both 2>&1 | tee exp_fixnn/logs/tsf_arima_ets.log
  exit 0
fi

# MSEVAL=1 evaluates the multi-seed forecasting checkpoints that were TRAINED but never SCORED.
# 180 *_ms_s* checkpoints exist and only 15 have an eval json, so the whole Paper-1 TSF ladder is
# single-seed while the GTSQ side carries five seeds and randomisation p-values.
#
# MUST be submitted to a GPU node -- full test-set inference of the autoregressive decoder is too
# slow on CPU (jobs/run_eval.sh says so explicitly), and this script's own header is
# cpu-single. Submit with the partition overridden:
#   MSEVAL=1 sbatch --partition=gpu-single --gres=gpu:1 --time=12:00:00 \
#       keepgate/exp_fixnn/run_analysis.sh
if [ -n "${MSEVAL:=}" ]; then
  cd "${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
  PYX=python
  # The counters must live in THIS shell. Piping the loop into `tee` put them in a subshell, so the
  # tally always printed 0/0/0 and `[ "$bad" -eq 0 ] || exit 1` could never fire -- a run where all
  # 165 evals failed still exited 0, which is precisely the failure the guard exists to catch.
  # ${VAR:-alt} yields VAR when VAR is SET, so the array id was appended twice (mseval_a2020.log).
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    _LOG="keepgate/exp_fixnn/logs/mseval_a${SLURM_ARRAY_TASK_ID}.log"
  else
    _LOG="keepgate/exp_fixnn/logs/mseval${NSHARD:+_s${SHARD:-0}of$NSHARD}${REVERSE:+_rev}.log"
  fi
  exec > >(tee "$_LOG") 2>&1
  ok=0; bad=0; skipped=0
  # SHARD/NSHARD split the 165 checkpoints across parallel jobs. One checkpoint takes ~15 min on
  # CPU for the big eICU test set (211k windows), so a single job needs ~41 h -- past any sensible
  # walltime. Sharding by a stable hash of the NAME keeps each job's share mixed across datasets
  # rather than giving one job all of eICU. Jobs still skip finished evals, so shards may overlap
  # harmlessly and a re-run costs nothing.
  # REVERSE=1 walks each shard's list from the far end. Because finished evals are skipped, a
  # forward worker and a reverse worker on the SAME shard converge from both sides and meet in the
  # middle -- roughly halving that shard's wall time without repartitioning the work or disturbing
  # jobs already running.
  # ARRAY mode: one checkpoint per SLURM array task. The pending list is recomputed at start, so
  # each task picks the Nth checkpoint that still lacks an eval json -- 97 independent ~15 min jobs
  # drain far faster than a dozen workers walking a shared list, and a failure costs one checkpoint
  # rather than a whole shard.
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mapfile -t PEND < <(for c in checkpoints/*_ms_s*/; do
        n=$(basename "$c"); [ -f "$c/best.pt" ] || continue
        [ -f "docs/tables/eval_${n}.json" ] || echo "$n"; done)
    NAME="${PEND[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$NAME" ]; then echo "array task $SLURM_ARRAY_TASK_ID: nothing left to do"; exit 0; fi
    CKS=("checkpoints/${NAME}/")
    NSHARD=1; SHARD=0; REVERSE=""
    echo "array task $SLURM_ARRAY_TASK_ID -> $NAME  (${#PEND[@]} pending at start)"
  fi
  SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
  [ ${#CKS[@]} -eq 0 ] && CKS=(checkpoints/*_ms_s*/)
  if [ -n "${REVERSE:-}" ]; then
    for ((k=${#CKS[@]}-1, j=0; k>=0; k--, j++)); do REV[$j]="${CKS[$k]}"; done
    CKS=("${REV[@]}")
  fi
  i=-1
  for CK in "${CKS[@]}"; do
    NAME=$(basename "$CK")
    i=$((i+1))
    if [ "$NSHARD" -gt 1 ]; then
        h=$(( $(printf "%s" "$NAME" | cksum | cut -d" " -f1) % NSHARD ))
      [ "$h" -eq "$SHARD" ] || continue
    fi
    OUT=docs/tables/eval_${NAME}.json
    [ -f "$OUT" ] && continue                       # re-runnable: already scored
    [ -f "$CK/best.pt" ] || { skipped=$((skipped+1)); continue; }
    # Dataset comes from the config's train_h5, NOT from the directory name. The *_ms_* configs
    # carry no `dataset` field, and suffix-matching the name is a guess: the 94-feature catalog is
    # identical across datasets, so a mismatch would score against the WRONG split silently instead
    # of erroring. train_h5 is recorded by the trainer and cannot drift from what was trained on.
    # `|| true` so a failing parse falls through to the SKIP branch below; under set -e the bare
    # assignment aborted the whole loop and the SKIP message was unreachable.
    DS=$($PYX -c "
import json,os,re
c=json.load(open('$CK/config.json'))
th=c.get('train_h5','')
m=re.search(r'([a-z0-9_]+)_seed\\d+_train', os.path.basename(th))
print(m.group(1) if m else '')" 2>/dev/null || true)
    if [ -z "$DS" ]; then echo "  SKIP $NAME: cannot resolve dataset from train_h5"; skipped=$((skipped+1)); continue; fi
    H5=data/dataset/${DS}_seed42_test__all_4h_fwd.h5
    if [ ! -f "$H5" ]; then echo "  SKIP $NAME: no $H5"; skipped=$((skipped+1)); continue; fi
    # --long-parquet is what produces the sepsis3 key. The 15 pre-existing *_ms_s* jsons all have
    # it; omitting it here would leave the downstream Sepsis-3 table at 15 of 180 and recreate
    # exactly the seed asymmetry this block exists to remove.
    LONG=data/long/${DS}_long.parquet
    LONG_ARG=""; [ -f "$LONG" ] && LONG_ARG="--long-parquet $LONG"
    echo "### $NAME  (dataset=$DS)"
    # -m, not a path: eval_forecast does `from harmonize.icu_dataset import ...`, so running it as
    # a script puts harmonize/ on sys.path instead of the repo root and every run dies on
    # ModuleNotFoundError. Every other call site in the repo uses -m.
    if $PYX -m harmonize.eval_forecast --checkpoint "$CK" --test-h5 "$H5" --out "$OUT" $LONG_ARG; then
      ok=$((ok+1))
    else
      bad=$((bad+1)); echo "  FAILED $NAME"
    fi
  done
  echo "MSEVAL done: ok=$ok failed=$bad skipped=$skipped"
  # A truncated run must not look like a complete one -- the multi-seed ladder would then be built
  # from an unknown subset of seeds without anyone noticing.
  [ "$bad" -eq 0 ] || exit 1
  exit 0
fi

# PERMUTE=1 answers "which physiology is the changer?". Two passes over the five markov-base seeds:
# every variable one at a time, then organ-system GROUPS permuted jointly. The grouped pass is the
# one expected to carry signal -- physiology's entire measured contribution is +0.0167 change-accuracy,
# so split across 93 variables each is ~0.0002, inside seed spread; and correlated proxies
# (creatinine vs urine output) hide each other under single-variable permutation. Scored on the
# headline macro-F1 and change-F1, because that is where the effect lives.
if [ -n "${PERMUTE:=}" ]; then
  # 08_probes derives the split from each run's own config and always scores test.npz, so there is
  # no --split flag to pass; inventing one here would have failed the job on argparse before a
  # single forward pass.
  # PERMUTE_PASS=groups re-runs only the grouped pass (the per-variable pass is settled: 0 of 93
  # variables clear twice their own seed s.d., so single-variable attribution is noise here).
  LOG=exp_fixnn/logs/permute_importance${PERMUTE_PASS:+_$PERMUTE_PASS}.log
  # exec, not a pipeline: piping the loop into tee puts it in a subshell where set -e kills only the
  # subshell and the block still exits 0 on a truncated pass. Identical to the MSEVAL bug above.
  exec > >(tee "$LOG") 2>&1
  for R in exp_fixnn/02_models_m2/mk_clin_k0/*/; do
    echo "########## $(basename $R) ##########"
    if [ "${PERMUTE_PASS:-all}" != "groups" ]; then
      echo "--- pass 1: every variable, one at a time ---"
      $PY 08_probes.py --run_dir "$R" --probe permute --permute_all
    fi
    echo "--- pass 2: organ-system groups, permuted jointly ---"
    $PY 08_probes.py --run_dir "$R" --probe permute --permute_groups
  done
  exit 0
fi

# CULTTIME=1 rebuilds the two culture-timing model groups. They were trained by hand with
# --drop_clinical and existed in no script, so the order-vs-result contrast in Section 4 could not be
# reproduced. culture_drawn is stamped at COLLECTION (leakage-safe, the clinician's suspicion now);
# culture_result_returned/culture_positive are stamped at ARRIVAL, a median 123 h later.
if [ -n "${CULTTIME:=}" ]; then
  CULT=exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult
  for S_ in 42 123 456 789 1337; do
    $PY 03_train.py --mask both --data_dir $CULT --output_dir exp_fixnn/02_models_m2/mk_cultdrawn \
        --drop_clinical culture_result_returned,culture_positive \
        --model gru --persist_path markov --markov_tau 0.5 --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 --batch_size 64 --seed $S_
    $PY 03_train.py --mask both --data_dir $CULT --output_dir exp_fixnn/02_models_m2/mk_cultresult \
        --drop_clinical culture_drawn \
        --model gru --persist_path markov --markov_tau 0.5 --use_gate --gate_lambda 0.5 \
        --d_model 128 --e_layers 2 --d_ff 512 --loss_balance uniform --lr 0.0005 --batch_size 64 --seed $S_
  done
  exit 0
fi

$PY exp_fixnn/analyse_gtsq.py "${1:-all}"
