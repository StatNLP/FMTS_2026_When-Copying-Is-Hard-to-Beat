#!/bin/bash
# keepgate/submit_keepgate.sh
# Orchestrate the Keep-Gate MIMIC companion (Paper 3) on SLURM — nothing runs on
# the login node. Chains three phases with Slurm dependencies:
#
#   Phase 0 (cpu-single):           preprocess each target  -> 01_split_sets/mimic_<t>_pred_noclin
#   Phase 1 (cpu-single, afterok):  non-neural baselines    -> baseline_results.json
#   Phase 2 (gpu-single, afterok):  model sweep             -> 02_models/ + results.csv
#       variants: persistence | gru base | gru skip{linear,copy} | gru gate{linear,copy}
#
# Usage (from repo root):
#   bash keepgate/submit_keepgate.sh
#   TARGETS=support SEEDS="42 123 456" bash keepgate/submit_keepgate.sh
#   SKIP_PREP=1 bash keepgate/submit_keepgate.sh        # reuse existing splits

set -euo pipefail

WS="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
NOTIFY_EMAIL="${NOTIFY_EMAIL:-fracarolli@cl.uni-heidelberg.de}"   # NB: not MAIL (reserved shell var)
export PYTHON="${PYTHON:-python}"
mkdir -p "$WS/keepgate/logs"

TARGETS="${TARGETS:-support sofa}"
SEEDS="${SEEDS:-42 123 456}"
CPU_PART="${CPU_PART:-cpu-single}"
GPU_PART="${GPU_PART:-gpu-single}"
SKIP_PREP="${SKIP_PREP:-0}"
DRYRUN="${DRYRUN:-0}"            # 1 = print the sbatch commands instead of submitting
RUN="keepgate/run_keepgate.sh"

s() {
    if [ "$DRYRUN" = "1" ]; then
        echo "[dry] sbatch --chdir=$WS --partition=... $*" >&2
        echo "DRYRUN"            # stand-in job id for dependency wiring
    else
        sbatch --chdir="$WS" --parsable --mail-user="$NOTIFY_EMAIL" --mail-type=FAIL "$@"
    fi
}

for t in $TARGETS; do
    DD="01_split_sets/mimic_${t}_pred_noclin"

    # ---- Phase 0: preprocess ----
    prep_dep=()
    if [ "$SKIP_PREP" != "1" ]; then
        pj=$(s --job-name="kg_prep_${t}" --partition="$CPU_PART" --time=00:30:00 \
                "$RUN" preprocess --target "$t")
        echo "preprocess ${t}: job ${pj}"
        prep_dep=(--dependency="afterok:${pj}")
    fi

    # ---- Phase 1: non-neural baselines ----
    s --job-name="kg_base_${t}" --partition="$CPU_PART" --time=02:00:00 \
      "${prep_dep[@]}" "$RUN" baselines --data_dir "$DD" >/dev/null \
      && echo "baselines ${t}: submitted"

    # ---- Phase 2: model sweep ----
    for seed in $SEEDS; do
        common=(train --data_dir "$DD" --loss_balance uniform --seed "$seed")
        declare -a VARIANTS=(
            "--model_type persistence"
            "--model_type gru"
            "--model_type gru --use_skip --persist_path linear"
            "--model_type gru --use_skip --persist_path copy"
            "--model_type gru --use_gate --persist_path linear"
            "--model_type gru --use_gate --persist_path copy"
        )
        for v in "${VARIANTS[@]}"; do
            s --job-name="kg_tr_${t}_s${seed}" --partition="$GPU_PART" \
              --gres=gpu:1 --time=02:00:00 \
              "${prep_dep[@]}" "$RUN" "${common[@]}" $v >/dev/null
        done
        echo "train ${t} seed ${seed}: 6 variants submitted"
    done
done

echo "=== Submitted. Monitor: squeue -u \$USER | grep kg_ ;  results in keepgate/02_models/results.csv ==="
