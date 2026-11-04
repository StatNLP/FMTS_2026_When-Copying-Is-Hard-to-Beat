# When Copying Is Hard to Beat — reproduction code

Code to **re-run the experiments** behind *When Copying Is Hard to Beat: Evaluation
Sensitivity in Clinical Assessment Forecasting* (FMTS @ NeurIPS 2026). This repository is
organised around reproducing the numbers, not around reading the paper; the paper is linked
at the bottom.

Contact: Marius Fracarolli, `fracarolli@cl.uni-heidelberg.de`.

## Start here

Every number in the paper comes out of **one** evaluation pass:

```bash
python keepgate/exp_fixnn/eval_deploy.py fmts \
    --split_name test --report seedmean --bootstrap 2000
```

One cohort, one observation mask, one label version, candidate selection on validation
before test is read once, and one reporting object (`--report seedmean`) driving the point
estimates, the patient-clustered bootstrap, the departure counts and the emitted LaTeX. If a
number in the paper disagrees with this pass, the pass is right.

## What you need that is not here

**The data.** The questionnaire (GTSQ) and clinical records are single-centre routine-care
data from the intensive care unit of University Medicine Mannheim and cannot be released.
Nothing in this repository will run end-to-end without them, and no checkpoint, cached
prediction or per-cell prediction is included — the persistence row is by construction
identical to the previous day's record, so releasing predictions would release the data.

The **aggregate evaluation artefacts** that back Table 1 and Figure 3 — metrics,
patient-clustered intervals, departure counts and the run manifest recording the input
cutoff, label version, mask, reporting object and seeds of every row — ship with the paper
as supplementary material. They are what lets you check a reported figure against its
interval without the data.

MIMIC-III, MIMIC-IV and eICU (used for the companion experiments) are available from
PhysioNet under their data use agreements.

## Environment

```bash
pip install -r requirements.txt
```

Python 3.13, PyTorch, scikit-learn 1.8, pandas, numpy. The Chronos rows additionally need
`chronos`; MIRA needs its own environment (`requirements-moirai.txt` covers the Moirai/MIRA
side: transformers 4.40.1, torchdiffeq) plus vendored model code.

Job scripts in `jobs/` are SLURM wrappers. They read `PROJECT_ROOT` (defaulting to the
working directory) and `CONDA_MODULE`, so nothing is pinned to our cluster — but partitions,
accounts and wall times are ours and will need editing.

## Pipeline

| step | command | output |
|---|---|---|
| 1. questionnaire → day level | `python keepgate/01_preprocess_gtsq.py` | day-level GTSQ table |
| 2. splits and tensors | `python keepgate/02_preprocess.py --gtsq_csv <day-level csv> --split_by patient --use_clinical --use_cultures --clinical_source complete --gap_hours 6 --seq_len 24 --save_dir 01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit` | `01_split_sets/<name>/{train,val,test}.npz` |
| 3. companion targets | `python keepgate/02b_preprocess_mimic.py`, `02c_preprocess_eicu.py` | MIMIC / eICU SOFA and organ-support targets |
| 4. training | `bash jobs/run_fmts.sh` — `PART=rows\|temporal\|gap24\|rawlab\|tstrict` | `02_models/<run>/` |
| 5. classical + rule baselines | `python keepgate/05_baselines.py` | Markov, GBT, ARIMA, ETS, the one-line ward rule |
| 6. pretrained forecasters | `bash jobs/run_fm.sh` | zero-shot and fine-tuned Chronos / Chronos-2 / MIRA rows |
| 7. the evaluation pass | `python keepgate/exp_fixnn/eval_deploy.py fmts --split_name test --report seedmean --bootstrap 2000` | `fmts_{metrics,departures,ci,manifest}.*` and Table 1 |

`--split_by patient` is not optional: the paper's claims are about unseen **patients**, and an
encounter-level split is not a patient-level split (one patient can hold encounters in two
splits). `02_preprocess.py` writes the overlap check it performed.

## Which script produces which part of the paper

| part of the paper | command |
|---|---|
| Table 1, and every number in Sections 3–4 | `exp_fixnn/eval_deploy.py fmts` |
| Figure 1 (carry-forward schematic) and Figure 3 (departures from persistence) | `exp_fixnn/make_figure.py`; Figure 3 from `fmts_departures.csv` |
| Appendix A.8 (per-item results, 45 items) | `exp_fixnn/analyse_gtsq.py peritem`, from the per-cell test predictions and the per-seed metrics of the evaluation pass |
| Appendix B (pretrained forecasters) | `keepgate/fm_baseline.py`, scored by the pass above |
| Appendix C (change score, temporal holdout) | `exp_fixnn/report.py readout`, `exp_fixnn/report.py drift` |
| cohort flow, patient linkage, split disjointness | `exp_fixnn/report.py cohort` |
| duplicate divergence, circularity controls | `exp_fixnn/analyse_gtsq.py`, `exp_fixnn/circularity.py` |
| the selected configuration per model family | `exp_fixnn/hp_search.py` |
| the one-line clinical rule | `exp_fixnn/ward_predictor.py` |
| derived vs explicitly recorded cells, class support | `exp_fixnn/report.py maskaudit`, `absclass` |

Useful flags: `--mask_from` transfers an observation mask between splits so the
explicit-cell rows rescore *identical* predictions; `--no_hierarchy_fill` builds the
explicit-only cohort.

## Two task definitions the result depends on

Stated here because they decide whether the headline is reproducible, and both are easy to
get wrong:

1. **Where the physiology window ends.** Ending it at the previous form's reference time
   (`--gap_hours 24`) is the strictly pre-round setting and the primary benchmark. The
   6-hour setting (`--gap_hours 6`) ends at 08:00 for a form whose reference time is 14:00,
   so part of the window is after the round began; it is reported as a sensitivity check,
   not as a matched 24-hour forecast.
2. **One mask, one label version, everywhere.** Every retained row uses the same observation
   mask and the corrected label version. Mixing them changes the numbers without changing
   the model.

## Entry points this paper does not use

Reachable and working, and simply not on the path to any reported number. They are kept so
these files stay identical to the code that produced the results:

- `exp_fixnn/eval_deploy.py alarm` and `decision` — deployment-threshold views of the same runs
- `keepgate/05_baselines.py --diagnose` — why Markov/GBT win accuracy but lose macro-F1
- `keepgate/gtsq_eval.py:weighted_f1` — support-weighted F1; the paper reports macro-F1 and accuracy

## Layout

```
keepgate/              preprocessing, training, baselines, evaluation
keepgate/exp_fixnn/    the evaluation pass, the audits and the analysis scripts
harmonize/             the shared 5-database harmonisation pipeline (Stages 1-3)
harmonize/scores/      SOFA, SAPS-II, KDIGO, Sepsis-3, NEWS2
configs/features.yaml  the 94-channel feature catalogue every stage keys on
configs/sweeps/        the hyperparameter grids as they were run
jobs/                  SLURM wrappers
```

`harmonize/` is shared with the companion repository for the per-variable combination paper
([ML4H 2026](https://github.com/StatNLP/ML4H_2026_Per-Variable-Forecast-Combination)); the
two releases carry the same Stage 1-3 code.

## Paper

https://github.com/StatNLP/FMTS_2026_When-Copying-Is-Hard-to-Beat — *When Copying Is Hard to
Beat: Evaluation Sensitivity in Clinical Assessment Forecasting*, Marius Fracarolli and
Stefan Riezler, Foundation Models for Temporal Systems (FMTS) @ NeurIPS 2026.
