"""KDIGO 2012 acute-kidney-injury (AKI) daily staging.

A consensus override target analogous to Sepsis-3 (ΔSOFA ≥ 2): the rare, high-value event
the keep gate should catch is new AKI / a stage escalation. Staged from serum creatinine
(mg/dL) relative to a per-stay baseline, plus RRT and an optional urine criterion
(mL/kg/h). Creatinine-based staging is the robust ICU-retrospective default; the urine
criterion is applied only where urine (and weight) are observed.

KDIGO 2012 (stage 0–3):
  Stage 1: creatinine 1.5–1.9× baseline  OR  ≥0.3 mg/dL rise (48 h)  OR  urine <0.5 mL/kg/h
  Stage 2: creatinine 2.0–2.9× baseline                              OR  urine <0.5 mL/kg/h (longer)
  Stage 3: creatinine ≥3.0× baseline OR ≥4.0 mg/dL OR RRT            OR  urine <0.3 mL/kg/h / anuria

Daily aggregation: creatinine is the day's worst (max); the 48 h rise compares the day's max
against the prior 2 days' minimum; urine is the day's total (mL/24 h) ÷ weight ÷ 24.
"""

import numpy as np


def kdigo_stage(creatinine, baseline, creat_rise_48h=None,
                urine_ml_kg_h=None, on_rrt=None, urine_stage2_assume_sustained=False):
    """Per-element KDIGO stage (0–3) as float; NaN where no criterion is evaluable.

    All inputs are array-likes broadcastable to a common shape (or None to skip a criterion).
    creatinine / baseline in mg/dL; creat_rise_48h in mg/dL; urine_ml_kg_h in mL/kg/h;
    on_rrt a 0/1 indicator.
    """
    cr = np.asarray(creatinine, dtype=np.float32)
    base = np.asarray(baseline, dtype=np.float32)
    rrt = (np.zeros_like(cr) if on_rrt is None
           else np.asarray(on_rrt, dtype=np.float32) * np.ones_like(cr))

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(base > 0, cr / base, np.nan)
        stage = np.zeros_like(cr, dtype=np.float32)
        stage = np.where(ratio >= 1.5, 1.0, stage)
        stage = np.where(ratio >= 2.0, 2.0, stage)
        # ≥4.0 mg/dL only with an acute component (ratio ≥1.5) — otherwise standalone ≥4.0
        # flags chronic kidney disease (stable high creatinine) as stage 3 every day.
        stage = np.where((ratio >= 3.0) | ((cr >= 4.0) & (ratio >= 1.5)), 3.0, stage)

        if creat_rise_48h is not None:                         # absolute 0.3 mg/dL rise → ≥ stage 1
            rise = np.asarray(creat_rise_48h, dtype=np.float32)
            stage = np.where((rise >= 0.3) & (stage < 1), 1.0, stage)

        if urine_ml_kg_h is not None:                          # oliguria (only where observed)
            # WHY THERE IS NO URINE-BASED STAGE 2 (deliberate, not an omission):
            # KDIGO's urine criteria are DURATION-based, and stages 1 and 2 share the SAME rate
            # threshold -- stage 1 = <0.5 mL/kg/h for 6-12 h, stage 2 = <0.5 mL/kg/h for >=12 h.
            # They differ ONLY in how long the oliguria lasted. This pipeline aggregates to one row
            # per icu_day (a 24 h total + a count of charted hours), so `urine_ml_kg_h` is a MEAN
            # rate over the charted hours; the duration of a low-output block is not recoverable.
            # Stage 1 and stage 2 are therefore not identifiable from each other here, and emitting
            # stage 2 would require assuming the daily mean persisted >=12 h -> systematic over-call.
            # Stage 3 IS approximable because it uses a DIFFERENT rate threshold (<0.3), which a
            # daily mean can separate from <0.5. Net effect: urine-based stage 2 is under-called
            # (those stay-days land in stage 1); creatinine/RRT staging is unaffected.
            u = np.asarray(urine_ml_kg_h, dtype=np.float32)
            stage = np.where((u < 0.5) & (stage < 1), 1.0, stage)
            if urine_stage2_assume_sustained:
                # OPT-IN sensitivity analysis only (default OFF). Emits stage 2 from urine by
                # ASSUMING the daily mean rate was sustained >=12 h, which the data cannot show.
                # Over-calls stage 2 whenever oliguria was shorter than 12 h; provided so the
                # magnitude of the missing criterion can be bounded, not for headline results.
                stage = np.where((u < 0.5) & (stage < 2), 2.0, stage)
            stage = np.where(u < 0.3, np.maximum(stage, 3.0), stage)

    stage = np.where(rrt > 0, 3.0, stage)                      # any RRT that day → stage 3

    # Evaluable iff creatinine known, urine known, or on RRT; else unknown (NaN, masked).
    has_urine = (np.zeros_like(cr, dtype=bool) if urine_ml_kg_h is None
                 else ~np.isnan(np.asarray(urine_ml_kg_h, dtype=np.float32)))
    evaluable = ~np.isnan(cr) | (rrt > 0) | has_urine
    return np.where(evaluable, stage, np.nan).astype(np.float32)