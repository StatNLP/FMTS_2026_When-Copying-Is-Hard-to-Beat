"""
harmonize/train_head.py
Stage-4b — train an outcome-prediction head (mortality / LOS) from the
same Stage-3 HDF5 used for the forecaster. Labels are read from per-window
metadata:

    mortality:  1.0 if mortality_hours not NaN else 0.0
    los:        log(los_hours)   (clipped to [1, 720] h before logging)

Usage:
    python -m harmonize.train_head \
        --head mortality \
        --train-h5 data/dataset/umm_seed42_train__sofa_first_fwd.h5 \
        --val-h5   data/dataset/umm_seed42_val__sofa_first_fwd.h5 \
        --epochs 40 --batch-size 256 --lr 3e-4

`evaluate()` reports:
    mortality — BCE loss, AUROC, AUPRC, accuracy
    los       — log-space MSE / MAE, plus MAE in physical hours
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from harmonize.icu_dataset import ICUDataset
from harmonize.models import build_head
from harmonize.models.outcome_heads import FrozenBackboneHead, ForecastOutputHead
from harmonize.scores.kdigo import kdigo_stage

log = logging.getLogger("train_head")
ROOT = Path(__file__).resolve().parents[1]

LOS_LOG_OFFSET = 1.0   # log(LOS + 1) to keep gradients sane for very short stays


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

MORTALITY_HORIZON_HEADS = {"mortality7": 7.0, "mortality30": 30.0}


def labels_from_batch(head: str, batch: dict,
                      aki_ctx: dict | None = None,
                      susinf_map: dict | None = None,
                      score_ctx: dict | None = None,
                      mort_map: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (target, valid_mask) for this head.
    `valid_mask` is True for samples where the label is defined."""
    if head in MORTALITY_HORIZON_HEADS:
        # Horizon mortality (7d / 30d): died within H days of ICU admission. The label CANNOT
        # come from `mortality_hours` (= los_hours for the deceased, not a time-to-death), so it
        # is looked up per stay from the cohort (harmonize/scores/mortality.py). Survivors and
        # patients discharged alive before H are 0; eICU/HiRID are mildly right-censored.
        m = mort_map or {}
        sid = batch["stay_id"].cpu().numpy()
        y = torch.tensor([m.get(int(s), float("nan")) for s in sid], dtype=torch.float32)
        valid = torch.isfinite(y)
        return torch.nan_to_num(y), valid
    if head == "sepsis":
        # Sepsis-3 (window) = clinically-significant deterioration (ΔSOFA≥2 — the dsofa2
        # label) AND suspected infection (antibiotic + blood culture ±72h), with the
        # dysfunction falling within ±24 h of the infection onset. `susinf_map` is
        # stay_id -> susinf_onset_h (NaN = no suspected infection); the per-stay BOOLEAN it
        # used to hold mislabelled 60-81 % of the positive class (§37.38 D19). Only datasets
        # with abx timestamps + cultures are evaluable (UMM/MIMIC-III/IV; eICU/HiRID — §7.3).
        from harmonize.scores.sepsis3 import susinf_coincident
        lbl = batch["dsofa2"]
        valid = torch.isfinite(lbl)
        dys = torch.nan_to_num(lbl)                          # 0/1 deterioration
        m = susinf_map or {}
        onset = np.array([m.get(int(s), np.nan)
                          for s in batch["stay_id"].cpu().numpy()], dtype=np.float64)
        coincident = susinf_coincident(onset, batch["t0_hour"].cpu().numpy(),
                                       batch["x_in"].shape[1], batch["x_target"].shape[1])
        susinf = torch.from_numpy(coincident.astype(np.float32)).to(lbl.dtype)
        return dys * susinf, valid                           # 1 iff deterioration AND infection
    if head == "dsofa2":
        # Clinically-significant deterioration: daily SOFA(target) − daily SOFA(input) >= 2,
        # both sides the per-organ WORST summed over organs (NOT a min/"best" baseline — that
        # would give ~52 % positives instead of ~12 %, §37.52). Precomputed per window by
        # 10_scripts/recompute_sofa_scores.py and stored in the H5 (`dsofa2`) so head and
        # eval_forecast share one definition; NaN = not computable → masked out.
        lbl = batch["dsofa2"]
        valid = torch.isfinite(lbl)
        y = torch.nan_to_num(lbl)
        return y, valid
    if head == "dsofa3":
        # 3-class signed deterioration trajectory: 0=stable, 1=deteriorate (ΔSOFA≥+1),
        # 2=improve (ΔSOFA≤−1). Precomputed in the H5 (`dsofa3`, same worst−best ΔSOFA as
        # dsofa2 but signed + ±1-binned); NaN = not computable → masked. Class index is
        # returned as float; the CE step casts it to long.
        lbl = batch["dsofa3"]
        valid = torch.isfinite(lbl)
        y = torch.nan_to_num(lbl)
        return y, valid
    if head in ("aki", "aki_uo"):
        # New AKI (KDIGO >= 1) in the horizon from the GROUND-TRUTH trajectory.
        # Baseline = best (min) observed input creatinine (the KDIGO look-back minimum,
        # not just the last value); event = worst (max) observed target creatinine.
        # `aki`    : creatinine-only (the robust ICU-retrospective default).
        # `aki_uo` : ALSO the KDIGO oliguria criterion, but ONLY where urine is charted
        #            densely enough (>=50% of hours) that the recorded zeros count and
        #            missing hours don't fake oliguria — the charting density varies from
        #            ~1.0 (UMM, hourly incl. 0s) to ~0.1 (eICU, intermittent shift totals).
        ci_in, ci_tgt = aki_ctx["ci_in"], aki_ctx["ci_tgt"]
        cmu, csd = aki_ctx["creat_mean"], aki_ctx["creat_std"]
        xi = batch["x_in"][..., ci_in] * csd + cmu       # (B, T_in)  mg/dL
        mi = batch["mask_in"][..., ci_in].bool()
        xt = batch["x_target"][..., ci_tgt] * csd + cmu  # (B, T_out) mg/dL
        mt = batch["mask_target"][..., ci_tgt].bool()
        base = torch.where(mi, xi, torch.full_like(xi, float("inf"))).min(dim=1).values
        worst = torch.where(mt, xt, torch.full_like(xt, float("-inf"))).max(dim=1).values
        has_tgt = mt.any(dim=1)
        base_np = torch.where(mi.any(dim=1), base, torch.full_like(base, float("nan"))).detach().cpu().numpy()
        worst_np = worst.detach().cpu().numpy()
        urine = None
        if head == "aki_uo" and aki_ctx.get("ui_tgt") is not None:
            ui, wi = aki_ctx["ui_tgt"], aki_ctx["wi"]
            umu, usd = aki_ctx["urine_mean"], aki_ctx["urine_std"]
            muo = batch["mask_target"][..., ui].bool()
            uo = batch["x_target"][..., ui] * usd + umu
            n_ch = muo.sum(dim=1)
            # mean charted urine rate (mL/h) — charted 0s included — ÷ weight = mL/kg/h
            rate = torch.where(muo, uo, torch.zeros_like(uo)).sum(dim=1) / n_ch.clamp(min=1)
            wt = batch["static"][..., wi].clamp(min=20.0)   # kg — static is RAW, not z-scored
            dense = n_ch >= 12                               # >=50% of 24h charted → reliable
            urine = torch.where(dense, rate / wt,
                                torch.full_like(rate, float("nan"))).detach().cpu().numpy()
        # D16: kdigo_stage accepts creat_rise_48h (the absolute >= 0.3 mg/dL rise, KDIGO's most
        # sensitive stage-1 trigger) but it was never passed, so only the RELATIVE criteria
        # (>= 1.5x / 2.0x / 3.0x baseline) and the urine criterion ever fired. The rise is already
        # in hand: `base` is the minimum observed INPUT creatinine and `worst` the maximum observed
        # TARGET creatinine, and window_in + window_out = 24 + 24 h is exactly KDIGO's 48 h window,
        # so worst - base IS the 48 h rise. Gated so every published AKI number stays reproducible
        # with --no-aki-absolute-rise.
        rise_np = None
        if (aki_ctx or {}).get("absolute_rise", True):
            rise_np = (worst_np - base_np)
        stage = kdigo_stage(worst_np, base_np, creat_rise_48h=rise_np,
                            urine_ml_kg_h=urine)   # (B,) 0..3, NaN where N/A
        y = torch.from_numpy((np.nan_to_num(stage, nan=0.0) >= 1).astype(np.float32))
        valid = torch.from_numpy(has_tgt.cpu().numpy() & np.isfinite(base_np))
        return y, valid
    if head == "mortality":
        mh = batch["mortality_hours"]
        valid = ~torch.isnan(mh)              # both classes contribute when set
        # mortality = 1 if mortality_hours is not nan; 0 if nan (survivor)
        # But Stage-3 stores NaN for survivors AND for unknown — distinguish via
        # los_hours: survivors have a finite LOS, unknowns shouldn't appear.
        y = torch.where(torch.isnan(mh), torch.zeros_like(mh), torch.ones_like(mh))
        return y, torch.ones_like(y, dtype=torch.bool)  # always valid for UMM
    if head in ("los3", "los7"):
        # Binary length-of-stay at the conventional thresholds. Regression on log-LoS answers a
        # different question and is not comparable to the benchmark literature. NOTE: only
        # meaningful with --first-day-only -- across all windows a stay must already be long to
        # contribute a late window, so "LoS > 3d" is 91-98 % positive and the window's own
        # t0_hour leaks the answer (corr 0.12-0.36 with LoS).
        thr = 72.0 if head == "los3" else 168.0
        los = batch["los_hours"]
        y = (los > thr).float()
        return y, torch.isfinite(los)
    if head == "los":
        los = batch["los_hours"].clamp(min=1.0, max=720.0)  # 1h..30d
        y = torch.log(los + LOS_LOG_OFFSET)
        valid = torch.isfinite(y)
        return y, valid
    if head in ("sofa", "saps"):
        # Severity-score REGRESSION on the worst-of-target-window ground truth — the same
        # quantity eval_forecast's SOFA / SAPS-II MAE compares the forecast against, so the
        # discriminative head-MAE is apples-to-apples with the forecast read-off (R2 dissociation).
        from harmonize.scores.sofa import daily_sofa, fill_ffb
        from harmonize.scores.sapsii import sapsii_batch
        sc = score_ctx
        xt = batch["x_target"].cpu().numpy()
        mt = batch["mask_target"].cpu().numpy().astype(bool)
        phys = fill_ffb(xt * sc["tgt_stds"] + sc["tgt_means"], mt)    # denormalise + symmetric fill
        if head == "sofa":
            score = daily_sofa(phys, sc["feats_tgt"], sc["dataset"])  # (B,) worst-of-window SOFA total
        else:
            age = (batch["static"].cpu().numpy()[:, sc["age_idx"]]
                   if sc.get("age_idx") is not None else None)        # RAW age (static is un-normed)
            # NB: admission_type (0/6/8) + chronic_disease (0/9/10/17) are non-window per-stay constants
            # not derivable from 24 h of physiology, so they are NOT in the head target (they would only
            # add irreducible between-stay variance → inflate MAE). They ARE added in the eval read-off,
            # where they cancel in pred−gt, so the head MAE stays apples-to-apples with the read-off
            # (both = physiology-forecast error). The full SAPS-II (incl. adm/chronic) is validated against
            # Le Gall mortality separately (sapsii_mortality analysis).
            score = sapsii_batch(phys, sc["feats_tgt"], age, None)    # (B,) worst-of-window SAPS-II (physio+age)
        score = np.asarray(score, dtype=np.float32)
        valid = torch.from_numpy(np.isfinite(score) & mt.any(axis=(1, 2)))
        return torch.from_numpy(np.nan_to_num(score, nan=0.0)), valid
    raise ValueError(f"unknown head: {head!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(head: str, model: nn.Module, loader: DataLoader,
             device: torch.device, aki_ctx: dict | None = None,
             susinf_map: dict | None = None, score_ctx: dict | None = None,
             mort_map: dict | None = None) -> dict:
    model.eval()
    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []
    for batch in loader:
        x_in   = batch["x_in"].to(device)
        m_in   = batch["mask_in"].to(device)
        static = batch["static"].to(device)
        y, valid = labels_from_batch(head, batch, aki_ctx, susinf_map, score_ctx, mort_map)
        y = y.to(device); valid = valid.to(device)
        pred = model(x_in, m_in, static)
        all_pred.append(pred[valid].cpu())
        all_true.append(y[valid].cpu())
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    if head == "dsofa3":
        pred_cls = pred.argmax(axis=1)          # pred: (N, 3) logits
        f1s = []
        for c in range(3):
            tp = float(((pred_cls == c) & (true == c)).sum())
            fp = float(((pred_cls == c) & (true != c)).sum())
            fn = float(((pred_cls != c) & (true == c)).sum())
            prec = tp / max(tp + fp, 1.0); rec = tp / max(tp + fn, 1.0)
            f1s.append(2 * prec * rec / max(prec + rec, 1e-7))
        return {"macro_f1": float(np.mean(f1s)),
                "accuracy": float((pred_cls == true).mean()),
                "f1_stable": f1s[0], "f1_deteriorate": f1s[1], "f1_improve": f1s[2],
                "support": [int((true == c).sum()) for c in range(3)],
                "n": int(len(true))}
    if head in ("mortality", "aki", "aki_uo", "dsofa2", "sepsis", "los3", "los7") or head in MORTALITY_HORIZON_HEADS:
        prob = 1.0 / (1.0 + np.exp(-pred))
        bce = -(true * np.log(prob.clip(1e-7)) + (1 - true) * np.log((1 - prob).clip(1e-7))).mean()
        auroc, auprc = _binary_curve(true, prob)
        acc = ((prob >= 0.5).astype(np.float32) == true).mean()
        # F1 at the default 0.5 threshold (≈0 for conservatively-calibrated rare events) and
        # the best achievable F1 over a threshold sweep (the achievable operating point).
        def _f1(thr):
            yh = prob >= thr
            tp = float(((yh == 1) & (true == 1)).sum())
            fp = float(((yh == 1) & (true == 0)).sum())
            fn = float(((yh == 0) & (true == 1)).sum())
            pr = tp / max(tp + fp, 1.0); rc = tp / max(tp + fn, 1.0)
            return 2 * pr * rc / max(pr + rc, 1e-9)
        return {"bce": float(bce), "auroc": float(auroc), "auprc": float(auprc),
                "accuracy": float(acc), "f1_at_0.5": float(_f1(0.5)),
                "best_f1": float(max(_f1(t) for t in np.linspace(0.05, 0.95, 19))),
                "pos_rate": float(true.mean()), "n": int(len(true))}
    if head in ("sofa", "saps"):
        return {"mae": float(np.abs(pred - true).mean()),
                "rmse": float(((pred - true) ** 2).mean() ** 0.5),
                "mean_true": float(true.mean()), "n": int(len(true))}
    # LOS
    mse = float(((pred - true) ** 2).mean())
    mae_log = float(np.abs(pred - true).mean())
    mae_hours = float(np.abs(np.exp(pred) - np.exp(true)).mean())  # back to hours
    return {"mse_log": mse, "mae_log": mae_log, "mae_hours": mae_hours,
            "rmse_log": mse ** 0.5, "n": int(len(true))}


def _binary_curve(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """AUROC + AUPRC, tie-safe.

    The previous hand-rolled version sorted by -p and integrated the resulting curve with
    `np.trapezoid`. That is wrong whenever predictions tie: `argsort` returns an arbitrary order
    among equal scores and the integral then treats that arbitrary order as signal. Measured on
    664 rows with every prediction identical it returned 0.4826, and on label-sorted rows it
    returned **1.0000**, where the correct answer is 0.5 in both cases. Because the heads select
    their epoch on `auroc`, that did not merely misreport — it could pick the wrong checkpoint.
    Ties are the normal case for a degenerate input (e.g. the all-zero forecast rung of the
    §16.10a ladder) and can occur for a saturated head anywhere.

    scikit-learn handles ties correctly (it assigns the average rank) and is already a hard
    dependency, so there is no reason to hand-roll this.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y).astype(float).ravel()
    p = np.asarray(p).astype(float).ravel()
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")       # AUROC is undefined with one class present
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    # Seed every source of randomness the head touches: weight init, dropout, and the DataLoader
    # shuffle. Without this a rerun of the same configuration gives a different number, which is
    # how 45 head cells came to carry no reproducible provenance.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info("Head=%s  device=%s  seed=%d", args.head, device, args.seed)

    import datetime
    from harmonize.common import archive_existing_run
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.output_dir) / args.run_name
    if not args.eval_only:
        archive_existing_run(out_dir)   # never silently overwrite a previous run's model
    elif not (out_dir / "best.pt").exists():
        raise SystemExit(f"--eval-only needs an existing checkpoint: {out_dir / 'best.pt'}")
    elif not (out_dir / "final_metrics.json").exists():
        # A run killed by the wall clock has best.pt but no final_metrics.json: its checkpoint is
        # whatever epoch happened to be best when it died, and nothing recorded that selection.
        # Scoring it would report an unselected checkpoint with sentinel provenance
        # (best_epoch=-1, best_val=-inf), which reads like data. Refuse.
        raise SystemExit(
            f"--eval-only needs a COMPLETED run: {out_dir / 'final_metrics.json'} is missing, so "
            f"{out_dir.name} never finished its epoch selection. Retrain it first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ICUDataset(args.train_h5)
    val_ds   = ICUDataset(args.val_h5)
    if args.first_day_only:
        # One window per stay, the admission window. Subset wraps the dataset without touching
        # ICUDataset, so the preloading and feature space are unchanged.
        import h5py
        from torch.utils.data import Subset

        class _Sub(Subset):
            """Subset that forwards dataset attributes (n_features_in, features_target, ...),
            which the head builder reads off the dataset object."""
            def __getattr__(self, name):
                return getattr(self.dataset, name)

        def _admission(h5):
            with h5py.File(h5, "r") as f:
                return np.flatnonzero(f["t0_hour"][:] == 0)
        i_tr, i_va = _admission(args.train_h5), _admission(args.val_h5)
        log.info("--first-day-only: train %d -> %d, val %d -> %d windows",
                 len(train_ds), len(i_tr), len(val_ds), len(i_va))
        if len(i_tr) == 0 or len(i_va) == 0:
            raise SystemExit("--first-day-only left an empty split; is t0_hour present?")
        train_ds, val_ds = _Sub(train_ds, i_tr.tolist()), _Sub(val_ds, i_va.tolist())
        if args.test_h5:
            test_ds = ICUDataset(args.test_h5)
            i_te = _admission(args.test_h5)
            if len(i_te) == 0:
                raise SystemExit("--first-day-only left an empty test split")
            log.info("--first-day-only: test %d -> %d windows", len(test_ds), len(i_te))
            test_ds = _Sub(test_ds, i_te.tolist())
    elif args.test_h5:
        test_ds = ICUDataset(args.test_h5)
    if args.input_traj_from is not None:
        # Substitute the input window with a precomputed trajectory. _TrajDS forwards every
        # attribute and every other field, so the head builder, the label extraction and the
        # first-day subsetting above are all untouched; only x_in changes (and mask_in becomes
        # all-ones, because a forecast has no missingness).
        paths = [Path(s) for s in str(args.input_traj_from).split(",")]
        if len(paths) < 2:
            raise SystemExit("--input-traj-from needs train,val[,test] paths (comma-separated)")

        class _TrajDS(torch.utils.data.Dataset):
            def __init__(self, base, npy):
                self.base = base
                self.traj = np.load(npy, mmap_mode="r")
                if len(self.traj) != len(base):
                    raise SystemExit(
                        f"{Path(npy).name}: {len(self.traj)} rows != {len(base)} windows. The "
                        "trajectory must be row-aligned to this split AFTER --first-day-only, so "
                        "dump it with `ensemble_eval --rows first-day`.")
            def __len__(self):
                return len(self.base)
            def __getattr__(self, name):
                # NEVER forward dunders. torch's _MapDatasetFetcher prefers a batched
                # `__getitems__` when the dataset has one; forwarding that to `self.base` handed
                # the fetcher the BASE dataset's batched getter, which bypassed __getitem__ below
                # entirely and silently returned the original x_in. The trajectory substitution
                # then did nothing, and 88 ladder cells came out bit-identical across 17 rungs.
                if name.startswith("__"):
                    raise AttributeError(name)
                return getattr(self.base, name)
            def __getitems__(self, idxs):
                # Defined explicitly so the fetcher's batched path also goes through __getitem__.
                return [self[i] for i in idxs]
            def __getitem__(self, k):
                item = self.base[k]
                tr = torch.from_numpy(np.asarray(self.traj[k], dtype=np.float32))
                item["x_in"] = tr
                item["mask_in"] = torch.ones_like(tr, dtype=item["mask_in"].dtype)
                return item

        train_ds, val_ds = _TrajDS(train_ds, paths[0]), _TrajDS(val_ds, paths[1])
        if test_loader_needed := (args.test_h5 is not None):
            if len(paths) < 3:
                raise SystemExit("--test-h5 with --input-traj-from needs a third trajectory path")
            test_ds = _TrajDS(test_ds, paths[2])
        log.info("--input-traj-from: head reads the PREDICTED trajectory, not the input window "
                 "(train=%s, val=%s%s)", paths[0].name, paths[1].name,
                 f", test={paths[2].name}" if test_loader_needed and len(paths) > 2 else "")
        # VERIFY the substitution survives the DataLoader. It silently did not once: torch's
        # fetcher prefers a batched `__getitems__`, `__getattr__` forwarded that to the base
        # dataset, and the head trained on the original window while the log said otherwise —
        # producing 17 bit-identical "rungs". Checking one batch costs milliseconds and makes
        # that failure loud instead of plausible.
        _probe = next(iter(DataLoader(train_ds, batch_size=min(4, len(train_ds)), num_workers=0)))
        _want = np.asarray(train_ds.traj[0], dtype=np.float32)
        if not np.allclose(_probe["x_in"][0].numpy(), _want, atol=1e-6):
            raise SystemExit(
                "--input-traj-from did NOT reach the batch: the DataLoader returned different "
                "x_in than the trajectory file. Every rung would train on the same input. "
                "Check that the dataset wrapper does not forward __getitems__ to its base.")
        log.info("    verified: the trajectory reaches the batch")

    _g = torch.Generator(); _g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=_g,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    test_loader = None
    if args.test_h5:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True,
                                 persistent_workers=args.num_workers > 0)

    log.info("Train windows: %d | Val windows: %d", len(train_ds), len(val_ds))
    log.info("F_in=%d  F_static=%d  T_in=%d",
             train_ds.n_features_in, train_ds.n_features_static, train_ds.window_in)

    aki_ctx = None
    if args.head in ("aki", "aki_uo"):
        dataset = Path(args.train_h5).name.split("_seed")[0]
        norm_path = args.norm_stats or (ROOT / "data" / "splits" / f"{dataset}_norm_stats.parquet")
        norm = pd.read_parquet(norm_path).set_index("variable")
        feats_in, feats_tgt = list(train_ds.features_in), list(train_ds.features_target)
        if "creatinine" not in feats_in or "creatinine" not in feats_tgt:
            raise SystemExit(f"AKI head needs 'creatinine' in features (dataset {dataset})")
        aki_ctx = {"ci_in": feats_in.index("creatinine"),
                   "ci_tgt": feats_tgt.index("creatinine"),
                   "creat_mean": float(norm.loc["creatinine", "train_mean"]),
                   "creat_std":  float(norm.loc["creatinine", "train_std"]),
                   "ui_tgt": None,
                   "absolute_rise": not args.no_aki_absolute_rise}
        feats_static = list(train_ds.features_static)
        # KDIGO oliguria criterion needs target urine + body weight (both z-scored)
        if ("urine_output" in feats_tgt and "weight_admission" in feats_static
                and "urine_output" in norm.index and "weight_admission" in norm.index):
            aki_ctx.update({
                "ui_tgt": feats_tgt.index("urine_output"),
                "wi": feats_static.index("weight_admission"),
                "urine_mean": float(norm.loc["urine_output", "train_mean"]),
                "urine_std":  float(norm.loc["urine_output", "train_std"]),
                "weight_mean": float(norm.loc["weight_admission", "train_mean"]),
                "weight_std":  float(norm.loc["weight_admission", "train_std"])})
        log.info("AKI label: creatinine in/tgt=%d/%d  urine criterion=%s",
                 aki_ctx["ci_in"], aki_ctx["ci_tgt"],
                 "on" if aki_ctx["ui_tgt"] is not None else "off")

    susinf_map = None
    if args.head == "sepsis":
        if args.long_parquet is None or not Path(args.long_parquet).exists():
            raise SystemExit("--head sepsis requires --long-parquet (suspected-infection map); "
                             "sepsis-evaluable on UMM / MIMIC-III / MIMIC-IV (abx+culture cluster) and "
                             "eICU (MLHC-2024 union) — §7.3")
        from harmonize.scores.sepsis3 import susinf_per_stay
        ldf = pd.read_parquet(args.long_parquet, columns=["stay_id", "variable", "hour", "value"])
        # eICU has no abx-start+culture clustering → auto-use the MLHC-2024 *union* definition
        # (abx OR culture), mirroring eval_forecast.load_susinf; --susinf-mode union forces it elsewhere.
        _union = (args.susinf_mode == "union") or ("eicu" in Path(args.long_parquet).name.lower())
        _si = susinf_per_stay(ldf, union=_union).set_index("stay_id")
        # Keep the ONSET HOUR, not just the flag: the label needs it to enforce Sepsis-3's
        # ±24 h coincidence (§37.38 D19). NaN onset == no suspected infection.
        si = _si["susinf_onset_h"].where(_si["has_susinf"], np.nan)
        susinf_map = {int(k): float(v) for k, v in si.items()}
        share = (float(np.isfinite(list(susinf_map.values())).mean())
                 if susinf_map else 0.0)
        log.info("Sepsis label: suspected-infection map for %d stays (susinf share %.1f%%)",
                 len(susinf_map), 100 * share)
        if share < 0.005:
            raise SystemExit("susinf share ~0 — dataset not sepsis-evaluable (no abx+culture)")

    mort_map = None
    if args.head in MORTALITY_HORIZON_HEADS:
        from harmonize.scores.mortality import load_mortality_horizon_map, is_censored
        dataset = Path(args.train_h5).name.split("_seed")[0]
        days = MORTALITY_HORIZON_HEADS[args.head]
        mort_map = load_mortality_horizon_map(dataset, days)
        rate = sum(mort_map.values()) / max(len(mort_map), 1)
        log.info("%s label: %d stays, %.1f%% positive (%.0f-day mortality)%s",
                 args.head, len(mort_map), 100 * rate, days,
                 "  [right-censored: in-hospital deaths only -> slight under-estimate]"
                 if is_censored(dataset) else "")

    score_ctx = None
    if args.head in ("sofa", "saps"):
        dataset = Path(args.train_h5).name.split("_seed")[0]
        norm_path = args.norm_stats or (ROOT / "data" / "splits" / f"{dataset}_norm_stats.parquet")
        norm = pd.read_parquet(norm_path).set_index("variable")
        feats_tgt = list(train_ds.features_target)
        tgt_means = np.array([float(norm.loc[f, "train_mean"]) if f in norm.index else 0.0
                              for f in feats_tgt], np.float32)
        tgt_stds = np.array([float(norm.loc[f, "train_std"]) if f in norm.index else 1.0
                             for f in feats_tgt], np.float32)
        feats_static = list(train_ds.features_static)
        score_ctx = {"tgt_means": tgt_means, "tgt_stds": tgt_stds, "feats_tgt": feats_tgt,
                     "dataset": dataset,
                     "age_idx": feats_static.index("age") if "age" in feats_static else None}
        log.info("%s head: worst-of-window regression (%d target feats, dataset=%s, age=%s)",
                 args.head, len(feats_tgt), dataset, score_ctx["age_idx"] is not None)

    if args.backbone_fm is not None:
        # Frozen-FM-backbone variant (§9.7): classify on a pretrained (or fine-tuned)
        # foundation model's frozen encoder representation — extends the R2 dissociation
        # to FMs ("does a 10-60x larger pretrained representation encode the outcome?").
        from harmonize.eval_forecast import build_fm
        backbone, bcfg = build_fm(args.backbone_fm, train_ds, device,
                                  args.backbone_chronos_id, "median")
        d_repr = backbone.repr_dim(device)
        n_st = train_ds.n_features_static if args.head_static else 0
        model = FrozenBackboneHead(backbone, d_model=d_repr, n_static=n_st,
                                   dropout=args.dropout).to(device)
        log.info("Frozen-FM head on %s (%s, d_model=%d, +static=%s) — FM frozen, classifier trains",
                 args.backbone_fm, args.backbone_chronos_id, d_repr, bool(n_st))
    elif args.backbone_from is not None:
        # Frozen-backbone variant: classify on a trained forecast model's (frozen) encoder
        # representation — probes whether the FORECAST'S learned representation encodes the
        # label, vs. the from-scratch heads which prove only the raw input carries it.
        from harmonize.eval_forecast import load_model
        backbone, bcfg = load_model(args.backbone_from, train_ds, device)
        n_st = train_ds.n_features_static if args.head_static else 0
        model = FrozenBackboneHead(backbone, d_model=int(bcfg.get("d_model", args.d_model)),
                                   n_static=n_st, dropout=args.dropout).to(device)
        log.info("Frozen-backbone head on %s (d_model=%s, +static=%s) — backbone frozen, classifier trains",
                 args.backbone_from, bcfg.get("d_model"), bool(n_st))
    elif args.forecast_output_from is not None:
        # Two-stage forecast→detect: classify on the FROZEN forecaster's GENERATED trajectory
        # (its student rollout output), with a from-scratch learned read-off — the third R2
        # probe (vs hand-coded read-off ≈0.6 and frozen-representation head ≈0.7). Tests whether
        # a learned read-off recovers the deterioration the hand-coded SOFA/KDIGO extraction loses.
        from harmonize.eval_forecast import load_model
        forecaster, fcfg = load_model(args.forecast_output_from, train_ds, device)
        n_st = train_ds.n_features_static if args.head_static else 0
        model = ForecastOutputHead(
            forecaster, n_features_target=train_ds.n_features_target,
            n_features_static=train_ds.n_features_static, window_out=train_ds.window_out,
            d_model=args.d_model, n_static=n_st, dropout=args.dropout).to(device)
        log.info("Forecast→detect head on %s (F_target=%d, T_out=%d, +static=%s) — forecaster frozen, "
                 "read-off trains on the GENERATED trajectory", args.forecast_output_from,
                 train_ds.n_features_target, train_ds.window_out, bool(n_st))
    else:
        model = build_head(
            args.head,
            n_features_in     = train_ds.n_features_in,
            n_features_static = train_ds.n_features_static,
            window_in         = train_ds.window_in,
            d_model           = args.d_model,
            dropout           = args.dropout,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params (trainable): %s", f"{n_params:,}")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)), pct_start=0.1,
    )

    if args.head == "dsofa3":
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        sel_key = "macro_f1"; sel_dir = +1   # maximise (robust to the stable-class majority)
    elif args.head in ("mortality", "aki", "aki_uo", "dsofa2", "sepsis", "los3", "los7") or args.head in MORTALITY_HORIZON_HEADS:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        sel_key = "auroc"; sel_dir = +1     # maximise
    elif args.head in ("sofa", "saps"):
        loss_fn = nn.MSELoss(reduction="none")
        sel_key = "mae"; sel_dir = -1        # minimise the worst-of-window score MAE
    else:  # los
        loss_fn = nn.MSELoss(reduction="none")
        sel_key = "mae_hours"; sel_dir = -1  # minimise

    cfg = vars(args).copy()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    best_score = -float("inf") * sel_dir
    best_epoch = -1
    train_log: list[dict] = []

    for epoch in range(1, 0 if args.eval_only else args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_steps = 0
        for batch in train_loader:
            x_in   = batch["x_in"].to(device, non_blocking=True)
            m_in   = batch["mask_in"].to(device, non_blocking=True)
            static = batch["static"].to(device, non_blocking=True)
            y, valid = labels_from_batch(args.head, batch, aki_ctx, susinf_map, score_ctx, mort_map)
            y = y.to(device); valid = valid.to(device)
            if not valid.any():
                continue
            pred = model(x_in, m_in, static)
            y_loss = y.long() if args.head == "dsofa3" else y   # CE needs class indices
            loss = (loss_fn(pred, y_loss) * valid.float()).sum() / valid.sum().clamp(min=1)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            running += loss.item()
            n_steps += 1

        train_loss = running / max(n_steps, 1)
        val = evaluate(args.head, model, val_loader, device, aki_ctx, susinf_map, score_ctx, mort_map)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": train_loss,
               **{f"val_{k}": v for k, v in val.items()},
               "elapsed_s": round(elapsed, 1)}
        train_log.append(row)
        log.info("Ep %2d/%2d  train=%.4f  val=%s  (%.1fs)",
                 epoch, args.epochs, train_loss,
                 ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in val.items()),
                 elapsed)
        improved = (val[sel_key] - best_score) * sel_dir > 0
        if improved:
            best_score = val[sel_key]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
            log.info("   -> new best val_%s=%.4f saved", sel_key, best_score)
        with open(out_dir / "train_log.jsonl", "w") as f:
            for r in train_log:
                f.write(json.dumps(r) + "\n")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    final_val = evaluate(args.head, model, val_loader, device, aki_ctx, susinf_map, score_ctx, mort_map)
    finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    prior = {}
    fm = out_dir / "final_metrics.json"
    if args.eval_only and fm.exists():
        prior = json.loads(fm.read_text())
    final = {"run_name": args.run_name, "seed": args.seed,
             "started_at": prior.get("started_at", started_at),
             "finished_at": finished_at,
             "best_epoch": prior.get("best_epoch", best_epoch),
             "selection": prior.get("selection", sel_key),
             "best_val": prior.get("best_val", best_score), "val": final_val}
    if test_loader is not None:
        # Selection already happened on val (or in the original run); this reads test once.
        final["test"] = evaluate(args.head, model, test_loader, device,
                                 aki_ctx, susinf_map, score_ctx, mort_map)
        final["test_h5"] = str(args.test_h5)
        log.info("TEST  %s", {k: round(v, 4) for k, v in final["test"].items()
                              if isinstance(v, (int, float))})
    fm.write_text(json.dumps(final, indent=2))
    from harmonize.common import append_run_manifest
    if args.eval_only:
        log.info("Done (eval-only). Wrote %s", fm)
        return
    append_run_manifest(args.run_name, out_dir, started_at, finished_at,
                        model=f"head:{args.head}", dataset=Path(args.train_h5).stem.split("_seed")[0],
                        epochs_run=best_epoch, best_metric=sel_key, best_val=best_score)
    log.info("Done. best val_%s=%.4f at epoch %d", sel_key, best_score, best_epoch)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 4b — outcome head trainer")
    p.add_argument("--head", required=True,
                   choices=["mortality", "mortality7", "mortality30", "los", "los3", "los7",
                            "aki", "aki_uo", "dsofa2", "dsofa3", "sepsis", "sofa", "saps"])
    p.add_argument("--susinf-mode", choices=["cluster", "union"], default="cluster",
                   help="(sepsis head) suspected infection: 'cluster' = abx start AND blood culture within "
                        "±72h (Sepsis-3 / Singer); 'union' = abx OR culture (MLHC-2024 eICU definition)")
    p.add_argument("--train-h5", required=True)
    p.add_argument("--val-h5",   required=True)
    p.add_argument("--test-h5", default=None,
                   help="score the selected checkpoint on this split as well. The point estimate "
                        "and the epoch are still chosen on --val-h5; test is read once and written "
                        "to the 'test' key of final_metrics.json.")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training: load the run's existing best.pt and (re)score it. Requires "
                        "the run directory to already hold best.pt. Use with --test-h5 to add a "
                        "test read to a completed run without retraining it.")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--run-name", default="head_run")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="seeds python/numpy/torch and the DataLoader shuffle. Before this existed "
                        "the heads were unseeded, so their numbers were not reproducible and had "
                        "no measurable spread; the §16.10a gains of +0.010..+0.039 were single "
                        "unseeded runs. Vary it to get a spread.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--backbone-from", type=Path, default=None,
                   help="Frozen-backbone variant (§24.5): classify on a trained forecast "
                        "checkpoint's frozen encoder representation instead of a from-scratch "
                        "MLP. Path to the forecast checkpoint dir (same feature space).")
    p.add_argument("--forecast-output-from", type=Path, default=None,
                   help="Two-stage forecast→detect variant (§24.5): classify on a trained forecast "
                        "checkpoint's GENERATED trajectory (student rollout output) with a from-scratch "
                        "learned read-off — tests whether a learned read-off recovers the deterioration "
                        "the hand-coded SOFA/KDIGO extraction loses. Path to the checkpoint dir.")
    p.add_argument("--input-traj-from", type=Path, default=None,
                   help="replace the head's INPUT WINDOW with a precomputed forecast trajectory "
                        "(.npy of shape (N, T, F), row-aligned to this split in dataset order; "
                        "produced by `ensemble_eval --dump-combined`, or any cached base-model "
                        "dump). The head architecture is unchanged, so this arm is exactly "
                        "comparable to `raw` — the only difference is whether the 24xF tensor is "
                        "the observed past or a predicted future. Pass one path per split as "
                        "train,val[,test].")
    p.add_argument("--backbone-fm", choices=["chronos", "chronos2"], default=None,
                   help="Frozen-FM-backbone variant (§9.7): classify on a pretrained foundation "
                        "model's frozen encoder representation. Use --backbone-chronos-id for the "
                        "HF id or a fine-tuned checkpoint dir.")
    p.add_argument("--backbone-chronos-id", default="amazon/chronos-bolt-base",
                   help="HF id / local fine-tuned dir for --backbone-fm")
    p.add_argument("--first-day-only", action="store_true",
                   help="Keep one window per stay, the admission window (t0_hour == 0). This is "
                        "the deployable setting -- one prediction per patient at admission -- and "
                        "it is REQUIRED for los3/los7: across all windows the window's own position "
                        "in the stay leaks the label.")
    p.add_argument("--head-static", action="store_true",
                   help="concatenate the raw static (demographics) to the frozen-backbone "
                        "representation before the classifier — 'representation + demographics' "
                        "variant (fair vs the from-scratch heads, esp. mortality/LOS). Default off "
                        "= pure representation probe.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-aki-absolute-rise", action="store_true", default=False,
                   help="D16: drop KDIGO's absolute >=0.3 mg/dL 48h rise criterion and score "
                        "only the relative ones, as every AKI number published before "
                        "2026-11-02 did. Use only to reproduce those.")
    p.add_argument("--norm-stats", type=Path, default=None,
                   help="defaults to data/splits/<ds>_norm_stats.parquet (aki head only)")
    p.add_argument("--long-parquet", type=Path, default=None,
                   help="Stage-1 long parquet → suspected-infection map (REQUIRED for --head sepsis)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
