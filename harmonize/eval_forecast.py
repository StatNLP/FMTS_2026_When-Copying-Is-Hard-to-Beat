"""
harmonize/eval_forecast.py
Stage 5 — downstream clinical evaluation of a trained forecast model.

The core claim of Staniek et al. (2024) is that downstream labels are
deterministic functions of the forecasted physiological trajectory. This
script loads a trained checkpoint, generates 24h-ahead forecasts on the test
split, and compares clinical scores computed on the PREDICTED trajectory
against those on the GROUND-TRUTH trajectory:

  - Forecast error      : masked MSE/MAE per-variable + global (sanity vs val)
  - SOFA                : predicted vs ground-truth window SOFA (total + 5/6
                          components), MAE + ΔSOFA≥2 detection (acc/sens/spec)
  - SAPS-II             : predicted vs ground-truth worst-of-window score, MAE
  - Sepsis-3 (optional) : with --long-parquet, joins susinf_per_stay (antibiotic
                          + blood culture ±72h) and reports sepsis-onset
                          detection on the forecast vs ground-truth SOFA.

Predicted SOFA uses the dense model output directly (the decoder emits every
timestep). Ground-truth SOFA uses LOCF on x_target seeded from the last input
step (carry last *observed* value forward, not a model fill) — via
`_window_sofa` / `daily_sofa` in `harmonize.scores.sofa`.

Usage:
    python -m harmonize.eval_forecast \
        --checkpoint checkpoints/umm__4h_d128_l1_dr0_2_s42 \
        --test-h5    data/dataset/umm_seed42_test__all_4h_fwd.h5 \
        [--long-parquet data/long/umm_long.parquet]    # enables Sepsis-3
        [--out docs/tables/eval_<run>.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from harmonize.icu_dataset import ICUDataset
from harmonize.models import build_model
from harmonize.scores.sofa import (daily_sofa, denormalise, fill_ffb, locf, sofa_components,
                                   sofa_hour_mask, _window_sofa)
from harmonize.scores.sapsii import sapsii_batch
from harmonize.scores.kdigo import kdigo_stage

#: channels where absence of a record means "not administered" (a known zero), not "not measured".
#: Used by 10_scripts/sofa_clean_comparison.py, which is the ONE place the medication-zero
#: analysis lives. It was briefly duplicated here as a `medzero_*` arm that zeroed only the
#: prediction while the ground truth kept its train-mean phantom infusion, so the two reported
#: numbers disagreed; the arm was removed rather than fixed twice.
MEDICATION_ZERO_CHANNELS = ("norepinephrine", "dopamine", "epinephrine", "dobutamine",
                            "vasopressin", "milrinone", "insulin", "furosemide", "heparin",
                            "midazolam", "propofol", "fentanyl", "morphine", "dexmedetomidine",
                            "amiodarone")

log = logging.getLogger("eval_forecast")
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build(cfg: dict, ds: ICUDataset, device: torch.device):
    return build_model(
        cfg["model"],
        n_features_in=ds.n_features_in,
        n_features_target=ds.n_features_target,
        n_features_static=ds.n_features_static,
        window_in=ds.window_in,
        window_out=ds.window_out,
        target_idx_in_input=ds.target_indices_in_input(),
        d_model=cfg.get("d_model", 128), d_ff=cfg.get("d_ff", -1),
        n_heads=cfg.get("n_heads", 4),
        n_layers_enc=cfg.get("n_layers", 2),
        n_layers_dec=(cfg.get("n_layers_dec") if cfg.get("n_layers_dec") is not None
                      else cfg.get("n_layers", 2)),   # honour 2/1 checkpoints; old cfgs fall back to n_layers
        dropout=cfg.get("dropout", 0.1),
        chronos_id=cfg.get("chronos_id", "amazon/chronos-bolt-base"),
        point=cfg.get("point", "median"),
        d_state=cfg.get("d_state", 16), d_conv=cfg.get("d_conv", 4),
        expand=cfg.get("expand", 2),
        # PatchTST's patching geometry determines pos/embed/head shapes. Without forwarding these
        # the model is rebuilt at the defaults (8/4) and load_state_dict fails on every checkpoint
        # trained with another setting (e.g. the `pl4st2` sweep runs) — so those checkpoints were
        # simply unloadable through this path.
        patch_len=cfg.get("patch_len", 8), stride=cfg.get("stride", 4),
    ).to(device)


def load_model(checkpoint_dir: Path, ds: ICUDataset, device: torch.device):
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    model = _build(cfg, ds, device)
    state = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(state)
    # dense_pteacher: `exogenous_idx` (which target channels are medications, fed from GT in the
    # AR rollout) is a non-persistent buffer — NOT in config.json or best.pt — so a freshly built
    # eval model has it empty and silently treats meds as ordinary channels. Reconstruct it from the
    # feature catalog exactly as train_forecast does, so the eval feeds the true future meds (the
    # whole point of the model). Without this the inference-time med-feeding in the eval loop is a no-op.
    if cfg.get("model") == "dense_pteacher" and getattr(model, "exogenous_idx", None) is not None:
        try:
            from harmonize.common import FeatureCatalog
            cand = next((p for p in (Path("configs/features.yaml"),
                                     checkpoint_dir.resolve().parents[1] / "configs" / "features.yaml")
                         if p.exists()), None)
            if cand is not None:
                cat = FeatureCatalog.from_yaml(cand)
                idx = []
                for i, f in enumerate(ds.features_target):
                    try:
                        if cat.measurement_type(f) == "medication":
                            idx.append(i)
                    except KeyError:
                        pass
                model.exogenous_idx = torch.tensor(idx, dtype=torch.long, device=device)
        except Exception as e:  # never let eval crash over the med-index reconstruction
            import logging; logging.getLogger(__name__).warning("exogenous_idx reconstruction failed: %s", e)
    model.eval()
    return model, cfg


def build_fm(model: str, ds: ICUDataset, device: torch.device,
             chronos_id: str, point: str):
    """Build a zero-shot foundation-model baseline directly (no checkpoint).

    FMs like Chronos carry no trained weights of ours, so Stage-5 eval
    instantiates them straight from `--fm` flags with a synthetic config.
    """
    run_name = f"{model}_{chronos_id.split('/')[-1]}_{point}".replace("-", "_")
    cfg = {"model": model, "run_name": run_name,
           "chronos_id": chronos_id, "point": point}
    m = _build(cfg, ds, device)
    m.eval()
    return m, cfg


def build_precomputed(pred_file: Path, ds: ICUDataset, device: torch.device,
                      n_eval: int | None = None):
    """Build a PrecomputedForecast from a .npy of z-scored predictions (e.g. Moirai
    forecasts produced in py311_moirai by harmonize/infer_moirai.py)."""
    from harmonize.models.precomputed import PrecomputedForecast
    run_name = pred_file.stem  # e.g. moirai_1.1-R-small_eicu
    cfg = {"model": "precomputed", "run_name": run_name, "pred_file": str(pred_file)}
    m = PrecomputedForecast(
        n_features_in=ds.n_features_in, n_features_target=ds.n_features_target,
        n_features_static=ds.n_features_static, window_in=ds.window_in,
        window_out=ds.window_out, target_idx_in_input=ds.target_indices_in_input(),
        pred_file=str(pred_file),
    ).to(device)
    # The .npy is consumed in dataset row order, so a length mismatch silently mis-pairs every
    # window with another patient's forecast and every clinical metric downstream is computed on
    # the wrong rows. Too-few rows already errors at consumption; too-many passed unnoticed.
    # `n_eval` is the number of windows this eval will actually consume (--limit), which is what
    # the .npy must match: comparing against len(ds) instead made the documented --limit
    # subsample protocol (the MIRA tables, §9.7) impossible to re-score at all.
    want = len(ds) if n_eval is None else min(int(n_eval), len(ds))
    if m.n_rows != want:
        raise ValueError(
            f"pred file {pred_file} has {m.n_rows} rows but this eval consumes {want} windows "
            f"(dataset {len(ds)}, --limit {n_eval}) — predictions would be mis-paired with "
            f"windows. Re-run inference on this split.")
    m.eval()
    return m, cfg


# ---------------------------------------------------------------------------
# Inference → denormalised predicted + ground-truth trajectories
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, ds: ICUDataset, norm_stats: pd.DataFrame,
                  device: torch.device, batch_size: int = 256,
                  limit: int | None = None, permute_static_idx: int | None = None,
                  phys_mask=None, per_stay_out=None):
    """Returns per-window arrays + accumulated forecast-error stats.

    pred_phys, gt_locf_phys : (N, T_out, F_target) physical units
    plus masked SE/AE accumulators for forecast metrics.
    `limit` caps the number of windows (for quick CPU smoke tests).
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats_in, feats_tgt = ds.features_in, ds.features_target

    pred_chunks, gt_chunks, static_chunks = [], [], []
    stay_ids, t0_hours = [], []
    in_day_chunks, gt_day_chunks, pred_day_chunks = [], [], []
    sofa_hm_chunks = []            # (B, T) per batch: hours where a SOFA driver is measured
    saps_wm_chunks = []            # (B, 12) per batch: SAPS-II components scorable in the window
    pred_matched_chunks = []       # prediction given the ground truth's own mask+LOCF treatment
    in_creat_chunks = []
    n_feat = len(feats_tgt)
    sq = np.zeros(n_feat); ab = np.zeros(n_feat); obs = np.zeros(n_feat)
    # per-horizon (per forecast-hour) accumulators → "is hour 12 worse than hour 1?"
    t_out = ds.window_out
    sq_h = np.zeros(t_out); obs_h = np.zeros(t_out)
    # JOINT (hour x variable) accumulators. `feat_mse` and `horizon_mse` are the two MARGINALS of
    # this matrix, and the marginals cannot answer which variables drive the horizon growth — a
    # variable can be hard everywhere (flat in h) or only far out (steep in h) and look identical
    # after summing. Cheap: T_out x F floats, accumulated in the pass that already exists.
    sq_hf = np.zeros((t_out, n_feat)); ab_hf = np.zeros((t_out, n_feat))
    obs_hf = np.zeros((t_out, n_feat))
    seen = 0

    # index map target→input for LOCF seed
    seed_idx = [feats_in.index(f) if f in feats_in else -1 for f in feats_tgt]

    # volatility / change-detection accumulators (z-space; change measured vs the LOCF anchor,
    # i.e. the LOCF/persistence-forecast prediction — so the LOCF reference falls out of the same pass).
    VOL_TAU = 0.5                                       # |Δ from anchor| > 0.5 SD = a real change
    vt = vt2 = vp = vp2 = vtp = vn = 0.0                # vol-ratio + corr over all observed positions
    ch_n = np.zeros(n_feat); ch_mod = np.zeros(n_feat)  # changed-subset: count, model SE,
    ch_pers = np.zeros(n_feat); ch_dir = np.zeros(n_feat)  #   LOCF SE, correct-direction
    ps_stay, ps_se_all, ps_ob_all, ps_se_phys, ps_ob_phys = [], [], [], [], []  # per-window MSE dump

    for batch in loader:
        if limit is not None and seen >= limit:
            break
        seen += batch["x_in"].shape[0]
        x_in = batch["x_in"].to(device)
        m_in = batch["mask_in"].to(device)
        static = batch["static"].to(device)
        if permute_static_idx is not None:
            # demographics ablation: break the static feature's link to the window by
            # permuting it across the batch → ΔMSE vs baseline = its forecast importance.
            # "all" permutes every static column (each independently) → positive control:
            # if ΔMSE(all) ≈ ΔMSE(weight alone), the other statics carry no forecast signal.
            cols = (range(static.shape[1]) if permute_static_idx == "all"
                    else [permute_static_idx])
            for _c in cols:
                static[:, _c] = static[torch.randperm(static.shape[0], device=device), _c]
        x_tgt = batch["x_target"]; m_tgt = batch["mask_target"]
        # dense_pteacher: feed the TRUE future EXOGENOUS (medication) channels at eval too,
        # exactly as train_forecast.evaluate does — otherwise the med channels fall back to the
        # model's own unsupervised predictions, a train/eval mismatch that unfairly degrades the
        # pteacher forecast (the meds are a *known* exogenous care-plan input by design, and are
        # excluded from the loss). No-op for models without `exogenous_idx` (denseAR, Informer, …).
        _exo = getattr(model, "exogenous_idx", None)
        if _exo is not None and _exo.numel() > 0:
            pred = model(x_in, m_in, static, x_target=x_tgt.to(device),
                         teacher_forcing_ratio=0.0).cpu().numpy()   # (B,T,F) z-score
        else:
            pred = model(x_in, m_in, static).cpu().numpy()      # (B,T,F) z-score
        xt = x_tgt.numpy(); mt = m_tgt.numpy().astype(bool)

        # forecast error on observed target positions (z-score space)
        diff = np.where(mt, pred - np.nan_to_num(xt), 0.0)
        se = diff**2
        sq  += se.sum(axis=(0, 1)); ab += np.abs(diff).sum(axis=(0, 1))
        obs += mt.sum(axis=(0, 1))
        sq_h  += se.sum(axis=(0, 2)); obs_h += mt.sum(axis=(0, 2))
        sq_hf += se.sum(axis=0); ab_hf += np.abs(diff).sum(axis=0); obs_hf += mt.sum(axis=0)

        # --- volatility / change vs the LOCF anchor (the forecasting baseline), z-space ---
        xin_z = x_in.detach().cpu().numpy(); min_z = m_in.detach().cpu().numpy().astype(bool)
        anchor = np.full((xin_z.shape[0], n_feat), np.nan, np.float32)
        for j, k in enumerate(seed_idx):
            if k < 0:
                continue
            mcol = min_z[:, :, k]; has = mcol.any(axis=1)
            last = mcol.shape[1] - 1 - mcol[:, ::-1].argmax(axis=1)     # last observed input step
            anchor[has, j] = xin_z[has, last[has], k]
        a0 = np.nan_to_num(anchor)[:, None, :]                          # (B,1,F)
        valid = mt & ~np.isnan(anchor)[:, None, :]                     # observed target & anchor exists
        dev_t = np.where(valid, np.nan_to_num(xt) - a0, 0.0)           # true change from anchor
        dev_p = np.where(valid, pred - a0, 0.0)                        # predicted change (LOCF→0)
        vt += dev_t.sum(); vt2 += (dev_t**2).sum(); vp += dev_p.sum()
        vp2 += (dev_p**2).sum(); vtp += (dev_t * dev_p).sum(); vn += float(valid.sum())
        chg = valid & (np.abs(dev_t) > VOL_TAU)
        ch_n    += chg.sum(axis=(0, 1))
        ch_mod  += np.where(chg, (pred - np.nan_to_num(xt))**2, 0.0).sum(axis=(0, 1))
        ch_pers += np.where(chg, dev_t**2, 0.0).sum(axis=(0, 1))        # LOCF SE = (anchor−true)²
        ch_dir  += np.where(chg, (np.sign(dev_p) == np.sign(dev_t)).astype(float), 0.0).sum(axis=(0, 1))

        # per-window z-space MSE dump (for per-hospital / per-stay aggregation)
        if per_stay_out is not None:
            ps_stay.append(batch["stay_id"].numpy())
            ps_se_all.append(se.sum(axis=(1, 2))); ps_ob_all.append(mt.sum(axis=(1, 2)))
            if phys_mask is not None:
                ps_se_phys.append(se[:, :, phys_mask].sum(axis=(1, 2)))
                ps_ob_phys.append(mt[:, :, phys_mask].sum(axis=(1, 2)))

        # denormalise
        pred_phys = denormalise(pred, feats_tgt, norm_stats)
        gt_phys   = denormalise(xt,   feats_tgt, norm_stats)
        xi_phys   = denormalise(x_in.cpu().numpy(), feats_in, norm_stats)

        # GT: LOCF seeded from last input step
        seed = np.full((len(gt_phys), n_feat), np.nan, dtype=np.float32)
        for j, k in enumerate(seed_idx):
            if k >= 0:
                seed[:, j] = xi_phys[:, -1, k]
        gt_locf = locf(np.where(mt, gt_phys, np.nan).astype(np.float32), axis=1, seed=seed)

        # MATCHED prediction: the same treatment the ground truth just received -- keep only the
        # cells that were actually measured, then carry forward -- so that the two objects being
        # compared are constructed the same way. `pred_phys` is dense at every timestep while
        # `gt_locf` is carried forward from the 2-13 % of cells that were measured.
        #
        # It must NOT share the ground truth's seed. `gt_locf` is seeded from the last observed
        # INPUT value; seeding the prediction the same way makes the two identical on every
        # channel-window with no observed target cell -- 53 % of them on UMM, 59 % on HiRID -- so
        # the error there would be exactly zero by construction. Seed with the model's own first
        # prediction instead: that is a genuine forecast, and the arm then measures the model
        # rather than the fill rule.
        pred_matched = locf(np.where(mt, pred_phys, np.nan).astype(np.float32), axis=1,
                            seed=pred_phys[:, 0, :])


        # daily SOFA per window (Σ per-organ worst, symmetric ffill+bfill imputation) for
        # the worst-best ΔSOFA detection. pred is dense → no fill needed.
        dname = ds.dataset_name
        m_in_np = m_in.cpu().numpy().astype(bool)
        in_day_chunks.append(daily_sofa(fill_ffb(xi_phys, m_in_np), feats_in, dname).astype(np.float32))
        gt_day_chunks.append(daily_sofa(fill_ffb(gt_phys, mt), feats_tgt, dname).astype(np.float32))
        pred_day_chunks.append(daily_sofa(pred_phys, feats_tgt, dname).astype(np.float32))
        # Which hours rest on a real measurement rather than carry-forward. Accumulated as
        # (B, T) rather than the full (B, T, F) mask: the latter is ~0.5 GB on eICU.
        sofa_hm_chunks.append(sofa_hour_mask(mt, feats_tgt, ds.dataset_name))   # (B, T, 6)
        # SAPS-II scores a missing variable as 0 POINTS, not NaN, so coverage is invisible in the
        # total and has to be reported separately (§37.38 D20). Worst-of-window ⇒ per-window mask.
        from harmonize.scores.sapsii import sapsii_window_mask
        saps_wm_chunks.append(sapsii_window_mask(mt, feats_tgt))                # (B, 12)

        # AKI baseline: best (min) observed input creatinine (KDIGO look-back minimum)
        if "creatinine" in feats_in:
            cii = feats_in.index("creatinine")
            b = np.where(m_in_np[:, :, cii], xi_phys[:, :, cii], np.inf).min(axis=1)
            in_creat_chunks.append(np.where(np.isfinite(b), b, np.nan).astype(np.float32))
        else:
            in_creat_chunks.append(np.full(len(xi_phys), np.nan, dtype=np.float32))

        pred_chunks.append(pred_phys.astype(np.float32))
        pred_matched_chunks.append(pred_matched.astype(np.float32))
        gt_chunks.append(gt_locf.astype(np.float32))
        static_chunks.append(static.cpu().numpy())
        stay_ids.append(batch["stay_id"].numpy())
        t0_hours.append(batch["t0_hour"].numpy())

    denom = np.clip(obs, 1, None)
    fc = {
        "val_mse_per_var": float(np.mean(sq / denom)),
        "val_mae_per_var": float(np.mean(ab / denom)),
        "val_mse":         float(sq.sum() / denom.sum()),
        "val_mae":         float(ab.sum() / denom.sum()),
        "n_obs":           int(obs.sum()),
        # per-variable breakdown (aligned to features_target) → feeds §10.6
        # per-variable analysis (analyze_error_structure.py reads feat_mse)
        "feat_mse":        {f: float(sq[j] / denom[j]) for j, f in enumerate(feats_tgt)},
        "feat_mae":        {f: float(ab[j] / denom[j]) for j, f in enumerate(feats_tgt)},
        "feat_n_obs":      {f: int(obs[j])             for j, f in enumerate(feats_tgt)},
        # per-forecast-hour MSE (z-score) — horizon degradation, h1..hT_out
        "horizon_mse":     (sq_h / np.clip(obs_h, 1, None)).tolist(),
        "horizon_ratio_last_first": float((sq_h[-1] / max(obs_h[-1], 1)) /
                                          max(sq_h[0] / max(obs_h[0], 1), 1e-9)),
        # Staniek's convention: the SUMMED masked squared error, not the per-observation mean.
        # A positive-constant rescale of ours, so it preserves every ranking but not the absolute
        # scale — reported so the MLHC-2024 numbers are directly comparable (§12.2 verdict).
        "mse_old":         float(sq.sum()),
        "mse_old_per_var": float(sq.mean()),
    }
    # (hour x variable) matrices — see the accumulator comment above. Stored as
    # {feature: [h1..hT]} so a JSON reader gets named rows without a shape convention to remember.
    # Each cell is normalised by ITS OWN observation count, so an unobserved cell is 0/1 = 0.0
    # (numerator is zero too). `hour_feat_n_obs` is therefore not needed to read a cell — it is
    # kept only as the weight for re-aggregating the matrix back to a marginal.
    fc["hour_feat_mse"] = {f: [float(sq_hf[h, j] / max(obs_hf[h, j], 1.0)) for h in range(t_out)]
                           for j, f in enumerate(feats_tgt)}
    fc["hour_feat_mae"] = {f: [float(ab_hf[h, j] / max(obs_hf[h, j], 1.0)) for h in range(t_out)]
                           for j, f in enumerate(feats_tgt)}
    fc["hour_feat_n_obs"] = {f: [int(obs_hf[h, j]) for h in range(t_out)]
                             for j, f in enumerate(feats_tgt)}
    # cells with no observation are reported as 0.0 MSE; n_obs is the mask that makes them
    # distinguishable from genuinely perfect predictions (the §27.7 dilution trap in miniature).
    fc["hour_feat_cells_observed"] = int((obs_hf > 0).sum())
    fc["hour_feat_shape"] = [t_out, n_feat]
    # volatility / change-detection summary (the axis where persistence is 0 by construction)
    vol: dict = {}
    if vn > 0:
        var_t = max(vt2 / vn - (vt / vn) ** 2, 0.0)
        var_p = max(vp2 / vn - (vp / vn) ** 2, 0.0)
        cov = vtp / vn - (vt / vn) * (vp / vn)
        nchg = float(ch_n.sum())
        vol = {"tau_sd": VOL_TAU, "change_rate": float(nchg / vn),
               "vol_ratio_pred_over_true": float(var_p ** 0.5 / max(var_t ** 0.5, 1e-9)),
               "vol_corr": float(cov / max((var_t * var_p) ** 0.5, 1e-9))}
        if nchg > 0:
            cmm = float(ch_mod.sum() / nchg); cmp_ = float(ch_pers.sum() / nchg)
            vol.update({"change_mse_model": cmm, "change_mse_locf": cmp_,
                        "change_skill_vs_locf": float(1.0 - cmm / max(cmp_, 1e-9)),
                        "dir_acc_on_changed": float(ch_dir.sum() / nchg),
                        "feat_change_skill": {f: float(1.0 - (ch_mod[j] / max(ch_n[j], 1)) /
                                              max(ch_pers[j] / max(ch_n[j], 1), 1e-9))
                                              for j, f in enumerate(feats_tgt) if ch_n[j] > 50}})
    fc["volatility"] = vol
    if per_stay_out is not None and ps_stay:
        df_ps = pd.DataFrame({"stay_id": np.concatenate(ps_stay),
                              "mse_all": np.concatenate(ps_se_all) / np.clip(np.concatenate(ps_ob_all), 1, None),
                              "n_obs": np.concatenate(ps_ob_all)})
        if ps_se_phys:
            df_ps["mse_phys"] = (np.concatenate(ps_se_phys) /
                                 np.clip(np.concatenate(ps_ob_phys), 1, None))
        df_ps.to_csv(per_stay_out, index=False)
        log.info("per-window MSE dump → %s (%d windows)", per_stay_out, len(df_ps))
    return (np.concatenate(pred_chunks), np.concatenate(gt_chunks),
            np.concatenate(static_chunks),
            np.concatenate(stay_ids), np.concatenate(t0_hours), fc,
            np.concatenate(in_day_chunks), np.concatenate(gt_day_chunks),
            np.concatenate(pred_day_chunks), np.concatenate(in_creat_chunks),
            np.concatenate(sofa_hm_chunks), np.concatenate(pred_matched_chunks),
            np.concatenate(saps_wm_chunks))


# ---------------------------------------------------------------------------
# Clinical-score comparison
# ---------------------------------------------------------------------------

def _detection_metrics(gt_bool, pred_bool, score=None, prefix=""):
    """Full binary-detection metric set (not just accuracy): accuracy, sensitivity
    (= recall), specificity, precision, F1, prevalence, and — if a continuous score
    is given — AUROC. Used for ΔSOFA≥2, new-AKI and Sepsis-3 onset detection."""
    gt = np.asarray(gt_bool, dtype=bool)
    pr = np.asarray(pred_bool, dtype=bool)
    n  = max(len(gt), 1)
    tp = int((gt & pr).sum());  tn = int((~gt & ~pr).sum())
    fp = int((~gt & pr).sum()); fn = int((gt & ~pr).sum())
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    m = {
        f"{prefix}accuracy":      (tp + tn) / n,
        f"{prefix}sensitivity":   sens,
        f"{prefix}recall":        sens,
        f"{prefix}specificity":   tn / max(tn + fp, 1),
        f"{prefix}precision":     prec,
        f"{prefix}f1":            (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0,
        f"{prefix}gt_prevalence": float(gt.mean()),
    }
    if score is not None:
        try:
            from sklearn.metrics import roc_auc_score
            s = np.asarray(score, dtype=float)
            ok = ~np.isnan(s)
            m[f"{prefix}auroc"] = (float(roc_auc_score(gt[ok], s[ok]))
                                   if gt[ok].any() and (~gt[ok]).any() else float("nan"))
        except Exception:
            m[f"{prefix}auroc"] = float("nan")
    return m


def sofa_comparison(pred_phys, gt_phys, feats_tgt, dataset,
                    in_day=None, gt_day=None, pred_day=None, sofa_hour_obs=None):
    """Per-window SOFA on predicted (dense) vs ground-truth (LOCF) trajectory.

    With `sofa_hour_obs` (an (N, T) bool of hours carrying a real SOFA measurement) the same
    metrics are additionally computed on observed hours only, prefixed `obs_`. That removes the
    circularity in the default numbers: the ground truth is LOCF-filled on 77-87 % of cells, so a
    persistence forecast reproduces it exactly there and scores zero error for free. Compare the
    two blocks -- an advantage that is an artefact of the fill shrinks or disappears under `obs_`.
    """
    p = _window_sofa(pred_phys, feats_tgt, dataset).add_prefix("pred_")
    g = _window_sofa(gt_phys,   feats_tgt, dataset).add_prefix("gt_")
    df = pd.concat([p, g], axis=1)
    metrics = {}
    for comp in ["sofa_max", "sofa_mean", "delta_sofa",
                 "sofa_resp", "sofa_coag", "sofa_liver", "sofa_renal", "sofa_cns",
                 "sofa_cv_mean"]:   # cardiovascular (stored as *_mean; was missing from the loop)
        pc, gc = f"pred_{comp}", f"gt_{comp}"
        m = df[[pc, gc]].dropna()
        if len(m):
            metrics[f"{comp}_mae"] = float((m[pc] - m[gc]).abs().mean())
            metrics[f"{comp}_mse"] = float(((m[pc] - m[gc]) ** 2).mean())
    # --- GCS-free SOFA, for every dataset -------------------------------------------------
    # UMM already scores 5 components (no CNS), the other four score 6, so a SOFA column that
    # mixes them is not like-for-like. Emit the 5-component score for ALL datasets under a
    # `noCNS_` prefix: the GCS-bearing table then covers the four databases with usable GCS
    # (eICU 61.8 %, HiRID/MIMIC ~99.7 %) and the GCS-free table covers all five (§37.41).
    # Totals only — the per-organ CNS term is identically 0 here, so it carries no information.
    pn = _window_sofa(pred_phys, feats_tgt, dataset, skip_cns=True).add_prefix("pred_")
    gn = _window_sofa(gt_phys,   feats_tgt, dataset, skip_cns=True).add_prefix("gt_")
    dfn = pd.concat([pn, gn], axis=1)
    for comp in ["sofa_max", "sofa_mean", "delta_sofa"]:
        pc, gc = f"pred_{comp}", f"gt_{comp}"
        m = dfn[[pc, gc]].dropna()
        if len(m):
            metrics[f"noCNS_{comp}_mae"] = float((m[pc] - m[gc]).abs().mean())
            metrics[f"noCNS_{comp}_mse"] = float(((m[pc] - m[gc]) ** 2).mean())
    metrics["noCNS_gt_sofa_max_mean"] = float(dfn["gt_sofa_max"].mean())
    metrics["noCNS_is_identical_to_default"] = bool(dataset == "umm")

    # ΔSOFA ≥ 2 detection — two definitions:
    #   dSOFA2_   : net change over the horizon (last − first target hour)
    #   dSOFA2wb_ : daily_sofa(target) − daily_sofa(input) ≥ 2, per-organ WORST on BOTH sides.
    #               The "wb" suffix is a historical misnomer ("worst−best"): a min/"best"
    #               baseline is NOT computed here and would give ~52 % positives, not ~12 %
    #               (§37.52 D31). Kept as a key name because it ships in every eval table.
    #               (needs the input-window baseline; reliable now that the renal urine
    #               artefact is fixed). Same definition as the ΔSOFA2 head → apples-to-apples.
    # --- observed-hours-only re-score ------------------------------------------------------
    if sofa_hour_obs is not None:
        hm = np.asarray(sofa_hour_obs, dtype=bool)
        po = _window_sofa(pred_phys, feats_tgt, dataset, hour_mask=hm).add_prefix("pred_")
        go = _window_sofa(gt_phys,   feats_tgt, dataset, hour_mask=hm).add_prefix("gt_")
        do = pd.concat([po, go], axis=1)
        for comp in ["sofa_max", "sofa_mean", "sofa_cv_mean",
                     "sofa_resp", "sofa_coag", "sofa_liver", "sofa_renal", "sofa_cns"]:
            pc, gc = f"pred_{comp}", f"gt_{comp}"
            mm = do[[pc, gc]].dropna()
            if len(mm):
                metrics[f"obs_{comp}_mae"] = float((mm[pc] - mm[gc]).abs().mean())
                metrics[f"obs_{comp}_mse"] = float(((mm[pc] - mm[gc]) ** 2).mean())
        from harmonize.scores.sofa import SOFA_COMPONENT_ORDER
        from harmonize.scores.sofa import sofa_total_mask
        metrics["obs_hour_frac_total"] = float(sofa_total_mask(hm, dataset).mean())
        for c, nm in enumerate(SOFA_COMPONENT_ORDER):
            metrics[f"obs_hour_frac_{nm}"] = float(hm[..., c].mean())
        metrics["obs_windows_scored"] = int(np.isfinite(do["gt_sofa_max"]).sum())
        metrics["obs_windows_total"] = int(len(do))

    gt_dys = (df["gt_delta_sofa"] >= 2).values
    pr_dys = (df["pred_delta_sofa"] >= 2).values
    metrics.update(_detection_metrics(
        gt_dys, pr_dys, score=df["pred_delta_sofa"].values, prefix="dSOFA2_"))
    if in_day is not None and gt_day is not None and pred_day is not None:
        ok = np.isfinite(in_day) & np.isfinite(gt_day) & np.isfinite(pred_day)
        if ok.any():
            pr_score = pred_day[ok] - in_day[ok]
            metrics.update(_detection_metrics(
                (gt_day[ok] - in_day[ok]) >= 2, pr_score >= 2,
                score=pr_score, prefix="dSOFA2wb_"))
        # daily-SOFA MAE: forecast vs ground-truth *worst-of-window* SOFA (`daily_sofa`) — the SAME
        # quantity the SOFA regression head predicts, so head-MAE vs this is apples-to-apples
        # (§10.4a/§R2). `sofa_max_mae` above is the _window_sofa per-hour-total view (different).
        okd = np.isfinite(gt_day) & np.isfinite(pred_day)
        if okd.any():
            metrics["daily_sofa_mae"]      = float(np.abs(pred_day[okd] - gt_day[okd]).mean())
            metrics["daily_sofa_mse"]      = float(((pred_day[okd] - gt_day[okd]) ** 2).mean())
            metrics["daily_sofa_gt_mean"]  = float(np.mean(gt_day[okd]))
            metrics["daily_sofa_pred_mean"] = float(np.mean(pred_day[okd]))
    return df, metrics


def news2_comparison(pred_phys, gt_phys, feats_tgt):
    """Per-window NEWS2 (worst-of-window) on predicted vs ground-truth physiology.

    NEWS2 is a *deterioration-detection* score (RCP 2017), so besides the MAE/MSE of the total we
    report how well the forecast reproduces the clinical **escalation band** (low / medium / high)
    and, most relevant to §15.10, whether it flags the high-risk band at all.
    """
    from harmonize.scores.news2 import (news2_batch, news2_component_batch,
                                        NEWS2_COMPONENT_FIELDS)
    p = news2_batch(pred_phys, feats_tgt)
    g = news2_batch(gt_phys, feats_tgt)
    ok = np.isfinite(p) & np.isfinite(g)
    m = {"news2_pred_mean": float(np.nanmean(p)), "news2_gt_mean": float(np.nanmean(g)),
         "news2_mae": float(np.nanmean(np.abs(p - g))),
         "news2_mse": float(np.nanmean((p - g) ** 2))}
    # escalation band: NEWS2 >= 7 is the 'high / emergency response' trigger
    gt_hi, pr_hi = (g >= 7), (p >= 7)
    m.update(_detection_metrics(gt_hi[ok], pr_hi[ok], score=p[ok], prefix="news2_high_"))
    pc = news2_component_batch(pred_phys, feats_tgt)
    gc = news2_component_batch(gt_phys, feats_tgt)
    for c in NEWS2_COMPONENT_FIELDS:
        m[f"news2_{c}_mae"] = float(np.nanmean(np.abs(pc[c] - gc[c])))
    return m


def sapsii_comparison(pred_phys, gt_phys, feats_tgt, static, feats_static, dataset, stay_id,
                      window_mask=None):
    """Per-window SAPS-II (worst-of-window) on predicted vs ground-truth.

    admission_type (0/6/8) + chronic_disease (0/9/10/17) points are per-stay statics that
    live only in the long parquet (not the H5 static set), so they are looked up by stay_id.
    """
    from harmonize.scores.sapsii import load_sapsii_static_maps, sapsii_component_batch
    age = None
    if "age" in feats_static:
        age = static[:, feats_static.index("age")]
    am, cm = load_sapsii_static_maps(dataset)
    adm     = np.array([am.get(int(s), 0.0) for s in stay_id], dtype=np.float32)
    chronic = np.array([cm.get(int(s), 0.0) for s in stay_id], dtype=np.float32)
    p = sapsii_batch(pred_phys, feats_tgt, age, adm, chronic)
    g = sapsii_batch(gt_phys,   feats_tgt, age, adm, chronic)
    metrics = {
        "sapsii_pred_mean": float(np.nanmean(p)),
        "sapsii_gt_mean":   float(np.nanmean(g)),
        "sapsii_mae":       float(np.nanmean(np.abs(p - g))),
        "sapsii_mse":       float(np.nanmean((p - g) ** 2)),
    }
    # per-component MAE/MSE — which SAPS-II component the forecast captures. age / admission_type /
    # chronic_disease are per-stay statics identical for pred & gt, so they cancel (MAE 0); the
    # physiology components (hr/sbp/temp/pao2_fio2/urine/bun/wbc/K/Na/HCO3/bilirubin/gcs) are informative.
    pc = sapsii_component_batch(pred_phys, feats_tgt, age, adm, chronic)
    gc = sapsii_component_batch(gt_phys,   feats_tgt, age, adm, chronic)
    for comp in pc:
        d = pc[comp] - gc[comp]
        metrics[f"sapsii_{comp}_mae"] = float(np.nanmean(np.abs(d)))
        metrics[f"sapsii_{comp}_mse"] = float(np.nanmean(d ** 2))
    # SAPS-II **without the GCS component**, the SAPS-II analogue of SOFA's UMM 5-component
    # treatment (`sofa.daily_sofa` zeroes CNS for UMM because the deep-sedation protocol makes GCS
    # unreliable; SAPS-II had no equivalent). SAPS-II points are additive, so subtracting the GCS
    # points is exact -- no re-scoring. Reported for EVERY dataset, not just UMM: GCS coverage runs
    # from 34.8 % (UMM) to 99.8 % (HiRID/MIMIC) (§37.39), and a missing GCS scores 0 points rather
    # than NaN, so the GCS-free score is the only SAPS-II variant that compares like with like
    # across the five databases.
    p_ng, g_ng = p - pc["gcs"], g - gc["gcs"]
    metrics.update({
        "sapsii_noGCS_pred_mean": float(np.nanmean(p_ng)),
        "sapsii_noGCS_gt_mean":   float(np.nanmean(g_ng)),
        "sapsii_noGCS_mae":       float(np.nanmean(np.abs(p_ng - g_ng))),
        "sapsii_noGCS_mse":       float(np.nanmean((p_ng - g_ng) ** 2)),
    })
    # Per-component COVERAGE. A missing SAPS-II variable scores 0 points rather than NaN, so the
    # total is always finite and says nothing about how much of the score was actually measured
    # (§37.38 D20). Publish it the way the SOFA hour mask is published, so a SAPS-II number is
    # never read without knowing how many of the 12 physiology components stood behind it.
    if window_mask is not None:
        from harmonize.scores.sapsii import SAPSII_PHYSIO_COMPONENTS
        wm = np.asarray(window_mask, dtype=bool)
        for c, comp in enumerate(SAPSII_PHYSIO_COMPONENTS):
            metrics[f"sapsii_{comp}_obs_frac"] = float(wm[:, c].mean())
        metrics["sapsii_components_scored_mean"] = float(wm.sum(axis=1).mean())
        metrics["sapsii_components_scored_n"]    = int(wm.shape[1])
        metrics["sapsii_all_components_frac"]    = float(wm.all(axis=1).mean())
    return metrics


def aki_comparison(pred_phys, gt_phys, feats_tgt, dataset,
                   in_creat=None, static=None, feats_static=None,
                   aki_absolute_rise: bool = False):
    """Per-window KDIGO AKI (creatinine-based) on predicted vs ground-truth trajectory.

    Same consensus-override logic as ΔSOFA≥2: the rare high-value event is **new AKI**
    (KDIGO stage ≥ 1) developing within the forecast horizon. Baseline = the window's
    first-hour creatinine; the stage is computed from the window's worst (max) creatinine
    (scores/kdigo.py). Urine / RRT criteria are omitted here — they need weight / procedure
    data outside the forecast feature space — so this is the robust creatinine-only staging.
    Reports stage MAE + the full new-AKI detection set (F1/precision/recall/AUROC)."""
    if "creatinine" not in feats_tgt:
        return {"aki_evaluable": False, "reason": "no creatinine in features_target"}
    ci = feats_tgt.index("creatinine")
    # baseline = best (min) input creatinine (KDIGO look-back minimum) if provided, else
    # the window's first-hour creatinine (legacy); worst = window max creatinine.
    with np.errstate(invalid="ignore"):
        gw = np.nanmax(gt_phys[:, :, ci], axis=1)
        pw = np.nanmax(pred_phys[:, :, ci], axis=1)
    base = in_creat if in_creat is not None else gt_phys[:, 0, ci]
    # D16: `gw - base` is KDIGO's 48 h absolute rise (window_in + window_out = 24 + 24 h) and
    # was never passed, so only the relative >= 1.5x/2.0x/3.0x criteria fired here too. Unlike the
    # label head, flipping this DEFAULT would silently change the `aki` block of all 526 committed
    # tables and leave them split across two label definitions -- so it is opt-in, and turning it
    # on means a deliberate full re-score. On the test splits it raises AKI prevalence from
    # 2.6-5.3 % to 9.5-15.3 % (2.9-3.7x), i.e. the relative-only label captures about a third of
    # KDIGO stage-1 events.
    rise = (gw - base) if aki_absolute_rise else None
    prise = (pw - base) if aki_absolute_rise else None
    gs = kdigo_stage(gw, base, creat_rise_48h=rise)
    ps = kdigo_stage(pw, base, creat_rise_48h=prise)             # creatinine-only
    ok = ~(np.isnan(ps) | np.isnan(gs))
    m = {"aki_evaluable": True, "aki_n": int(ok.sum())}
    if ok.sum():
        m["aki_stage_mae"] = float(np.abs(ps[ok] - gs[ok]).mean())
        m.update(_detection_metrics(gs[ok] >= 1, ps[ok] >= 1, score=ps[ok], prefix="aki_"))
    # aki_uo: + KDIGO oliguria (mean urine rate over the window / weight). The eval
    # trajectories are LOCF-imputed, so this is the dense-charting approximation of the
    # head's raw-masked gated label — reported as a separate, comparable metric.
    if ("urine_output" in feats_tgt and static is not None and feats_static is not None
            and "weight_admission" in feats_static):
        ui = feats_tgt.index("urine_output")
        wt = np.clip(static[:, feats_static.index("weight_admission")], 20, None)
        with np.errstate(invalid="ignore"):
            gu = np.nanmean(gt_phys[:, :, ui], axis=1) / wt
            pu = np.nanmean(pred_phys[:, :, ui], axis=1) / wt
        gsu = kdigo_stage(gw, base, creat_rise_48h=rise, urine_ml_kg_h=gu)
        psu = kdigo_stage(pw, base, creat_rise_48h=prise, urine_ml_kg_h=pu)
        oku = ~(np.isnan(psu) | np.isnan(gsu))
        if oku.sum():
            m["aki_uo_n"] = int(oku.sum())
            m.update(_detection_metrics(gsu[oku] >= 1, psu[oku] >= 1, score=psu[oku], prefix="aki_uo_"))
    return m


# ---------------------------------------------------------------------------
# Reusable single-model evaluation (factored out of main so a multi-model
# driver can load the dataset / norm-stats / susinf ONCE and loop over models)
# ---------------------------------------------------------------------------

def load_susinf(long_parquet: Path | None):
    """Precompute the per-stay suspected-infection table for Sepsis-3, or None.

    **eICU uses the MLHC-2024 *union* definition** (suspected infection = any IV
    antibiotic OR any blood culture): eICU microbiology is sparse (~343 culture
    stays after the Blood-site + 168 h filter), so Singer's strict
    antibiotic-AND-culture-within-±72 h is culture-limited and collapses susinf to
    ~0 — but eICU *does* have antibiotics for ~23 % of stays (`abx_*` from
    `extract_antibiotics`), exactly the arm the original paper used. All other
    datasets keep the strict ±window co-occurrence (they have both arms charted)."""
    if not (long_parquet and Path(long_parquet).exists()):
        return None
    from harmonize.scores.sepsis3 import susinf_per_stay
    long_df = pd.read_parquet(long_parquet, columns=["stay_id", "variable", "hour", "value"])
    union = "eicu" in Path(long_parquet).name.lower()
    return susinf_per_stay(long_df, union=union).set_index("stay_id")


def evaluate(model, cfg: dict, ds: ICUDataset, norm_stats: pd.DataFrame,
             dataset: str, device: torch.device, *, susinf=None,
             batch_size: int = 256, limit: int | None = None,
             permute_static_idx: int | None = None,
             phys_mask=None, per_stay_out=None, first_day_only: bool = False,
             aki_absolute_rise: bool = False) -> dict:
    """Run inference + all clinical-score comparisons for one model → summary dict.

    `susinf` is the precomputed per-stay suspected-infection table (or None);
    pass it in so a multi-model driver builds it once.
    """
    (pred, gt, static_arr, stay_id, t0, fc, in_day, gt_day, pred_day, in_creat,
     sofa_hm, pred_matched, saps_wm) = run_inference(
        model, ds, norm_stats, device, batch_size, limit=limit,
        permute_static_idx=permute_static_idx, phys_mask=phys_mask, per_stay_out=per_stay_out)
    log.info("Forecast (test): mse_per_var=%.4f mae_per_var=%.4f",
             fc["val_mse_per_var"], fc["val_mae_per_var"])

    # --first-day-only: keep one window per stay (the admission window, t0_hour == 0).
    # Sliding windows score STAY-level outcomes (mortality, LOS, admission scores) once per
    # window, so long-staying — i.e. sicker — patients are counted many times and the apparent
    # event prevalence is inflated. Restricting to the admission window gives one prediction per
    # stay at the true stay-level prevalence, which is also how SAPS-II / APACHE / NEWS2 are
    # defined (first 24 h) and therefore the literature-comparable view.
    if first_day_only:
        keep = np.asarray(t0) == 0
        if not keep.any():
            log.warning("--first-day-only: no windows with t0_hour==0; keeping all")
        else:
            n0 = len(keep)
            pred, gt, static_arr = pred[keep], gt[keep], static_arr[keep]
            stay_id, t0 = np.asarray(stay_id)[keep], np.asarray(t0)[keep]
            in_day = np.asarray(in_day)[keep] if in_day is not None else None
            gt_day = np.asarray(gt_day)[keep] if gt_day is not None else None
            pred_day = np.asarray(pred_day)[keep] if pred_day is not None else None
            in_creat = np.asarray(in_creat)[keep] if in_creat is not None else None
            sofa_hm = np.asarray(sofa_hm)[keep] if sofa_hm is not None else None
            saps_wm = np.asarray(saps_wm)[keep] if saps_wm is not None else None
            pred_matched = np.asarray(pred_matched)[keep] if pred_matched is not None else None
            log.info("--first-day-only: %d -> %d windows (one per stay, t0_hour==0)", n0, keep.sum())

    sofa_df, sofa_m = sofa_comparison(pred, gt, ds.features_target, dataset,
                                      in_day=in_day, gt_day=gt_day, pred_day=pred_day,
                                      sofa_hour_obs=sofa_hm)
    log.info("SOFA: total_max MAE=%.3f  ΔSOFA≥2 acc=%.3f sens=%.3f spec=%.3f (prev=%.3f)",
             sofa_m.get("sofa_max_mae", float('nan')), sofa_m["dSOFA2_accuracy"],
             sofa_m["dSOFA2_sensitivity"], sofa_m["dSOFA2_specificity"],
             sofa_m["dSOFA2_gt_prevalence"])

    saps_m = {}
    if static_arr.shape[1] > 0:
        saps_m = sapsii_comparison(pred, gt, ds.features_target,
                                   static_arr, ds.features_static, dataset, stay_id,
                                   window_mask=saps_wm)
        log.info("SAPS-II: pred=%.1f gt=%.1f MAE=%.2f | components scored %.2f/%d "
                 "(all 12 in %.1f%% of windows)",
                 saps_m["sapsii_pred_mean"], saps_m["sapsii_gt_mean"], saps_m["sapsii_mae"],
                 saps_m.get("sapsii_components_scored_mean", float("nan")),
                 saps_m.get("sapsii_components_scored_n", 0),
                 100 * saps_m.get("sapsii_all_components_frac", float("nan")))

    news_m = news2_comparison(pred, gt, ds.features_target)
    log.info("NEWS2: pred=%.1f gt=%.1f MAE=%.2f | high-risk(>=7) sens=%.3f spec=%.3f prev=%.3f",
             news_m["news2_pred_mean"], news_m["news2_gt_mean"], news_m["news2_mae"],
             news_m.get("news2_high_sensitivity", float("nan")),
             news_m.get("news2_high_specificity", float("nan")),
             news_m.get("news2_high_gt_prevalence", float("nan")))

    aki_m = aki_comparison(pred, gt, ds.features_target, dataset,
                           in_creat=in_creat, static=static_arr,
                           feats_static=ds.features_static,
                           aki_absolute_rise=aki_absolute_rise)
    if aki_m.get("aki_evaluable"):
        log.info("AKI (KDIGO): stage MAE=%.3f  new-AKI F1=%.3f sens=%.3f prec=%.3f auroc=%.3f (prev=%.3f)",
                 aki_m.get("aki_stage_mae", float('nan')), aki_m.get("aki_f1", float('nan')),
                 aki_m.get("aki_sensitivity", float('nan')), aki_m.get("aki_precision", float('nan')),
                 aki_m.get("aki_auroc", float('nan')), aki_m.get("aki_gt_prevalence", float('nan')))

    # --- MATCHED arm: every score recomputed with the prediction given the ground truth's own
    # mask+LOCF treatment, so the two objects being compared are constructed the same way.
    # Prefixed `matched_`; the unprefixed numbers above keep the dense prediction. Report both and
    # say which is which: the dense arm does not use the future mask (so it is the information-
    # honest one) but compares against an LOCF object; the matched arm is comparable but
    # retrospective. Neither alone is the whole answer.
    matched_m = {}
    if pred_matched is not None:
        try:
            _, m_sofa = sofa_comparison(pred_matched, gt, ds.features_target, dataset,
                                        in_day=in_day, gt_day=gt_day, pred_day=pred_day)
            matched_m.update({f"matched_{k}": v for k, v in m_sofa.items()
                              if not k.startswith("obs_")})
            if static_arr.shape[1] > 0:
                m_saps = sapsii_comparison(pred_matched, gt, ds.features_target,
                                           static_arr, ds.features_static, dataset, stay_id,
                                           window_mask=saps_wm)
                matched_m.update({f"matched_{k}": v for k, v in m_saps.items()})
            m_news = news2_comparison(pred_matched, gt, ds.features_target)
            matched_m.update({f"matched_{k}": v for k, v in m_news.items()})
            m_aki = aki_comparison(pred_matched, gt, ds.features_target, dataset,
                                   in_creat=in_creat, static=static_arr,
                                   feats_static=ds.features_static,
                                   aki_absolute_rise=aki_absolute_rise)
            matched_m.update({f"matched_{k}": v for k, v in m_aki.items()})
            log.info("MATCHED (prediction masked+LOCF like the GT): SOFA max MAE=%.3f (dense %.3f)"
                     "  SAPS-II MAE=%.2f (dense %.2f)",
                     matched_m.get("matched_sofa_max_mae", float("nan")),
                     sofa_m.get("sofa_max_mae", float("nan")),
                     matched_m.get("matched_sapsii_mae", float("nan")),
                     saps_m.get("sapsii_mae", float("nan")))
        except Exception as e:                       # a diagnostic must never take the eval down
            log.warning("matched-arm scoring skipped: %s", e)

    # n_windows was int(len(ds)) -- the size of the test H5 -- regardless of --limit, so every
    # subsample run wrote a table CLAIMING the full split. All ten MIRA tables are affected:
    # the five mira_<ds> ones scored 6000 windows while reporting 19541-211251, and
    # mira_eicu_full scored 150000 while reporting 211251. Record what was actually scored and
    # keep the split size beside it, so coverage is readable from the table alone.
    n_scored = int(len(ds)) if limit is None else int(min(limit, len(ds)))
    summary = {
        "run_name": cfg["run_name"], "model": cfg["model"], "dataset": dataset,
        "n_windows": n_scored,
        "n_windows_split": int(len(ds)),
        "eval_limit": None if limit is None else int(limit),
        # NEWS2 was computed and LOGGED but never written to the JSON, so a 16 h job's numbers
        # existed only in a SLURM log that has since been rotated away. Persist it like the rest.
        "forecast": fc, "sofa": sofa_m, "sapsii": saps_m, "news2": news_m, "aki": aki_m,
        "matched": matched_m,
    }

    if susinf is not None:
        from harmonize.scores.sepsis3 import susinf_coincident
        sofa_df["stay_id"] = stay_id
        # Sepsis-3 requires the ΔSOFA rise within ±24 h of the suspected-infection onset. Keying
        # on the per-stay flag alone put 60-81 % of the positive class outside that band
        # (§37.38 D19), so map the ONSET and test the overlap with this window's target interval.
        _onset = sofa_df["stay_id"].map(
            susinf["susinf_onset_h"].where(susinf["has_susinf"], np.nan)).to_numpy(dtype=float)
        sofa_df["has_susinf"] = susinf_coincident(_onset, t0, ds.window_in, ds.window_out)
        susinf_share = float(sofa_df["has_susinf"].mean())
        if susinf_share < 0.005:
            # No (usable) suspected-infection cohort — Sepsis-3 is NOT evaluable
            # here, NOT a model failure; a near-zero susinf share would otherwise
            # masquerade as 0% sensitivity. (eICU now passes via union mode —
            # abx OR culture, load_susinf — so this branch is for datasets that
            # genuinely lack both antibiotic and culture timing.)
            summary["sepsis3"] = {"evaluable": False, "susinf_share": susinf_share,
                                  "reason": "no antibiotic-start + blood-culture clustering "
                                            "(missing abx timestamps for this dataset)"}
            log.warning("Sepsis-3: NOT evaluable (susinf share %.3f%% — no abx timestamps)",
                        100 * susinf_share)
        else:
            # Sepsis-3 = suspected infection AND organ dysfunction (ΔSOFA≥2) in the window
            gt_sep = (sofa_df["has_susinf"] & (sofa_df["gt_delta_sofa"] >= 2)).values
            pr_sep = (sofa_df["has_susinf"] & (sofa_df["pred_delta_sofa"] >= 2)).values
            summary["sepsis3"] = {
                "evaluable": True, "susinf_share": susinf_share,
                **_detection_metrics(gt_sep, pr_sep,
                                     score=sofa_df["pred_delta_sofa"].values),
            }
            log.info("Sepsis-3: prev=%.3f acc=%.3f sens=%.3f spec=%.3f (susinf=%.1f%%)",
                     summary["sepsis3"]["gt_prevalence"], summary["sepsis3"]["accuracy"],
                     summary["sepsis3"]["sensitivity"], summary["sepsis3"]["specificity"],
                     100 * susinf_share)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Downstream clinical evaluation of a forecast model")
    ap.add_argument("--checkpoint", type=Path, default=None, help="checkpoint dir (with config.json, best.pt)")
    ap.add_argument("--fm", type=str, default=None,
                    help="build a zero-shot foundation-model baseline directly (no checkpoint), e.g. 'chronos'")
    ap.add_argument("--chronos-id", type=str, default="amazon/chronos-bolt-base",
                    help="HF model id for --fm chronos")
    ap.add_argument("--point", type=str, default="median", choices=["median", "mean"],
                    help="point-forecast reduction for FM baselines")
    ap.add_argument("--pred-file", type=Path, default=None,
                    help="evaluate precomputed z-scored predictions (.npy, N×T_out×F_target) "
                         "in H5 order — e.g. Moirai forecasts from harmonize/infer_moirai.py")
    ap.add_argument("--test-h5", required=True, type=Path)
    ap.add_argument("--norm-stats", type=Path, default=None,
                    help="defaults to data/splits/<ds>_norm_stats.parquet")
    ap.add_argument("--long-parquet", type=Path, default=None,
                    help="Stage-1 long parquet → enables Sepsis-3 susinf join")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    # D22: Informer's ProbSparse attention draws its key sample with torch.randint on every
    # forward pass (_prob_QK), from the global RNG, and this script never seeded torch -- so
    # every evaluation of an Informer checkpoint produced slightly different predictions
    # (max |p1-p2| = 0.015 over two identical passes on 32 UMM windows). Seeding here makes
    # any sampling-based architecture reproducible at eval time. It changes nothing for the
    # deterministic models, which is why it is safe to default on.
    ap.add_argument("--aki-absolute-rise", action="store_true", default=False,
                    help="D16: also apply KDIGO's absolute >=0.3 mg/dL 48h rise criterion. OFF by "
                         "default because it changes the aki block of every committed table; "
                         "turning it on means a deliberate full re-score (prevalence 2.6-5.3%% "
                         "-> 9.5-15.3%%).")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed for eval (D22: Informer ProbSparse samples keys at "
                         "inference; unseeded evals are not reproducible)")
    ap.add_argument("--first-day-only", action="store_true", default=False,
                    help="keep only the admission window (t0_hour==0): one prediction per stay at "
                         "the true stay-level prevalence — the literature-comparable view for "
                         "stay-level outcomes and admission scores (SAPS-II/NEWS2).")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of test windows (quick CPU smoke test)")
    ap.add_argument("--permute-static", type=str, default=None,
                    help="demographics ablation: permute this static feature "
                         "(age/sex/weight_admission/height) across the batch and report "
                         "the resulting forecast metrics. ΔMSE vs baseline = its importance.")
    ap.add_argument("--per-stay-mse-out", type=Path, default=None,
                    help="dump per-window z-space MSE (mse_all + mse_phys=physiology-only, i.e. "
                         "excluding medications) keyed by stay_id to this CSV → per-hospital/ward analysis.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset name from h5 filename: <ds>_seed..._<split>__<tag>.h5
    stem = args.test_h5.stem
    dataset = stem.split("_seed")[0]
    norm_path = args.norm_stats or (ROOT / "data" / "splits" / f"{dataset}_norm_stats.parquet")

    if not args.fm and not args.checkpoint and not args.pred_file:
        ap.error("one of --checkpoint, --fm or --pred-file is required")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = ICUDataset(str(args.test_h5), preload=True)
    norm_stats = pd.read_parquet(norm_path).set_index("variable")
    if args.pred_file:
        model, cfg = build_precomputed(args.pred_file, ds, device, n_eval=args.limit)
    elif args.fm:
        model, cfg = build_fm(args.fm, ds, device, args.chronos_id, args.point)
    else:
        model, cfg = load_model(args.checkpoint, ds, device)
    log.info("Loaded %s (%s) on %s — test windows: %d",
             cfg["run_name"], cfg["model"], device, len(ds))

    susinf = load_susinf(args.long_parquet)
    perm_idx = None
    if args.permute_static:
        if args.permute_static == "all":
            perm_idx = "all"
        elif args.permute_static not in ds.features_static:
            ap.error(f"--permute-static {args.permute_static!r} not in {ds.features_static} (or 'all')")
        else:
            perm_idx = ds.features_static.index(args.permute_static)
        # %s, not %d: perm_idx is the string "all" in the all-features mode, which made
        # logging raise TypeError on every --permute-static all run (swallowed by logging,
        # so the eval still finished but the line was lost behind a spurious traceback).
        log.info("Demographics ablation: permuting static feature %r (idx %s)",
                 args.permute_static, perm_idx)
    phys_mask = None
    if args.per_stay_mse_out:                       # physiology = non-medication target features
        import yaml
        cat = yaml.safe_load(open(ROOT / "configs" / "features.yaml"))
        ftype = {k: v.get("type") for k, v in cat.items() if isinstance(v, dict)}
        phys_mask = np.array([ftype.get(f) != "medication" for f in ds.features_target])
        log.info("per-stay MSE dump on; physiology mask keeps %d/%d (non-medication) features",
                 int(phys_mask.sum()), len(phys_mask))
    summary = evaluate(model, cfg, ds, norm_stats, dataset, device,
                       susinf=susinf, batch_size=args.batch_size, limit=args.limit,
                       permute_static_idx=perm_idx, phys_mask=phys_mask,
                       per_stay_out=args.per_stay_mse_out,
                       first_day_only=args.first_day_only,
                       aki_absolute_rise=args.aki_absolute_rise)
    summary["test_h5"] = str(args.test_h5)
    summary["aki_absolute_rise"] = bool(args.aki_absolute_rise)

    perm_tag = f"_permstatic_{args.permute_static}" if args.permute_static else ""
    out = args.out or (ROOT / "docs" / "tables" / f"eval_{cfg['run_name']}{perm_tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("Wrote evaluation summary → %s", out)


if __name__ == "__main__":
    main()
