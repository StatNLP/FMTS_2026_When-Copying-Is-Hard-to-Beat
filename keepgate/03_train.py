#!/usr/bin/env python3
"""Training script for GTSQ prediction with keep gate + learnable feature weights.

Features:
  --use_gate          Enable per-feature keep gate (gate blends pred with persistence)
  --gate_lambda       Weight for gate change-detection auxiliary loss
  --loss_balance lw   Learnable feature weights (Kendall 2018 uncertainty weighting)
  --no_gtsq_input     Only-clinical mode (no GTSQ input, no gate/skip)
  --fusion crossattn  Cross-attention fusion (GTSQ queries, clinical keys/values)
  --fusion film       FiLM fusion (clinical modulates GTSQ via learned scale+shift)

Usage:
  python 03_train.py --data_dir 01_split_sets/gtsq_pred_24h_gap6h_mean_noclin \\
      --model_type gru --use_gate --gate_lambda 0.5 --loss_balance lw
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from models import build_model
from models.gated_predictor import build_feature_slices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def writer(msg, log_file):
    ts = datetime.now().strftime("%y%m%d-%H%M")
    line = f"{ts}: {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def writer2(msg, result_file):
    with open(result_file, "a") as f:
        f.write(msg + "\n")


def load_split(npz_path, mask="target"):
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if mask == "both":
        im = out["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        out["gtsq_out_mask"] = (out["gtsq_out_mask"] == 1) & (im == 1)
        out["gtsq_out_mask"] = out["gtsq_out_mask"].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Class weights from training data
# ---------------------------------------------------------------------------

def compute_class_weights(gtsq_in, gtsq_in_mask, gtsq_means, gtsq_stds,
                          feature_slices, gtsq_ftypes, gtsq_nclasses):
    """Compute pos_weight (binary) and class_weight (ordinal) from training data."""
    binary_pos_weights = {}
    ordinal_class_weights = {}
    raw = gtsq_in * gtsq_stds + gtsq_means

    for g, slc in enumerate(feature_slices):
        mask = gtsq_in_mask[:, g] == 1
        vals = raw[mask, g]
        if len(vals) == 0:
            continue
        ftype = str(gtsq_ftypes[g])
        nclass = int(gtsq_nclasses[g])

        if ftype == 'binary':
            labels = np.clip(np.round(vals), 0, 1).astype(int)
            n_pos = (labels == 1).sum()
            n_neg = (labels == 0).sum()
            pw = min(n_neg / max(n_pos, 1), 10.0)
            binary_pos_weights[g] = float(pw)
        elif ftype == 'ordinal':
            labels = np.clip(np.round(vals), 0, nclass - 1).astype(int)
            n_total = len(labels)
            cw = []
            for c in range(nclass):
                n_c = (labels == c).sum()
                cw.append(n_total / (nclass * max(n_c, 1)))
            ordinal_class_weights[g] = cw

    return binary_pos_weights, ordinal_class_weights


# ---------------------------------------------------------------------------
# Learnable feature weights (Kendall 2018)
# ---------------------------------------------------------------------------

class LearnableWeightLoss(torch.nn.Module):
    """Homoscedastic uncertainty weighting: learns per-feature log-variance.

    Loss for feature g:  exp(-2·log_σ_g) · L_g  +  log_σ_g
                       = L_g / σ_g²  +  log(σ_g)

    Features with high loss → σ increases → lower weight (automatic balancing).
    The log(σ) regularizer prevents σ → ∞ (trivial zero-weight solution).

    Reference: Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.
    """

    def __init__(self, n_features):
        super().__init__()
        self.log_sigma = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, per_feature_losses):
        """per_feature_losses: list of (feature_idx, loss_tensor)."""
        total = torch.tensor(0.0, device=self.log_sigma.device)
        n = 0
        for g, loss_g in per_feature_losses:
            precision = torch.exp(-2.0 * self.log_sigma[g])
            total = total + precision * loss_g + self.log_sigma[g]
            n += 1
        return total / max(n, 1)

    def get_weights(self):
        """Return effective weights (precision) as numpy array."""
        with torch.no_grad():
            return torch.exp(-2.0 * self.log_sigma).cpu().numpy()


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(raw_output, gtsq_target_norm, gtsq_mask,
                 feature_slices, gtsq_ftypes, gtsq_nclasses,
                 gtsq_means, gtsq_stds,
                 binary_pos_weights, ordinal_class_weights,
                 loss_balance, device, learnable_weight_loss=None,
                 only_feature_idx=-1,
                 ordinal_loss=None, change_loss=None,
                 change_weight=5.0, focal_gamma=2.0, gtsq_input=None):
    """Type-specific loss over all features.

    Binary    → BCEWithLogitsLoss + pos_weight
    Ordinal   → CrossEntropyLoss  + class_weight
                ordinal_loss overrides the ordinal branch (review Tier B):
                  'emd'   squared earth-mover distance on the softmax CDF (distance-sensitive)
                  'ordce' distance-weighted CE,  weight = 1 + |argmax - target|

    change_loss reshapes the per-sample reduction (review Tier D; the keep gate keeps the
    persistence floor safe regardless):
      'reweight': upweight CHANGED feature-days (yesterday != tomorrow) by change_weight
      'focal':    focal down-weighting of easy/confident cells (gamma=focal_gamma)

    loss_balance modes: uniform (equal) | lw (learnable Kendall weights). With
    ordinal_loss=None and change_loss=None this equals plain per-feature mean BCE/CE
    (no behaviour change for canonical runs).
    """
    has_target = gtsq_mask.sum(dim=1) > 0
    means_t = torch.tensor(gtsq_means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(gtsq_stds, dtype=torch.float32, device=device)
    target_raw = gtsq_target_norm * stds_t + means_t
    input_raw = None
    if change_loss == 'reweight' and gtsq_input is not None:
        gin_last = gtsq_input[:, -1, :] if gtsq_input.ndim == 3 else gtsq_input
        input_raw = gin_last * stds_t + means_t

    per_feature_losses = []

    for g, slc in enumerate(feature_slices):
        if only_feature_idx >= 0 and g != only_feature_idx:
            continue   # per-feature single-target training (decoupled de-interference test)
        m = (gtsq_mask[:, g] == 1) & has_target
        if m.sum() < 1:
            continue
        ftype = str(gtsq_ftypes[g])
        nclass = int(gtsq_nclasses[g])
        out_g = raw_output[m, slc]
        # Per-sample CLASS weight actually applied inside the loss (None when unweighted).
        # Needed for the denominator: F.cross_entropy(..., weight=w, reduction='none') already
        # returns w_{y_i}*CE_i, so a plain .mean() would divide by N instead of sum(w) -- see below.
        wvec = None

        if ftype == 'binary':
            t = torch.clamp(torch.round(target_raw[m, g]), 0, 1)
            pw = torch.tensor(binary_pos_weights.get(g, 1.0),
                              dtype=torch.float32, device=device)
            ls = F.binary_cross_entropy_with_logits(
                out_g.squeeze(-1), t, pos_weight=pw, reduction='none')
            if change_loss == 'focal':
                p = torch.sigmoid(out_g.squeeze(-1).detach())
                pt = p * t + (1 - p) * (1 - t)
                ls = (1 - pt).pow(focal_gamma) * ls
        elif ftype == 'ordinal':
            t = torch.clamp(torch.round(target_raw[m, g]), 0, nclass - 1).long()
            cw = ordinal_class_weights.get(g)
            cw_t = (torch.tensor(cw, dtype=torch.float32, device=device)
                    if cw is not None else None)
            if cw_t is not None:
                wvec = cw_t[t]                     # the weight each sample contributes
            if ordinal_loss == 'emd':                         # squared earth-mover (ordinal)
                probs = F.softmax(out_g, dim=1)
                cdf_p = torch.cumsum(probs, dim=1)
                cdf_t = torch.cumsum(F.one_hot(t, nclass).float(), dim=1)
                ls = (cdf_p - cdf_t).pow(2).sum(dim=1)
                if cw_t is not None:
                    ls = ls * cw_t[t]
            elif ordinal_loss == 'ordce':                     # distance-weighted CE
                ce = F.cross_entropy(out_g, t, weight=cw_t, reduction='none')
                dist = (out_g.argmax(1) - t).abs().float().detach()
                ls = ce * (1.0 + dist)
            elif ordinal_loss == 'cumlink':                   # cumulative-link (CORAL-style) ordinal regression
                # BCE on the softmax CDF: predict P(y>k) for k=0..K-2 (monotone by cumsum
                # construction, so rank-consistent), target 1 if true rank > k. Proper
                # cumulative-link objective (cf. emd = squared error on the same CDF).
                probs = F.softmax(out_g, dim=1)
                pgt = (1.0 - torch.cumsum(probs, dim=1)[:, :nclass - 1]).clamp(1e-6, 1 - 1e-6)
                kk = torch.arange(nclass - 1, device=device)
                tgt = (t[:, None] > kk[None, :]).float()
                ls = F.binary_cross_entropy(pgt, tgt, reduction='none').sum(dim=1)
                if cw_t is not None:
                    ls = ls * cw_t[t]
            else:
                ls = F.cross_entropy(out_g, t, weight=cw_t, reduction='none')
            if change_loss == 'focal':
                pt = F.log_softmax(out_g, dim=1).gather(1, t[:, None]).squeeze(1).detach().exp()
                ls = (1 - pt).pow(focal_gamma) * ls
        else:  # continuous
            ls = F.mse_loss(out_g.squeeze(-1), gtsq_target_norm[m, g], reduction='none')

        # Denominator fix (reviewer: Michi, July 2026). With class weights, PyTorch's own
        # reduction='mean' normalises by SUM(w), not N. Because we call reduction='none' (to allow
        # focal / change-reweighting on top) the weight is already baked into `ls`, so a plain
        # .mean() divided by N -- inflating class-weighted features relative to unweighted ones and
        # silently interacting with the Kendall sigma. Normalise by the weight mass instead.
        # Nothing is double-counted: each cell contributes to exactly one feature, and features are
        # averaged once in the caller. (BCE keeps .mean(): PyTorch's pos_weight convention also
        # divides by N, so binary items are already consistent with the reference implementation.)
        if change_loss == 'reweight' and input_raw is not None:
            w = torch.where(torch.round(input_raw[m, g]) != torch.round(target_raw[m, g]),
                            torch.tensor(change_weight, dtype=torch.float32, device=device),
                            torch.tensor(1.0, dtype=torch.float32, device=device))
            denom = (w * wvec).sum() if wvec is not None else w.sum()
            loss_g = (ls * w).sum() / denom.clamp(min=1e-8)
        elif wvec is not None:
            loss_g = ls.sum() / wvec.sum().clamp(min=1e-8)
        else:
            loss_g = ls.mean()

        per_feature_losses.append((g, loss_g))

    if not per_feature_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if loss_balance == 'lw' and learnable_weight_loss is not None:
        return learnable_weight_loss(per_feature_losses)
    else:  # uniform
        return torch.stack([l for _, l in per_feature_losses]).mean()


# ---------------------------------------------------------------------------
# Gate change-detection loss
# ---------------------------------------------------------------------------

def compute_gate_loss(gate, gtsq_input, gtsq_target, gtsq_mask,
                      gtsq_means, gtsq_stds, device, focal_gamma=0.0, pos_weight=1.0):
    """Auxiliary loss: encourage gate to open where target ≠ input.

    gate:       [B, G] — predicted gate values ∈ [0,1]
    changed:    [B, G] — 1.0 where target class ≠ input class
    Loss:       masked BCE(gate, changed)  (focal + pos_weight when focal_gamma>0 /
                pos_weight!=1 — the dedicated change-DETECTOR mode, --change_detector).

    This teaches the gate to open for real changes and stay closed otherwise.
    For multi-day input (B, K, G), uses the most recent day (last timestep).
    """
    means = torch.tensor(gtsq_means, dtype=torch.float32, device=device)
    stds = torch.tensor(gtsq_stds, dtype=torch.float32, device=device)

    # Use most recent day for change detection
    if gtsq_input.ndim == 3:
        gtsq_last = gtsq_input[:, -1, :]
    else:
        gtsq_last = gtsq_input

    inp_raw = gtsq_last * stds + means
    tgt_raw = gtsq_target * stds + means

    inp_cls = torch.round(inp_raw)
    tgt_cls = torch.round(tgt_raw)

    changed = (inp_cls != tgt_cls).float()  # [B, G]
    mask = gtsq_mask  # [B, G]

    g = gate.clamp(1e-6, 1 - 1e-6)
    loss = F.binary_cross_entropy(g, changed, reduction='none')  # [B, G]
    if pos_weight != 1.0:                                   # up-weight the rare 'changed' class
        loss = loss * torch.where(changed > 0.5, float(pos_weight), 1.0)
    if focal_gamma > 0:                                     # focal: focus on hard/rare cells
        pt = g * changed + (1 - g) * (1 - changed)
        loss = (1 - pt).pow(focal_gamma) * loss
    masked_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
    return masked_loss


# ---------------------------------------------------------------------------
# Decode + evaluate
# ---------------------------------------------------------------------------

def gate_auroc(model, split_data, batch_size, gtsq_means, gtsq_stds, device, use_clinical):
    """AUROC(gate gamma -> cell truly changes) over a split — the change-DETECTOR metric.
    Used for model selection in --change_detector mode (prediction-F1 is meaningless there)."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    gi_all = split_data['gtsq_in']; N = len(gi_all)
    gm = np.asarray(gtsq_means); gs = np.asarray(gtsq_stds)
    gin = gi_all[:, -1, :] if gi_all.ndim == 3 else gi_all
    inp_cls = np.round(gin * gs + gm); tgt_cls = np.round(split_data['gtsq_out'] * gs + gm)
    changed = (inp_cls != tgt_cls); om = split_data['gtsq_out_mask'] == 1
    has_clin = 'clinical' in split_data and split_data['clinical'] is not None
    gam = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            gi = torch.tensor(gi_all[s:e], dtype=torch.float32, device=device)
            ci = (torch.tensor(split_data['clinical'][s:e], dtype=torch.float32, device=device)
                  if use_clinical and has_clin else None)
            _, g = model(clinical_input=ci, clinical_mark=None, gtsq_input=gi)
            gam.append(g.cpu().numpy())
    gam = np.concatenate(gam, 0)
    m = om & np.isfinite(gam)
    y = changed[m].astype(int); p = gam[m]
    model.train()
    return float(roc_auc_score(y, p)) if y.any() and not y.all() else 0.0


def decode_output(raw_np, feature_slices, gtsq_ftypes, gtsq_nclasses):
    """Decode raw model output to class predictions."""
    N = raw_np.shape[0]
    G = len(feature_slices)
    decoded = np.zeros((N, G), dtype=np.float32)

    for g, slc in enumerate(feature_slices):
        ftype = str(gtsq_ftypes[g])
        nclass = int(gtsq_nclasses[g])
        out_g = raw_np[:, slc]

        if ftype == 'binary':
            prob = 1.0 / (1.0 + np.exp(-out_g[:, 0]))
            decoded[:, g] = (prob > 0.5).astype(float)
        elif ftype == 'ordinal':
            decoded[:, g] = np.argmax(out_g, axis=1).astype(float)
        else:
            decoded[:, g] = out_g[:, 0]

    return decoded


DOMAIN_PATTERNS = {
    'Klinisch': ['i01000_', 'i02000_', 'i03000_', 'i04000_'],
    'Organ': ['i11100_', 'i09100_', 'i09000_'],
    'Lokalisierung': ['i05100_'],
}


def _assign_domain(fname):
    for domain, prefixes in DOMAIN_PATTERNS.items():
        for prefix in prefixes:
            if fname.startswith(prefix):
                return domain
    return 'Other'


# NOTE: this is the FROZEN training-time metric path (its numbers are already in every
# checkpoint JSON). Its decode + four-metric conventions are the canonical reference that
# Metric definitions (f1 / kappa / f1_change / cell-pooled accuracy) are the single source of truth
# in keepgate/gtsq_eval.py; 05_baselines + the analysis scripts import from there. This function keeps
# its own loop because it ALSO does logit-decode, domain aggregation and continuous MSE — but its
# per-feature f1/kappa/f1_change calls are identical to gtsq_eval and are LOCKED by
# keepgate/test_metric_parity.py (run it if you touch either; it fails on any drift).
def eval_gtsq(model, split_data, batch_size, feature_slices,
              gtsq_ftypes, gtsq_nclasses, gtsq_means, gtsq_stds,
              gtsq_features, device, use_clinical, use_day_of_stay=False, use_supervisor='none'):
    """Evaluate model on a split.

    Returns (f1_macro, mse_cont, per_feat, domain_f1, extra_metrics).

    extra_metrics contains:
      - kappa: Cohen's Kappa (macro-averaged over features)
      - f1_change: Change-Detection F1 (macro-avg, only on samples where y changed)
      - domain_kappa: per-domain Kappa
      - domain_f1_change: per-domain Change-Detection F1
    """
    model.eval()
    gtsq_in = split_data['gtsq_in']
    gtsq_out = split_data['gtsq_out']
    gtsq_out_mask = split_data['gtsq_out_mask']
    has_clinical = 'clinical' in split_data and split_data['clinical'] is not None
    N = len(gtsq_in)

    all_raw = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            gi = torch.tensor(gtsq_in[start:end], dtype=torch.float32, device=device)

            # clinical_mark is unused: every clinical encoder ignores the temporal
            # mark (the Transformer uses DataEmbedding_wo_temp; RNN/MLP/CNN drop it).
            ci = None
            if use_clinical and has_clinical:
                ci = torch.tensor(split_data['clinical'][start:end],
                                  dtype=torch.float32, device=device)
            dos = None
            if use_day_of_stay and 'day_of_stay' in split_data:
                dos = torch.tensor(split_data['day_of_stay'][start:end] / 7.0,
                                   dtype=torch.float32, device=device)
            sd = sn = None
            if use_supervisor in ('d', 'both') and 'supervisor_d' in split_data:
                sd = torch.tensor(split_data['supervisor_d'][start:end], device=device)
            if use_supervisor in ('next', 'both') and 'supervisor_next' in split_data:
                sn = torch.tensor(split_data['supervisor_next'][start:end], device=device)
            out, _ = model(clinical_input=ci, clinical_mark=None, gtsq_input=gi, day_of_stay=dos,
                           supervisor_d=sd, supervisor_next=sn)
            all_raw.append(out.cpu().numpy())

    raw_np = np.concatenate(all_raw, axis=0)
    decoded = decode_output(raw_np, feature_slices, gtsq_ftypes, gtsq_nclasses)

    # Denormalize targets
    target_raw = gtsq_out * gtsq_stds + gtsq_means
    G = len(feature_slices)

    # Denormalize input (most recent day) for persistence / change detection
    if gtsq_in.ndim == 3:
        input_last = gtsq_in[:, -1, :]
    else:
        input_last = gtsq_in
    input_raw = input_last * gtsq_stds + gtsq_means

    from sklearn.metrics import f1_score, cohen_kappa_score
    per_feat = {}
    f1_list = []
    mse_list = []
    kappa_list = []
    f1_change_list = []
    domain_f1s = defaultdict(list)
    domain_kappas = defaultdict(list)
    domain_f1_changes = defaultdict(list)

    for g in range(G):
        mask = gtsq_out_mask[:, g] == 1
        if mask.sum() < 1:
            continue
        ftype = str(gtsq_ftypes[g])
        nclass = int(gtsq_nclasses[g])
        fname = str(gtsq_features[g]) if gtsq_features is not None else f"f{g}"
        domain = _assign_domain(fname)

        if ftype in ('binary', 'ordinal'):
            if ftype == 'binary':
                t_cls = np.clip(np.round(target_raw[mask, g]), 0, 1).astype(int)
                i_cls = np.clip(np.round(input_raw[mask, g]), 0, 1).astype(int)
            else:
                t_cls = np.clip(np.round(target_raw[mask, g]), 0, nclass - 1).astype(int)
                i_cls = np.clip(np.round(input_raw[mask, g]), 0, nclass - 1).astype(int)
            p_cls = np.clip(decoded[mask, g], 0, nclass - 1).astype(int)

            # Standard macro F1
            f1 = float(f1_score(t_cls, p_cls, average='macro',
                                labels=list(range(nclass)), zero_division=0))

            # Cohen's Kappa
            try:
                kappa = float(cohen_kappa_score(t_cls, p_cls,
                                               labels=list(range(nclass))))
            except Exception:
                kappa = 0.0

            # Change-Detection F1: only on samples where input ≠ target
            changed_mask = (i_cls != t_cls)
            n_changed = int(changed_mask.sum())
            if n_changed >= 1:
                f1_ch = float(f1_score(t_cls[changed_mask], p_cls[changed_mask],
                                       average='macro',
                                       labels=list(range(nclass)),
                                       zero_division=0))
            else:
                f1_ch = float('nan')

            # Accuracy + change-accuracy. Recorded alongside F1 so the four headline metrics
            # (F1 / accuracy, each overall and on the changed cells) come straight out of the
            # run — no re-inference needed. Persistence change_acc is 0 by construction.
            acc = float((p_cls == t_cls).mean())
            acc_ch = (float((p_cls[changed_mask] == t_cls[changed_mask]).mean())
                      if n_changed >= 1 else float('nan'))

            per_feat[fname] = {
                'type': ftype, 'f1_macro': f1, 'domain': domain,
                'kappa': kappa, 'f1_change': f1_ch,
                'accuracy': acc, 'acc_change': acc_ch,
                'n_changed': n_changed, 'n_total': int(mask.sum()),
            }
            f1_list.append(f1)
            kappa_list.append(kappa)
            domain_f1s[domain].append(f1)
            domain_kappas[domain].append(kappa)
            if not np.isnan(f1_ch):
                f1_change_list.append(f1_ch)
                domain_f1_changes[domain].append(f1_ch)
        else:
            mse = float(np.mean((decoded[mask, g] - gtsq_out[mask, g]) ** 2))
            per_feat[fname] = {'type': ftype, 'mse': mse, 'domain': domain}
            mse_list.append(mse)

    f1_macro = float(np.mean(f1_list)) if f1_list else 0.0
    mse_cont = float(np.mean(mse_list)) if mse_list else 0.0
    domain_f1 = {d: float(np.mean(vs)) for d, vs in domain_f1s.items()}

    # Cell-pooled accuracy (weight every observed cell equally) and change-accuracy
    # (only the cells that actually change) — the accuracy counterparts of f1_macro /
    # f1_change, so all four headline metrics are persisted with the run.
    _n_ok = sum(v['accuracy'] * v['n_total'] for v in per_feat.values() if 'accuracy' in v)
    _n_all = sum(v['n_total'] for v in per_feat.values() if 'accuracy' in v)
    _c_ok = sum(v['acc_change'] * v['n_changed'] for v in per_feat.values()
                if v.get('n_changed', 0) >= 1 and not np.isnan(v.get('acc_change', np.nan)))
    _c_all = sum(v['n_changed'] for v in per_feat.values()
                 if v.get('n_changed', 0) >= 1 and not np.isnan(v.get('acc_change', np.nan)))
    extra_metrics = {
        'kappa': float(np.mean(kappa_list)) if kappa_list else 0.0,
        'f1_change': float(np.mean(f1_change_list)) if f1_change_list else 0.0,
        'accuracy': float(_n_ok / _n_all) if _n_all else 0.0,
        'acc_change': float(_c_ok / _c_all) if _c_all else 0.0,
        'domain_kappa': {d: float(np.mean(vs)) for d, vs in domain_kappas.items()},
        'domain_f1_change': {d: float(np.mean(vs))
                             for d, vs in domain_f1_changes.items()},
    }

    model.train()
    return f1_macro, mse_cont, per_feat, domain_f1, extra_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GTSQ Prediction — Gate + Learnable Weights")

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="02_models")

    # Model
    p.add_argument("--model_type", type=str, default="transformer",
                   choices=["transformer", "mlp", "cnn", "gru", "lstm", "persistence"])
    p.add_argument("--fusion", type=str, default="concat",
                   choices=["concat", "crossattn", "film"],
                   help="Fusion mechanism: concat (default), crossattn, or film")
    p.add_argument("--d_layers", type=int, default=2,
                   help="Number of cross-attention decoder layers (for --fusion crossattn)")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--activation", type=str, default="gelu")

    # Gate
    p.add_argument("--use_gate", action="store_true", default=False,
                   help="Enable per-feature keep gate")
    p.add_argument("--gate_lambda", type=float, default=0.5,
                   help="Weight for gate change-detection loss")
    p.add_argument("--select_metric", type=str, default="f1", choices=["f1", "acc"],
                   help="Checkpoint-selection metric on val: macro-F1 (default) or cell-pooled accuracy")
    p.add_argument("--mask", type=str, default="target", choices=["target", "both"],
                   help="Which cells count, for the LOSS and for every metric. `target` = the day-D+1 "
                        "value is observed (historical default). `both` = day D is observed too, so a "
                        "change is defined and persistence has something real to copy. Changing this "
                        "alters the training signal, not only the reporting.")
    p.add_argument("--copy_scale", type=float, default=8.0,
                   help="Logit magnitude of the persistence/markov copy path (models/gated_predictor: "
                        "the copied class gets +copy_scale, binaries +/-copy_scale). Large values make "
                        "the copy path near-saturated, so the gate must overcome a very confident vote "
                        "to change a cell; small values make departing from persistence cheap. Was "
                        "hardcoded at 8.0 until 2026-07-30.")

    # Skip (for comparison with old approach)
    p.add_argument("--use_skip", action="store_true", default=False,
                   help="Additive skip connection (old approach, for comparison)")
    p.add_argument("--persist_path", type=str, default="linear",
                   choices=["linear", "copy", "markov"],
                   help="Persistence reference: 'linear' (learned W_pers), 'copy' (exact one-hot "
                        "copy; gamma=0 recovers persistence), or 'markov' (copy + confident "
                        "first-order transition overrides, e.g. neu->akut; gamma=0 recovers a "
                        "no-regret reference that encodes deterministic clinical rules)")
    p.add_argument("--markov_min_n", type=int, default=10,
                   help="Minimum train support for a cell to enter the markov transition table. "
                        "Without it a single training row with purity 1.0 becomes a 'rule' (this "
                        "happened: Knochenmark_Ursache 2->1 from n=1).")
    p.add_argument("--markov_tau", type=float, default=0.9,
                   help="Confidence threshold for a markov override (deviate from copy only where "
                        "max P(next|state) >= tau). Higher = safer / fewer overrides.")

    # Input modes
    p.add_argument("--no_clinical", dest="use_clinical", action="store_false")
    p.add_argument("--no_gtsq_input", dest="use_gtsq_input", action="store_false")
    p.add_argument("--gtsq_enc_layers", type=int, default=2,
                   choices=[1, 2, 3],
                   help="Number of layers in the GTSQ MLP encoder (default: 2)")
    p.set_defaults(use_clinical=True, use_gtsq_input=True)

    # Loss
    p.add_argument("--loss_balance", type=str, default="uniform",
                   choices=["uniform", "lw"],
                   help="uniform = equal weight per feature (DEFAULT since 2026-07-30: the simpler "
                        "method, and it does not let the learned log-variance drive rare-but-learnable "
                        "features to zero weight); lw = learnable weights (Kendall 2018).")
    # Review-driven objectives (Tier B/D). Both default off -> identical to the canonical run;
    # both are persisted to config.json (vars(args)) so 07_aggregate isolates them.
    p.add_argument("--ordinal_loss", type=str, default=None, choices=["ce", "emd", "ordce", "cumlink"],
                   help="Tier B ordinal-aware objective: emd=squared earth-mover on the softmax "
                        "CDF, ordce=distance-weighted CE, cumlink=cumulative-link/CORAL-style BCE on "
                        "the softmax CDF; default (None)=plain CE.")
    p.add_argument("--change_loss", type=str, default=None, choices=["reweight", "focal"],
                   help="Tier D change-focused training: reweight upweights changed cells by "
                        "--change_weight, focal down-weights easy cells; default (None)=standard.")
    p.add_argument("--change_weight", type=float, default=5.0,
                   help="(--change_loss reweight) weight on changed feature-days")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="(--change_loss focal) focusing parameter gamma")
    p.add_argument("--change_detector", action="store_true", default=False,
                   help="Dedicated change-DETECTOR: train the gate gamma as a per-cell 'will it change "
                        "tomorrow?' classifier as the SOLE objective (prediction loss zeroed). Tests whether "
                        "a detection-only model beats the gate's AUROC(gamma->change)=0.79. Needs --use_gate.")
    p.add_argument("--detector_focal", type=float, default=0.0,
                   help="Focal-loss gamma for --change_detector (0 = plain BCE; 2 = focus on hard/rare cells).")
    p.add_argument("--detector_pos_weight", type=float, default=1.0,
                   help="Up-weight the rare 'changed' class in --change_detector (e.g. 5 or 10).")
    p.add_argument("--use_supervisor", type=str, default="none",
                   choices=["none", "d", "next", "both"],
                   help="rater identity as input: d=day-D attending (always available), "
                        "next=the attending who files day D+1 (needs the duty rota), both")
    p.add_argument("--use_day_of_stay", action="store_true", default=False,
                   help="Add the episode-relative ICU day (0-based) as an input covariate "
                        "(embedded, added to the fused representation; never a target). "
                        "Tests whether early, more-volatile stay-days are more predictable.")
    p.add_argument("--no_class_weights", action="store_true", default=False,
                   help="Disable class balancing (plain CE/BCE, no pos_weight/class_weight). "
                        "Diagnostic: isolates training-objective effect from representation.")

    # Training
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--samples_per_epoch", type=int, default=102400)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience_scheduler", type=int, default=3)
    p.add_argument("--patience_break", type=int, default=6)
    p.add_argument("--clip_grad", type=float, default=None)
    p.add_argument("--grad_warn_threshold", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)

    # Embedding (for Transformer clinical encoder)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--factor", type=int, default=1)

    # Sanity check: shuffle clinical data across patients
    p.add_argument("--only_feature_idx", type=int, default=-1,
                   help="If >=0, train the main loss on ONLY this GTSQ feature index "
                        "(0..G-1) -- decoupled single-target run to test whether "
                        "de-interference alone recovers persistence (colleague's hypothesis).")
    p.add_argument("--drop_clinical", default="",
                   help="comma-separated clinical variable names to remove (value+mask columns), "
                        "e.g. culture_result_returned,culture_positive to keep only the "
                        "collection-time culture signal")
    p.add_argument("--shuffle_patients", action="store_true", default=False,
                   help="Shuffle clinical data across patients (sanity check)")
    p.add_argument("--train_change_min", type=int, default=0,
                   help="Stage 4: RETRAIN on day-pairs with >= this many changed cells "
                        "(0=all, 1=>0 changes, 2=>1, 4=>3). Filters TRAIN only; val/test stay full.")
    p.add_argument("--train_subsample", type=int, default=0,
                   help="Stage 4 control: random subset of the FULL train cohort at this size, to "
                        "separate 'volatile-day distribution' from 'less data'.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    data_train = load_split(os.path.join(args.data_dir, "train.npz"), args.mask)
    data_val   = load_split(os.path.join(args.data_dir, "val.npz"), args.mask)
    data_test  = load_split(os.path.join(args.data_dir, "test.npz"), args.mask)
    norm = dict(np.load(os.path.join(args.data_dir, "norm_stats.npz"),
                        allow_pickle=True))

    # --- Stage-4 cohort selection: RETRAIN on the changing day-pairs -------------------
    # --train_change_min K keeps only training rows whose day-pair has >= K changed cells,
    # i.e. it removes the stable-day dominance from the TRAINING DISTRIBUTION (a different
    # question from filtering rows at scoring time, which leaves the model untouched).
    # --train_subsample N is the N-matched control: a random subset of the FULL cohort at the
    # same size, so "trained on volatile days" can be separated from "trained on less data".
    # val/test are never filtered -- the deployable number is always the full test set (you
    # cannot filter test by change-count without knowing the answer).
    if args.train_change_min > 0 or args.train_subsample > 0:
        gm = norm["gtsq_means"].astype(float); gs = norm["gtsq_stds"].astype(float)
        gi = data_train["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = data_train["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = np.round(gi * gs + gm); T = np.round(data_train["gtsq_out"] * gs + gm)
        obs = (im == 1) & (data_train["gtsq_out_mask"] == 1)
        n_chg = ((I != T) & obs).sum(1)
        keep = n_chg >= args.train_change_min
        if args.train_subsample > 0:                       # N-matched control on the full cohort
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(n_chg), size=min(args.train_subsample, len(n_chg)), replace=False)
            keep = np.zeros(len(n_chg), bool); keep[idx] = True
        n_before = len(n_chg)
        data_train = {k: (v[keep] if hasattr(v, "__len__") and len(v) == n_before else v)
                      for k, v in data_train.items()}
        print(f"[cohort] train rows {n_before} -> {int(keep.sum())} "
              f"(change_min={args.train_change_min}, subsample={args.train_subsample})")

    # --- Shuffle clinical data across patients (sanity check) ---
    if args.shuffle_patients:
        for split_data in [data_train, data_val, data_test]:
            if 'clinical' not in split_data or 'enc_ids' not in split_data:
                continue
            enc_ids = split_data['enc_ids']
            unique_ids = np.unique(enc_ids)
            # Build permutation: each patient gets another patient's clinical data
            perm = np.random.permutation(len(unique_ids))
            # Ensure no patient maps to itself
            for i in range(len(perm)):
                if perm[i] == i:
                    swap = (i + 1) % len(perm)
                    perm[i], perm[swap] = perm[swap], perm[i]
            id_map = dict(zip(unique_ids, unique_ids[perm]))
            # Apply: for each sample, replace clinical data with mapped patient's
            clin = split_data['clinical'].copy()
            for src_id, dst_id in id_map.items():
                src_mask = enc_ids == src_id
                dst_mask = enc_ids == dst_id
                # Take clinical rows from dst patient, truncate/pad to match count
                dst_clin = split_data['clinical'][dst_mask]
                src_idx = np.where(src_mask)[0]
                for j, idx in enumerate(src_idx):
                    clin[idx] = dst_clin[j % len(dst_clin)]
            split_data['clinical'] = clin

    gtsq_means    = norm['gtsq_means'].astype(np.float32)
    gtsq_stds     = norm['gtsq_stds'].astype(np.float32)
    gtsq_features = norm['gtsq_features']
    gtsq_ftypes   = norm['gtsq_ftypes']
    gtsq_nclasses = norm['gtsq_nclasses']
    G = len(gtsq_features)

    # Clinical info
    has_clinical = ('clinical' in data_train
                    and data_train['clinical'] is not None
                    and data_train['clinical'].shape[-1] > 0)
    if not has_clinical:
        args.use_clinical = False
    if not has_clinical and not args.use_gtsq_input:
        print("ERROR: No clinical data in this split AND --no_gtsq_input. Nothing to train on.")
        sys.exit(1)

    # --drop_clinical removes named channels from the physiology block before anything else sees it.
    # The block is [values | masks], so a variable's value AND mask column must go together or the
    # model reads a mask that no longer belongs to its value. Used to separate the culture signals:
    # culture_drawn is stamped at COLLECTION time (leakage-safe, current), while
    # culture_result_returned / culture_positive are stamped at arrival, a median 123 h later, so
    # inside a 24 h window they describe a culture drawn about five days earlier.
    if has_clinical and getattr(args, 'drop_clinical', ''):
        allv = [str(x) for x in norm['clinical_variables']]
        want = [w.strip() for w in args.drop_clinical.split(',') if w.strip()]
        miss = [w for w in want if w not in allv]
        if miss:
            print(f"ERROR: --drop_clinical names unknown variables: {miss}"); sys.exit(1)
        Vall = len(allv)
        keep = [i for i, v in enumerate(allv) if v not in want]
        cols = keep + [Vall + i for i in keep]
        if not keep:
            print("ERROR: --drop_clinical would remove every clinical variable; "
                  "use --no_clinical instead so the model is built without the branch"); sys.exit(1)
        # A silent skip here produced a FULL-clinical run wearing an ablation's _drop<hash> tag --
        # indistinguishable afterwards from a real ablation. A width mismatch means the split is not
        # the one these variable names describe, which is never recoverable by carrying on.
        for d in (data_train, data_val, data_test):
            if 'clinical' not in d or d['clinical'] is None:
                continue
            if d['clinical'].shape[-1] != 2 * Vall:
                print(f"ERROR: --drop_clinical expects clinical width {2 * Vall} "
                      f"(2 x {Vall} variables) but this split has {d['clinical'].shape[-1]}; "
                      f"the split does not match norm['clinical_variables']"); sys.exit(1)
            d['clinical'] = d['clinical'][:, :, cols]
        print(f"[clinical] dropped {want} -> {len(keep)} of {Vall} variables remain")
        # Recorded for provenance only. NOTE: 08_probes.reconstruct does NOT read this yet, so a
        # dropped-channel run still cannot be reloaded for gamma-based analyses (eval_deploy
        # swallows the width mismatch). Such runs are usable for their own metrics and nothing else.
        args.dropped_clinical_idx = keep

    V = data_train['clinical'].shape[-1] // 2 if has_clinical else 0
    seq_len = int(norm.get('seq_len', 24))
    gtsq_history = int(norm.get('gtsq_history', 1))

    # Feature slices
    gtsq_feature_info = list(zip(gtsq_ftypes, gtsq_nclasses))
    feature_slices, total_output_size = build_feature_slices(gtsq_feature_info)

    # Class weights
    binary_pw, ordinal_cw = compute_class_weights(
        data_train['gtsq_out'], data_train['gtsq_out_mask'],
        gtsq_means, gtsq_stds, feature_slices, gtsq_ftypes, gtsq_nclasses)
    if getattr(args, 'no_class_weights', False):
        binary_pw, ordinal_cw = {}, {}  # plain CE/BCE (pos_weight=1, no class_weight)

    # --- Determine input mode string ---
    if args.use_clinical and args.use_gtsq_input:
        input_mode = "clin+gtsq"
    elif args.use_clinical:
        input_mode = "onlyclin"
    else:
        input_mode = "gtsq"

    # --- Determine variant string ---
    model_abbrev = {'transformer': 'tran', 'persistence': 'pers'}.get(
        args.model_type, args.model_type)
    # persist_path is part of the identity for gate/skip (linear vs copy) — without it
    # the linear and copy variants share a run_dir and clobber each other's checkpoint.
    if args.use_gate:
        variant = f"gate_l{args.gate_lambda}_{args.persist_path}"
    elif args.use_skip:
        variant = f"sk_{args.persist_path}"
    else:
        variant = "base"
    loss_abbrev = args.loss_balance

    # --- Run directory ---
    ds_name = os.path.basename(args.data_dir.rstrip('/'))
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    shuf_tag = "_shuf" if args.shuffle_patients else ""
    # A dropped-channel run is a DIFFERENT model on different inputs; without a tag it lands on
    # the canonical run's name in the same --output_dir and overwrites its checkpoint/metrics.
    # Hash the NAMES, not the count: --drop_clinical culture_drawn and culture_positive both drop
    # one variable and would otherwise share a run_name and overwrite each other.
    drop_tag = ("_drop" + __import__("hashlib").md5(
        ",".join(sorted(v.strip() for v in args.drop_clinical.split(","))).encode()
    ).hexdigest()[:6]) if getattr(args, "drop_clinical", "") else ""
    # The masking convention changes the loss, the checkpoint selection AND the metrics, so two runs
    # that differ only in --mask are not comparable. Tag it into the name: a run written to the wrong
    # output tree is then still self-describing, instead of being distinguishable only by timestamp.
    mask_tag = "_m2" if args.mask == "both" else ""
    hist_tag = f"_h{gtsq_history}" if gtsq_history > 1 else ""
    fuse_tag = f"_f{args.fusion}" if args.fusion != "concat" else ""
    genc_tag = f"_genc{args.gtsq_enc_layers}" if getattr(args, 'gtsq_enc_layers', 2) != 2 else ""
    onlyf_tag = f"_onlyf{args.only_feature_idx}" if args.only_feature_idx >= 0 else ""
    # Review Tier B/D tags — keep new objectives in a distinct run dir so they never collide
    # with (or overwrite) a canonical run; empty for canonical runs (name unchanged).
    ol_tag = f"_ol{args.ordinal_loss}" if getattr(args, 'ordinal_loss', None) else ""
    cl_tag = f"_cl{args.change_loss}" if getattr(args, 'change_loss', None) else ""
    ncw_tag = "_ncw" if getattr(args, 'no_class_weights', False) else ""
    dos_tag = "_dos" if getattr(args, 'use_day_of_stay', False) else ""
    sup_tag = ("_sup" + args.use_supervisor) if getattr(args, "use_supervisor", "none") != "none" else ""
    cdet_tag = (f"_cdet{'f'+str(args.detector_focal) if args.detector_focal else ''}"
                f"{'w'+str(args.detector_pos_weight) if args.detector_pos_weight != 1.0 else ''}"
                if getattr(args, 'change_detector', False) else "")
    run_name = (f"{timestamp}_{ds_name}_{input_mode}_{model_abbrev}_{variant}{onlyf_tag}"
                f"_d{args.d_model}_e{args.e_layers}_ff{args.d_ff}"
                f"_lb{loss_abbrev}_lr{args.lr}_bs{args.batch_size}_s{args.seed}"
                f"{hist_tag}{fuse_tag}{shuf_tag}{drop_tag}{mask_tag}{genc_tag}{ol_tag}{cl_tag}{ncw_tag}{dos_tag}{sup_tag}{cdet_tag}")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "log.txt")
    result_file = os.path.join(run_dir, "resultlist.txt")
    model_path = os.path.join(run_dir, "model.pytorch")

    writer(f"Run: {run_name}", log_file)
    writer(f"Device: {device}", log_file)
    writer(f"Input: {input_mode}, Gate: {args.use_gate}, Skip: {args.use_skip}, "
           f"Loss: {args.loss_balance}, Shuffle: {args.shuffle_patients}", log_file)
    writer(f"G={G}, V={V}, seq_len={seq_len}, gtsq_history={gtsq_history}, "
           f"output_size={total_output_size}", log_file)

    # --- Build model ---
    args.enc_in = V * 2 if has_clinical else 0
    args.seq_len = seq_len
    args.gtsq_in = G
    args.gtsq_out = G
    args.gtsq_feature_info = gtsq_feature_info
    args.gtsq_history = gtsq_history
    args.gtsq_means = gtsq_means      # needed by 'copy' persistence path
    args.gtsq_stds = gtsq_stds

    # rater-embedding vocab size from the data (train-fitted in 02_preprocess; 0 = unknown)
    if getattr(args, 'use_supervisor', 'none') != 'none':
        _sup_max = 0
        for _d in (data_train, data_val, data_test):
            for _k in ('supervisor_d', 'supervisor_next'):
                if _k in _d and len(_d[_k]):
                    _sup_max = max(_sup_max, int(_d[_k].max()))
        args.n_supervisors = _sup_max + 1
        writer(f"use_supervisor={args.use_supervisor}  n_supervisors={args.n_supervisors} "
               f"(index 0 = unknown/unseen rater)", log_file)
    model = build_model(args).to(device)

    # 'markov' persistence reference: fill the confident first-order transition table
    # from TRAIN (parameter-free). Override copy only where max P(next|state) >= tau
    # AND the argmax is a *different* class (e.g. neu->akut); else stay (== copy).
    if args.persist_path == "markov" and (args.use_gate or args.use_skip):
        gi = data_train['gtsq_in']
        gi = gi[:, -1, :] if gi.ndim == 3 else gi
        in_c = np.round(gi * gtsq_stds + gtsq_means).astype(int)
        out_c = np.round(data_train['gtsq_out'] * gtsq_stds + gtsq_means).astype(int)
        om = data_train['gtsq_out_mask']
        table = model.markov_map.detach().cpu().clone()
        n_over = 0
        for g, (_, nc) in enumerate(gtsq_feature_info):
            nc = int(nc); m = om[:, g] == 1
            s = np.clip(in_c[m, g], 0, nc - 1); t = np.clip(out_c[m, g], 0, nc - 1)
            T = np.zeros((nc, nc)); np.add.at(T, (s, t), 1.0)
            for c in range(nc):
                tot = T[c].sum()
                if tot >= args.markov_min_n and T[c].max() / tot >= args.markov_tau \
                        and int(T[c].argmax()) != c:
                    table[g, c] = int(T[c].argmax()); n_over += 1
        model.markov_map.copy_(table.to(device))
        writer(f"markov_map: {n_over} confident transition overrides "
               f"(tau={args.markov_tau}, min_n={args.markov_min_n})", log_file)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer(f"Model: {args.model_type}, params: {trainable_params:,}", log_file)

    # Save config
    config = {k: v for k, v in vars(args).items()
              if not k.startswith('_')
              and k not in ('gtsq_feature_info', 'gtsq_means', 'gtsq_stds')}
    config['trainable_params'] = trainable_params
    config['run_dir'] = run_dir
    config['run_name'] = run_name
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # --- Evaluation kwargs ---
    eval_kw = dict(
        feature_slices=feature_slices, gtsq_ftypes=gtsq_ftypes,
        gtsq_nclasses=gtsq_nclasses, gtsq_means=gtsq_means, gtsq_stds=gtsq_stds,
        gtsq_features=gtsq_features, device=device, use_clinical=args.use_clinical,
        use_day_of_stay=getattr(args, 'use_day_of_stay', False),
        use_supervisor=getattr(args, 'use_supervisor', 'none'))

    # --- Persistence baseline: no training ---
    is_persistence = args.model_type == 'persistence'
    if is_persistence:
        writer("Persistence baseline — no training.", log_file)
        final = {}
        for split_name, split_data in [("train", data_train),
                                        ("val", data_val), ("test", data_test)]:
            f1_macro, mse_cont, per_feat, dom_f1, extra = eval_gtsq(
                model, split_data, args.batch_size, **eval_kw)
            final[split_name] = {"f1_macro": f1_macro, "mse_cont": mse_cont,
                                  "domain_f1": dom_f1, "extra": extra}
            writer(f"  {split_name}: F1_macro={f1_macro:.4f}  "
                   f"Kappa={extra['kappa']:.4f}  F1_change={extra['f1_change']:.4f}  "
                   f"Klinisch={dom_f1.get('Klinisch', 0):.4f}  "
                   f"Organ={dom_f1.get('Organ', 0):.4f}  "
                   f"Lok={dom_f1.get('Lokalisierung', 0):.4f}", log_file)
            with open(os.path.join(run_dir, f"gtsq_metrics_{split_name}.json"), "w") as f:
                json.dump(per_feat, f, indent=2, default=float)
        torch.save(model.state_dict(), model_path)
        _finalize_run(run_dir, config, final, 0, 0, log_file)
        return

    # --- Learnable weights ---
    learnable_weight_loss = None
    if args.loss_balance == 'lw':
        learnable_weight_loss = LearnableWeightLoss(G).to(device)

    # --- Optimizer ---
    all_params = list(model.parameters())
    if learnable_weight_loss is not None:
        all_params += list(learnable_weight_loss.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    # NOTE (resubmission TODO): with step_size=1 + gamma=0.1, every scheduler.step()
    # call below decays the LR by 10x, and step() fires each epoch once the plateau
    # patience is exceeded -> LR collapses ~10^-3 over 3 epochs and training early-stops
    # right after, so the reduced LR never does useful work. For the corrected re-run,
    # switch to ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3) called
    # every epoch as scheduler.step(f1_val), and raise patience_break above the scheduler
    # patience so the lowered LR can actually improve. (Changes training dynamics -> re-run
    # all seeds.) Kept as-is now to stay consistent with the submitted paper's numbers.
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    N_train = len(data_train['gtsq_in'])
    batch_size = args.batch_size
    samples_per_epoch = min(args.samples_per_epoch, N_train)
    remainder = samples_per_epoch % batch_size
    if remainder == 1:
        samples_per_epoch -= 1

    best_f1 = -1.0
    best_sel = -1e18   # checkpoint-selection metric (= f1_val, or single-feature metric)
    best_epoch = 0
    # Per-feature single-target runs must select/early-stop on the trained feature only,
    # not the macro F1 over 45 heads (44 of which stay random).
    monitor_fname = (str(gtsq_features[args.only_feature_idx])
                     if args.only_feature_idx >= 0 else None)

    writer2("Epoch loss F1_train F1_val "
            "F1_Klinisch_val F1_Organ_val F1_Lok_val "
            "grad_mean grad_max explosions "
            "gate_mean change_rate result", result_file)

    gtsq_in_train = data_train['gtsq_in']
    gtsq_out_train = data_train['gtsq_out']
    gtsq_out_mask_train = data_train['gtsq_out_mask']

    writer(f"Training: {N_train} samples, {samples_per_epoch} per epoch, "
           f"lr={args.lr}, batch={batch_size}", log_file)

    # --- Training loop ---
    for e in range(1, args.epochs + 1):
        writer(f"Epoch {e} started.", log_file)
        e_indices = np.random.choice(N_train, size=samples_per_epoch, replace=True)
        model.train()

        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_steps = 0
        grad_explosion_count = 0
        gate_sum = 0.0
        gate_count = 0
        loss_sum = 0.0
        loss_steps = 0

        for b_start in range(0, samples_per_epoch, batch_size):
            b_idx = e_indices[b_start:b_start + batch_size]

            gi = torch.tensor(gtsq_in_train[b_idx], dtype=torch.float32, device=device)
            gt = torch.tensor(gtsq_out_train[b_idx], dtype=torch.float32, device=device)
            gm = torch.tensor(gtsq_out_mask_train[b_idx], dtype=torch.float32, device=device)

            ci = None  # clinical_mark unused (see eval_gtsq note)
            if args.use_clinical and has_clinical:
                ci = torch.tensor(data_train['clinical'][b_idx],
                                  dtype=torch.float32, device=device)

            gtsq_input = gi if args.use_gtsq_input else None
            dos = None
            sup_d_b = sup_n_b = None
            if getattr(args, 'use_supervisor', 'none') in ('d', 'both') and 'supervisor_d' in data_train:
                sup_d_b = torch.tensor(data_train['supervisor_d'][b_idx], device=device)
            if getattr(args, 'use_supervisor', 'none') in ('next', 'both') and 'supervisor_next' in data_train:
                sup_n_b = torch.tensor(data_train['supervisor_next'][b_idx], device=device)
            if getattr(args, 'use_day_of_stay', False) and 'day_of_stay' in data_train:
                dos = torch.tensor(data_train['day_of_stay'][b_idx] / 7.0,
                                   dtype=torch.float32, device=device)
            output, gate = model(clinical_input=ci, clinical_mark=None,
                                 gtsq_input=gtsq_input, day_of_stay=dos,
                                 supervisor_d=sup_d_b, supervisor_next=sup_n_b)

            # Main loss
            main_loss = compute_loss(
                output, gt, gm, feature_slices, gtsq_ftypes, gtsq_nclasses,
                gtsq_means, gtsq_stds, binary_pw, ordinal_cw,
                args.loss_balance, device, learnable_weight_loss,
                only_feature_idx=args.only_feature_idx,
                ordinal_loss=getattr(args, 'ordinal_loss', None),
                change_loss=getattr(args, 'change_loss', None),
                change_weight=getattr(args, 'change_weight', 5.0),
                focal_gamma=getattr(args, 'focal_gamma', 2.0),
                gtsq_input=gi)

            # Gate auxiliary loss
            detector = getattr(args, 'change_detector', False)
            total_loss = 0.0 * main_loss if detector else main_loss   # detector: detection is the SOLE loss
            if gate is not None and (args.gate_lambda > 0 or detector):
                g_loss = compute_gate_loss(
                    gate, gi, gt, gm, gtsq_means, gtsq_stds, device,
                    focal_gamma=getattr(args, 'detector_focal', 0.0) if detector else 0.0,
                    pos_weight=getattr(args, 'detector_pos_weight', 1.0) if detector else 1.0)
                total_loss = total_loss + (1.0 if detector else args.gate_lambda) * g_loss

                gate_sum += gate.detach().mean().item()
                gate_count += 1

            loss_sum += total_loss.detach().item()
            loss_steps += 1

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float('inf'))
            grad_norm_sum += grad_norm
            grad_steps += 1
            if grad_norm > grad_norm_max:
                grad_norm_max = grad_norm
            if args.grad_warn_threshold and grad_norm > args.grad_warn_threshold:
                grad_explosion_count += 1
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

        # --- Epoch evaluation ---
        grad_norm_mean = grad_norm_sum / max(grad_steps, 1)
        gate_mean = gate_sum / max(gate_count, 1)
        loss_mean = loss_sum / max(loss_steps, 1)

        f1_train, _, _, _, _ = eval_gtsq(model, data_train, batch_size, **eval_kw)
        f1_val, mse_val, per_feat_val, dom_f1_val, extra_val = eval_gtsq(
            model, data_val, batch_size, **eval_kw)

        # Compute change rate in validation data (use most recent day)
        val_gtsq_in = data_val['gtsq_in']
        if val_gtsq_in.ndim == 3:
            val_gtsq_in = val_gtsq_in[:, -1, :]  # last day
        val_inp_raw = val_gtsq_in * gtsq_stds + gtsq_means
        val_tgt_raw = data_val['gtsq_out'] * gtsq_stds + gtsq_means
        val_mask = data_val['gtsq_out_mask']
        changed_val = (np.round(val_inp_raw) != np.round(val_tgt_raw)).astype(float)
        change_rate = float((changed_val * val_mask).sum() / val_mask.sum())

        res = 0
        # Checkpoint selection. Default is val macro-F1; --select_metric acc selects on cell-pooled
        # val accuracy instead. The two disagree systematically on this task (macro-F1 rewards rare
        # class recall, accuracy rewards copying), so the flag decides which side of that trade-off
        # the saved epoch sits on. Precedence: detector mode > single-feature monitor > this flag.
        sel_metric = extra_val.get('accuracy', 0.0) \
            if getattr(args, 'select_metric', 'f1') == 'acc' else f1_val
        if getattr(args, 'change_detector', False):        # select on detection AUROC, not F1
            sel_metric = gate_auroc(model, data_val, batch_size,
                                    gtsq_means, gtsq_stds, device, args.use_clinical)
        elif monitor_fname is not None:
            pf = per_feat_val.get(monitor_fname, {})
            sel_metric = pf['f1_macro'] if 'f1_macro' in pf else -pf.get('mse', 1e9)
        if sel_metric > best_sel:
            best_sel = sel_metric
            best_f1 = f1_val
            best_epoch = e
            res = 1
            torch.save(model.state_dict(), model_path)
            with open(os.path.join(run_dir, "gtsq_metrics_val.json"), "w") as f:
                json.dump(per_feat_val, f, indent=2, default=float)

        writer(f"  Epoch {e}: loss={loss_mean:.4f} F1_train={f1_train:.4f} F1_val={f1_val:.4f} "
               f"grad_mean={grad_norm_mean:.1f} "
               f"gate_mean={gate_mean:.3f} change_rate={change_rate:.3f} "
               f"explosions={grad_explosion_count}", log_file)

        writer2(f"{e} {loss_mean:.6f} {f1_train:.6f} {f1_val:.6f} "
                f"{dom_f1_val.get('Klinisch', 0):.6f} "
                f"{dom_f1_val.get('Organ', 0):.6f} "
                f"{dom_f1_val.get('Lokalisierung', 0):.6f} "
                f"{grad_norm_mean:.2f} {float(grad_norm_max):.2f} "
                f"{grad_explosion_count} "
                f"{gate_mean:.4f} {change_rate:.4f} {res}", result_file)

        # Scheduler + early stopping
        if e - best_epoch >= args.patience_scheduler:
            scheduler.step()
            writer(f"  LR step → {optimizer.param_groups[0]['lr']:.2e}", log_file)
        if e - best_epoch >= args.patience_break:
            writer(f"  Early stop at epoch {e} (best: {best_epoch})", log_file)
            break

    # --- Final evaluation on all splits ---
    model.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    final = {}
    for split_name, split_data in [("train", data_train),
                                    ("val", data_val), ("test", data_test)]:
        f1_macro, mse_cont, per_feat, dom_f1, extra = eval_gtsq(
            model, split_data, batch_size, **eval_kw)
        final[split_name] = {"f1_macro": f1_macro, "mse_cont": mse_cont,
                              "domain_f1": dom_f1, "extra": extra}
        writer(f"\n  {split_name}: F1_macro={f1_macro:.4f}  "
               f"Kappa={extra['kappa']:.4f}  F1_change={extra['f1_change']:.4f}  "
               f"Klinisch={dom_f1.get('Klinisch', 0):.4f}  "
               f"Organ={dom_f1.get('Organ', 0):.4f}  "
               f"Lok={dom_f1.get('Lokalisierung', 0):.4f}", log_file)
        with open(os.path.join(run_dir, f"gtsq_metrics_{split_name}.json"), "w") as f:
            json.dump(per_feat, f, indent=2, default=float)

    # Log learned weights
    if learnable_weight_loss is not None:
        weights = learnable_weight_loss.get_weights()
        writer("\nLearned feature weights (precision = 1/σ²):", log_file)
        for g, fname in enumerate(gtsq_features):
            writer(f"  {fname}: {weights[g]:.4f}", log_file)
        np.save(os.path.join(run_dir, "learned_weights.npy"), weights)

    _finalize_run(run_dir, config, final, best_epoch, e, log_file)


def _finalize_run(run_dir, config, final, best_epoch, last_epoch, log_file):
    """Append to central results.csv and save final config."""
    config['best_epoch'] = best_epoch
    config['last_epoch'] = last_epoch
    for split in ('train', 'val', 'test'):
        if split in final:
            config[f'F1_{split}'] = final[split]['f1_macro']
            config[f'MSE_cont_{split}'] = final[split]['mse_cont']
            for dom in ('Klinisch', 'Organ', 'Lokalisierung'):
                config[f'F1_{dom}_{split}'] = final[split]['domain_f1'].get(dom, 0)
            extra = final[split].get('extra', {})
            config[f'Kappa_{split}'] = extra.get('kappa', 0)
            config[f'F1_change_{split}'] = extra.get('f1_change', 0)
            # accuracy counterparts of F1 / F1_change, so the four headline metrics
            # (F1, accuracy — each overall and on changed cells) are read straight
            # from config.json by 07_aggregate (no re-inference).
            config[f'Acc_{split}'] = extra.get('accuracy', 0)
            config[f'Acc_change_{split}'] = extra.get('acc_change', 0)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Append to central CSV
    csv_path = os.path.join(os.path.dirname(run_dir), "results.csv")
    header_needed = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    row = {
        'run': os.path.basename(run_dir),
        'best_epoch': best_epoch,
        'epochs': last_epoch,
        'GTSQ_F1_train': f"{final.get('train', {}).get('f1_macro', 0):.6f}",
        'GTSQ_F1_val':   f"{final.get('val', {}).get('f1_macro', 0):.6f}",
        'GTSQ_F1_test':  f"{final.get('test', {}).get('f1_macro', 0):.6f}",
        'F1_Klinisch_val':  f"{final.get('val', {}).get('domain_f1', {}).get('Klinisch', 0):.6f}",
        'F1_Organ_val':     f"{final.get('val', {}).get('domain_f1', {}).get('Organ', 0):.6f}",
        'F1_Lok_val':       f"{final.get('val', {}).get('domain_f1', {}).get('Lokalisierung', 0):.6f}",
        'F1_Klinisch_test': f"{final.get('test', {}).get('domain_f1', {}).get('Klinisch', 0):.6f}",
        'F1_Organ_test':    f"{final.get('test', {}).get('domain_f1', {}).get('Organ', 0):.6f}",
        'F1_Lok_test':      f"{final.get('test', {}).get('domain_f1', {}).get('Lokalisierung', 0):.6f}",
        'GTSQ_MSE_cont_val':  f"{final.get('val', {}).get('mse_cont', 0):.6f}",
        'GTSQ_MSE_cont_test': f"{final.get('test', {}).get('mse_cont', 0):.6f}",
        'Kappa_val':    f"{final.get('val', {}).get('extra', {}).get('kappa', 0):.6f}",
        'Kappa_test':   f"{final.get('test', {}).get('extra', {}).get('kappa', 0):.6f}",
        'F1_change_val':  f"{final.get('val', {}).get('extra', {}).get('f1_change', 0):.6f}",
        'F1_change_test': f"{final.get('test', {}).get('extra', {}).get('f1_change', 0):.6f}",
    }
    with open(csv_path, "a") as f:
        if header_needed:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")

    writer(f"Results appended to {csv_path}", log_file)
    writer(f"Done. Best epoch: {best_epoch}", log_file)


if __name__ == "__main__":
    main(parse_args())
