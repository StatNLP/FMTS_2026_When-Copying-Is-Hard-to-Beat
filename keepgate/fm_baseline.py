"""Zero-shot time-series foundation models on the GTSQ next-day task (closes the 'you did not test
large pretrained models' gap).

Design. Each of the 45 items is a univariate DAILY series for one encounter. The model sees the
observed history up to day D (K=7 split; mean 5.3 real days) and forecasts day D+1; the continuous
forecast is decoded to a class (binary: threshold 0.5; ordinal: round+clip to the item's range).
Scored on exactly the canonical test pairs with the same four metrics -> rows are directly
comparable to Table 1 (the harness reproduces persistence exactly, which is the check that the
decode path is right). Default split is the CORRECTED one; set FM_SPLIT to score another.

MIRA is the one comparator that is both clinical (pretrained on ICU/EHR signals) and
triplet-native, so the questionnaire's real observation days are supplied rather than compacted; it
still sees the questionnaire only, since the physiology stream is not part of this split.

Honest caveat this is built to expose: these models forecast CONTINUOUS series and were pretrained
on long contexts; here the series are categorical and a handful of days long. If they collapse onto
persistence, that is the finding, not a bug -- for a near-constant series the last value IS the
optimal forecast, which is the paper's whole point.
"""
import os
import argparse, os, sys, numpy as np, torch
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gtsq_eval as ge

SPLIT = os.environ.get("FM_SPLIT", "exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_noclin_h7")
# Feature metadata comes from the SPLIT we actually score (not bayes_floor's noclin split), so a
# column-order/cardinality change in this split can never be silently mis-decoded with the wrong NC.
_NORM = np.load(os.path.join(SPLIT, "norm_stats.npz"), allow_pickle=True)
FE=[str(x) for x in _NORM["gtsq_features"]]; FT=[str(x) for x in _NORM["gtsq_ftypes"]]
NC=[int(x) for x in _NORM["gtsq_nclasses"]]; G=len(FE)

def load(split="test"):
    d=np.load(os.path.join(SPLIT,f"{split}.npz"),allow_pickle=True)
    n=np.load(os.path.join(SPLIT,"norm_stats.npz"),allow_pickle=True)
    mu,sd=n['gtsq_means'],n['gtsq_stds']
    H=d['gtsq_in']*sd+mu; Hm=d['gtsq_in_mask']          # (N,7,G) back to class scale
    Y=d['gtsq_out']*sd+mu; Ym=d['gtsq_out_mask']
    # per-CELL both-days mask, matching the rest of the paper (a pair is kept if any cell qualifies)
    cell=(Ym==1)&(Hm[:,-1,:]==1)
    keep=cell.any(1)
    return np.rint(H[keep]), Hm[keep], np.rint(Y[keep]).astype(int), cell[keep]

def decode(p,g):
    # continuous forecast -> class: binary thresholds at 0.5; ordinal rounds+clips (shared helper)
    return (p>0.5).astype(int) if FT[g]=="binary" else ge.round_to_class(p, FT[g], NC[g])

def score(P,I,T,Mout):
    # all FOUR metrics: change-F1 was computed and thrown away, so the FM rows of Table 1 carried
    # '--' in that column while every other row had a number.
    return ge.four_metrics(P,I,T,Mout,NC)

def _phys(split="test"):
    """Per-channel 24-hour mean of the physiology panel, or None when the split carries none.
    Uses the SAME keep-mask as load() so rows line up with H/I/T."""
    d = np.load(os.path.join(SPLIT, f"{split}.npz"), allow_pickle=True)
    if "clinical" not in d.files or d["clinical"].shape[-1] == 0:
        return None
    Ym = d["gtsq_out_mask"]; Hm = d["gtsq_in_mask"]
    H1 = Hm[:, -1, :] if Hm.ndim == 3 else Hm
    keep = ((Ym == 1) & (H1 == 1)).any(1)
    C = d["clinical"].astype(np.float32)[keep]        # (N, 24, P)
    with np.errstate(invalid="ignore"):
        M = np.nanmean(np.where(np.isfinite(C), C, np.nan), axis=1)
    return np.nan_to_num(M, nan=0.0)


def _flat_series(split):
    """Flatten a split into univariate (context, target) pairs: one per (day-pair, item).

    The task is next-day-from-history for each item independently, which is exactly the shape
    Chronos-Bolt is trained on. Rows with no observed history or an unobserved target are dropped --
    they carry no supervision and would otherwise teach the model to predict the padding value.
    """
    H, Hm, T, CM = load(split)
    K = H.shape[1]
    # Chronos-Bolt's instance_norm runs BEFORE the mask is applied, so zero-filled unobserved days
    # are folded into loc/scale -- and the zero-shot path feeds a COMPACTED observed-only series with
    # no mask at all. Training on padded input and scoring on compacted input normalises the two
    # differently. Compact here too, left-padding with NaN (see the next comment) so the two paths normalise
    # identically.
    ctx = H.transpose(0, 2, 1).reshape(-1, K).astype(np.float32)
    cmask = (Hm.transpose(0, 2, 1).reshape(-1, K) == 1)
    # Pad with NaN, not with a repeated value. Chronos-Bolt's InstanceNorm uses nanmean/nanstd
    # (chronos_bolt.py:87-88), so NaN slots are EXCLUDED from loc/scale -- which makes the padded
    # batch normalise exactly like the variable-length compacted series the zero-shot path feeds.
    # Padding with obs[0] instead put the repeated value into loc/scale: 52% of series are shorter
    # than K, and 37% of the non-constant ones (the change events) were normalised differently at
    # train and serve time.
    comp = np.full_like(ctx, np.nan)
    for i in range(len(ctx)):
        obs = ctx[i][cmask[i]]
        if len(obs):
            comp[i, K - len(obs):] = obs
    ctx = comp
    cmask = np.isfinite(ctx)
    tgt = T.reshape(-1, 1).astype(np.float32)
    tmask = CM.reshape(-1, 1).astype(bool)
    keep = tmask[:, 0] & cmask.any(1)
    return ctx[keep], cmask[keep], tgt[keep], tmask[keep]


def _finetune_bolt(a, model_id):
    """Fine-tune Chronos-Bolt on the GTSQ TRAIN split; early-stop on VAL. Returns the pipeline.

    Chronos-Bolt's forward computes its own quantile loss when a target is supplied, so this is an
    ordinary training loop -- no re-implementation of the objective, which is what makes fine-tuning
    tractable here at all. Test is never touched: the epoch is chosen on val loss.
    """
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=a.device, dtype=torch.float32)
    inner = pipe.model
    dev = a.device
    # Chronos-Bolt standardises each series by its own std and floors a zero std at eps=1e-5. On this
    # data 88% of item-histories are constant and 67% are all-zero, so a target one class away from a
    # constant context normalises to ~1e5 and its loss term swamps the batch -- and those are exactly
    # the CHANGE events. Left at the default the loss sits at ~2016 and does not move in 10 epochs,
    # and the model collapses onto persistence. The floor belongs on the scale of the data: these are
    # integer classes, so one class is the smallest meaningful spread.
    if hasattr(inner, "instance_norm"):
        inner.instance_norm.eps = a.ft_scale_floor
        print(f"  instance-norm scale floor set to {a.ft_scale_floor} (default 1e-5) -- see comment",
              flush=True)
    Xtr, Mtr, Ytr, TMtr = _flat_series("train")
    Xva, Mva, Yva, TMva = _flat_series("val")
    print(f"  finetune: {len(Xtr):,} train series, {len(Xva):,} val, {a.finetune} epochs, "
          f"lr {a.ft_lr}, batch {a.ft_batch}", flush=True)
    opt = torch.optim.AdamW(inner.parameters(), lr=a.ft_lr, weight_decay=0.01)
    def _epoch(X, M_, Y, TM, train):
        inner.train(train)
        idx = np.random.permutation(len(X)) if train else np.arange(len(X))
        tot = n = 0.0
        for s0 in range(0, len(idx), a.ft_batch):
            b_ = idx[s0:s0 + a.ft_batch]
            ctx = torch.tensor(X[b_], device=dev)
            msk = torch.tensor(M_[b_], device=dev)
            tg = torch.tensor(Y[b_], device=dev)
            tgm = torch.tensor(TM[b_], device=dev)
            with torch.set_grad_enabled(train):
                out = inner(context=ctx, mask=msk, target=tg, target_mask=tgm)
                loss = out.loss
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
                opt.step()
            tot += float(loss) * len(b_); n += len(b_)
        return tot / max(n, 1)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(1, a.finetune + 1):
        tr = _epoch(Xtr, Mtr, Ytr, TMtr, True)
        va = _epoch(Xva, Mva, Yva, TMva, False)
        flag = ""
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}
            flag = "  <- best"
        else:
            bad += 1
        print(f"    epoch {ep:3d}  train {tr:.5f}  val {va:.5f}{flag}", flush=True)
        if bad >= 3:
            print("    early stop (val loss not improving for 3 epochs)"); break
    if best_state is not None:
        inner.load_state_dict(best_state)
    inner.eval()
    return pipe


def _finetune_mira(a, model, mu, sd):
    """Fine-tune MIRA on the GTSQ train split; early-stop on val. Mutates and returns `model`.

    MIRA scores prediction[t] against labels[t] directly (horizon_length=1, so calc_ar_loss does not
    shift), and the zero-shot path reads logits[:, -1, 0] as the next-day forecast. So position t must
    predict the value at t+1: labels[t] = v[t+1] for the interior, and labels[K-1] = the day-D+1
    target. Getting this off by one would train the model to copy its own input -- which would look
    like a plausible loss curve and produce a persistence-shaped forecaster.
    """
    dev = a.device
    def build(split):
        H, Hm, T, CM = load(split)
        K = H.shape[1]
        v = ((H - mu) / np.where(sd == 0, 1.0, sd)).transpose(0, 2, 1).reshape(-1, K).astype(np.float32)
        obs = (Hm.transpose(0, 2, 1).reshape(-1, K) == 1)
        y = ((T - mu) / np.where(sd == 0, 1.0, sd)).reshape(-1).astype(np.float32)
        tm = CM.reshape(-1).astype(bool)
        lab = np.zeros_like(v); lab[:, :-1] = v[:, 1:]; lab[:, -1] = y
        # MIRA emits exactly ONE output position (modeling_mira: hidden_states_all[:, -1, :] ->
        # [B,1,D]) while labels are [B,K], so the loss BROADCASTS that single prediction against
        # every label. Scoring the interior positions therefore trains the day-D+1 head to
        # reproduce its own input history -- about 6/7 of the loss -- and produces a
        # persistence-shaped forecaster with a perfectly plausible loss curve. Only the final
        # position is a real next-day target, so only it may carry weight.
        lm = np.zeros_like(v)
        lm[:, -1] = (obs[:, -1] & tm).astype(np.float32)
        keep = (lm.sum(1) > 0)
        return v[keep], obs[keep], lab[keep], lm[keep]
    Xtr, Otr, Ltr, LMtr = build("train"); Xva, Ova, Lva, LMva = build("val")
    K = Xtr.shape[1]
    print(f"  finetune MIRA: {len(Xtr):,} train series, {len(Xva):,} val, {a.finetune} epochs, "
          f"lr {a.ft_lr}, batch {a.ft_batch}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.ft_lr, weight_decay=0.01)
    def _epoch(X, O, L, LM, train):
        model.train(train)
        idx = np.random.permutation(len(X)) if train else np.arange(len(X))
        tot = n = 0.0
        for s0 in range(0, len(idx), a.ft_batch):
            b_ = idx[s0:s0 + a.ft_batch]; B = len(b_)
            v = torch.tensor(X[b_], device=dev)
            m = torch.tensor(O[b_].astype(np.float32), device=dev)
            m = torch.where((m.sum(1) == 0).unsqueeze(1), torch.ones_like(m), m)
            t = torch.arange(K, dtype=torch.float32, device=dev).expand(B, -1).clone()
            lab = torch.tensor(L[b_], device=dev)
            lm = torch.tensor(LM[b_], device=dev)
            with torch.set_grad_enabled(train):
                out = model(input_ids=v.unsqueeze(-1), time_values=t, attention_mask=m,
                            next_target_time_values=None, labels=lab, loss_masks=lm,
                            return_dict=True)
                loss = out.loss
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * B; n += B
        return tot / max(n, 1)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(1, a.finetune + 1):
        tr = _epoch(Xtr, Otr, Ltr, LMtr, True); va = _epoch(Xva, Ova, Lva, LMva, False)
        flag = ""
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: t_.detach().cpu().clone() for k, t_ in model.state_dict().items()}
            flag = "  <- best"
        else:
            bad += 1
        print(f"    epoch {ep:3d}  train {tr:.5f}  val {va:.5f}{flag}", flush=True)
        if bad >= 3:
            print("    early stop"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def main(a):
    if getattr(a, "use_phys", False):
        raise SystemExit("--use_phys is declared but not implemented; refusing to run so the job "
                         "cannot be mistaken for one that used physiology.")
    if getattr(a, "finetune", 0) and not (a.model.startswith("chronos-bolt") or a.model == "mira"):
        raise SystemExit(f"--finetune is implemented for chronos-bolt-* and mira, not {a.model}; "
                         "refusing to run so the job cannot be mistaken for a fine-tune.")
    H,Hm,T,CM=load(a.split_name); N=len(H); I=H[:,-1,:].astype(int)     # day D = last history slot
    PHYS = _phys(a.split_name) if a.model == "chronos2-phys" else None
    if a.model == "chronos2-phys" and PHYS is None:
        raise SystemExit("chronos2-phys needs a split WITH physiology: set FM_SPLIT to a clin_cult_h7 split")
    print(f"test pairs {N} | context {H.shape[1]}d (mean observed {Hm.any(2).sum(1).mean():.2f}) | feats {G}",flush=True)
    if a.model=="persistence":
        P=I.copy()
    elif a.model=="mira":
        # MIRA (microsoft/MIRA) is triplet-native: it takes per-observation TIMESTAMPS, so an
        # irregular questionnaire history enters with its true day indices and the gaps are real gaps
        # rather than being silently compacted, which is what Chronos has to do. It is also the only
        # comparator pretrained on ICU/EHR signals. Run under py311_mira (see run_mira.sh).
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "10_scripts/external/MIRA"))
        from MIRA.models.modeling_mira import MIRAForPrediction
        dev = a.device
        model = MIRAForPrediction.from_pretrained(a.weights).to(dev).eval()
        # MIRA expects z-scored input (its internal normalisation is the identity then), so we feed
        # the split's z-scored history and denormalise the forecast back to class scale afterwards.
        n = np.load(os.path.join(SPLIT, "norm_stats.npz"), allow_pickle=True)
        mu, sd = n["gtsq_means"], n["gtsq_stds"]
        Hz = (H - mu) / np.where(sd == 0, 1.0, sd)                 # (N,K,G) back to z-scale
        Kd = H.shape[1]
        if a.finetune:
            model = _finetune_mira(a, model, mu, sd)
        P = np.zeros_like(I)
        with torch.no_grad():
            for g in range(G):
                out = np.zeros(N, dtype=np.float32)
                for s0 in range(0, N, a.batch):
                    sl = slice(s0, min(s0 + a.batch, N)); B = sl.stop - sl.start
                    v = torch.tensor(Hz[sl, :, g], dtype=torch.float32, device=dev)
                    m = torch.tensor((Hm[sl, :, g] == 1).astype(np.float32), device=dev)
                    t = torch.arange(Kd, dtype=torch.float32, device=dev).expand(B, -1).clone()
                    empty = m.sum(1) == 0                           # never observed -> keep day D
                    m = torch.where(empty.unsqueeze(1), torch.ones_like(m), m)
                    r = model(input_ids=v.unsqueeze(-1), time_values=t, attention_mask=m,
                              next_target_time_values=None, return_dict=True)
                    out[sl] = r.logits[:, -1, 0].float().cpu().numpy()
                P[:, g] = decode(out * (sd[g] if sd[g] else 1.0) + mu[g], g)
                if (g + 1) % 10 == 0:
                    print(f"   feat {g+1}/{G}", flush=True)
    elif a.model in ("chronos2-mv", "chronos2-phys"):
        # chronos2-phys additionally hands the model the PHYSIOLOGY as extra variates. Note what this
        # can and cannot show: the split carries physiology as a single 24-hour window (24 x 186), not
        # as a 7-day trajectory, so on the daily axis it becomes ONE summary column at day D and the
        # model sees a snapshot, never a trend. The shuffle probe showed the trained gate's gain comes
        # from patient-specific physiology over time, so a snapshot is the weaker test -- a negative
        # result here does not rule out that a trajectory would help.
        # Chronos-2 is genuinely MULTIVARIATE: given a (n_variates, history) array it shares
        # information across variates. Here the 45 questionnaire items of one encounter are the
        # variates, so the model may use e.g. a rising organ score to predict the working diagnosis --
        # something the univariate Chronos-Bolt and MIRA runs cannot do by construction. Unobserved
        # days are passed as NaN so the model sees the real gaps.
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=a.device,
                                                   dtype=torch.float32)
        qs = list(pipe.quantiles); mid = qs.index(0.5)
        P = np.zeros_like(I)
        for s0 in range(0, N, a.batch):
            sl = slice(s0, min(s0 + a.batch, N))
            batch = []
            for i in range(sl.start, sl.stop):
                arr = H[i].T.astype(np.float32).copy()          # (G, K) variates x days
                arr[Hm[i].T == 0] = np.nan                      # real gaps, not compacted
                if a.model == "chronos2-phys" and PHYS is not None:
                    ext = np.full((PHYS.shape[1], arr.shape[1]), np.nan, dtype=np.float32)
                    ext[:, -1] = PHYS[i]                        # snapshot lands on day D only
                    arr = np.vstack([arr, ext])
                batch.append(arr)
            out = pipe.predict(batch, prediction_length=1)       # list of (G, n_quantiles, 1)
            for k, o in enumerate(out):
                med = o[:G, mid, 0].float().cpu().numpy()   # extra variates are inputs, not targets
                P[sl.start + k] = [decode(np.array([med[g]]), g)[0] for g in range(G)]
            print(f"   {min(sl.stop, N)}/{N}", flush=True)
    else:
        from chronos import BaseChronosPipeline
        name={"chronos-bolt-base":"amazon/chronos-bolt-base","chronos-bolt-small":"amazon/chronos-bolt-small",
              "chronos2":"amazon/chronos-2"}[a.model]
        # --finetune trains the backbone on the GTSQ train split first; the same pipeline then
        # predicts with the fine-tuned weights, so the decode path below is unchanged.
        pipe=(_finetune_bolt(a,name) if a.finetune else
              BaseChronosPipeline.from_pretrained(name,device_map=a.device,torch_dtype=torch.float32))
        P=np.zeros_like(I)
        _want_qf = (a.decode=="cathead") or bool(a.save_qf)
        QF=np.zeros((N,G,9),dtype=np.float32) if _want_qf else None
        for g in range(G):
            ctx=[]
            for i in range(N):
                obs=Hm[i,:,g]==1
                s=H[i,obs,g] if obs.sum()>0 else H[i,-1:,g]
                ctx.append(torch.tensor(s,dtype=torch.float32))
            # --decode cathead needs the whole predictive distribution, not the median: the point
            # of the check is that collapsing a forecast distribution to one number and rounding it
            # is only ONE way to turn a numerical forecaster into a classifier, and not obviously
            # the right one for nominal targets. QL is the fan handed to the categorical head.
            QL=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] if _want_qf else [0.5]
            out=[]
            for s0 in range(0,N,a.batch):
                q,_=pipe.predict_quantiles(inputs=ctx[s0:s0+a.batch],prediction_length=1,quantile_levels=QL)
                # Chronos-Bolt returns a (batch, horizon, n_quantiles) tensor; Chronos-2 returns a
                # LIST of per-series tensors. Subscripting the list with a tuple raised TypeError and
                # took the whole job down, so normalise before indexing.
                if isinstance(q,(list,tuple)):
                    q=torch.stack([torch.as_tensor(x) for x in q])
                out.append(q[:,0,:].cpu().numpy())          # (batch, n_quantiles)
            Q=np.concatenate(out)
            if _want_qf:
                QF[:,g,:]=Q
            P[:,g]=decode(Q[:,QL.index(0.5)],g)
            if (g+1)%10==0: print(f"   feat {g+1}/{G}",flush=True)
    if a.decode=="cathead" and a.qf_train and a.qf_val and os.path.exists(a.qf_train) \
            and os.path.exists(a.qf_val):
        # CATEGORICAL ADAPTATION (OPEN_ITEMS O13). The zero-shot rows decode a numerical forecast by
        # thresholding at 0.5 (binary) or rounding to the nearest class (ordinal). That imposes a
        # metric on the label space -- fine for an ordinal severity, wrong for a nominal cause, where
        # "distance 2" between two causes means nothing. Here the model's predictive DISTRIBUTION is
        # kept and a per-item multinomial logistic regression maps its nine quantiles to class
        # probabilities. The mapping is FITTED ON TRAIN ONLY and its single hyperparameter C is
        # chosen on validation, so the comparison is decoding-vs-decoding on the same forecasts, not
        # a stronger model. If this does not beat rounding, the numerical-objective limitation is a
        # property of the forecasts, not of the arithmetic used to read them.
        from sklearn.linear_model import LogisticRegression
        qtr=np.load(a.qf_train); qva=np.load(a.qf_val)
        Ttr=qtr["T"]; Mtr=qtr["M"]; Tva=qva["T"]; Mva=qva["M"]
        if a.cathead_features in ("state","state+quantiles"):
            # CONTROL (reviewer R5b): the same fitted read-out, same protocol, but given only the
            # item's CURRENT state instead of the model's nine forecast quantiles. If this matches
            # the quantile head, the extra transitions the read-out recovers did not need the
            # pretrained forecasts at all, and the decoding result is about fitting a categorical
            # map, not about the forecaster. One-hot over the item's class range.
            def _oh(Icls,g,nc):
                z=np.zeros((len(Icls),nc),dtype=np.float32)
                z[np.arange(len(Icls)),np.clip(Icls[:,g],0,nc-1).astype(int)]=1.0
                return z
            Xtr=None; Xva=None; XTE=None                  # built per item below
            Itr_c=qtr["I"]; Iva_c=qva["I"]
        else:
            Xtr=qtr["Q"]; Xva=qva["Q"]
        Pc=I.copy(); nsel=[]
        for g in range(G):
            mtr=Mtr[:,g]; mva=Mva[:,g]
            ytr=Ttr[mtr,g].astype(int); yva=Tva[mva,g].astype(int)
            if len(np.unique(ytr))<2 or mtr.sum()<50:
                continue                                    # single-class item: rounding is fine
            if a.cathead_features=="state":
                ncg=max(2,int(NC[g]))
                xtr=_oh(Itr_c,g,ncg)[mtr]; xva=_oh(Iva_c,g,ncg)[mva]; xte=_oh(I,g,ncg)
            elif a.cathead_features=="state+quantiles":
                # INCREMENTAL test: the state one-hot concatenated with the forecast fan. A
                # state-only control shows the forecasts are not NEEDED; only this shows whether
                # they ADD anything once the current state is already available.
                ncg=max(2,int(NC[g]))
                xtr=np.hstack([_oh(Itr_c,g,ncg)[mtr], qtr["Q"][mtr,g,:]])
                xva=np.hstack([_oh(Iva_c,g,ncg)[mva], qva["Q"][mva,g,:]])
                xte=np.hstack([_oh(I,g,ncg), QF[:,g,:]])
            else:
                xtr=Xtr[mtr,g,:]; xva=Xva[mva,g,:]; xte=QF[:,g,:]
            best=(-1,None)
            for C in (0.01,0.1,1.0,10.0):
                clf=LogisticRegression(max_iter=2000,C=C)
                clf.fit(xtr,ytr)
                sc=f1_score(yva,clf.predict(xva),average="macro",zero_division=0)
                if sc>best[0]: best=(sc,clf)
            nsel.append(best[0]); Pc[:,g]=best[1].predict(xte)
        print(f"\n  categorical head [{a.cathead_features}] fitted on {len(nsel)}/{G} items "
              f"(val macro-F1 mean {np.mean(nsel):.4f})")
        acc2,f12,ca2,f1c2=score(Pc,I,T,CM)
        print(f"  DECODE numeric  : acc {score(P,I,T,CM)[0]:.4f}  macro-F1 {score(P,I,T,CM)[1]:.4f}")
        print(f"  DECODE cathead  : acc {acc2:.4f}  macro-F1 {f12:.4f}  change-acc {ca2:.4f}  change-F1 {f1c2:.4f}")
        P=Pc
    elif a.decode=="cathead":
        print("  --decode cathead: quantile caches not present yet; this pass only writes "
              f"--save_qf ({a.save_qf}). Re-run once train and val caches exist.")
    if a.save_qf and QF is not None:
        # cache the quantile fan + labels/mask so the head can be refitted without a forward pass
        os.makedirs(os.path.dirname(a.save_qf) or ".", exist_ok=True)
        np.savez_compressed(a.save_qf, Q=QF, T=T, M=CM, I=I); print(f"saved quantiles -> {a.save_qf}")
    # cache the decoded predictions so any later analysis (CIs, per-item, per-class) can be redone
    # without paying for another forward pass through the model
    if a.save_pred:
        os.makedirs(os.path.dirname(a.save_pred) or ".", exist_ok=True)
        np.save(a.save_pred, P); print(f"saved predictions -> {a.save_pred}")
    acc,f1,ca,f1c=score(P,I,T,CM)
    agree=float((P[CM]==I[CM]).mean())
    print(f"\nRESULT {a.model:<20} cell-acc {acc:.4f}  macro-F1 {f1:.4f}  change-acc {ca:.4f}"
          f"  change-F1 {f1c:.4f}"
          f"   [agrees with persistence on {agree*100:.1f}% of cells]")
    # References computed on the split actually being scored. They used to be hardcoded from the
    # encounter split, which silently invited a comparison against a different cohort once FM_SPLIT
    # pointed elsewhere.
    _R = I.copy()
    for _j, _f in enumerate(FE):
        if "akut" in str(_f).lower():
            _R[:, _j] = np.where(I[:, _j] == 1, 2, I[:, _j])
    _pa, _pf, _pc, _pfc = score(I, I, T, CM)
    _ra, _rf, _rc, _rfc = score(_R, I, T, CM)
    print(f"  refs on THIS split (mask=both, n={int(CM.sum()):,}): "
          f"persistence {_pa:.4f}/{_pf:.4f}/{_pc:.4f}/{_pfc:.4f} | "
          f"+rule {_ra:.4f}/{_rf:.4f}/{_rc:.4f}/{_rfc:.4f}")

    if a.bootstrap:
        # Paired bootstrap over day-pairs against BOTH baselines. Paired because the predictors agree
        # on ~99% of cells: an unpaired interval would be far wider than any difference between them.
        R = I.copy()
        for g in range(G):
            if "akut" in FE[g].lower():
                R[:, g] = np.where(I[:, g] == 1, 2, I[:, g])
        rng = np.random.RandomState(0); n = len(I)
        rows = {"persistence": I, "persistence+rule": R, a.model: P}
        boot = {k: [] for k in rows}
        for _ in range(a.bootstrap):
            idx = rng.randint(0, n, n)
            M = np.ones((n, G), bool)
            for k, Pr in rows.items():
                acc_, f1_, ca_, _ = ge.four_metrics(Pr[idx], I[idx], T[idx], M, NC)
                boot[k].append((f1_, acc_, ca_))
        boot = {k: np.array(v) for k, v in boot.items()}
        names = ["macro-F1", "cell-acc", "chg-acc"]
        print(f"\n  BOOTSTRAP ({a.bootstrap} resamples over day-pairs), 95% intervals")
        for k in rows:
            b = boot[k]
            print(f"    {k:22s}" + "  ".join(
                f"{names[j]} {b[:,j].mean():.4f} [{np.percentile(b[:,j],2.5):.4f},"
                f"{np.percentile(b[:,j],97.5):.4f}]" for j in range(3)))
        for base in ("persistence", "persistence+rule"):
            d = boot[a.model] - boot[base]
            print(f"    vs {base:19s}" + "  ".join(
                f"{names[j]} {d[:,j].mean():+.4f} [{np.percentile(d[:,j],2.5):+.4f},"
                f"{np.percentile(d[:,j],97.5):+.4f}]" for j in range(3)))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="persistence")
    p.add_argument("--device",default="cpu"); p.add_argument("--batch",type=int,default=2048)
    p.add_argument("--weights",default="MIRA-Mode/MIRA",help="MIRA checkpoint (--model mira)")
    p.add_argument("--save_pred",default=None,help="cache decoded predictions to this .npy")
    p.add_argument("--decode",default="numeric",choices=["numeric","cathead"],
                   help="numeric = threshold/round the median forecast (the reported rows); "
                        "cathead = per-item multinomial logistic map from the nine forecast "
                        "quantiles to classes, fitted on train and selected on val (O13)")
    p.add_argument("--cathead_features",default="quantiles",choices=["quantiles","state","state+quantiles"],
                   help="what the fitted categorical read-out sees: the model's nine forecast "
                        "quantiles, or ONLY the item's current state (control: does the read-out "
                        "need the pretrained forecasts at all?)")
    p.add_argument("--save_qf",default=None,help="cache the quantile fan (.npz) for --decode cathead")
    p.add_argument("--qf_train",default=None,help="train quantile cache written by --save_qf")
    p.add_argument("--qf_val",default=None,help="val quantile cache written by --save_qf")
    p.add_argument("--split_name",default="test",choices=["train","val","test"],
                   help="which split to forecast. Needed to cache TRAIN predictions for the "
                        "supervised calibration step (report.py fmcal), which must never see test.")
    p.add_argument("--bootstrap",type=int,default=0,help="paired bootstrap resamples for 95%% CIs")
    # Declared so the runners can pass them, but NOT yet implemented. They exit with an error rather
    # than being ignored: a job that silently fell back to zero-shot would be harvested and reported
    # as a fine-tune. (run_fm.sh advertised a MODEL flag it never read, and that is exactly how
    # chronos2-mv results turned out to be chronos-bolt re-runs.)
    p.add_argument("--finetune",type=int,default=0,
                   help="fine-tune the backbone on the GTSQ train split for up to N epochs, "
                        "early-stopping on val loss (chronos-bolt-* and mira)")
    p.add_argument("--ft_lr",type=float,default=1e-4,help="fine-tune learning rate")
    p.add_argument("--ft_batch",type=int,default=256,help="fine-tune batch size (series per step)")
    p.add_argument("--ft_scale_floor",type=float,default=1.0,
                   help="minimum per-series scale for Chronos-Bolt's instance norm. The library "
                        "default (1e-5) explodes on constant integer-class histories; 1.0 = one class")
    p.add_argument("--use_phys",action="store_true",
                   help="feed physiology alongside the questionnaire; for MIRA the native irregular "
                        "triplets rather than a daily summary [TODO]")
    main(p.parse_args())
