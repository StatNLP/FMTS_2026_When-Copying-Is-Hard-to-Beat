"""
harmonize/icu_dataset.py
PyTorch Dataset wrapping the Stage-3 forecast HDF5 tensors
(schema 1.0-forecast — see docs/2605_comprehensive_documentation.md).

Produces per-sample dicts:

    x_in        (T_in,  F_in)      float32
    mask_in     (T_in,  F_in)      bool
    x_target    (T_out, F_target)  float32        — may contain NaN
    mask_target (T_out, F_target)  bool
    static      (F_static,)        float32
    static_mask (F_static,)        bool
    stay_id     scalar int64
    ds_id       scalar int64
    t0_hour     scalar float32
    los_hours   scalar float32
    mortality_hours scalar float32 (NaN for survivors)

Notes
-----
- The HDF5 handle is opened lazily once per worker (fork-safe via
  __getstate__ / __setstate__).
- NaNs in `x_target` are preserved; the training loop uses `mask_target`
  to compute the masked-MSE loss.
- `x_in` is normally NaN-free (Stage-3 applied imputation), but if the
  Stage-3 imputation strategy was `nan` or `learned`, NaNs are kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_str_array(arr) -> list[str]:
    """Decode an HDF5 attr that is an array of bytes/object strings."""
    out: list[str] = []
    for s in arr:
        if isinstance(s, bytes):
            out.append(s.decode())
        else:
            out.append(str(s))
    return out


class ICUDataset(Dataset):
    """PyTorch Dataset over a single Stage-3 forecast HDF5 file.

    Args:
        h5_path: Path to data/dataset/{ds}_seed{seed}_{split}__{tag}.h5.
        datasets: Optional list of dataset names to include (filter by
            the per-window ds_id field). For per-dataset files this is
            usually a no-op; useful for combined files.
        nan_to_zero_in: If True (default), NaNs remaining in x_in are
            replaced with 0.0 before returning (compatible with most
            models). Set False for mask-aware / imputation-as-task models.
        preload: If True, load all arrays into RAM at init time.
            Eliminates HDF5 random-access overhead (gzip-compressed HDF5
            reads are ~28 000× slower than RAM on GPFS). Highly recommended
            for training: MIMIC-IV train is ~620 MB, well within the 64 GB
            job allocation. Default True.
    """

    def __init__(
        self,
        h5_path: str | Path,
        datasets: Sequence[str] | None = None,
        nan_to_zero_in: bool = True,
        preload: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        self.nan_to_zero_in = nan_to_zero_in
        self._preloaded = False

        with h5py.File(self.h5_path, "r") as f:
            a = f.attrs
            self.features_in     = _decode_str_array(a["features_in"])
            self.features_target = _decode_str_array(a["features_target"])
            self.features_static = _decode_str_array(a["features_static"])
            self.window_in       = int(a["window_in"])
            self.window_out      = int(a["window_out"])
            self.stride          = str(a["stride"])
            self.imputation      = str(a["imputation"])
            self.dataset_name    = str(a["dataset"])
            self.split           = str(a["split"])
            self.seed            = int(a["seed"])
            self.version         = str(a["version"])

            n_total = f["x_in"].shape[0]
            ds_ids  = f["ds_id"][:]

            if preload:
                import time as _time
                _t0 = _time.perf_counter()
                self._x_in        = f["x_in"][:]
                self._mask_in     = f["mask_in"][:]
                self._x_target    = f["x_target"][:]
                self._mask_target = f["mask_target"][:]
                self._static      = f["static"][:]
                self._static_mask = f["static_mask"][:]
                self._stay_id     = f["stay_id"][:]
                self._ds_id       = ds_ids
                self._t0_hour     = f["t0_hour"][:]
                self._los_hours   = f["los_hours"][:]
                self._mort_hours  = (f["mortality_hours"][:]
                                     if "mortality_hours" in f
                                     else np.full(n_total, np.nan, dtype=np.float32))
                self._sofa        = (f["sofa_scores"][:]
                                     if "sofa_scores" in f else None)
                self._dsofa2      = (f["dsofa2"][:]
                                     if "dsofa2" in f else None)
                self._dsofa3      = (f["dsofa3"][:]
                                     if "dsofa3" in f else None)
                _mb = sum(a.nbytes for a in [
                    self._x_in, self._mask_in, self._x_target, self._mask_target,
                    self._static, self._static_mask,
                ]) / 1e6
                import logging as _log
                _log.getLogger(__name__).info(
                    "ICUDataset preloaded %d windows (%.0f MB) in %.1fs — %s/%s",
                    n_total, _mb, _time.perf_counter() - _t0,
                    self.dataset_name, self.split,
                )
                self._preloaded = True

        idx = np.arange(n_total, dtype=np.int64)
        if datasets is not None:
            allowed = set()
            for ds in datasets:
                if ds == self.dataset_name:
                    allowed.update(np.unique(ds_ids).tolist())
            if allowed:
                idx = idx[np.isin(ds_ids, list(allowed))]
        self._indices = idx

    # ----- fork safety (only needed for streaming mode) -----
    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_h5"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._h5 = None

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    # ----- Dataset interface -----
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        i = int(self._indices[k])

        if self._preloaded:
            x_in        = self._x_in[i]
            mask_in     = self._mask_in[i]
            x_target    = self._x_target[i]
            mask_target = self._mask_target[i]
            static      = self._static[i]
            static_mask = self._static_mask[i]
            stay_id     = int(self._stay_id[i])
            ds_id       = int(self._ds_id[i])
            t0_hour     = float(self._t0_hour[i])
            los_hours   = float(self._los_hours[i])
            mort_hours  = float(self._mort_hours[i])
            sofa        = (self._sofa[i] if getattr(self, "_sofa", None) is not None else None)
            dsofa2      = (self._dsofa2[i] if getattr(self, "_dsofa2", None) is not None else None)
            dsofa3      = (self._dsofa3[i] if getattr(self, "_dsofa3", None) is not None else None)
        else:
            f = self._open()
            x_in        = f["x_in"][i]
            mask_in     = f["mask_in"][i]
            x_target    = f["x_target"][i]
            mask_target = f["mask_target"][i]
            static      = f["static"][i]
            static_mask = f["static_mask"][i]
            stay_id     = int(f["stay_id"][i])
            ds_id       = int(f["ds_id"][i])
            t0_hour     = float(f["t0_hour"][i])
            los_hours   = float(f["los_hours"][i])
            mort_hours  = (float(f["mortality_hours"][i])
                           if "mortality_hours" in f else float("nan"))
            sofa        = (f["sofa_scores"][i] if "sofa_scores" in f else None)
            dsofa2      = (f["dsofa2"][i] if "dsofa2" in f else None)
            dsofa3      = (f["dsofa3"][i] if "dsofa3" in f else None)

        if self.nan_to_zero_in:
            x_in = np.nan_to_num(x_in, nan=0.0)

        out = {
            "x_in":        torch.from_numpy(x_in.astype(np.float32)),
            "mask_in":     torch.from_numpy(mask_in),
            "x_target":    torch.from_numpy(x_target.astype(np.float32)),
            "mask_target": torch.from_numpy(mask_target),
            "static":      torch.from_numpy(static.astype(np.float32)),
            "static_mask": torch.from_numpy(static_mask),
            "stay_id":     torch.tensor(stay_id,  dtype=torch.int64),
            "ds_id":       torch.tensor(ds_id,    dtype=torch.int64),
            "t0_hour":     torch.tensor(t0_hour,  dtype=torch.float32),
            "los_hours":   torch.tensor(los_hours, dtype=torch.float32),
            "mortality_hours": torch.tensor(mort_hours, dtype=torch.float32),
            # row position in the H5 — lets a caller line a sample up with a precomputed
            # per-window array (e.g. the ensemble predictions used as the AR decoder input)
            "row_idx":     torch.tensor(int(i), dtype=torch.int64),
        }
        if sofa is not None:
            out["sofa_scores"] = torch.from_numpy(np.asarray(sofa, dtype=np.float32))
        if dsofa2 is not None:
            out["dsofa2"] = torch.tensor(float(dsofa2), dtype=torch.float32)
        if dsofa3 is not None:
            out["dsofa3"] = torch.tensor(float(dsofa3), dtype=torch.float32)
        return out

    # ----- convenience -----
    @property
    def n_features_in(self) -> int:
        return len(self.features_in)

    @property
    def n_features_target(self) -> int:
        return len(self.features_target)

    @property
    def n_features_static(self) -> int:
        return len(self.features_static)

    def target_indices_in_input(self) -> list[int]:
        """Indices of features_target within features_in (target ⊆ input)."""
        idx: list[int] = []
        for f in self.features_target:
            if f in self.features_in:
                idx.append(self.features_in.index(f))
            else:
                raise ValueError(f"target feature {f!r} not in features_in")
        return idx
