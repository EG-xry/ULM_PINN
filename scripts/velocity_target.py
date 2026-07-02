#!/usr/bin/env python3
"""velocity_target.py - ADDITIVE helpers for the in vivo PINN anti collapse recipe.

Two things live here, neither affecting the existing pipeline:

1. build_voxel_mean_velocity_field(): turn noisy per detection finite difference
   velocity LABELS into a spatiotemporally-binned (voxel mean) velocity field.
   On dense in vivo data the per detection target is ~95% noise/multi-valued, so a
   smooth net collapses to the mean. Averaging within (x,z,t) voxels removes the
   within-cell scatter and exposes the coherent flow the PINN can actually learn.

2. evaluate_fit(): a held out GUARD metric (R^2 + correlation between prediction and
   target). Use as train_pinn(epoch_callback=...) so a collapse (R^2~0) is visible in
   the logs instead of hiding behind a saturated Huber loss.

All coordinates are the normalised [0,1]^3 (x,z,t) used throughout training.
"""
import numpy as np
import torch


def build_voxel_mean_velocity_field(pos, vel, nx=96, nz=96, nt=8, min_pts=8):
    """Spatiotemporal voxel mean velocity target (denoising).

    Parameters
    ----------
    pos : (N,3) array, normalised (x,z,t) in [0,1]
    vel : (N,2) array, per detection (u,v) velocity labels
    nx, nz, nt : voxel grid resolution in x, z, t
    min_pts : keep only points whose voxel has >= min_pts detections (drop
              singleton/under-sampled voxels whose "mean" is still noisy)

    Returns
    -------
    dict with:
      pos_kept, vel_target_kept : per-point positions and their VOXEL-MEAN velocity
                                  (the denoised training target), kept points only
      keep_mask                 : bool mask into the input arrays
      coherent_fraction         : Var(voxel_mean)/Var(raw) on kept points - the
                                  fraction of per detection velocity variance that is
                                  coherent (position-predictable) at this resolution
      n_voxels_used, frac_kept  : bookkeeping
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    xi = np.clip((pos[:, 0] * nx).astype(np.int64), 0, nx - 1)
    zi = np.clip((pos[:, 1] * nz).astype(np.int64), 0, nz - 1)
    ti = np.clip((pos[:, 2] * nt).astype(np.int64), 0, nt - 1)
    key = (ti * nx + xi) * nz + zi          # unique voxel id per point
    nbins = nx * nz * nt

    cnt = np.bincount(key, minlength=nbins).astype(np.float64)
    su = np.bincount(key, weights=vel[:, 0], minlength=nbins)
    sv = np.bincount(key, weights=vel[:, 1], minlength=nbins)
    safe = np.maximum(cnt, 1.0)
    mu = su / safe
    mv = sv / safe

    vel_target = np.column_stack([mu[key], mv[key]])   # each point -> its voxel mean
    point_count = cnt[key]
    keep = point_count >= int(min_pts)

    pos_k = pos[keep].astype(np.float32)
    tgt_k = vel_target[keep].astype(np.float32)
    raw_k = vel[keep]
    denom = raw_k.var(axis=0) + 1e-12
    coherent = float(np.mean(tgt_k.var(axis=0) / denom))

    return dict(
        pos_kept=pos_k,
        vel_target_kept=tgt_k,
        keep_mask=keep,
        coherent_fraction=coherent,
        n_voxels_used=int((cnt >= int(min_pts)).sum()),
        frac_kept=float(keep.mean()),
        grid=(nx, nz, nt),
        min_pts=int(min_pts),
    )


def train_val_split(n, val_frac=0.1, seed=0):
    """Random index split for the held out guard."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    nval = max(1, int(round(val_frac * n)))
    return perm[nval:], perm[:nval]      # train_idx, val_idx


@torch.no_grad()
def evaluate_fit(model, pos_val, vel_val_target, normalizer, device, batch=200000):
    """Held out fit metric (the GUARD). Returns physical-space R^2 + Pearson corr
    between the model's predicted velocity and the (binned) target.

    R^2 ~ 0  => collapse to the mean (BAD).
    R^2 > 0  => the field is being learned.
    Mirrors inference: star the inputs, model(eval+normalizer) returns physical (u,v).
    """
    was_training = model.training
    model.eval()
    pos_val = np.asarray(pos_val, dtype=np.float32)
    preds = np.empty((len(pos_val), 2), dtype=np.float64)
    for i in range(0, len(pos_val), batch):
        p = pos_val[i:i + batch]
        x = torch.tensor(p[:, 0], dtype=torch.float32, device=device)
        z = torch.tensor(p[:, 1], dtype=torch.float32, device=device)
        t = torch.tensor(p[:, 2], dtype=torch.float32, device=device)
        xs = torch.stack([normalizer.star(x, 'x'),
                          normalizer.star(z, 'z'),
                          normalizer.star(t, 't')], dim=1).float()
        u, v, _ = model(xs)            # eval+normalizer -> physical units
        preds[i:i + batch, 0] = u.detach().cpu().numpy().ravel()
        preds[i:i + batch, 1] = v.detach().cpu().numpy().ravel()
    if was_training:
        model.train()

    tgt = np.asarray(vel_val_target, dtype=np.float64)
    out = {}
    r2s = []
    for c, name in [(0, 'u'), (1, 'v')]:
        p = preds[:, c]; y = tgt[:, c]
        ss_res = float(np.sum((p - y) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        corr = float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 else 0.0
        out[f'r2_{name}'] = r2
        out[f'corr_{name}'] = corr
        r2s.append(r2)
    out['r2'] = float(np.mean(r2s))
    out['pred_speed_mean'] = float(np.sqrt((preds ** 2).sum(1)).mean())
    out['target_speed_mean'] = float(np.sqrt((tgt ** 2).sum(1)).mean())
    return out
