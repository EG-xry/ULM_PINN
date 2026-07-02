#!/usr/bin/env python3
"""
render_brainmaps.py - render super-resolution ULM maps from a tracks file.

Reads the `tracks.npz` written by main.py (normalized tracks + original
bounds) and renders three maps, following the PALA ULM_Track2MatOut convention:
  - density   (hot,        counts^(1/3))
  - velocity  (jet,        mean speed in mm/s, density-shadowed)
  - direction (blue/red,   axial flow sign)

Physical scale (--res, --lambda_mm) and rendering knobs (blur, gamma, power)
are exposed as options.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

PAD = 20      # SR-pixel padding around the track bounding box
DPI = 300


def moving_average(a, span):
    """Centered moving average with symmetric window shrink at the ends."""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    if span % 2 == 0:
        span -= 1
    if span < 3 or n < 3:
        return a.copy()
    span = min(span, n if n % 2 == 1 else n - 1)
    half = (span - 1) // 2
    csum = np.concatenate(([0.0], np.cumsum(a)))
    i = np.arange(n)
    w = np.minimum(half, np.minimum(i, n - 1 - i))
    lo, hi = i - w, i + w
    return (csum[hi + 1] - csum[lo]) / (hi - lo + 1)


def prepare_tracks(npz_path, interp_factor, smooth_factor):
    """Load tracks.npz and return per track [x, z, vx, vz, t] arrays in physical units."""
    z = np.load(npz_path, allow_pickle=True)
    raw = np.asarray(z["tracks"], dtype=np.float64)        # [track_id, x, z, t] normalized
    Xmin, Xmax, Zmin, Zmax, Tmin, Tmax = np.asarray(z["orig_bounds"], dtype=np.float64)
    xspan, zspan, tspan = Xmax - Xmin, Zmax - Zmin, Tmax - Tmin

    # group rows into per track blocks in O(N log N), sort by track id, split at id changes
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    cuts = np.flatnonzero(np.diff(raw[:, 0]) != 0) + 1
    out = []
    for blk in np.split(raw, cuts):
        if blk.shape[0] < 2:
            continue
        blk = blk[np.argsort(blk[:, 3], kind="stable")]
        x = blk[:, 1] * xspan + Xmin
        z_ = blk[:, 2] * zspan + Zmin
        t = blk[:, 3] * tspan + Tmin
        if interp_factor and interp_factor > 0 and blk.shape[0] >= 3:
            xs = moving_average(x, smooth_factor)
            zs = moving_average(z_, smooth_factor)
            base = np.arange(1, len(x) + 1, dtype=np.float64)
            new = np.arange(1.0, len(x) + 1e-9, interp_factor, dtype=np.float64)
            xi, zi, ti = np.interp(new, base, xs), np.interp(new, base, zs), np.interp(new, base, t)
        else:
            xi, zi, ti = x, z_, t
        dt = np.diff(ti)
        dt[np.abs(dt) < 1e-12] = 1e-12
        vx = np.concatenate(([0.0], np.diff(xi) / dt))
        vz = np.concatenate(([0.0], np.diff(zi) / dt))
        vx[0], vz[0] = vx[1] if len(vx) > 1 else 0.0, vz[1] if len(vz) > 1 else 0.0
        out.append(np.column_stack([xi, zi, vx, vz, ti]))
    return out


def build_matouts(tracks, res):
    """Accumulate density, axial direction, and speed maps (ULM_Track2MatOut semantics)."""
    srscale = 1.0 / res
    x0 = min(t[:, 0].min() for t in tracks)
    x1 = max(t[:, 0].max() for t in tracks)
    z0 = min(t[:, 1].min() for t in tracks)
    z1 = max(t[:, 1].max() for t in tracks)
    sr_w = int(np.ceil((x1 - x0) / srscale)) + 1 + 2 * PAD
    sr_h = int(np.ceil((z1 - z0) / srscale)) + 1 + 2 * PAD
    size = (sr_h + 1, sr_w + 1)
    xo, zo = x0 - PAD * srscale, z0 - PAD * srscale

    density = np.zeros(size)
    vel_sum = np.zeros(size)
    zdir_sum = np.zeros(size)
    for tr in tracks:
        px = np.rint((tr[:, 0] - xo) / srscale).astype(np.int64)
        pz = np.rint((tr[:, 1] - zo) / srscale).astype(np.int64)
        vx_grid, vz_grid = tr[:, 2] / srscale, tr[:, 3] / srscale
        inb = (pz >= 0) & (pz < size[0]) & (px >= 0) & (px < size[1])
        pz, px = pz[inb], px[inb]
        if pz.size == 0:
            continue
        speed_s = moving_average(np.hypot(vz_grid, vx_grid), 10)[inb]
        sgn = np.sign(np.mean(vz_grid)) if vz_grid.size else 0.0
        uniq = np.unique(pz * size[1] + px)
        uz, ux = uniq // size[1], uniq % size[1]
        density[uz, ux] += 1.0
        contrib = speed_s[:uniq.size]
        vel_sum[uz, ux] += contrib
        zdir_sum[uz, ux] += contrib * sgn

    nz = density > 0
    matout_vel = np.zeros_like(vel_sum); matout_vel[nz] = vel_sum[nz] / density[nz]
    matout_zdir = np.zeros_like(zdir_sum); matout_zdir[nz] = zdir_sum[nz] / density[nz]
    llx = np.arange(size[1] + 1) * srscale + xo
    llz = np.arange(size[0] + 1) * srscale + zo
    return density, matout_zdir, matout_vel, llx, llz


def hot_mirror_cmap():
    """Blue (down) / red (up) direction colormap: mirrored 'hot'."""
    hot = plt.cm.hot(np.linspace(0, 1, 128))
    bl = hot[::-1].copy()
    bl[:, [0, 2]] = bl[:, [2, 0]]
    return LinearSegmentedColormap.from_list("hot_mirror", np.vstack([bl, hot])[5:-5])


def _orient(arr, llx, llz, swap):
    if arr.ndim == 3:
        disp = arr if swap else np.transpose(arr, (1, 0, 2))
    else:
        disp = arr if swap else arr.T
    if swap:
        return disp, [llx[0], llx[-1], llz[-1], llz[0]], "x (λ)", "z (λ)"
    return disp, [llz[0], llz[-1], llx[-1], llx[0]], "z (λ)", "x (λ)"


def render_density(density, llx, llz, out_png, int_power, blur, swap):
    base = density ** int_power
    if blur > 0:
        base = gaussian_filter(base, blur)
    disp, extent, xl, yl = _orient(base, llx, llz, swap)
    vmax = disp.max() * 0.8 if np.any(disp > 0) else 1.0
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    im = ax.imshow(disp, cmap="hot", extent=extent, aspect="equal", origin="upper",
                   vmin=0, vmax=vmax, interpolation="bilinear")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    cb = plt.colorbar(im, ax=ax, shrink=0.7); cb.set_label("counts$^{1/3}$")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)


def render_direction(density, matout_zdir, llx, llz, out_png, blur, swap):
    zs = gaussian_filter(matout_zdir, 0.8)
    dd = (density ** 0.25) * np.sign(zs)
    dd -= np.sign(dd) * 0.5
    dmax = np.max(np.abs(dd)) if np.any(dd != 0) else 1.0
    alpha_base = (density > 0).astype(float)
    if blur > 0:
        alpha_base = np.clip(gaussian_filter(alpha_base, blur), 0, 1)
    dd_disp, extent, xl, yl = _orient(dd, llx, llz, swap)
    alpha, _, _, _ = _orient(alpha_base, llx, llz, swap)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    ax.set_facecolor("black")
    im = ax.imshow(dd_disp, cmap=hot_mirror_cmap(), extent=extent, aspect="equal",
                   origin="upper", vmin=-dmax * 0.7, vmax=dmax * 0.7, interpolation="bilinear")
    im.set_alpha(alpha)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    cb = plt.colorbar(im, ax=ax, shrink=0.7); cb.set_label("up (+) / down (-)")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", facecolor="white"); plt.close(fig)


def render_velocity(density, matout_vel, llx, llz, out_png, lambda_mm, blur, shadow_gamma, swap):
    vel_mms = matout_vel * lambda_mm
    nz = vel_mms[vel_mms > 0]
    if nz.size:
        vlo = max(0.0, np.percentile(nz, 5))
        vhi = np.percentile(nz, 95)
        if vhi <= vlo:
            vhi = nz.max()
    else:
        vlo, vhi = 0.0, 1.0
    mn = np.clip((vel_mms - vlo) / max(vhi - vlo, 1e-12), 0, 1)
    if blur > 0:
        mn = gaussian_filter(mn, blur)
    mn = np.clip(mn ** (1.0 / 1.5), 0, 1)
    rgb = plt.cm.jet(mn)[:, :, :3]
    dens = gaussian_filter(density, blur) if blur > 0 else density
    shadow = np.clip(dens / max(dens.max() * 0.3, 1e-12), 0, 1) ** shadow_gamma
    rgb = np.clip(rgb * shadow[:, :, None], 0, 1) ** 0.6
    rgb_disp, extent, xl, yl = _orient(rgb, llx, llz, swap)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    ax.imshow(rgb_disp, extent=extent, aspect="equal", origin="upper")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vlo, vhi)); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.7); cb.set_label("mm/s")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ============================================================================
# TUNABLE PARAMETERS
# ----------------------------------------------------------------------------
# Physical scale (--res, --lambda_mm) and rendering knobs (blur, gamma, power,
# orientation) are defined below. This function is the single place to tune the
# rendered maps
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description="Render ULM super-resolution maps from a tracks file",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tracks", default="ulm_pinn_out/tracks.npz", help="tracks.npz from main.py")
    p.add_argument("--out_dir", default="ulm_pinn_out/maps", help="Output directory for the PNGs")
    p.add_argument("--res", type=float, default=10.0, help="Super-resolution factor (SR pixels per wavelength)")
    p.add_argument("--lambda_mm", type=float, default=1540.0 / 15e6 * 1000.0,
                   help="Ultrasound wavelength in mm (default: 15 MHz)")
    p.add_argument("--interp_factor", type=float, default=0.04, help="Track interpolation step")
    p.add_argument("--smooth_factor", type=int, default=20, help="Moving-average window for track smoothing")
    p.add_argument("--int_power", type=float, default=1.0 / 3.0, help="Density compression exponent")
    p.add_argument("--density_blur", type=float, default=0.0, help="Gaussian blur (SR px) on the density map")
    p.add_argument("--vel_blur", type=float, default=0.5, help="Gaussian blur (SR px) on the velocity map")
    p.add_argument("--vel_gamma", type=float, default=0.25, help="Density-shadow gamma on the velocity map")
    p.add_argument("--swap_xz", action="store_true", help="Put x on the horizontal axis (landscape maps)")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tracks = prepare_tracks(args.tracks, args.interp_factor, args.smooth_factor)
    if not tracks:
        print("No tracks to render.")
        return
    print(f"rendering {len(tracks):,} tracks -> {args.out_dir}", flush=True)
    density, zdir, vel, llx, llz = build_matouts(tracks, args.res)
    render_density(density, llx, llz, os.path.join(args.out_dir, "density.png"),
                   args.int_power, args.density_blur, args.swap_xz)
    render_velocity(density, vel, llx, llz, os.path.join(args.out_dir, "velocity.png"),
                    args.lambda_mm, args.vel_blur, args.vel_gamma, args.swap_xz)
    render_direction(density, zdir, llx, llz, os.path.join(args.out_dir, "direction.png"),
                     args.density_blur, args.swap_xz)
    print("saved density.png, velocity.png, direction.png", flush=True)


if __name__ == "__main__":
    main()
