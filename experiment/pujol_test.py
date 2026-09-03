"""
Pujol bias test (Pujol et al. 2018/2019, arXiv:1806.10537, "Method 1" /
per-galaxy response): estimate multiplicative (m) and additive (c) bias of
autometacal's *metacal-calibrated* shear estimate, sample-efficiently.

Unlike the ensemble multi-shear-bin test in cosmos_calibration_test.py, this
method needs only ONE reference shear point (default g=0): for each
simulated galaxy, apply +-`step` around that point directly to the TRUE
(pre-PSF, pre-noise) galaxy model, reusing the *exact same* noise
realization for all 5 copies (noshear, 1p, 1m, 2p, 2m) so the noise
contribution cancels exactly in the finite difference (paper Section 3.1).
The per-galaxy response R and offset a are then simply averaged over many
galaxies (Eqs. 4, 6, 8):

  R_ab,i ~= (ghat_a^{+}_i - ghat_a^{-}_i) / (2 * step)       [per galaxy]
  a_i     = ghat^{noshear}_i - R_i @ g_ref - g^I_i           [per galaxy]
  1 + m_a = <R_aa>                                           [ensemble]
  c_a     = <a_a>                                            [ensemble]

(R is generalized here to the full 2x2 matrix rather than the paper's
diagonal-only presentation -- a natural, low-risk extension matching
autometacal's own response-matrix convention elsewhere in this repo.)

`ghat` is autometacal's own metacal-CALIBRATED shear estimate, not the raw
moments ellipticity: for each galaxy, `get_metacal_response` is called ONCE
at the baseline (noshear) point to get that galaxy's own deconvolve/
reconvolve autodiff response `R_metacal`; the same `R_metacal` is then used
to calibrate (`R_metacal^-1 @ e`) the raw moments measured at all 5 Pujol
points (metacal itself relies on the same small-perturbation-constant-R
assumption, so reusing R_metacal across the 5 points rather than
recomputing it 5x is consistent with how metacal is meant to be used, and
is ~5x cheaper). `g^I` (intrinsic ellipticity) is likewise converted from
the raw moments distortion `e` to reduced-shear units
(`tf_ngmix.gmix.e1e2_to_g1g2`) to stay dimensionally consistent with
`ghat`. Earlier versions of this test fed raw, uncalibrated `e` directly
into R/a -- that conflates two different things: (1) get_moment_ellipticities
returns a *distortion* e, not reduced shear g (e = 2g/(1+g^2) for small g,
so an unbiased distortion estimator alone gives R=2, not 1 -- this alone
produced a spurious m~-0.5 that had nothing to do with real bias), and (2)
even after fixing that, testing the *uncalibrated* estimator isn't the
interesting question here -- autometacal's whole point is that metacal
calibration should remove shape-measurement bias, so the bias worth
measuring is on `ghat`, post-calibration.

The paper reports this needs only ~1e4 galaxies to reach a precision on m
that a naive multi-shear-bin linear fit would need ~1e7 images for (Section
6) for their per-galaxy response, computed cheaply for their raw estimator.
Here, getting `R_metacal` costs one ~5s `get_metacal_response` call per
galaxy (jax_galsim's deconvolve/reconvolve + autodiff, unjitted -- see
plans.md), so 10000 galaxies is ~14 hours, not the few minutes the paper's
own (uncalibrated) estimator would need; n_gals defaults much lower here
accordingly. Scale it up deliberately, e.g. in a background run.

Noise is plain additive Gaussian (not the CCDNoise/Poisson model used in
cosmos_calibration_test.py) -- this is required, not just simpler: Poisson
noise is signal-dependent, so it would differ slightly between the 5
sheared copies of a galaxy (their pixel values differ slightly under
different shears) and break the exact noise-cancellation the whole method
relies on. Matches the paper's own Section 7.3 Gaussian-noise treatment.
"""
import argparse
import dataclasses
import os
import time
import warnings

# GPU memory hygiene: JAX's default XLA_PYTHON_CLIENT_PREALLOCATE=true grabs
# ~75% of *available* GPU memory on the very first op, regardless of actual
# need (verified directly: 5.1GB out of 6GB total on this machine's shared
# laptop GPU, for a single 10x10 array). Must be set before `import jax`
# (or anything that imports it, e.g. autometacal) to take effect. See
# CLAUDE.md's GPU notes for the full picture, including that this
# workload's small, sequential, non-batched calls currently run *faster* on
# CPU than GPU (dispatch overhead dominates) -- this setting doesn't change
# that, it just stops the GPU backend from being needlessly greedy.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import galsim
import jax
import jax.numpy as jnp
import autometacal

# SAFETY: GalSim's gsparams.maximum_fft_size does NOT actually prevent large
# FFT allocations -- reading GSObject.drawFFT_makeKImage's source directly:
# it only *warns* (GalSimFFTSizeWarning) when the required size exceeds the
# cap, then proceeds to allocate the (potentially huge) array anyway. A
# handful of pathological COSMOS parametric fits want FFTs up to ~44000^2
# complex128 (~43GB) -- verified directly, and it came within seconds of
# exhausting this machine's 30GB RAM before being caught and killed. The
# warning fires BEFORE the allocation, so converting it into a raised
# exception (standard `warnings` mechanism) stops the allocation from ever
# happening; the per-galaxy try/except below then just skips that object.
warnings.filterwarnings('error', category=galsim.errors.GalSimFFTSizeWarning)


@dataclasses.dataclass
class Config:
  # imaging (see cosmos_calibration_test.py for the same defaults' provenance)
  pixel_scale: float = 0.187
  stamp_size: int = 51
  exptime: float = 50.0          # -> noise_std = sqrt(sky_level * exptime), SNR ~20 for a typical resolved galaxy
  sky_level: float = 400.0

  # psf
  psf_type: str = 'Kolmogorov'
  psf_fwhm: float = 0.7
  psf_beta: float = 4.8
  psf_e1: float = 0.0
  psf_e2: float = 0.0            # circular PSF by default (datasets/cfis.py had 0.025; simplified for now)
  # reconvolution target = true PSF dilated by this factor (standard metacal
  # practice: the reconvolution PSF should be safely larger than the
  # observed one, not identical -- separate from, and on top of,
  # metacal.py's own tiny internal 1.001 dilation for numerical stability)
  reconv_psf_dilation: float = 1.02

  # galaxies
  cosmos_sample: str = '25.2'
  mag_zp: float = 32.0
  min_hlr: float = 0.3           # arcsec; see cosmos_calibration_test.py for why this matters

  # Pujol test
  g_ref: tuple = (0.0, 0.0)      # reference shear point to test bias at
  step: float = 0.02             # shear perturbation (paper: g=(+-0.02,0), g=(0,+-0.02))
  n_gals: int = 200

  # GPU batching: stamp generation (real GalSim) stays a CPU loop, but the
  # metacal derivative and ellipticity measurements run `chunk_size` galaxies
  # at a time as one jax.vmap dispatch (get_metacal_response_batched). This
  # needs a *fixed* FFT grid size (fft_size) shared by the whole chunk --
  # galaxies that wouldn't fit it are rejected at generation time (see
  # metacal.fits_fixed_fft_size) rather than silently aliasing. Verified
  # directly: fft_size=128 matches unbatched adaptive sizing to ~1e-7 for
  # 45x45 stamps at scale=0.263; chunk_size=4 gave a real ~5.8x/galaxy
  # speedup over sequential on a 6GB laptop GPU (memory-bound, not compute-
  # bound -- this machine is a prototyping stand-in for a much larger GPU;
  # bump chunk_size up there, that's the whole point of it being a knob
  # rather than hardcoded).
  chunk_size: int = 4
  fft_size: int = 128

  # ellipticity measurement
  weight_fwhm: float = 1.2

  seed: int = 31415


GSPARAMS = galsim.GSParams(maximum_fft_size=4096)


def make_psf(cfg):
  if cfg.psf_type == 'Kolmogorov':
    psf = galsim.Kolmogorov(fwhm=cfg.psf_fwhm, flux=1.0, gsparams=GSPARAMS)
  elif cfg.psf_type == 'Moffat':
    psf = galsim.Moffat(beta=cfg.psf_beta, fwhm=cfg.psf_fwhm, flux=1.0, gsparams=GSPARAMS)
  elif cfg.psf_type == 'Gaussian':
    psf = galsim.Gaussian(fwhm=cfg.psf_fwhm, flux=1.0, gsparams=GSPARAMS)
  else:
    raise ValueError(f"unknown psf_type '{cfg.psf_type}'")
  return psf.shear(g1=cfg.psf_e1, g2=cfg.psf_e2)


def make_galaxy(cat, idx, cfg):
  gal = cat.makeGalaxy(idx, gal_type='parametric', gsparams=GSPARAMS)
  mag = cat.param_cat['mag_auto'][cat.orig_index[idx]]
  flux = 10 ** (-(mag - cfg.mag_zp) / 2.5) * cfg.exptime
  return gal.withFlux(flux)


def select_resolved_indices(cat, cfg):
  hlr = cat.param_cat['hlr'][cat.orig_index, 0]
  idxs = np.where(hlr >= cfg.min_hlr)[0]
  print(f"Resolution cut (hlr >= {cfg.min_hlr}\"): {len(idxs)}/{cat.nobjects} "
        f"catalog objects pass ({100*len(idxs)/cat.nobjects:.1f}%).")
  return idxs


_SHEAR_OFFSETS = {
    'noshear': (0.0, 0.0),
    '1p': (1.0, 0.0), '1m': (-1.0, 0.0),
    '2p': (0.0, 1.0), '2m': (0.0, -1.0),
}


def _render(gal0, psf, dg1, dg2, noise, cfg, g1_ref, g2_ref):
  """ Real GalSim (CPU): shear the true galaxy model, convolve, draw, add
  the (shared, per-galaxy) noise realization. """
  sheared = gal0.shear(g1=g1_ref + cfg.step * dg1, g2=g2_ref + cfg.step * dg2)
  obj = galsim.Convolve([sheared, psf])
  image = obj.drawImage(nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale).array
  return image.astype('float32') + noise


def run_pujol_test(cfg):
  cat = galsim.COSMOSCatalog(sample=cfg.cosmos_sample)
  resolved_idxs = select_resolved_indices(cat, cfg)
  np_rng = np.random.RandomState(cfg.seed)
  psf = make_psf(cfg)
  psf_image = psf.drawImage(nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale).array
  psf_image = psf_image.astype('float32')
  # dilate at the pixel/interpolated-image level (Lanczos(11), matching
  # autometacal's own convention) rather than GalSim's exact analytic
  # .dilate() -- a real PSF model is always a pixelized image, not an
  # analytic profile, so this is what the actual pipeline (and metacal.py's
  # own internal reconv-psf dilation) actually does: resample the
  # interpolated image onto a finer grid and read it back at the normal
  # grid spacing.
  reconv_psf_image = np.asarray(
      autometacal.galflow.dilate(psf_image, cfg.reconv_psf_dilation, scale=cfg.pixel_scale)
  )
  g1_ref, g2_ref = cfg.g_ref
  g_ref_vec = jnp.array([g1_ref, g2_ref], dtype=jnp.float32)
  noise_std = np.sqrt(cfg.sky_level * cfg.exptime)

  def method(image):
    return autometacal.get_moment_ellipticities(image, scale=cfg.pixel_scale, fwhm=cfg.weight_fwhm)

  measure_batched = jax.jit(jax.vmap(method))  # cheap raw-moments path, batched over a chunk
  e1e2_to_g1g2_batched = jax.jit(jax.vmap(
      lambda e: jnp.stack(autometacal.tf_ngmix.e1e2_to_g1g2(e[0], e[1]))
  ))

  print(f"Pujol test (metacal-calibrated, GPU-batched): {cfg.n_gals} galaxies, "
        f"reference shear g={cfg.g_ref}, step={cfg.step}, noise_std={noise_std:.1f} ADU, "
        f"chunk_size={cfg.chunk_size}, fft_size={cfg.fft_size}. Stamp generation (real "
        f"GalSim) is a CPU loop; get_metacal_response_batched + the moment "
        f"measurements run one jax.vmap dispatch per chunk (GPU).")

  Rs, a_s = [], []
  n_done = 0
  n_rejected = 0
  t0 = time.time()

  # buffers for the current chunk: one entry per surviving candidate galaxy
  buf = {k: [] for k in ('noshear', '1p', '1m', '2p', '2m', 'noise', 'intrinsic')}

  def flush_chunk():
    nonlocal Rs, a_s
    B = len(buf['noshear'])
    if B == 0:
      return
    psf_batch = jnp.stack([psf_image] * B)
    reconv_batch = jnp.stack([reconv_psf_image] * B)
    noise_batch = jnp.stack(buf['noise'])
    gal_batch = jnp.stack(buf['noshear'])

    e_noshear, R_metacal, _, _, _ = autometacal.metacal.get_metacal_response_batched(
        gal_batch, psf_batch, reconv_batch, noise_batch, method, cfg.fft_size, scale=cfg.pixel_scale,
    )
    ghat = {'noshear': jnp.linalg.solve(R_metacal, e_noshear[..., None])[..., 0]}
    for name in ('1p', '1m', '2p', '2m'):
      e_i = measure_batched(jnp.stack(buf[name]))
      ghat[name] = jnp.linalg.solve(R_metacal, e_i[..., None])[..., 0]

    e_intrinsic = measure_batched(jnp.stack(buf['intrinsic']))
    g_intrinsic = e1e2_to_g1g2_batched(e_intrinsic)

    R_pujol = jnp.stack([
        jnp.stack([(ghat['1p'][:, 0] - ghat['1m'][:, 0]) / (2 * cfg.step),
                   (ghat['2p'][:, 0] - ghat['2m'][:, 0]) / (2 * cfg.step)], axis=-1),
        jnp.stack([(ghat['1p'][:, 1] - ghat['1m'][:, 1]) / (2 * cfg.step),
                   (ghat['2p'][:, 1] - ghat['2m'][:, 1]) / (2 * cfg.step)], axis=-1),
    ], axis=1)  # (B, 2, 2)
    a = ghat['noshear'] - jnp.einsum('bij,j->bi', R_pujol, g_ref_vec) - g_intrinsic

    Rs.append(np.asarray(R_pujol))
    a_s.append(np.asarray(a))
    for k in buf:
      buf[k].clear()

  while n_done < cfg.n_gals:
    idx = int(resolved_idxs[np_rng.randint(len(resolved_idxs))])
    n_done += 1
    try:
      gal0 = make_galaxy(cat, idx, cfg)
      intrinsic_img = gal0.drawImage(
          nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale,
      ).array.astype('float32')
      noise = np_rng.normal(scale=noise_std, size=(cfg.stamp_size, cfg.stamp_size)).astype('float32')
      noshear_img = _render(gal0, psf, 0.0, 0.0, noise, cfg, g1_ref, g2_ref)

      if not autometacal.metacal.fits_fixed_fft_size(
          noshear_img, psf_image, reconv_psf_image, cfg.fft_size, scale=cfg.pixel_scale,
      ):
        n_rejected += 1
        continue

      buf['noshear'].append(noshear_img)
      buf['intrinsic'].append(intrinsic_img)
      buf['noise'].append(noise)
      for name, (dg1, dg2) in _SHEAR_OFFSETS.items():
        if name != 'noshear':
          buf[name].append(_render(gal0, psf, dg1, dg2, noise, cfg, g1_ref, g2_ref))
    except Exception as exc:  # pragma: no cover -- a handful of pathological catalog entries is expected
      n_rejected += 1
      print(f"  [idx={idx}] failed: {type(exc).__name__}: {exc}")
      continue

    if len(buf['noshear']) == cfg.chunk_size:
      flush_chunk()

    if n_done % max(cfg.chunk_size, 10) == 0 or n_done == cfg.n_gals:
      elapsed = time.time() - t0
      rate = n_done / elapsed
      eta = (cfg.n_gals - n_done) / rate if rate > 0 else float('nan')
      print(f"  {n_done}/{cfg.n_gals}  ({elapsed:.0f}s elapsed, {1/rate:.2f}s/galaxy, ETA {eta:.0f}s, "
            f"{n_rejected} rejected)")

  flush_chunk()  # remaining partial chunk

  Rs = np.concatenate(Rs, axis=0) if Rs else np.zeros((0, 2, 2))
  a_s = np.concatenate(a_s, axis=0) if a_s else np.zeros((0, 2))
  print(f"{len(Rs)}/{cfg.n_gals} galaxies succeeded ({n_rejected} rejected: pathological "
        f"FFT size or other failure).")
  return Rs, a_s


def summarize(Rs, a_s):
  n = len(Rs)
  mean_R = Rs.mean(axis=0)
  mean_a = a_s.mean(axis=0)
  m = np.diagonal(mean_R) - 1
  c = mean_a

  diag = np.diagonal(Rs, axis1=1, axis2=2)  # (n, 2)
  m_err = diag.std(axis=0) / np.sqrt(n)
  c_err = a_s.std(axis=0) / np.sqrt(n)

  print()
  print(f"N = {n}")
  print(f"mean R =\n{mean_R}")
  print(f"m1 = {m[0]:+.5f} +/- {m_err[0]:.5f}")
  print(f"m2 = {m[1]:+.5f} +/- {m_err[1]:.5f}")
  print(f"c1 = {c[0]:+.6f} +/- {c_err[0]:.6f}")
  print(f"c2 = {c[1]:+.6f} +/- {c_err[1]:.6f}")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  d = Config()
  p.add_argument('--pixel-scale', type=float, default=d.pixel_scale)
  p.add_argument('--stamp-size', type=int, default=d.stamp_size)
  p.add_argument('--exptime', type=float, default=d.exptime)
  p.add_argument('--sky-level', type=float, default=d.sky_level)
  p.add_argument('--psf-type', choices=['Kolmogorov', 'Moffat', 'Gaussian'], default=d.psf_type)
  p.add_argument('--psf-fwhm', type=float, default=d.psf_fwhm)
  p.add_argument('--psf-beta', type=float, default=d.psf_beta)
  p.add_argument('--psf-e1', type=float, default=d.psf_e1)
  p.add_argument('--psf-e2', type=float, default=d.psf_e2)
  p.add_argument('--reconv-psf-dilation', type=float, default=d.reconv_psf_dilation)
  p.add_argument('--cosmos-sample', default=d.cosmos_sample)
  p.add_argument('--mag-zp', type=float, default=d.mag_zp)
  p.add_argument('--min-hlr', type=float, default=d.min_hlr)
  p.add_argument('--g-ref', type=float, nargs=2, default=list(d.g_ref))
  p.add_argument('--step', type=float, default=d.step)
  p.add_argument('--n-gals', type=int, default=d.n_gals)
  p.add_argument('--chunk-size', type=int, default=d.chunk_size,
                  help='galaxies per GPU batch dispatch; bump this up on a bigger GPU')
  p.add_argument('--fft-size', type=int, default=d.fft_size,
                  help='fixed FFT grid size shared by a whole chunk; galaxies needing more are rejected')
  p.add_argument('--weight-fwhm', type=float, default=d.weight_fwhm)
  p.add_argument('--seed', type=int, default=d.seed)
  p.add_argument('--output', default='pujol_test_results.npz')
  return p.parse_args()


def main():
  args = parse_args()
  cfg = Config(
      pixel_scale=args.pixel_scale, stamp_size=args.stamp_size, exptime=args.exptime,
      sky_level=args.sky_level, psf_type=args.psf_type, psf_fwhm=args.psf_fwhm,
      psf_beta=args.psf_beta, psf_e1=args.psf_e1, psf_e2=args.psf_e2,
      reconv_psf_dilation=args.reconv_psf_dilation,
      cosmos_sample=args.cosmos_sample, mag_zp=args.mag_zp, min_hlr=args.min_hlr,
      g_ref=tuple(args.g_ref), step=args.step, n_gals=args.n_gals,
      chunk_size=args.chunk_size, fft_size=args.fft_size,
      weight_fwhm=args.weight_fwhm, seed=args.seed,
  )
  Rs, a_s = run_pujol_test(cfg)
  summarize(Rs, a_s)
  np.savez(args.output, R=Rs, a=a_s)
  print(f"\nSaved per-galaxy R, a to {args.output}")


if __name__ == '__main__':
  main()
