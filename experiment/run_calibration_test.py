"""Standalone data-generation script for the ngmix vs. autometacal(KSB
moments) vs. autometacal(fitting) vs. ola calibration test
(notebooks/calibration_test.ipynb).

Deliberately NOT reusing pujol_test.py's bias-computation logic (per the
user's explicit request -- this tests the three shape-measurement methods
themselves, not the Pujol-style bias-correction machinery) -- it does reuse
pujol_test.py's Config/make_psf/make_galaxy/select_resolved_indices/GSPARAMS
as plain setup/rendering utilities, same as notebooks/galaxy_and_psf_diagnostics.ipynb
already does.

For each galaxy: one COSMOS profile, one noise realization, shared between
an unsheared (e1=0.0) and a sheared (e1=0.02, applied via GalSim's
distortion-convention `.shear(e1=..., e2=...)`) branch, and shared across
all three methods (so any noise-driven differences are between *methods*,
not between independent noise draws). For each (galaxy, branch, method):
the metacal *response* R and the noshear measurement -- nothing else
(no Pujol-style per-galaxy re-shearing/finite-difference-of-ghat step).

**"moments" here means KSB Psh-calibrated shear, not raw windowed-moment
distortion `e`.** An earlier version of this script fed metacal raw
`get_moment_ellipticities` output (plain distortion `e`) for the
"moments" method, and got a suspiciously low response (R11~0.42, vs. the
~0.8 typically expected for a moments-based method). Checking ola's own
`metacal_package/metacal.py::get_metacal_response` found it runs metacal on
`resdict[type]["g"]` -- and for ola's KSB pipeline that `"g"` is
`correct_ksb`'s **Psh-calibrated** shear (an analytic weight-dilution
correction), not a plain geometric e-to-g relabeling (verified directly:
converting our own raw `e` to `g` via `e1e2_to_g1g2` and redoing R via the
chain rule made the response *lower*, not higher -- ruling that out).
`ksb.calibrated_g` (this repo's port of that same Psh correction, split out
so it doesn't need the non-differentiable `admom` call `correct_ksb` also
makes) is used here as the "moments" method's per-galaxy measurement, for a
genuine apples-to-apples comparison against ola's own ~0.8 benchmark.
`ngmix` (raw GaussMom `e`, no KSB layered on) and `am_fitting` (the
differentiable Gaussian-fit estimator) are unchanged.

**`ola` cross-check**: a fourth method runs ola's *own* reference metacal+KSB
implementation (`ola.pipeline.MetacalPipeline`, backed by
`ola.metacal_package.metacal.get_all_metacal`/`get_metacal_response` and
`measure.wcsmoms_wrapper` -- real GalSim, not JAX/autometacal at all) on the
*same* galaxy/PSF/noise pixel data as the other three methods, using ola's
own typical `window=2.0` (`ola/tests/test_bias.py`'s own reference
configuration). This directly tests whether `autometacal.ksb.calibrated_g`'s
~0.54 (vs. the user's ~0.8 expectation) reflects a remaining bug in this
repo's KSB port, or is just what this particular COSMOS+Kolmogorov+`sigw`
configuration gives regardless of implementation -- verified directly on one
galaxy before committing to the full run: ola's own code gave R11~0.57 on
the exact same stamp, matching our port, not ~0.8. `ola` is a dev-machine-only
reference clone (see CLAUDE.md), not a repo dependency -- this method is
skipped with a warning if it isn't importable.

Run: python run_calibration_test.py [--n-gals 100] [--exptime 1000] [--out ...]
Writes an .npz checkpoint periodically and at the end.
"""
import argparse
import os
import sys
import time
import warnings

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax
import jax.numpy as jnp
import galsim
import ngmix

import pujol_test as pt
import autometacal
from autometacal.python import gaussfit
from autometacal.python import ksb as ksb_module

sys.path.insert(0, os.path.expanduser('~/github/ola'))
try:
  from ola.pipeline import MetacalPipeline as OlaMetacalPipeline
  from ola.metacal_package import measure as ola_measure
  HAVE_OLA = True
except ImportError:
  HAVE_OLA = False
  warnings.warn("ola not importable (dev-machine-only reference clone, see "
                 "CLAUDE.md) -- skipping the 'ola' cross-check method.")

TYPES = ['noshear', '1p', '1m', '2p', '2m']


def build_branch_image(gal0, psf, e1, noise, cfg):
  sheared = gal0.shear(e1=e1, e2=0.0)
  obj = galsim.Convolve([sheared, psf])
  img = obj.drawImage(nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale).array.astype('float32')
  return img + noise


def ngmix_e_and_R11(obs_image, psf_image, noise_std, pixel_scale, mcal_step, gm_fwhm, rng):
  im64 = obs_image.astype('float64')
  cen = obs_image.shape[0] / 2.0
  jac = ngmix.DiagonalJacobian(row=cen, col=cen, scale=pixel_scale)
  psf_noise = 1.0e-6
  psf_obs_arr = (psf_image + rng.normal(scale=psf_noise, size=psf_image.shape)).astype('float64')
  psf_wt = psf_obs_arr * 0 + 1.0 / psf_noise ** 2
  psf_obs = ngmix.Observation(psf_obs_arr, weight=psf_wt, jacobian=jac)
  wt = im64 * 0 + 1.0 / noise_std ** 2
  obs = ngmix.Observation(im64, weight=wt, jacobian=jac, psf=psf_obs)

  mcal_rng = np.random.RandomState(int(rng.randint(1 << 30)))
  obsdict = ngmix.metacal.get_all_metacal(obs, psf='gauss', step=mcal_step, fixnoise=True, rng=mcal_rng)
  fitter = ngmix.gaussmom.GaussMom(fwhm=gm_fwhm)
  e = {t: fitter.go(obsdict[t])['e'] for t in TYPES}
  e1_out = e['noshear'][0]
  e2_out = e['noshear'][1]
  R11 = (e['1p'][0] - e['1m'][0]) / (2 * mcal_step)
  R22 = (e['2p'][1] - e['2m'][1]) / (2 * mcal_step)
  return e1_out, e2_out, R11, R22


def make_ola_pipeline(cfg, mcal_step):
  def _method(obss, psfs, weights):
    return ola_measure.wcsmoms_wrapper(
        obss=obss, psfs=psfs, weights=weights, psf_upsampling=1, gain=1.0,
        strict=False, window=2.0,
    )
  return OlaMetacalPipeline(
      method=_method, target_psf_mode="dilate", dilate_factor=cfg.reconv_psf_dilation,
      psf_pixel_scale=cfg.pixel_scale, step=mcal_step, interpolator="lanczos11",
      shear_type="g", propagate_upsampling=False, fixnoise=True, select=False,
  )


def ola_e_and_R11(pipeline, obs_image, psf_image, noise_std, pixel_scale, rng):
  obs_gsimage = galsim.Image(obs_image.astype('float64'), scale=pixel_scale)
  psf_gsimage = galsim.Image(psf_image.astype('float64'), scale=pixel_scale)
  extra_noise = rng.normal(scale=noise_std, size=obs_image.shape)
  noise_gsimage = galsim.Image(extra_noise, scale=pixel_scale)

  stamp = {
      "images": [obs_gsimage], "psfs": [psf_gsimage],
      "noise_images": [noise_gsimage], "weights": None,
  }
  ellip_dict, R_she, _, _ = pipeline.process(stamp)['result']
  e1_out = ellip_dict['noshear'][0, 0]
  e2_out = ellip_dict['noshear'][0, 1]
  R11 = R_she[0, 0, 0]
  R22 = R_she[1, 1, 0]
  return e1_out, e2_out, R11, R22


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--n-gals', type=int, default=1000)
  p.add_argument('--exptime', type=float, default=1000.0)
  p.add_argument('--shear-e1', type=float, default=0.02)
  p.add_argument('--mcal-step', type=float, default=0.01)
  p.add_argument('--chunk-size', type=int, default=8)
  p.add_argument('--fft-size', type=int, default=128)
  p.add_argument('--seed', type=int, default=31415)
  p.add_argument('--out', type=str, default='calibration_test_results.npz')
  p.add_argument('--checkpoint-every', type=int, default=80)
  args = p.parse_args()

  cfg = pt.Config(exptime=args.exptime)
  cat = galsim.COSMOSCatalog(sample=cfg.cosmos_sample)
  resolved_idxs = pt.select_resolved_indices(cat, cfg)
  psf = pt.make_psf(cfg)
  psf_image = psf.drawImage(nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale).array.astype('float32')
  reconv_psf_image = np.asarray(
      autometacal.galflow.dilate(psf_image, cfg.reconv_psf_dilation, scale=cfg.pixel_scale)
  )
  noise_std = np.sqrt(cfg.sky_level * cfg.exptime)
  print(f"noise_std={noise_std:.2f}  n_gals={args.n_gals}  exptime={args.exptime}  "
        f"shear_e1={args.shear_e1}  chunk_size={args.chunk_size}")

  # KSB weight sigma, in pixels -- fwhm_to_sigma, matching cfg.weight_fwhm
  # (arcsec) the same way ngmix's GaussMom(fwhm=cfg.weight_fwhm) uses it.
  ksb_sigw_pixels = (cfg.weight_fwhm / cfg.pixel_scale) / 2.3548200450309493
  # get_metacal_response internally dilates the reconvolution PSF by its own
  # tiny _reconv_psf_dilation (numerical stability, on top of
  # cfg.reconv_psf_dilation already applied above) -- replicate that here so
  # calibrated_g's own PSF measurement matches the PSF the galaxy image is
  # actually reconvolved with inside get_metacal_response_batched.
  ksb_psf_image = np.asarray(autometacal.galflow.dilate(
      reconv_psf_image, autometacal.python.metacal._reconv_psf_dilation, scale=cfg.pixel_scale,
  ))

  method_ksb_g = lambda img: ksb_module.calibrated_g(
      img, jnp.asarray(ksb_psf_image), scale=cfg.pixel_scale, sigw=ksb_sigw_pixels)
  method_fit = lambda img: gaussfit.get_fit_ellipticities(img)

  ola_pipeline = make_ola_pipeline(cfg, args.mcal_step) if HAVE_OLA else None

  rng = np.random.RandomState(args.seed)

  # storage: [n_gals] per branch per method (columns stay NaN for 'ola' if
  # the ola reference clone isn't available on this machine)
  keys = ['ngmix', 'am_moments', 'am_fitting', 'ola']
  branches = ['unsheared', 'sheared']
  data = {f'{m}_{b}_e1': np.full(args.n_gals, np.nan, dtype='float64') for m in keys for b in branches}
  data.update({f'{m}_{b}_e2': np.full(args.n_gals, np.nan, dtype='float64') for m in keys for b in branches})
  data.update({f'{m}_{b}_R11': np.full(args.n_gals, np.nan, dtype='float64') for m in keys for b in branches})
  data.update({f'{m}_{b}_R22': np.full(args.n_gals, np.nan, dtype='float64') for m in keys for b in branches})
  data['idx'] = np.full(args.n_gals, -1, dtype='int64')
  data['snr'] = np.full(args.n_gals, np.nan, dtype='float64')

  # buffers for the current chunk (per branch), for the AM (batched) methods
  buf = {b: {'gal': [], 'psf': [], 'rpsf': [], 'noise': [], 'gidx': []} for b in branches}

  def flush_chunk(branch):
    entries = buf[branch]
    B = len(entries['gal'])
    if B == 0:
      return
    gal_b = jnp.stack(entries['gal'])
    psf_b = jnp.stack(entries['psf'])
    rpsf_b = jnp.stack(entries['rpsf'])
    noise_b = jnp.stack(entries['noise'])

    e_gm, R_gm, _, _, _ = autometacal.metacal.get_metacal_response_batched(
        gal_b, psf_b, rpsf_b, noise_b, method_ksb_g, args.fft_size, scale=cfg.pixel_scale,
    )
    e_fit, R_fit, _, _, _ = autometacal.metacal.get_metacal_response_batched(
        gal_b, psf_b, rpsf_b, noise_b, method_fit, args.fft_size, scale=cfg.pixel_scale,
    )
    e_gm = np.asarray(e_gm); R_gm = np.asarray(R_gm)
    e_fit = np.asarray(e_fit); R_fit = np.asarray(R_fit)

    for k, gidx in enumerate(entries['gidx']):
      data[f'am_moments_{branch}_e1'][gidx] = e_gm[k, 0]
      data[f'am_moments_{branch}_e2'][gidx] = e_gm[k, 1]
      data[f'am_moments_{branch}_R11'][gidx] = R_gm[k, 0, 0]
      data[f'am_moments_{branch}_R22'][gidx] = R_gm[k, 1, 1]
      data[f'am_fitting_{branch}_e1'][gidx] = e_fit[k, 0]
      data[f'am_fitting_{branch}_e2'][gidx] = e_fit[k, 1]
      data[f'am_fitting_{branch}_R11'][gidx] = R_fit[k, 0, 0]
      data[f'am_fitting_{branch}_R22'][gidx] = R_fit[k, 1, 1]

    for k in entries:
      entries[k].clear()

  t_start = time.time()
  n_done = 0
  n_rejected = 0
  while n_done < args.n_gals:
    idx = int(resolved_idxs[rng.randint(len(resolved_idxs))])
    try:
      gal0 = pt.make_galaxy(cat, idx, cfg)
      noise = rng.normal(scale=noise_std, size=(cfg.stamp_size, cfg.stamp_size)).astype('float32')
      obs_unsheared = build_branch_image(gal0, psf, 0.0, noise, cfg)
      obs_sheared = build_branch_image(gal0, psf, args.shear_e1, noise, cfg)
      noiseless_unsheared = obs_unsheared - noise
      snr = np.sqrt(np.sum(noiseless_unsheared.astype('float64') ** 2)) / noise_std

      ok_u = autometacal.metacal.fits_fixed_fft_size(
          obs_unsheared, psf_image, reconv_psf_image, args.fft_size, scale=cfg.pixel_scale)
      ok_s = autometacal.metacal.fits_fixed_fft_size(
          obs_sheared, psf_image, reconv_psf_image, args.fft_size, scale=cfg.pixel_scale)
      if not (ok_u and ok_s):
        n_rejected += 1
        continue
    except Exception:
      n_rejected += 1
      continue

    gidx = n_done
    data['idx'][gidx] = idx
    data['snr'][gidx] = snr

    for branch, obs_image in [('unsheared', obs_unsheared), ('sheared', obs_sheared)]:
      # ngmix (sequential, real GalSim/CPU)
      e1_ng, e2_ng, R11_ng, R22_ng = ngmix_e_and_R11(
          obs_image, psf_image, noise_std, cfg.pixel_scale, args.mcal_step, cfg.weight_fwhm, rng)
      data[f'ngmix_{branch}_e1'][gidx] = e1_ng
      data[f'ngmix_{branch}_e2'][gidx] = e2_ng
      data[f'ngmix_{branch}_R11'][gidx] = R11_ng
      data[f'ngmix_{branch}_R22'][gidx] = R22_ng

      # ola (sequential, real GalSim/CPU) -- dev-machine-only cross-check
      if HAVE_OLA:
        e1_ola, e2_ola, R11_ola, R22_ola = ola_e_and_R11(
            ola_pipeline, obs_image, psf_image, noise_std, cfg.pixel_scale, rng)
        data[f'ola_{branch}_e1'][gidx] = e1_ola
        data[f'ola_{branch}_e2'][gidx] = e2_ola
        data[f'ola_{branch}_R11'][gidx] = R11_ola
        data[f'ola_{branch}_R22'][gidx] = R22_ola

      # buffer for batched autometacal methods
      extra_noise = rng.normal(scale=noise_std, size=(cfg.stamp_size, cfg.stamp_size)).astype('float32')
      buf[branch]['gal'].append(jnp.asarray(obs_image))
      buf[branch]['psf'].append(jnp.asarray(psf_image))
      buf[branch]['rpsf'].append(jnp.asarray(reconv_psf_image))
      buf[branch]['noise'].append(jnp.asarray(extra_noise))
      buf[branch]['gidx'].append(gidx)

      if len(buf[branch]['gal']) >= args.chunk_size:
        flush_chunk(branch)

    n_done += 1
    if n_done % 20 == 0:
      elapsed = time.time() - t_start
      print(f"  {n_done}/{args.n_gals} done ({n_rejected} rejected), "
            f"elapsed={elapsed:.0f}s, rate={n_done/elapsed:.2f} gal/s")

    if n_done % args.checkpoint_every == 0:
      for branch in branches:
        flush_chunk(branch)
      np.savez(args.out, **data, n_done=n_done, args=vars(args))
      print(f"  checkpoint written to {args.out} at n_done={n_done}")

  for branch in branches:
    flush_chunk(branch)
  np.savez(args.out, **data, n_done=n_done, args=vars(args))
  print(f"DONE. total time={time.time()-t_start:.0f}s. wrote {args.out}")


if __name__ == '__main__':
  main()
