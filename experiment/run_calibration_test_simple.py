"""Controlled, idealized calibration test: simple circular exponential-disk
galaxies (no intrinsic ellipticity), all the same size, ola's own "standard"
PSF (`ola.pipeline.detector.make_optical_psf`), ola's own pixel scale
(0.1"/pixel), and very low noise.

This strips away everything the COSMOS-based `run_calibration_test.py` test
mixes together (real, varied galaxy morphology, a wide size/SNR
distribution including marginally-resolved objects, realistic noise) to
isolate one question: with the sample held maximally simple/favorable
(round, single size chosen for a fixed resolution=0.7 -- see
`source_resolution`/ola's own `R2 = 1 - Tpsf/Tgal` convention --, near-zero
noise), is the multiplicative bias m consistent with zero for each method?
If not here, the issue is in the metacal/measurement machinery itself, not
sample selection or noise -- see CLAUDE.md and this repo's calibration-test
history for the COSMOS-test's own diagnosis (which pointed at sample
resolvedness, not an implementation bug).

The galaxy size (half_light_radius) is solved once, up front, via bisection
on resolution=0.7 (ola's own R2 convention, sigw=2.0 pixels, matching their
test_bias.py's own defaults), not hardcoded.

Reuses ngmix_e_and_R11/ola_e_and_R11/make_ola_pipeline/HAVE_OLA from
run_calibration_test.py (same measurement code, only the galaxy/PSF/noise
generation differs here).

Run: python run_calibration_test_simple.py [--n-gals 100] [--out ...]
"""
import argparse
import time

import numpy as np
import jax.numpy as jnp
import galsim
from scipy.optimize import brentq

import autometacal
from autometacal.python import gaussfit
from autometacal.python import ksb as ksb_module

import run_calibration_test as base

PIXEL_SCALE = 0.1
BOXSIZE = 64
SIGW = 2.0  # ola's own default (matches their PIXEL_SCALE=0.1 test config)
TARGET_RESOLUTION = 0.7
RECONV_PSF_DILATION = 1.02
FFT_SIZE = 256


def make_psf():
  if not base.HAVE_OLA:
    raise RuntimeError("ola not importable -- this script needs it for make_optical_psf")
  from ola.pipeline.detector import make_optical_psf
  return make_optical_psf()


def solve_hlr_for_resolution(psf, target_resolution, pixel_scale=PIXEL_SCALE, boxsize=BOXSIZE, sigw=SIGW):
  psf_image = psf.drawImage(nx=boxsize, ny=boxsize, scale=pixel_scale).array.astype('float32')
  Tpsf, _ = ksb_module.size_moments(jnp.asarray(psf_image), pixel_scale, sigw=sigw)
  Tpsf = float(Tpsf)

  def resolution_for_hlr(hlr):
    gal = galsim.Exponential(half_light_radius=hlr, flux=1.0e6)
    obj = galsim.Convolve([gal, psf])
    img = obj.drawImage(nx=boxsize, ny=boxsize, scale=pixel_scale).array.astype('float32')
    Tgal, _ = ksb_module.size_moments(jnp.asarray(img), pixel_scale, sigw=sigw)
    return float(ksb_module.source_resolution(Tgal, Tpsf)) - target_resolution

  hlr = brentq(resolution_for_hlr, 0.05, 3.0, xtol=1e-5)
  return hlr, Tpsf


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--n-gals', type=int, default=100)
  p.add_argument('--flux', type=float, default=1.0e6)
  p.add_argument('--noise-std', type=float, default=1.0)
  p.add_argument('--g1-true', type=float, default=0.02)
  p.add_argument('--mcal-step', type=float, default=0.01)
  p.add_argument('--seed', type=int, default=31415)
  p.add_argument('--out', type=str, default='results/calibration_test_simple_results.npz')
  args = p.parse_args()

  psf = make_psf()
  hlr, Tpsf = solve_hlr_for_resolution(psf, TARGET_RESOLUTION)
  psf_image = psf.drawImage(nx=BOXSIZE, ny=BOXSIZE, scale=PIXEL_SCALE).array.astype('float32')
  reconv_psf_image = np.asarray(
      autometacal.galflow.dilate(psf_image, RECONV_PSF_DILATION, scale=PIXEL_SCALE)
  )
  ksb_psf_image = np.asarray(autometacal.galflow.dilate(
      reconv_psf_image, autometacal.python.metacal._reconv_psf_dilation, scale=PIXEL_SCALE,
  ))

  print(f"psf HLR={psf.calculateHLR():.4f}\"  Tpsf={Tpsf:.5f} arcsec^2")
  print(f"solved galaxy hlr={hlr:.5f}\" for target resolution={TARGET_RESOLUTION}")
  print(f"n_gals={args.n_gals}  flux={args.flux}  noise_std={args.noise_std}  g1_true={args.g1_true}")

  gal0 = galsim.Exponential(half_light_radius=hlr, flux=args.flux)

  def render(g1, noise):
    sheared = gal0.shear(g1=g1, g2=0.0)
    obj = galsim.Convolve([sheared, psf])
    img = obj.drawImage(nx=BOXSIZE, ny=BOXSIZE, scale=PIXEL_SCALE).array.astype('float32')
    return img + noise

  snr0 = np.sqrt(np.sum(
      gal0.drawImage(nx=BOXSIZE, ny=BOXSIZE, scale=PIXEL_SCALE).array.astype('float64') ** 2
  )) / args.noise_std
  print(f"approx noiseless-flux SNR: {snr0:.1f}")

  method_ksb_g = lambda img: ksb_module.calibrated_g(
      img, jnp.asarray(ksb_psf_image), scale=PIXEL_SCALE, sigw=SIGW)
  method_fit = lambda img: gaussfit.get_fit_ellipticities(img)

  ola_pipeline = base.make_ola_pipeline(
      type('Cfg', (), {'pixel_scale': PIXEL_SCALE, 'reconv_psf_dilation': RECONV_PSF_DILATION})(),
      args.mcal_step,
  ) if base.HAVE_OLA else None

  rng = np.random.RandomState(args.seed)
  keys = ['ngmix', 'am_moments', 'am_fitting', 'ola']
  branches = ['unsheared', 'sheared']
  data = {f'{m}_{b}_e1': np.full(args.n_gals, np.nan) for m in keys for b in branches}
  data.update({f'{m}_{b}_e2': np.full(args.n_gals, np.nan) for m in keys for b in branches})
  data.update({f'{m}_{b}_R11': np.full(args.n_gals, np.nan) for m in keys for b in branches})
  data.update({f'{m}_{b}_R22': np.full(args.n_gals, np.nan) for m in keys for b in branches})

  buf = {b: {'gal': [], 'noise': []} for b in branches}
  am_e = {m: {b: np.full(args.n_gals, np.nan) for b in branches} for m in ['am_moments', 'am_fitting']}
  am_R11 = {m: {b: np.full(args.n_gals, np.nan) for b in branches} for m in ['am_moments', 'am_fitting']}
  am_R22 = {m: {b: np.full(args.n_gals, np.nan) for b in branches} for m in ['am_moments', 'am_fitting']}

  t0 = time.time()
  for i in range(args.n_gals):
    noise = rng.normal(scale=args.noise_std, size=(BOXSIZE, BOXSIZE)).astype('float32')
    obs_unsheared = render(0.0, noise)
    obs_sheared = render(args.g1_true, noise)

    for branch, obs_image in [('unsheared', obs_unsheared), ('sheared', obs_sheared)]:
      e1_ng, e2_ng, R11_ng, R22_ng = base.ngmix_e_and_R11(
          obs_image, psf_image, args.noise_std, PIXEL_SCALE, args.mcal_step, 2 * hlr, rng)
      data[f'ngmix_{branch}_e1'][i] = e1_ng
      data[f'ngmix_{branch}_e2'][i] = e2_ng
      data[f'ngmix_{branch}_R11'][i] = R11_ng
      data[f'ngmix_{branch}_R22'][i] = R22_ng

      if base.HAVE_OLA:
        e1_ola, e2_ola, R11_ola, R22_ola = base.ola_e_and_R11(
            ola_pipeline, obs_image, psf_image, args.noise_std, PIXEL_SCALE, rng)
        data[f'ola_{branch}_e1'][i] = e1_ola
        data[f'ola_{branch}_e2'][i] = e2_ola
        data[f'ola_{branch}_R11'][i] = R11_ola
        data[f'ola_{branch}_R22'][i] = R22_ola

      extra_noise = rng.normal(scale=args.noise_std, size=(BOXSIZE, BOXSIZE)).astype('float32')
      e_gm, R_gm, _, _, _ = autometacal.get_metacal_response(
          jnp.asarray(obs_image), jnp.asarray(psf_image), jnp.asarray(reconv_psf_image),
          jnp.asarray(extra_noise), method_ksb_g, scale=PIXEL_SCALE,
      )
      e_fit, R_fit, _, _, _ = autometacal.get_metacal_response(
          jnp.asarray(obs_image), jnp.asarray(psf_image), jnp.asarray(reconv_psf_image),
          jnp.asarray(extra_noise), method_fit, scale=PIXEL_SCALE,
      )
      data[f'am_moments_{branch}_e1'][i] = float(e_gm[0])
      data[f'am_moments_{branch}_e2'][i] = float(e_gm[1])
      data[f'am_moments_{branch}_R11'][i] = float(R_gm[0, 0])
      data[f'am_moments_{branch}_R22'][i] = float(R_gm[1, 1])
      data[f'am_fitting_{branch}_e1'][i] = float(e_fit[0])
      data[f'am_fitting_{branch}_e2'][i] = float(e_fit[1])
      data[f'am_fitting_{branch}_R11'][i] = float(R_fit[0, 0])
      data[f'am_fitting_{branch}_R22'][i] = float(R_fit[1, 1])

    if (i + 1) % 10 == 0:
      elapsed = time.time() - t0
      print(f"  {i+1}/{args.n_gals} done, elapsed={elapsed:.0f}s, rate={(i+1)/elapsed:.2f} gal/s")

  args_dict = vars(args)
  args_dict['pixel_scale'] = PIXEL_SCALE
  args_dict['hlr'] = hlr
  args_dict['Tpsf'] = Tpsf
  args_dict['target_resolution'] = TARGET_RESOLUTION
  args_dict['snr0'] = snr0
  np.savez(args.out, **data, n_done=args.n_gals, args=args_dict)
  print(f"DONE. total time={time.time()-t0:.0f}s. wrote {args.out}")


if __name__ == '__main__':
  main()
