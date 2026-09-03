"""
Simulate single-galaxy stamps from the GalSim COSMOS parametric catalog at a
handful of small constant shear values, run them through autometacal's
autodiff metacal response, and report the calibrated shear per true-shear
value -- a first end-to-end test of the JAX/jax_galsim migration.

Config defaults mirror what was already used elsewhere in this repo before
the migration: `datasets/cfis.py` (retired in Phase 4) for the COSMOS
catalog + imaging setup (pixel scale, stamp size, PSF, sky level, magnitude
zeropoint), and `tests/test_metacal_ngmix.py` for the moments weight
function scale. See plans.md for the full migration history.

Performance note: `autometacal.get_metacal_response` currently runs
unjitted at ~5s/stamp on this machine -- `jax.jit` fails outright on it
(jax_galsim's automatic FFT-size selection reads concrete values from the
traced input images, which isn't jit-compatible; a separate, previously
undocumented limitation from the known vmap-over-shears issue in plans.md).
So `n_gals_per_shear` is deliberately modest by default; scale it up
consciously once you've confirmed the pipeline behaves as expected.
"""
import argparse
import dataclasses
import os
import time
import warnings

# GPU memory hygiene: see pujol_test.py's identical comment / CLAUDE.md's
# GPU notes -- must be set before `import autometacal` (which imports jax).
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import galsim
import autometacal

# SAFETY: GalSim's gsparams.maximum_fft_size does NOT actually prevent large
# FFT allocations -- reading GSObject.drawFFT_makeKImage's source directly:
# it only *warns* (GalSimFFTSizeWarning) when the required size exceeds the
# cap, then proceeds to allocate the (potentially huge) array anyway. A
# handful of pathological COSMOS parametric fits want FFTs up to ~44000^2
# complex128 (~43GB) -- verified directly in pujol_test.py's development,
# and it came within seconds of exhausting this machine's 30GB RAM before
# being caught and killed. The warning fires BEFORE the allocation, so
# converting it into a raised exception (standard `warnings` mechanism)
# stops the allocation from ever happening; the per-galaxy try/except below
# then just skips that object.
warnings.filterwarnings('error', category=galsim.errors.GalSimFFTSizeWarning)


@dataclasses.dataclass
class Config:
  # imaging
  pixel_scale: float = 0.187     # arcsec/pixel (datasets/cfis.py)
  stamp_size: int = 51           # pixels (datasets/cfis.py)
  # relative exposure multiplier (scales both galaxy flux and sky counts,
  # so SNR ~ sqrt(exptime) -- higher = deeper/less noisy). exptime=1.0
  # reproduces datasets/cfis.py's original flux/sky calibration verbatim,
  # but that alone gives total-flux SNR ~3 for a typical resolved galaxy --
  # far too low for individual shape measurements to be anything but
  # noise-dominated (verified directly: same galaxy, same true shear,
  # repeated noise draws gave wildly different sign/magnitude ellipticity).
  # 50.0 (SNR ~20 for a typical resolved galaxy) was the smallest value
  # found to give stable, correctly-signed per-galaxy measurements.
  exptime: float = 50.0
  sky_level: float = 400.0       # ADU/pixel at exptime=1 (datasets/cfis.py)
  gain: float = 1.0              # e-/ADU
  read_noise: float = 0.0        # e-/pixel

  # psf
  psf_type: str = 'Kolmogorov'   # 'Kolmogorov', 'Moffat', or 'Gaussian'
  psf_fwhm: float = 0.7          # arcsec (datasets/cfis.py)
  psf_beta: float = 4.8          # only used if psf_type == 'Moffat' (experiment/data_generator.py)
  psf_e1: float = 0.0            # datasets/cfis.py
  psf_e2: float = 0.0            # circular PSF by default (datasets/cfis.py had 0.025; simplified for now)
  # reconvolution target = true PSF dilated by this factor (standard metacal
  # practice; see pujol_test.py's identical comment)
  reconv_psf_dilation: float = 1.02

  # galaxies
  cosmos_sample: str = '25.2'    # datasets/cfis.py
  mag_zp: float = 32.0           # datasets/cfis.py

  # shear grid + statistics
  shear_values: tuple = (
      (0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02),
  )
  n_gals_per_shear: int = 50

  # galaxy selection: the raw COSMOS parametric catalog is dominated by small,
  # faint (often high-z) galaxies -- median HLR ~0.2" at this pixel
  # scale/PSF, well below the 0.7" PSF FWHM, i.e. essentially unresolved and
  # unmeasurable for shear (verified directly: an unfiltered draw gave a
  # near-constant measured ellipticity dominated by the PSF's own shape,
  # not the applied shear). Real weak-lensing pipelines always apply a
  # resolution cut before shape measurement; min_hlr does the same here,
  # using the catalog's own precomputed `hlr` column (cheap, no per-object
  # calculateHLR() FFT needed).
  min_hlr: float = 0.3           # arcsec
  shape_noise_cancel: bool = True  # pair each draw with a 90deg-rotated copy; see simulate_and_measure

  # ellipticity measurement
  weight_fwhm: float = 1.2       # gaussmom weight fwhm, arcsec (tests/test_metacal_ngmix.py)

  seed: int = 31415


# A handful of raw COSMOS parametric-catalog fits have pathologically large
# profile components (e.g. an oversized de Vaucouleurs bulge) that make real
# GalSim want a huge internal FFT to render -- one was observed requesting a
# 65536x65536 FFT (96GB). Capping maximum_fft_size makes GalSim raise a
# catchable GalSimFFTSizeError for these instead of hanging/OOMing; such
# objects are then just skipped by the per-galaxy try/except below.
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


def draw_noisy(obj, cfg, rng):
  image = obj.drawImage(nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale)
  noise = galsim.CCDNoise(
      rng, sky_level=cfg.sky_level * cfg.exptime, gain=cfg.gain, read_noise=cfg.read_noise,
  )
  image.addNoise(noise)
  return image.array.astype('float32')


def make_noise_stamp(cfg, rng):
  """ An independent noise realization at the same level as the galaxy
  stamp's own noise, for the metacal fixnoise trick. """
  blank = galsim.Image(cfg.stamp_size, cfg.stamp_size, scale=cfg.pixel_scale, init_value=0.0)
  noise = galsim.CCDNoise(
      rng, sky_level=cfg.sky_level * cfg.exptime, gain=cfg.gain, read_noise=cfg.read_noise,
  )
  blank.addNoise(noise)
  return blank.array.astype('float32')


def select_resolved_indices(cat, cfg):
  """ Indices of catalog objects with `hlr` (overall Sersic-equivalent
  half-light radius, first of the 3 `hlr` columns) above `cfg.min_hlr`. """
  hlr = cat.param_cat['hlr'][cat.orig_index, 0]
  idxs = np.where(hlr >= cfg.min_hlr)[0]
  print(f"Resolution cut (hlr >= {cfg.min_hlr}\"): {len(idxs)}/{cat.nobjects} "
        f"catalog objects pass ({100*len(idxs)/cat.nobjects:.1f}%).")
  return idxs


def simulate_and_measure(cfg):
  cat = galsim.COSMOSCatalog(sample=cfg.cosmos_sample)
  rng = galsim.BaseDeviate(cfg.seed)
  np_rng = np.random.RandomState(cfg.seed)
  resolved_idxs = select_resolved_indices(cat, cfg)

  psf = make_psf(cfg)
  psf_image = psf.drawImage(nx=cfg.stamp_size, ny=cfg.stamp_size, scale=cfg.pixel_scale).array
  psf_image = psf_image.astype('float32')
  # dilate at the pixel/interpolated-image level (Lanczos(11), matching
  # autometacal's own convention) -- see pujol_test.py's identical comment
  # for why (a real PSF model is a pixelized image, not an analytic
  # profile, so this is what the actual pipeline does).
  reconv_psf_image = np.asarray(
      autometacal.galflow.dilate(psf_image, cfg.reconv_psf_dilation, scale=cfg.pixel_scale)
  )

  def method(image):
    return autometacal.get_moment_ellipticities(image, scale=cfg.pixel_scale, fwhm=cfg.weight_fwhm)

  # Shape-noise cancellation: for each draw, measure the galaxy at its
  # random orientation AND rotated 90deg, both sheared identically, and
  # average the pair. A 90deg rotation negates a spin-2 quantity's intrinsic
  # ellipticity (e' = e*exp(i*2*90deg) = -e) while the shear response is
  # (to first order) orientation-independent, so averaging the pair cancels
  # most of the intrinsic-shape scatter that otherwise swamps a small
  # applied shear at modest N (verified directly: without this, n=15/bin
  # gave wrong-signed g1 recovery despite a clean single-galaxy control
  # test -- likely just a few outlier intrinsic shapes dominating a small
  # unweighted mean). Costs 2x metacal-response calls per sample.
  calls_per_sample = 2 if cfg.shape_noise_cancel else 1
  n_total = len(cfg.shear_values) * cfg.n_gals_per_shear
  print(f"Simulating {n_total} galaxy samples ({len(cfg.shear_values)} shear values x "
        f"{cfg.n_gals_per_shear} each), {calls_per_sample}x metacal-response call(s) per "
        f"sample. At ~5s/call (unjitted get_metacal_response) that's roughly "
        f"{n_total * calls_per_sample * 5 / 60:.1f} minutes.")

  results = {'g_true': [], 'idx': [], 'e': [], 'R': [], 'Rpsf': [], 'epsf': [], 'Repsf': [], 'flags': []}

  def measure(gal, g1_true, g2_true):
    sheared = gal.shear(g1=g1_true, g2=g2_true)
    obj = galsim.Convolve([sheared, psf])
    gal_image = draw_noisy(obj, cfg, rng)
    noise_image = make_noise_stamp(cfg, rng)
    return autometacal.get_metacal_response(gal_image, psf_image, reconv_psf_image, noise_image, method)

  t_start = time.time()
  n_done = 0
  for g1_true, g2_true in cfg.shear_values:
    for _ in range(cfg.n_gals_per_shear):
      idx = int(resolved_idxs[np_rng.randint(len(resolved_idxs))])
      try:
        gal0 = make_galaxy(cat, idx, cfg)
        outs = [measure(gal0, g1_true, g2_true)]
        if cfg.shape_noise_cancel:
          outs.append(measure(gal0.rotate(90 * galsim.degrees), g1_true, g2_true))

        e, R, Rpsf, epsf, Repsf = (
            np.mean([np.asarray(o[k]) for o in outs], axis=0) for k in range(5)
        )
        flag = int(np.any(np.isnan(e)) or np.any(np.isnan(R)))
      except Exception as exc:  # pragma: no cover -- a handful of pathological catalog entries is expected
        print(f"  [idx={idx}] measurement failed: {type(exc).__name__}: {exc}")
        e, R, Rpsf, epsf, Repsf, flag = None, None, None, None, None, 1

      results['g_true'].append((g1_true, g2_true))
      results['idx'].append(idx)
      results['e'].append(e)
      results['R'].append(R)
      results['Rpsf'].append(Rpsf)
      results['epsf'].append(epsf)
      results['Repsf'].append(Repsf)
      results['flags'].append(flag)

      n_done += 1
      if n_done % 10 == 0 or n_done == n_total:
        elapsed = time.time() - t_start
        rate = n_done / elapsed
        eta = (n_total - n_done) / rate if rate > 0 else float('nan')
        print(f"  {n_done}/{n_total}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

  return results


def summarize(results):
  g_true = np.array(results['g_true'])
  flags = np.array(results['flags'])
  ok = flags == 0

  print()
  print(f"{'g1_true':>8} {'g2_true':>8} {'n_ok':>6} {'g1_cal':>10} {'g2_cal':>10}")
  rows = []
  for g1_true, g2_true in sorted(set(map(tuple, g_true))):
    sel = ok & (g_true[:, 0] == g1_true) & (g_true[:, 1] == g2_true)
    if sel.sum() == 0:
      print(f"{g1_true:8.3f} {g2_true:8.3f} {0:6d} {'--':>10} {'--':>10}")
      continue
    e = np.array([results['e'][i] for i in np.where(sel)[0]])
    R = np.array([results['R'][i] for i in np.where(sel)[0]])
    mean_e = e.mean(axis=0)
    mean_R = R.mean(axis=0)
    g_cal = np.linalg.solve(mean_R, mean_e)
    rows.append((g1_true, g2_true, sel.sum(), g_cal[0], g_cal[1]))
    print(f"{g1_true:8.3f} {g2_true:8.3f} {sel.sum():6d} {g_cal[0]:10.4f} {g_cal[1]:10.4f}")

  rows = np.array(rows)

  def fit_mc(g_true_col, g_cal_col, mask):
    x, y = rows[mask, g_true_col], rows[mask, 2 + g_true_col]
    if len(np.unique(x)) < 2:
      return None
    slope, intercept = np.polyfit(x, y, 1)
    return slope - 1, intercept

  print()
  g1_mask = rows[:, 1] == 0.0
  g2_mask = rows[:, 0] == 0.0
  fit1 = fit_mc(0, 0, g1_mask)
  fit2 = fit_mc(1, 1, g2_mask)
  if fit1:
    print(f"g1: m1 = {fit1[0]:+.4f}   c1 = {fit1[1]:+.5f}")
  if fit2:
    print(f"g2: m2 = {fit2[0]:+.4f}   c2 = {fit2[1]:+.5f}")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  defaults = Config()
  p.add_argument('--pixel-scale', type=float, default=defaults.pixel_scale)
  p.add_argument('--stamp-size', type=int, default=defaults.stamp_size)
  p.add_argument('--exptime', type=float, default=defaults.exptime)
  p.add_argument('--sky-level', type=float, default=defaults.sky_level)
  p.add_argument('--gain', type=float, default=defaults.gain)
  p.add_argument('--read-noise', type=float, default=defaults.read_noise)
  p.add_argument('--psf-type', choices=['Kolmogorov', 'Moffat', 'Gaussian'], default=defaults.psf_type)
  p.add_argument('--psf-fwhm', type=float, default=defaults.psf_fwhm)
  p.add_argument('--psf-beta', type=float, default=defaults.psf_beta)
  p.add_argument('--psf-e1', type=float, default=defaults.psf_e1)
  p.add_argument('--psf-e2', type=float, default=defaults.psf_e2)
  p.add_argument('--reconv-psf-dilation', type=float, default=defaults.reconv_psf_dilation)
  p.add_argument('--cosmos-sample', default=defaults.cosmos_sample)
  p.add_argument('--mag-zp', type=float, default=defaults.mag_zp)
  p.add_argument('--n-gals-per-shear', type=int, default=defaults.n_gals_per_shear)
  p.add_argument('--min-hlr', type=float, default=defaults.min_hlr)
  p.add_argument('--no-shape-noise-cancel', dest='shape_noise_cancel', action='store_false',
                  default=defaults.shape_noise_cancel)
  p.add_argument('--weight-fwhm', type=float, default=defaults.weight_fwhm)
  p.add_argument('--seed', type=int, default=defaults.seed)
  p.add_argument('--output', default='cosmos_calibration_results.npz')
  return p.parse_args()


def main():
  args = parse_args()
  cfg = Config(
      pixel_scale=args.pixel_scale, stamp_size=args.stamp_size, exptime=args.exptime,
      sky_level=args.sky_level, gain=args.gain, read_noise=args.read_noise,
      psf_type=args.psf_type, psf_fwhm=args.psf_fwhm, psf_beta=args.psf_beta,
      psf_e1=args.psf_e1, psf_e2=args.psf_e2, reconv_psf_dilation=args.reconv_psf_dilation,
      cosmos_sample=args.cosmos_sample,
      mag_zp=args.mag_zp, n_gals_per_shear=args.n_gals_per_shear, min_hlr=args.min_hlr,
      shape_noise_cancel=args.shape_noise_cancel,
      weight_fwhm=args.weight_fwhm, seed=args.seed,
  )
  results = simulate_and_measure(cfg)
  summarize(results)

  np.savez(
      args.output,
      g_true=np.array(results['g_true']),
      idx=np.array(results['idx']),
      flags=np.array(results['flags']),
      e=np.array([x if x is not None else [np.nan, np.nan] for x in results['e']]),
      R=np.array([x if x is not None else np.full((2, 2), np.nan) for x in results['R']]),
      Rpsf=np.array([x if x is not None else np.full((2, 2), np.nan) for x in results['Rpsf']]),
      epsf=np.array([x if x is not None else [np.nan, np.nan] for x in results['epsf']]),
      Repsf=np.array([x if x is not None else np.full((2, 2), np.nan) for x in results['Repsf']]),
  )
  print(f"\nSaved per-galaxy results to {args.output}")


if __name__ == '__main__':
  main()
