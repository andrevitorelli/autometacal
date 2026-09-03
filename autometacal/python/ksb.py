"""JAX port of ola's `metacal_package/moments.py` (KSB moments, resolution,
S/N, and the KSB PSF/shear polarisability correction).

Two pieces, kept deliberately separate:

- `moments`/`size_moments`: fixed-centroid, Gaussian-windowed moments plus the
  KSB susceptibility tensors (`Psm`, `Psh`). A plain differentiable function
  of the input image -- safe to push `jax.grad`/`jax.jacrev` through, e.g. as
  the `method` callable for `autometacal.metacal.get_metacal_response`. This
  is ola's `moments()` with its data-dependent centroid-recentering while-loop
  removed (the centroid defaults to the image center, or can be supplied
  externally, e.g. from `admom` below) -- that loop is what would make ola's
  original version non-differentiable.
- `admom`: the actual adaptive-moments algorithm (Hirata & Seljak 2003 --
  the same algorithm both `galsim.hsm.FindAdaptiveMom` and `ngmix.admom`
  implement), reimplemented here in JAX (via `jax.lax.while_loop`) purely so
  `correct_ksb`'s S/N estimate doesn't need real GalSim's HSM (C++, not
  JAX/GPU-friendly). This one is genuinely iterative by construction (that's
  what "adaptive" means: the weight function's own covariance is updated each
  step until it matches the object's), so it is not meant to be
  differentiated through -- nothing in this repo's metacal response calls it.

Author: andrevitorelli (JAX port); original: ola/metacal_package/moments.py
(Henk Hoekstra-derived KSB code) and ngmix.admom (Hirata & Seljak 2003
adaptive moments).
"""
import collections

import jax.numpy as jnp
from jax import lax

from .galflow import dtype_real

# admom flag codes (mirrors ngmix.flags' role, not its exact values)
ADMOM_OK = 0
ADMOM_LOW_DET = 1
ADMOM_NONPOS_FLUX = 2
ADMOM_CEN_SHIFT = 3
ADMOM_NONPOS_SIZE = 4
ADMOM_MAXITER = 5

_LOW_DETVAL = 1.0e-8


def _weighted_sum(image, xvec, yvec):
  """`sum_{i,j} yvec[i] * image[i,j] * xvec[j]`, ola's `image_weighted_sum`."""
  return yvec @ (image @ xvec)


def moments(image, sigw=2.0, xc=None, yc=None, oversampling=1.0):
  """Gaussian-windowed second moments + KSB susceptibility tensors.

  JAX port of ola's `metacal_package/moments.py::moments`, with the
  centroid fixed (no iterative recentering -- see module docstring) so this
  stays a plain differentiable function of `image`.

  Args:
    image: array (nx, ny), square stamp
    sigw: float
      weight function sigma, in pixels (before `oversampling`)
    xc, yc: float, optional
      weight center, in pixels; defaults to the stamp center
    oversampling: float
      matches ola's `oversampling` (used for a PSF stamp drawn at finer than
      native pixel scale; scales `sigw` up and `Psm`/`Psh` by
      `oversampling**2`)

  Returns:
    dict with keys `xc`, `yc`, `e` (array (2,), distortion e1/e2), `T`
    (in pixel^2), `Psm`, `Psh` (each array (2, 2), KSB PSF-smearing and
    shear polarisability tensors)
  """
  sigw = sigw * oversampling
  nx, ny = image.shape
  x = jnp.arange(nx, dtype=dtype_real)
  y = jnp.arange(ny, dtype=dtype_real)
  # nx // 2, not nx / 2.0 (ola's own pre-iteration initial guess): GalSim's
  # drawImage centers an object at pixel index nx // 2 for odd nx (verified
  # directly against a centered Gaussian's peak pixel), matching
  # `gaussmom.py`'s own `centre_x/y` convention. ola's while-loop corrects
  # its coarser initial guess away from this; without that loop here, the
  # default needs to be right from the start.
  if xc is None:
    xc = nx // 2
  if yc is None:
    yc = ny // 2
  sigmasq = sigw ** 2

  xp = x - xc
  yp = y - yc
  xw = jnp.exp(-xp ** 2 / 2.0 / sigmasq)
  yw = jnp.exp(-yp ** 2 / 2.0 / sigmasq)
  w_factor = 1.0 / (2.0 * jnp.pi * sigmasq)

  xsq = xp ** 2
  ysq = yp ** 2

  q11 = _weighted_sum(image, xw * xsq, yw) * w_factor
  q12 = _weighted_sum(image, xw * xp, yw * yp) * w_factor
  q22 = _weighted_sum(image, xw, yw * ysq) * w_factor
  stamp_sum = _weighted_sum(image, xw, yw)
  T = (q11 + q22) / (stamp_sum * w_factor)

  denom = q11 + q22
  e1 = (q11 - q22) / denom
  e2 = (2.0 * q12) / denom

  # derivatives of the Gaussian weight w.r.t. sigmasq, precomputed as factors
  w_factor_p = -0.5 / sigmasq * w_factor
  w_factor_pp = 0.25 / sigmasq ** 2 * w_factor

  xcube, ycube = xp * xsq, yp * ysq
  xquad, yquad = xsq ** 2, ysq ** 2

  xp_w, xsq_w, xcube_w, xquad_w = xp * xw, xsq * xw, xcube * xw, xquad * xw
  yp_w, ysq_w, ycube_w, yquad_w = yp * yw, ysq * yw, ycube * yw, yquad * yw

  Sqsum_x = _weighted_sum(image, xsq_w, yw)
  Sqsum_y = _weighted_sum(image, xw, ysq_w)
  DD_sum = Sqsum_x + Sqsum_y
  DD1_sum = Sqsum_x - Sqsum_y
  DD2_sum = 2.0 * _weighted_sum(image, xp_w, yp_w)

  Quadsum_x = _weighted_sum(image, xquad_w, yw)
  Quadsum_y = _weighted_sum(image, xw, yquad_w)
  Cubesum_x_yp = _weighted_sum(image, xcube_w, yp_w)
  Cubesum_y_xp = _weighted_sum(image, xp_w, ycube_w)
  Squaresum = _weighted_sum(image, xsq_w, ysq_w)
  DD1sq_sum = Quadsum_x - 2.0 * Squaresum + Quadsum_y
  DD2sq_sum = 4.0 * Squaresum
  DD1_DD2_sum = 2.0 * (Cubesum_x_yp - Cubesum_y_xp)
  DD_DD1_sum = Quadsum_x - Quadsum_y
  DD_DD2_sum = 2.0 * (Cubesum_x_yp + Cubesum_y_xp)

  # formula 4-4 (Hoekstra KSB notes / ola)
  Xsm11 = 2.0 * w_factor * stamp_sum + 4.0 * w_factor_p * DD_sum + 2.0 * w_factor_pp * DD1sq_sum
  Xsm22 = 2.0 * w_factor * stamp_sum + 4.0 * w_factor_p * DD_sum + 2.0 * w_factor_pp * DD2sq_sum
  Xsm12 = 2.0 * w_factor_pp * DD1_DD2_sum

  # formula 5-3
  Xsh11 = 2.0 * w_factor * DD_sum + 2.0 * w_factor_p * DD1sq_sum
  Xsh22 = 2.0 * w_factor * DD_sum + 2.0 * w_factor_p * DD2sq_sum
  Xsh12 = 2.0 * w_factor_p * DD1_DD2_sum

  # formula 4-5
  em1 = (4.0 * w_factor_p * DD1_sum + 2.0 * w_factor_pp * DD_DD1_sum) / denom
  em2 = (4.0 * w_factor_p * DD2_sum + 2.0 * w_factor_pp * DD_DD2_sum) / denom

  # formula 5-4
  eh1 = 2.0 * w_factor_p * DD_DD1_sum / denom + 2.0 * e1
  eh2 = 2.0 * w_factor_p * DD_DD2_sum / denom + 2.0 * e2

  # formula 4-3
  psm11 = Xsm11 / denom - e1 * em1
  psm22 = Xsm22 / denom - e2 * em2
  psm12 = Xsm12 / denom - 0.5 * (e1 * em2 + e2 * em1)

  # formula 5-2
  psh11 = Xsh11 / denom - e1 * eh1
  psh22 = Xsh22 / denom - e2 * eh2
  psh12 = Xsh12 / denom - 0.5 * (e1 * eh2 + e2 * eh1)

  Psm = jnp.array([[psm11, psm12], [psm12, psm22]], dtype=dtype_real) * oversampling ** 2
  Psh = jnp.array([[psh11, psh12], [psh12, psh22]], dtype=dtype_real) * oversampling ** 2

  return {
      'xc': xc, 'yc': yc,
      'e': jnp.array([e1, e2], dtype=dtype_real),
      'T': T,
      'Psm': Psm,
      'Psh': Psh,
  }


def size_moments(image, scale, sigw=2.0, oversampling=1.0):
  """`moments`, with `T` converted from pixel^2 to physical (`scale`^2)
  units -- JAX port of ola's `size_moments` (`image.wcs.pixelArea()` ->
  `scale**2`, since this repo only ever uses square, non-rotated pixels).

  Returns:
    T: float, in `scale`^2 units
    moms: dict, see `moments`
  """
  moms = moments(image, sigw=sigw, oversampling=oversampling)
  T = moms['T'] * scale ** 2
  return T, moms


def sigma_sky(image, edge_width=2):
  """Sky-noise estimate from the stamp's edge pixels via MAD, JAX port of
  ola's `sigma_sky` (`1.4826 * median_abs_deviation`, robust to any object
  flux still present near the edges).
  """
  stripe1 = image[:edge_width, :]
  stripe2 = image[-edge_width:, :]
  stripe3 = image[edge_width:-edge_width, :edge_width]
  stripe4 = image[edge_width:-edge_width, -edge_width:]
  stripe = jnp.concatenate(
      [stripe1.ravel(), stripe2.ravel(), stripe3.ravel(), stripe4.ravel()]
  )
  med = jnp.median(stripe)
  mad = jnp.median(jnp.abs(stripe - med))
  return 1.4826 * mad


def source_resolution(Tobs, Tpsf):
  """`R_2`-type resolution factor, JAX port of ola's `source_resolution`.
  `nan` (not an error) when the object isn't resolved (`Tobs <= Tpsf`),
  matching ola's own convention.
  """
  return jnp.where(Tobs > Tpsf, 1.0 - Tpsf / Tobs, jnp.nan)


def ellipticity_error(flux, resolution, sigmaobs, sigmasky):
  """Per-component ellipticity error, JAX port of ola's `ellipticity_error`
  (Tewes et al. 2019-style)."""
  valid = (flux > 0) & (resolution > 0) & (sigmaobs > 0) & (sigmasky > 0)
  val = jnp.sqrt(4.0 * jnp.pi) * sigmasky * sigmaobs / (resolution * flux)
  return jnp.where(valid, val, jnp.nan)


_AdmomState = collections.namedtuple(
    '_AdmomState', ['i', 'irr', 'icc', 'irc', 'row', 'col', 'e1old', 'e2old', 'Told', 'flags']
)


def admom(image, guess_T, row0=None, col0=None, maxiter=200, shiftmax=5.0, etol=1e-5, Ttol=1e-3):
  """Adaptive moments (Hirata & Seljak 2003), JAX port of the algorithm
  behind both `galsim.hsm.FindAdaptiveMom` and `ngmix.admom` -- used as a
  JAX/GPU-friendly, dependency-free stand-in for GalSim's (C++, eager-only)
  HSM in `correct_ksb`'s S/N estimate. See module docstring for why this is
  iterative (and not meant to be differentiated through) while `moments`
  above is not.

  Args:
    image: array (nx, ny), square stamp
    guess_T: float, initial guess for T = Irr+Icc (pixel^2)
    row0, col0: float, optional initial centroid guess (pixels); defaults to
      the stamp center
    maxiter, shiftmax, etol, Ttol: see `ngmix.admom.run_admom` (same
      defaults, same meaning: `shiftmax` in pixels, `etol`/`Ttol` are the
      absolute/relative convergence tolerances on e1,e2/T)

  Returns:
    dict with keys `flux` (total flux, same convention as ngmix's own
    `flux`/GalSim's `moments_amp` -- approximately the object's true flux
    for a Gaussian-like light profile), `sigma` (`(Irr*Icc-Irc**2)**0.25`,
    matching `galsim.hsm`'s `moments_sigma`), `T`, `e` (array (2,)), `row`,
    `col`, `flags` (0 = converged OK, see `ADMOM_*` constants), `numiter`
  """
  nx, ny = image.shape
  yy, xx = jnp.mgrid[0:nx, 0:ny]
  yy = yy.astype(dtype_real)
  xx = xx.astype(dtype_real)
  image = image.astype(dtype_real)

  if row0 is None:
    row0 = nx // 2
  if col0 is None:
    col0 = ny // 2
  row0 = jnp.asarray(row0, dtype=dtype_real)
  col0 = jnp.asarray(col0, dtype=dtype_real)
  row_orig, col_orig = row0, col0

  T0 = jnp.asarray(guess_T, dtype=dtype_real)
  irr0 = T0 / 2.0
  icc0 = T0 / 2.0
  irc0 = jnp.zeros((), dtype=dtype_real)

  def eval_weight(irr, icc, irc, row, col):
    det = irr * icc - irc ** 2
    det = jnp.where(det > _LOW_DETVAL, det, _LOW_DETVAL)
    dcc = icc / det
    drr = irr / det
    drc = irc / det
    norm = 1.0 / (2.0 * jnp.pi * jnp.sqrt(det))
    vmod = yy - row
    umod = xx - col
    chi2 = dcc * vmod ** 2 + drr * umod ** 2 - 2.0 * drc * vmod * umod
    w = norm * jnp.exp(-0.5 * chi2)
    return w, det, norm, vmod, umod

  def cond_fn(state):
    return (state.flags == ADMOM_OK) & (state.i < maxiter)

  def body_fn(state):
    det_w = state.irr * state.icc - state.irc ** 2
    low_det = det_w <= _LOW_DETVAL

    # censums: re-center under the current weight
    w_cen, _, _, _, _ = eval_weight(state.irr, state.icc, state.irc, state.row, state.col)
    wdata_cen = w_cen * image
    sums5_cen = jnp.sum(wdata_cen)
    nonpos_flux_cen = sums5_cen <= 0.0
    new_row = jnp.sum(wdata_cen * yy) / jnp.where(nonpos_flux_cen, 1.0, sums5_cen)
    new_col = jnp.sum(wdata_cen * xx) / jnp.where(nonpos_flux_cen, 1.0, sums5_cen)
    cen_shift = (jnp.abs(new_row - row_orig) > shiftmax) | (jnp.abs(new_col - col_orig) > shiftmax)

    # momsums: weighted second moments at the new center, current weight covariance
    w_mom, _, _, vmod, umod = eval_weight(state.irr, state.icc, state.irc, new_row, new_col)
    wdata_mom = w_mom * image
    sums5_mom = jnp.sum(wdata_mom)
    nonpos_flux_mom = sums5_mom <= 0.0
    finv = 1.0 / jnp.where(nonpos_flux_mom, 1.0, sums5_mom)
    M1 = jnp.sum(wdata_mom * (umod ** 2 - vmod ** 2)) * finv
    M2 = jnp.sum(wdata_mom * (2.0 * vmod * umod)) * finv
    Tm = jnp.sum(wdata_mom * (umod ** 2 + vmod ** 2)) * finv

    Irr_meas = 0.5 * (Tm - M1)
    Icc_meas = 0.5 * (Tm + M1)
    Irc_meas = 0.5 * M2
    nonpos_size = Tm <= 0.0

    e1 = (Icc_meas - Irr_meas) / jnp.where(nonpos_size, 1.0, Tm)
    e2 = (2.0 * Irc_meas) / jnp.where(nonpos_size, 1.0, Tm)

    converged = (
        (jnp.abs(e1 - state.e1old) < etol)
        & (jnp.abs(e2 - state.e2old) < etol)
        & (jnp.abs(Tm / jnp.where(state.Told == 0, 1.0, state.Told) - 1.0) < Ttol)
    )

    # deweight: solve for the new weight covariance N s.t. inv(N) = inv(measured) - inv(weight)
    detm = Irr_meas * Icc_meas - Irc_meas ** 2
    low_detm = detm <= _LOW_DETVAL
    idetm = 1.0 / jnp.where(low_detm, 1.0, detm)
    idetw = 1.0 / jnp.where(low_det, 1.0, det_w)
    Nrr = Icc_meas * idetm - state.icc * idetw
    Ncc = Irr_meas * idetm - state.irr * idetw
    Nrc = -Irc_meas * idetm + state.irc * idetw
    detn = Nrr * Ncc - Nrc ** 2
    low_detn = detn <= _LOW_DETVAL
    idetn = 1.0 / jnp.where(low_detn, 1.0, detn)
    new_irr = Ncc * idetn
    new_icc = Nrr * idetn
    new_irc = -Nrc * idetn

    fail_flags = jnp.where(
        low_det, ADMOM_LOW_DET,
        jnp.where(
            nonpos_flux_cen, ADMOM_NONPOS_FLUX,
            jnp.where(
                cen_shift, ADMOM_CEN_SHIFT,
                jnp.where(
                    nonpos_flux_mom, ADMOM_NONPOS_FLUX,
                    jnp.where(
                        nonpos_size, ADMOM_NONPOS_SIZE,
                        jnp.where(low_detm | low_detn, ADMOM_LOW_DET, ADMOM_OK),
                    ),
                ),
            ),
        ),
    )
    failed = fail_flags != ADMOM_OK

    out_flags = jnp.where(converged, ADMOM_OK + 100, jnp.where(failed, fail_flags, ADMOM_OK))
    # sentinel ADMOM_OK+100 marks "converged" internally; unpacked back to
    # ADMOM_OK once the loop exits (see below) -- keeps the loop condition
    # (flags == ADMOM_OK) simple: only "still running" uses ADMOM_OK inside
    # the loop.
    stop_flags = jnp.where(converged | failed, out_flags, ADMOM_OK)

    return _AdmomState(
        i=state.i + 1,
        irr=jnp.where(converged | failed, state.irr, new_irr),
        icc=jnp.where(converged | failed, state.icc, new_icc),
        irc=jnp.where(converged | failed, state.irc, new_irc),
        row=jnp.where(failed, state.row, new_row),
        col=jnp.where(failed, state.col, new_col),
        e1old=jnp.where(converged | failed, state.e1old, e1),
        e2old=jnp.where(converged | failed, state.e2old, e2),
        Told=jnp.where(converged | failed, state.Told, Tm),
        flags=stop_flags,
    )

  init = _AdmomState(
      i=jnp.zeros((), dtype=jnp.int32),
      irr=irr0, icc=icc0, irc=irc0,
      row=row0, col=col0,
      e1old=jnp.full((), jnp.nan, dtype=dtype_real),
      e2old=jnp.full((), jnp.nan, dtype=dtype_real),
      Told=jnp.full((), jnp.nan, dtype=dtype_real),
      flags=jnp.asarray(ADMOM_OK, dtype=jnp.int32),
  )

  final = lax.while_loop(cond_fn, body_fn, init)

  converged = final.flags == (ADMOM_OK + 100)
  maxed_out = (final.flags == ADMOM_OK) & (final.i >= maxiter)
  flags = jnp.where(converged, ADMOM_OK, jnp.where(maxed_out, ADMOM_MAXITER, final.flags))

  w_final, det_final, norm_final, vmod, umod = eval_weight(
      final.irr, final.icc, final.irc, final.row, final.col
  )
  wdata_final = w_final * image
  sums5 = jnp.sum(wdata_final)
  wsum = jnp.sum(w_final)
  ok = flags == ADMOM_OK

  # factor of 2: sum(weight*data)/(norm*wsum) is ngmix's own `flux` (verified
  # bit-for-bit against ngmix.admom.run_admom, in pure pixel-grid units, same
  # as galsim.hsm's default -- no use_sky_coords/pixel-scale conversion on
  # either side), but at convergence (weight == object's own Gaussian) this
  # recovers only *half* the true flux -- a derivable property of
  # matched-Gaussian-weight photometry (for a weight normalized to unit
  # integral, sum(weight*data)/norm = F/2 in the continuum limit, since
  # integral(pdf^2) = norm/2, not norm), not a bug in either implementation.
  # GalSim's `moments_amp` (what `correct_ksb`/ola actually use as "flux")
  # instead targets the true total flux directly -- verified directly: this
  # factor of 2 reproduces `galsim.hsm.FindAdaptiveMom`'s `moments_amp` to
  # ~1e-5 relative on a well-resolved test Gaussian.
  flux = jnp.where(ok, 2.0 * sums5 / jnp.where(ok, norm_final * wsum, 1.0), jnp.nan)
  sigma = jnp.where(ok, det_final ** 0.25, jnp.nan)
  T_out = jnp.where(ok, final.irr + final.icc, jnp.nan)
  e_out = jnp.array([
      jnp.where(ok, (final.icc - final.irr) / jnp.where(ok, T_out, 1.0), jnp.nan),
      jnp.where(ok, (2.0 * final.irc) / jnp.where(ok, T_out, 1.0), jnp.nan),
  ])

  return {
      'flux': flux, 'sigma': sigma, 'T': T_out, 'e': e_out,
      'row': final.row, 'col': final.col,
      'flags': flags, 'numiter': final.i,
  }


def correct_ksb(obs_image, psf_image, scale, psf_upsampling=1.0, sigw=2.0, gain=1.0,
                 sigmasky=None, admom_guess_T=None, admom_kwargs=None):
  """KSB-corrected shear + S/N + resolution for one exposure, JAX port of
  ola's `correct_ksb`.

  Args:
    obs_image: array (nx, ny), observed (PSF-convolved, noisy) galaxy stamp
    psf_image: array (nx, ny), PSF model stamp
    scale: float, pixel scale (shared by `obs_image` and `psf_image`)
    psf_upsampling: float, matches ola's `psf_upsampling` (`moments.py`'s
      `oversampling`, for a PSF stamp rendered at finer than `scale`)
    sigw: float, KSB weight sigma in pixels (see `moments`)
    gain: float, detector gain (electrons/ADU), for the S/N formula
    sigmasky: float, optional
      sky noise sigma; estimated from `obs_image`'s edges via `sigma_sky` if
      not given
    admom_guess_T: float, optional
      initial T guess (pixel^2) for the `admom` S/N-estimation pass;
      defaults to `size_moments`'s (fixed-centroid) `T` measurement
    admom_kwargs: dict, optional
      extra kwargs forwarded to `admom` (`maxiter`, `shiftmax`, `etol`,
      `Ttol`)

  Returns:
    dict with keys `uncal_e` (raw windowed e1/e2), `g` (KSB-calibrated
    shear), `e_err`, `flux`, `Tgal`, `SN`, `xc`, `yc` -- same as ola's
    `correct_ksb`
  """
  admom_kwargs = admom_kwargs or {}

  Tobs, obs_moments = size_moments(obs_image, scale, sigw=sigw)
  Tpsf, psf_moments = size_moments(psf_image, scale, sigw=sigw, oversampling=psf_upsampling)

  if admom_guess_T is None:
    admom_guess_T = jnp.maximum(obs_moments['T'], 1.0)
  am = admom(obs_image, admom_guess_T, **admom_kwargs)
  flux = jnp.where(am['flags'] == ADMOM_OK, jnp.maximum(am['flux'], 0.0), 0.0)
  sigmaobs = jnp.where(am['flags'] == ADMOM_OK, jnp.maximum(am['sigma'], 0.0), 0.0)

  # S/N, Tewes et al. 2019 effective-area formula (same as ola)
  A_eff = jnp.pi * (3.0 * sigmaobs * jnp.sqrt(2.0 * jnp.log(2.0))) ** 2
  if sigmasky is None:
    sigmasky = sigma_sky(obs_image)
  e_count = gain * flux
  SN = jnp.where(
      (flux == 0) & (A_eff == 0), 0.0,
      e_count / jnp.sqrt(e_count + A_eff * (gain * sigmasky) ** 2),
  )

  resolution = source_resolution(Tobs, Tpsf)
  e_err = ellipticity_error(flux, resolution, sigmaobs, sigmasky)

  calib_e = jnp.linalg.solve(
      obs_moments['Psh'],
      obs_moments['e'] - obs_moments['Psm'] @ jnp.linalg.solve(psf_moments['Psm'], psf_moments['e']),
  )

  return {
      'uncal_e': obs_moments['e'],
      'g': calib_e,
      'e_err': e_err,
      'flux': flux,
      'Tgal': Tobs,
      'SN': SN,
      'xc': obs_moments['xc'],
      'yc': obs_moments['yc'],
  }
