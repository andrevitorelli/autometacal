"""
Gaussian Mixtures implementations from ngmix ported to JAX

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""

import jax.numpy as jnp
from autometacal.python.galflow import dtype_real

pi = 3.141592653589793


#############utilites conversions
def fwhm_to_sigma(fwhm):
  """
  convert fwhm to sigma for a gaussian
  """
  return fwhm / 2.3548200450309493


def fwhm_to_T(fwhm):
  """
  convert fwhm to T for a gaussian
  """
  sigma = fwhm_to_sigma(fwhm)
  return 2. * sigma * sigma


def g1g2_to_e1e2(g1, g2):
  """
  convert g to e
  """
  g = jnp.sqrt(g1 * g1 + g2 * g2)
  # guard the g=0 division: jnp.where evaluates both branches, so the
  # denominator must stay finite even where the result gets discarded
  safe_g = jnp.where(g == 0.0, 1.0, g)

  eta = 2 * jnp.arctanh(safe_g)
  e = jnp.minimum(jnp.tanh(eta), 0.99999999)
  fac = jnp.where(g == 0.0, 0.0, e / safe_g)

  return fac * g1, fac * g2


def e1e2_to_g1g2(e1, e2):
  """
  convert e (distortion) to g (reduced shear): the exact closed-form
  inverse of g1g2_to_e1e2, g = e / (1 + sqrt(1 - |e|^2)) -- unlike
  g1g2_to_e1e2 this has no g==0 singularity to guard (well-defined and
  continuous down to e=0, where fac=0.5). `jnp.maximum(...,0.)` guards only
  against |e| slightly exceeding 1 from measurement noise on a noisy image.
  """
  esq = e1 * e1 + e2 * e2
  fac = 1.0 / (1.0 + jnp.sqrt(jnp.maximum(1.0 - esq, 0.0)))
  return fac * e1, fac * e2


###################evaluate pixels#####################################
def gmix_eval_pixel(gmix, pixel):
  """
  evaluate a mixture of 2-d gaussians at the given pixel positions
  Args:
    gmix: array (n_gauss, 13)
      gauss2d structure:
      0 ='p',
      1 = 'row',
      2 = 'col',
      3 = 'irr',
      4 = 'irc',
      5 = 'icc',
      6 = 'det',
      7 = 'norm_set',
      8 = 'drr',
      9 = 'drc',
      10 ='dcc',
      11 ='norm',
      12 ='pnorm'
    pixel: array (n_pixels, 5)
      struct with coords u, v
      0 = u,
      1 = v,
      2 = area
      3 = val
      4 = ierr
  Returns:
    array (n_pixels,)
      model evaluated at each pixel's (u, v) position
  """
  gmix = gmix[:, None, :]
  # v->row, u->col in gauss
  vdiff = pixel[None, :, 1] - gmix[..., 1]
  udiff = pixel[None, :, 0] - gmix[..., 2]

  chi2 = (
      vdiff * vdiff * gmix[..., 8]
      + udiff * udiff * gmix[..., 10]
      - 2.0 * gmix[..., 9] * vdiff * udiff
  )

  return jnp.sum(gmix[..., -1] * jnp.exp(-0.5 * chi2) * pixel[None, :, 2], axis=0)


####################create gmixes ######################
def create_gmix(pars, model):
  """
  Build a profile from a mixture of gaussians

  Args:
    pars: array (6,)
      [row, col, g1, g2, T, flux] model parameters
    model: str
      model name ('gauss' or 'exp')

  returns: array (n_gauss, 13)
    mixture of gaussians, see `gmix_eval_pixel` for the column layout
  """
  if model == 'gauss':
    fvals, pvals = _fvals_gauss, _pvals_gauss
  elif model == 'exp':
    fvals, pvals = _fvals_exp, _pvals_exp
  else:
    raise ValueError(f"unknown model '{model}'")

  n_gauss = fvals.shape[0]
  row, col, g1, g2, T, flux = pars
  e1, e2 = g1g2_to_e1e2(g1, g2)

  T_i_2 = 0.5 * T * fvals
  p = flux * pvals

  irr = T_i_2 * (1 - e1)
  irc = T_i_2 * e2
  icc = T_i_2 * (1 + e1)
  det = irr * icc - irc * irc
  norm_set = jnp.ones((n_gauss,), dtype=dtype_real)

  drr = irr / det
  drc = irc / det
  dcc = icc / det
  norm = 1.0 / (2. * pi * jnp.sqrt(det))

  # renormalize p so that pnorm (the peak amplitude actually used for
  # evaluation) matches the requested total flux
  rat = (1. / norm) / jnp.sum(p)
  p = p * rat
  pnorm = p * norm

  row_arr = jnp.full((n_gauss,), row, dtype=dtype_real)
  col_arr = jnp.full((n_gauss,), col, dtype=dtype_real)

  return jnp.stack(
      [p, row_arr, col_arr, irr, irc, icc, det, norm_set, drr, drc, dcc, norm, pnorm],
      axis=-1,
  )


#predefs
_pvals_exp = jnp.array(
    [
        0.00061601229677880041,
        0.0079461395724623237,
        0.053280454055540001,
        0.21797364640726541,
        0.45496740582554868,
        0.26521634184240478,
    ],
    dtype=dtype_real,
)

_fvals_exp = jnp.array(
    [
        0.002467115141477932,
        0.018147435573256168,
        0.07944063151366336,
        0.27137669897479122,
        0.79782256866993773,
        2.1623306025075739,
    ],
    dtype=dtype_real,
)

_pvals_gauss = jnp.array([1.0], dtype=dtype_real)
_fvals_gauss = jnp.array([1.0], dtype=dtype_real)
