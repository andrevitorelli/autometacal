"""
Moments functions from ngmix ported to JAX


Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0
"""
import jax.numpy as jnp
from .gmix import gmix_eval_pixel


######measure weighted moments

def get_moments(weights, pixels):
  """
  Get gaussian-weighted moments from a ngmix-like pixel structure
  (1d array of pixels = (u,v, pixel area, pixel value, pixel weights))

  Args:
    weights: array (n_gauss, 13)
      gaussian mixture to be used as the weight function
    pixels: array (n_pixels, 5)
      flattened pixel list, see `make_pixels`

  Returns:
    Q11, Q12, Q22: the gaussian-weighted second moments of the image
  """
  w = gmix_eval_pixel(weights, pixels)

  norm = jnp.sum(w * pixels[:, 3])
  Q11 = jnp.sum(w * pixels[:, 3] * pixels[:, 0] * pixels[:, 0]) / norm
  Q12 = jnp.sum(w * pixels[:, 3] * pixels[:, 0] * pixels[:, 1]) / norm
  Q22 = jnp.sum(w * pixels[:, 3] * pixels[:, 1] * pixels[:, 1]) / norm

  return Q11, Q12, Q22
