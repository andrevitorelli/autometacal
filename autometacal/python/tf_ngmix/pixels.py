"""
Observation implementations from ngmix ported to JAX

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""

import jax.numpy as jnp
from autometacal.python.galflow import dtype_real


def make_pixels(image, weights, centre, pixel_scale):
  """ Build an ngmix-style flat pixel list from a single image.

  Args:
    image: array (nx, ny)
    weights: array (nx, ny)
      per-pixel weights
    centre: (centre_x, centre_y)
      pixel-coordinate centre used to build the (u, v) grid
    pixel_scale: float

  Returns:
    array (nx*ny, 5)
      flattened (u, v, area, value, weight) pixel list
  """
  nx, ny = image.shape
  centre_x, centre_y = centre

  grid_x, grid_y = jnp.meshgrid(jnp.arange(nx, dtype=dtype_real), jnp.arange(ny, dtype=dtype_real))
  u = (grid_x - jnp.asarray(centre_x, dtype=dtype_real)) * pixel_scale
  v = (grid_y - jnp.asarray(centre_y, dtype=dtype_real)) * pixel_scale
  area = jnp.full((nx * ny,), pixel_scale * pixel_scale, dtype=dtype_real)

  return jnp.stack([
      u.reshape(-1),
      v.reshape(-1),
      area,
      image.reshape(-1).astype(dtype_real),
      weights.reshape(-1).astype(dtype_real),
  ], axis=-1)
