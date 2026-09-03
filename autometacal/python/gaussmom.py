from autometacal.python.tf_ngmix.moments import get_moments
from autometacal.python.tf_ngmix.gmix import create_gmix, fwhm_to_T
from autometacal.python.tf_ngmix.pixels import make_pixels
import jax.numpy as jnp
from autometacal.python.galflow import dtype_real


def get_moment_ellipticities(image, scale, fwhm, **kwargs):
  """
  Gets an ellipticity estimate from the gaussian moments of a stamp.

  Args:
    image: array (nx, ny)
    scale: float
      the pixel scale of the image in arcsec/pixel
    fwhm: float
      the full width at half maximum of the gaussian filter in arcseconds
    centre_x, centre_y: floats
      centre of the image in pixels; if omitted, the stamp's centre pixel is used
    weights: array (nx, ny)
      per-pixel weights; if omitted, all pixels are weighted equally

  Returns:
    array (2,)
      [g1, g2] ellipticity, according to the |e| = (a - b)/(a + b) convention
  """
  Q11, Q12, Q22 = moments(image, scale, fwhm, **kwargs)

  q1 = Q11 - Q22
  q2 = 2 * Q12
  T = Q11 + Q22

  return jnp.stack([q1 / T, q2 / T])


def moments(image, scale, fwhm, **kwargs):
  """
  Gets the gaussian-weighted moments of a stamp.

  Args: see `get_moment_ellipticities`.

  Returns:
    Gaussian-weighted moments: Q11, Q12 and Q22 for the image.
  """
  nx, ny = image.shape
  defaults = {
      'centre_x': nx // 2,
      'centre_y': ny // 2,
      'weights': jnp.ones((nx, ny), dtype=dtype_real),
  }
  defaults.update(kwargs)

  pixels = make_pixels(
      image,
      defaults['weights'],
      (defaults['centre_x'], defaults['centre_y']),
      scale,
  )

  T = fwhm_to_T(fwhm)
  wt = create_gmix(jnp.array([0., 0., 0., 0., T, 1.], dtype=dtype_real), 'gauss')

  return get_moments(wt, pixels)
