# This module tests that jax_galsim's interpolants are actually
# differentiable through autometacal.galflow.shear -- i.e. that
# jax.jacrev(shear) gives the true derivative w.r.t. the shear, not just
# "doesn't crash/NaN". Cross-checked against central finite differences.
#
# Note: this pipeline runs in float32 by default (see plans.md). Finite
# differences below a step of ~1e-4 become meaningless in float32 (the
# rendered image stops changing at all -- verified directly while
# investigating this), so the step sizes here are deliberately not tiny.
import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpy as np
from numpy.testing import assert_allclose

from autometacal.python.galflow import shear as am_shear

N = 32
_x = np.arange(N) - N / 2 + 0.5
_xx, _yy = np.meshgrid(_x, _x)
# asymmetric test stamp: a symmetric input sits exactly at the documented
# zero-shear NaN singularity (see plans.md), which is a separate, known issue
# unrelated to interpolant differentiability
_IMAGE = (
    np.exp(-(_xx**2 + _yy**2) / (2 * 4.0**2))
    + 0.3 * np.exp(-((_xx - 3) ** 2 + (_yy - 1) ** 2) / (2 * 2.0**2))
).astype(np.float32)


def finite_diff_jacobian(f, g0, h=1e-2):
  cols = []
  for i in range(g0.shape[0]):
    gp = g0.at[i].add(h)
    gm = g0.at[i].add(-h)
    cols.append((f(gp) - f(gm)) / (2 * h))
  return jnp.stack(cols, axis=1)


def check_interpolant_gradient(x_interpolant, g0, rtol=0.1, atol=0.05):
  def f(g):
    return am_shear(_IMAGE, g[0], g[1], x_interpolant=x_interpolant).flatten()

  autodiff_jacobian = jax.jacrev(f)(g0)
  numdiff_jacobian = finite_diff_jacobian(f, g0)

  assert not np.any(np.isnan(np.asarray(autodiff_jacobian)))
  assert_allclose(
      np.asarray(autodiff_jacobian), np.asarray(numdiff_jacobian), rtol=rtol, atol=atol,
  )


def test_interpolation_gradients_lanczos11():
  """ Lanczos(11): autometacal's default interpolant (matches ola). """
  check_interpolant_gradient(galsim.Lanczos(11), jnp.array([0.02, -0.01]))


def test_interpolation_gradients_quintic():
  """ Quintic: jax_galsim/GalSim's own default interpolant. """
  check_interpolant_gradient(galsim.Quintic(), jnp.array([0.02, -0.01]))


def test_interpolation_gradients_near_zero_shear():
  """ Gradients should also be well-behaved close to (but not at) g=0, on an
  asymmetric stamp -- unlike the literal g=0 case on a symmetric stamp,
  which is a documented, separate singularity (see plans.md).
  """
  check_interpolant_gradient(galsim.Lanczos(11), jnp.array([1e-3, -1e-3]))


if __name__ == '__main__':
  test_interpolation_gradients_lanczos11()
  test_interpolation_gradients_quintic()
  test_interpolation_gradients_near_zero_shear()
