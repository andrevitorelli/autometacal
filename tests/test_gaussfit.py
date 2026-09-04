# This module tests autometacal.gaussfit, the JAX/implicit-diff port of the
# pre-migration TF fitting.py's differentiable Gaussian-fit approach
# (fixed_point_layer_implicit/fwd_solver -> jax.lax.custom_root).
import numpy as np
import jax
import jax.numpy as jnp
import galsim
import autometacal
from autometacal.python import gaussfit

from numpy.testing import assert_allclose

scale = 0.2
stamp_size = 51


def test_fit_recovers_known_shear_on_clean_gaussian():
  """ For a clean (noiseless), off-center sheared Gaussian, the fit should
  recover the exact distortion ellipticity implied by the applied shear
  (e = 2g/(1+|g|^2), same direction as g) -- this is an exact match case,
  not an approximation, since the image genuinely is an elliptical
  Gaussian (the model family being fit).
  """
  g1, g2 = 0.15, -0.08
  image = galsim.Gaussian(sigma=3.0, flux=1.0e5).shear(g1=g1, g2=g2).shift(0.6, -0.3).drawImage(
      nx=41, ny=41, scale=1.0, method='no_pixel',
  ).array.astype('float32')

  e = np.asarray(gaussfit.get_fit_ellipticities(jnp.asarray(image)))

  gsq = g1 ** 2 + g2 ** 2
  fac = 2.0 / (1.0 + gsq)
  expected = np.array([fac * g1, fac * g2])
  assert_allclose(e, expected, atol=2e-3)


def test_fit_differentiable_and_matches_finite_difference():
  """ Gradients of the fitted ellipticity w.r.t. the image, computed via
  jax.grad (implicit function theorem through jax.lax.custom_root), must
  match central finite differences -- this is the entire point of using
  custom_root rather than differentiating through the LM iterations.

  This module inherits ksb.admom's jax_enable_x64 incompatibility (see
  gaussfit.py's module docstring), so this necessarily runs in float32 --
  jnp.asarray() silently downcasts float64 input with x64 off. Tolerance
  is loosened accordingly (same float32 precision-floor caveat documented
  elsewhere in this repo, e.g. test_interpolation_gradients.py).
  """
  image = galsim.Gaussian(sigma=3.0, flux=1.0e5).shear(g1=0.1, g2=-0.05).shift(0.4, -0.2).drawImage(
      nx=41, ny=41, scale=1.0, method='no_pixel',
  ).array.astype('float32')
  im = jnp.asarray(image)
  assert im.dtype == jnp.float32

  def e1_of_image(x):
    return gaussfit.get_fit_ellipticities(x)[0]

  analytic = jax.grad(e1_of_image)(im)
  assert jnp.all(jnp.isfinite(analytic))

  eps = 1.0
  for (i, j) in [(20, 21), (19, 20), (15, 15), (25, 25)]:
    fd = (e1_of_image(im.at[i, j].add(eps)) - e1_of_image(im.at[i, j].add(-eps))) / (2 * eps)
    assert_allclose(float(analytic[i, j]), float(fd), rtol=0.15, atol=1e-7)


def test_fit_robust_on_realistic_noisy_stamp():
  """ On a realistic PSF-convolved, noisy exponential galaxy (not a pure
  Gaussian -- the model family being fit), the fit should converge to a
  finite, bounded, sane ellipticity, not diverge/NaN/saturate. This
  guards against the initial-guess degeneracy found during development
  (naive unweighted second moments gave a near-singular covariance guess
  for some real galaxy shapes, causing the LM solve to saturate at
  |e| = 1 -- fixed by seeding the initial guess from ksb.admom instead).
  """
  psf = galsim.Kolmogorov(fwhm=0.7)
  gal = galsim.Exponential(half_light_radius=0.4, flux=3.0e5).shear(g1=0.05, g2=-0.03)
  obj = galsim.Convolve(gal, psf)
  rng = np.random.RandomState(2)
  image = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale, method='no_pixel').array.astype('float32')
  image = image + rng.normal(scale=8.0, size=image.shape).astype('float32')

  e = np.asarray(gaussfit.get_fit_ellipticities(jnp.asarray(image)))
  assert np.all(np.isfinite(e))
  assert np.all(np.abs(e) < 0.9)


def test_fit_jit_and_vmap_compatible():
  image = galsim.Gaussian(sigma=3.0, flux=1.0e5).shear(g1=0.1, g2=0.0).drawImage(
      nx=41, ny=41, scale=1.0, method='no_pixel',
  ).array.astype('float32')
  im = jnp.asarray(image)

  jitted = jax.jit(gaussfit.get_fit_ellipticities)
  e_jit = np.asarray(jitted(im))

  vmapped = jax.vmap(gaussfit.get_fit_ellipticities)
  e_vmap = np.asarray(vmapped(jnp.stack([im, im])))

  assert_allclose(e_jit, e_vmap[0], atol=1e-5)
  assert_allclose(e_vmap[0], e_vmap[1], atol=1e-5)


def test_fit_as_metacal_response_method():
  """ End-to-end: usable as the `method` callable for
  `get_metacal_response`, matching gaussmom's calling convention (image ->
  ellipticities). The response matrix should be finite and reasonably
  well-conditioned (not necessarily close to gaussmom's own R -- a less
  diluted estimator is expected to need a smaller metacal correction, i.e.
  R closer to the identity, though this test only checks sanity, not that
  specific property).
  """
  psf = galsim.Kolmogorov(fwhm=0.7)
  gal = galsim.Exponential(half_light_radius=0.4, flux=3.0e5).shear(g1=0.05, g2=-0.03)
  obj = galsim.Convolve(gal, psf)
  rng = np.random.RandomState(2)
  obs_image = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale, method='no_pixel').array.astype('float32')
  noise = rng.normal(scale=8.0, size=obs_image.shape).astype('float32')
  obs_noisy = obs_image + noise
  psf_image = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale, method='no_pixel').array.astype('float32')

  method = lambda img: gaussfit.get_fit_ellipticities(img)
  e, R, Rpsf, epsf, Repsf = autometacal.get_metacal_response(
      jnp.asarray(obs_noisy), jnp.asarray(psf_image), jnp.asarray(psf_image), jnp.asarray(noise),
      method, scale=scale,
  )
  assert np.all(np.isfinite(np.asarray(e)))
  assert np.all(np.isfinite(np.asarray(R)))
  assert 0.5 < float(R[0, 0]) < 1.5
  assert 0.5 < float(R[1, 1]) < 1.5


if __name__ == '__main__':
  test_fit_recovers_known_shear_on_clean_gaussian()
  test_fit_differentiable_and_matches_finite_difference()
  test_fit_robust_on_realistic_noisy_stamp()
  test_fit_jit_and_vmap_compatible()
  test_fit_as_metacal_response_method()
