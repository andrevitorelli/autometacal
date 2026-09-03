# This module tests our Gaussian-moments ellipticity estimator
# (autometacal.get_moment_ellipticities, the JAX port of tf_ngmix/gaussmom)
# against ngmix's own GaussMom, on the same stamps.
#
# Replaces the old test_tf_ngmix.py, which referenced a
# `autometacal.datasets.galaxies.make_data` that didn't exist anywhere in
# the repo (already dead code before this migration) and depended on
# `datasets/`, retired in Phase 4.
import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose

scale = 0.2
stamp_size = 51


def make_stamp(rng, g1, g2, hlr=0.6):
  gal = galsim.Exponential(half_light_radius=hlr).shear(g1=g1, g2=g2)
  image = gal.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
  return image.astype('float32')


def test_gaussmom_matches_ngmix():
  """ Gaussian-weighted moment ellipticities should match ngmix's own
  GaussMom, on the same noiseless stamps, to high precision -- both compute
  the same weighted-moments algorithm, just in different frameworks.
  """
  rng = np.random.RandomState(31415)
  weight_fwhm = scale * stamp_size / 2

  g1s = rng.uniform(-0.7, 0.7, 20)
  g2s = rng.uniform(-0.7, 0.7, 20)

  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

  for g1, g2 in zip(g1s, g2s):
    image = make_stamp(rng, g1, g2)

    obs = ngmix.Observation(
        image.astype('float64'),
        jacobian=ngmix.DiagonalJacobian(row=stamp_size // 2, col=stamp_size // 2, scale=scale),
    )
    e_ngmix = fitter.go(obs)['e']
    e_ours = autometacal.get_moment_ellipticities(image, scale=scale, fwhm=weight_fwhm)

    assert_allclose(np.asarray(e_ours), e_ngmix, rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
  test_gaussmom_matches_ngmix()
