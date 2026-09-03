# This module tests our metacal image generation against ngmix's own
# (both are galsim-based under the hood, but via independent code paths).
import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose


def make_data(rng, noise, shear):
  """
  simulate an exponential object with moffat psf

  Parameters
  ----------
  rng: np.random.RandomState
    The random number generator
  noise: float
    Noise for the image
  shear: (g1, g2)
    The shear in each component

  Returns
  -------
  ngmix.Observation
  """
  psf_noise = 1.0e-6

  scale = 0.263
  stamp_size = 45
  psf_fwhm = 0.9
  gal_hlr = 0.5

  psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm).shear(g1=0.0, g2=0.0)
  obj0 = galsim.Exponential(half_light_radius=gal_hlr).shear(g1=shear[0], g2=shear[1])
  obj = galsim.Convolve(psf, obj0)

  psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
  im = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array

  psf_im = psf_im + rng.normal(scale=psf_noise, size=psf_im.shape)
  im = im + rng.normal(scale=noise, size=im.shape)

  cen = np.array(im.shape) / 2.0
  psf_cen = np.array(psf_im.shape) / 2.0

  jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
  psf_jacobian = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

  wt = im * 0 + 1.0 / noise**2
  psf_wt = psf_im * 0 + 1.0 / psf_noise**2

  psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jacobian)
  obs = ngmix.Observation(im, weight=wt, jacobian=jacobian, psf=psf_obs)

  return obs


args = {'seed': 31415, 'noise': 1e-5, 'psf': 'gauss'}
shear_true = [0.0, 0.0]


def test_generate_mcal_image():
  """ Checks that `generate_mcal_image` reproduces ngmix's own metacal images
  (noshear, 1p, 2p) for the same observation, within cross-implementation
  tolerance. ngmix's own metacal defaults to `lanczos15` (real GalSim,
  float64) while autometacal defaults to `Lanczos(11)` (jax_galsim, float32,
  matching ola rather than ngmix) -- exact agreement isn't expected, but the
  two should agree well relative to the galaxy's peak flux (~0.0235 here).
  """
  rng = np.random.RandomState(args['seed'])
  obs = make_data(rng=rng, noise=args['noise'], shear=shear_true)

  obsdict = ngmix.metacal.get_all_metacal(
      obs, psf=args['psf'], step=0.01, fixnoise=False, rng=rng,
  )

  im = obs.image.astype('float32')
  psf = obs.psf.image.astype('float32')
  rpsf = obsdict['noshear'].psf.image.astype('float32')

  zero = np.array([0., 0.], dtype='float32')
  step1p = np.array([0.01, 0.], dtype='float32')
  step2p = np.array([0., 0.01], dtype='float32')

  mcal_noshear = autometacal.generate_mcal_image(im, psf, rpsf, zero, zero)
  mcal_1p = autometacal.generate_mcal_image(im, psf, rpsf, step1p, zero)
  mcal_2p = autometacal.generate_mcal_image(im, psf, rpsf, step2p, zero)

  peak = obsdict['noshear'].image.max()
  atol = 5e-2 * peak
  assert_allclose(mcal_noshear, obsdict['noshear'].image, atol=atol)
  assert_allclose(mcal_1p, obsdict['1p'].image, atol=atol)
  assert_allclose(mcal_2p, obsdict['2p'].image, atol=atol)


if __name__ == '__main__':
  test_generate_mcal_image()
