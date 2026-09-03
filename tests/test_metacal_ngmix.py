# This is the "goal test" from plans.md: proof that the autodiff metacal
# shear response works on a single galaxy stamp, cross-checked against both
# a from-scratch finite-difference oracle and against ngmix's own
# (independent, real-GalSim-based) metacal response on the same observation.
import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose

args = {
    'seed': 31415,
    'noise': 1e-6,
    'psf': 'gauss',
    'shear_true': [0.01, 0.00],
    'weight_fwhm': 1.2,
    'step': 0.01,
}


def make_data(rng, noise, shear):
  """ simulate an exponential object with moffat psf. Deliberately shears the
  psf too (unlike test_metacal.py) so the galaxy+psf combo isn't perfectly
  circularly symmetric -- see plans.md's zero-shear-NaN finding.
  """
  scale = 0.263
  stamp_size = 45
  psf_fwhm = 0.9
  gal_hlr = 0.5
  psf_noise = 1.0e-6

  psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm).shear(g1=0.02, g2=-0.01)
  obj0 = galsim.Exponential(half_light_radius=gal_hlr).shear(g1=shear[0], g2=shear[1])
  obj = galsim.Convolve(psf, obj0)

  psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
  im = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array

  psf_im = psf_im + rng.normal(scale=psf_noise, size=psf_im.shape)
  im = im + rng.normal(scale=noise, size=im.shape)

  cen = (np.array(im.shape) - 1.0) / 2.0
  psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0

  jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
  psf_jacobian = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

  wt = im * 0 + 1.0 / noise**2
  psf_wt = psf_im * 0 + 1.0 / psf_noise**2

  psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jacobian)
  obs = ngmix.Observation(im, weight=wt, jacobian=jacobian, psf=psf_obs)

  return obs


def ngmix_R11(obs, rng):
  """ ngmix's own shear response, via its own (independent) metacal images
  and its own GaussMom moments -- central-differenced exactly like
  `get_metacal_response_finitediff`, but through ngmix's own code path.
  """
  obsdict = ngmix.metacal.get_all_metacal(
      obs, psf=args['psf'], step=args['step'], fixnoise=False, rng=rng,
      types=['1p', '1m'],
  )
  fitter = ngmix.gaussmom.GaussMom(fwhm=args['weight_fwhm'])
  e1p = fitter.go(obsdict['1p'])['e']
  e1m = fitter.go(obsdict['1m'])['e']
  return (e1p[0] - e1m[0]) / (2 * args['step'])


def test_metacal_response():
  """ Compares, on the same single galaxy stamp: (1) autometacal's autodiff
  response against its own finite-difference oracle, and (2) both against
  ngmix's independent metacal response. This is the goal test from
  plans.md -- proof that the autodiff response is right, not just that it
  runs.
  """
  rng = np.random.RandomState(args['seed'])
  obs = make_data(rng=rng, noise=args['noise'], shear=args['shear_true'])

  im = obs.image.astype('float32')
  psf = obs.psf.image.astype('float32')
  # ngmix's own noshear reconvolution psf, so both pipelines target the same psf
  rpsf = ngmix.metacal.get_all_metacal(
      obs, psf=args['psf'], step=args['step'], fixnoise=False, rng=rng, types=['noshear'],
  )['noshear'].psf.image.astype('float32')
  noise_im = rng.normal(scale=args['noise'], size=im.shape).astype('float32')

  def method(x):
    return autometacal.get_moment_ellipticities(x, scale=0.263, fwhm=args['weight_fwhm'])

  e, R, Rpsf, epsf, Repsf = autometacal.get_metacal_response(im, psf, rpsf, noise_im, method)
  ellip_dict, Rfd, Rpsffd, epsffd, Repsffd = autometacal.get_metacal_response_finitediff(
      im, psf, rpsf, noise_im, method, step=args['step'],
  )

  for arr in (e, R, Rpsf, epsf, Repsf, Rfd, Rpsffd, epsffd, Repsffd):
    assert not np.any(np.isnan(np.asarray(arr))), "response contains NaN"

  # autodiff vs its own finite-difference oracle
  assert_allclose(np.asarray(R), np.asarray(Rfd), atol=0.1, rtol=0.1)

  # both vs ngmix's independent metacal response, same observation
  r11_ngmix = ngmix_R11(obs, np.random.RandomState(args['seed'] + 1))
  assert_allclose(float(R[0, 0]), r11_ngmix, atol=0.2, rtol=0.2)
  assert_allclose(float(Rfd[0, 0]), r11_ngmix, atol=0.2, rtol=0.2)


if __name__ == '__main__':
  test_metacal_response()
