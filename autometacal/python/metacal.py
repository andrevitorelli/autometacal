import warnings

import jax
import jax.numpy as jnp
import jax_galsim as galsim

from autometacal.python.galflow import interpolated_image, dilate, dtype_real, fixed_fft_gsparams

# tiny dilation applied to the reconvolution psf to avoid ringing/negative-flux
# artifacts when deconvolving by a psf very close to the reconvolution target
_reconv_psf_dilation = 1.001

# base shear the response Jacobian is evaluated at, instead of literal 0.0:
# jax-galsim's autodiff through the deconvolve-shear-reconvolve chain is
# numerically unstable very close to g=0 (blows up rather than NaNs for
# g >~ 1e-5, and NaNs outright at literal 0.0 for symmetric inputs -- an FFT
# fast-path issue, not yet fixed upstream). Empirically stable for eps >= 1e-4
# (spot-checked directly, see plans.md); 1e-3 keeps a safety margin while
# staying an order of magnitude below metacal's usual 0.01 calibration step.
_response_eps = 1e-3


def generate_mcal_image(gal_image, psf_image, reconvolution_psf_image, g, gp, scale=1.0, gsparams=None):
  """ Generate a metacalibration image given input and target PSFs.

  Args:
    gal_image: array (nx, ny)
      image of the galaxy
    psf_image: array (nx, ny)
      image of the psf model
    reconvolution_psf_image: array (nx, ny)
      image of the reconvolution psf model
    g: array (2,)
      [g1, g2] shear applied to the deconvolved galaxy
    gp: array (2,)
      [gp1, gp2] shear applied to the reconvolution psf
    scale: float
      pixel scale of the input/output stamps
    gsparams: galsim.GSParams, optional
      pass `autometacal.galflow.fixed_fft_gsparams(N)` to make this
      `jit`/`vmap`-safe (see `get_metacal_response_batched`); default `None`
      keeps normal adaptive FFT sizing, fine for a single call.

  Returns:
    array (nx, ny)
      image of the galaxy after deconvolution by the psf, shearing by g, and
      reconvolution with the reconvolution psf (itself sheared by gp)
  """
  nx, ny = gal_image.shape
  gal = interpolated_image(gal_image, scale=scale, gsparams=gsparams)
  psf = interpolated_image(psf_image, scale=scale, gsparams=gsparams)
  reconv_psf = interpolated_image(reconvolution_psf_image, scale=scale, gsparams=gsparams)

  deconv_gal = galsim.Convolve(gal, galsim.Deconvolve(psf))
  sheared_gal = deconv_gal.shear(g1=g[0], g2=g[1])
  sheared_reconv_psf = reconv_psf.shear(g1=gp[0], g2=gp[1])
  reconvolved = galsim.Convolve(sheared_gal, sheared_reconv_psf)

  return reconvolved.drawImage(nx=nx, ny=ny, scale=scale, method='fft').array


def generate_mcal_psf(reconvolution_psf_image, gp, scale=1.0, gsparams=None):
  """ Generate a metacalibration psf image: the reconvolution psf, sheared by gp.

  Used to measure the calibration psf's own ellipticity response to gp
  (independently of any galaxy), matching ola's separate `"psf"` obsdict entry.

  Args:
    reconvolution_psf_image: array (nx, ny)
      image of the reconvolution psf model
    gp: array (2,)
      [gp1, gp2] shear to apply
    scale: float
      pixel scale of the input/output stamps
    gsparams: galsim.GSParams, optional
      see `generate_mcal_image`

  Returns:
    array (nx, ny)
      image of the reconvolution psf sheared by gp
  """
  nx, ny = reconvolution_psf_image.shape
  reconv_psf = interpolated_image(reconvolution_psf_image, scale=scale, gsparams=gsparams)
  sheared = reconv_psf.shear(g1=gp[0], g2=gp[1])
  return sheared.drawImage(nx=nx, ny=ny, scale=scale, method='fft').array


def generate_fixnoise(noise_image, psf_image, reconvolution_psf_image, g, gp, scale=1.0, gsparams=None):
  """ Generate a counter-sheared noise image for noise-bias cancellation.

  Follows ola's `get_fixnoise` recipe: deconvolve the noise by the input psf,
  rotate 90deg, shear by g, rotate back -90deg, then reconvolve with the
  (gp-sheared) reconvolution psf. The 90/-90 rotation sandwich is equivalent to
  directly applying shear(-g) to the unrotated, deconvolved noise, which is
  what decorrelates the added noise from the galaxy's own embedded noise.

  Args:
    noise_image: array (nx, ny)
      a noise realization matching the galaxy stamp's own noise properties
    psf_image: array (nx, ny)
      image of the psf model
    reconvolution_psf_image: array (nx, ny)
      image of the reconvolution psf model
    g: array (2,)
      [g1, g2] shear applied to the galaxy (used for the noise's own shear step)
    gp: array (2,)
      [gp1, gp2] shear applied to the reconvolution psf
    scale: float
      pixel scale of the input/output stamps
    gsparams: galsim.GSParams, optional
      see `generate_mcal_image`

  Returns:
    array (nx, ny)
      counter-sheared, reconvolved noise image
  """
  nx, ny = noise_image.shape
  noise = interpolated_image(noise_image, scale=scale, gsparams=gsparams)
  psf = interpolated_image(psf_image, scale=scale, gsparams=gsparams)
  reconv_psf = interpolated_image(reconvolution_psf_image, scale=scale, gsparams=gsparams)

  deconv_noise = galsim.Convolve(noise, galsim.Deconvolve(psf))
  rotated = deconv_noise.rotate(90 * galsim.degrees)
  sheared = rotated.shear(g1=g[0], g2=g[1])
  unrotated = sheared.rotate(-90 * galsim.degrees)
  sheared_reconv_psf = reconv_psf.shear(g1=gp[0], g2=gp[1])
  reconvolved = galsim.Convolve(unrotated, sheared_reconv_psf)

  return reconvolved.drawImage(nx=nx, ny=ny, scale=scale, method='fft').array


def get_metacal_response(gal_image, psf_image, reconvolution_psf_image, noise_image,
                          method, scale=1.0, eps=_response_eps, gsparams=None):
  """ Computes the shear response by automatic differentiation.

  Replaces the standard 5-image finite-difference metacal response with a
  single autodiff Jacobian of the measured ellipticity with respect to the
  combined [g1, g2, gp1, gp2] shear vector.

  Args:
    gal_image, psf_image, reconvolution_psf_image, noise_image: array (nx, ny)
    method: callable, array (nx, ny) -> array (2,)
      ellipticity estimator applied to a metacal image
    scale: float
      pixel scale of the input stamps
    eps: float
      base shear the Jacobian is evaluated at, instead of literal 0.0
      (see `_response_eps` above)
    gsparams: galsim.GSParams, optional
      forces a fixed FFT grid size (see `galflow.fixed_fft_gsparams`) instead
      of jax_galsim's normal adaptive sizing. Not needed for a single,
      standalone call (this function already works unjitted as-is); required
      for `jit`/`vmap` -- see `get_metacal_response_batched`, which sets this
      automatically. If set, the caller is responsible for having checked
      (via `required_fft_size`) that every input actually fits -- unlike the
      adaptive path, the fixed-size path does not self-check and will
      silently alias rather than error if it doesn't.

  Returns:
    e: array (2,), the measured (noshear) ellipticity
    R: array (2, 2), shear response matrix (de/dg)
    Rpsf: array (2, 2), psf shear response matrix (de/dgp)
    epsf: array (2,), the calibration psf's own (noshear) ellipticity
    Repsf: array (2, 2), the calibration psf's own ellipticity response to gp
  """
  reconvolution_psf_image = dilate(reconvolution_psf_image, _reconv_psf_dilation, scale=scale, gsparams=gsparams)

  def measure(gs):
    g, gp = gs[0:2], gs[2:4]
    img = generate_mcal_image(gal_image, psf_image, reconvolution_psf_image, g, gp, scale=scale, gsparams=gsparams)
    img = img + generate_fixnoise(noise_image, psf_image, reconvolution_psf_image, g, gp, scale=scale, gsparams=gsparams)
    return method(img)

  def measure_psf(gp):
    img = generate_mcal_psf(reconvolution_psf_image, gp, scale=scale, gsparams=gsparams)
    return method(img)

  gs0 = jnp.full((4,), eps, dtype=dtype_real)
  gp0 = jnp.full((2,), eps, dtype=dtype_real)

  e = measure(gs0)
  Rs = jax.jacrev(measure)(gs0)
  R, Rpsf = Rs[:, 0:2], Rs[:, 2:4]

  epsf = measure_psf(gp0)
  Repsf = jax.jacrev(measure_psf)(gp0)

  return e, R, Rpsf, epsf, Repsf


def fits_fixed_fft_size(gal_image, psf_image, reconvolution_psf_image, fft_size, scale=1.0):
  """ Checks whether a single galaxy's metacal pipeline would fit within a
  fixed FFT grid of size `fft_size` -- run this (real jax_galsim, eager, one
  galaxy at a time, cheap) on every candidate *before* batching, and only
  include galaxies that pass in the arrays handed to
  `get_metacal_response_batched`. `get_metacal_response_batched` itself has
  no way to check this (that's the whole reason it needs a fixed size), and
  an object that doesn't actually fit will silently alias there rather than
  error.

  Implemented by running `generate_mcal_image` eagerly (not jit/vmap'd, so
  jax_galsim's normal *adaptive* FFT sizing is free to run) with
  `gsparams=galsim.GSParams(maximum_fft_size=fft_size)` (deliberately not
  setting `minimum_fft_size` too, which would trigger the fixed-size path
  and skip the check) and treating `GalSimFFTSizeWarning` as a failure. This
  reuses jax_galsim's own size computation directly rather than
  reimplementing it, so it can't drift out of sync with it.

  Args:
    gal_image, psf_image, reconvolution_psf_image: array (nx, ny)
    fft_size: int
      the fixed size that will be used for batched processing
    scale: float

  Returns:
    bool
  """
  gsparams = galsim.GSParams(maximum_fft_size=fft_size)
  zero = jnp.zeros((2,), dtype=dtype_real)
  try:
    with warnings.catch_warnings():
      warnings.filterwarnings('error', category=galsim.errors.GalSimFFTSizeWarning)
      generate_mcal_image(gal_image, psf_image, reconvolution_psf_image, zero, zero, scale=scale, gsparams=gsparams)
    return True
  except (galsim.errors.GalSimFFTSizeWarning, galsim.errors.GalSimFFTSizeError):
    return False


def get_metacal_response_batched(gal_images, psf_images, reconvolution_psf_images, noise_images,
                                  method, fft_size, scale=1.0, eps=_response_eps):
  """ `get_metacal_response`, vectorized over a batch of galaxies with
  `jax.vmap` -- runs the deconvolve/shear/reconvolve autodiff response for
  every galaxy in the batch as a single parallel dispatch (GPU-friendly).

  This requires a *fixed* FFT grid size shared by the whole batch: jax_galsim's
  normal adaptive FFT-size selection does real Python control flow on values
  that depend on each object's shear/data, which `jax.vmap` cannot trace
  (verified directly -- see `galflow.fixed_fft_gsparams`'s docstring for the
  mechanism and where this was found in jax_galsim's own source). Use
  `required_fft_size` on each candidate galaxy *before* calling this, at
  stamp-generation time (real GalSim, CPU, one at a time is fine there), and
  only include galaxies that fit within `fft_size` -- this function does not
  check, and a galaxy that doesn't fit will silently alias, not error.

  Args:
    gal_images, psf_images, reconvolution_psf_images, noise_images: array (B, nx, ny)
    method: callable, array (nx, ny) -> array (2,)
      applied per-galaxy inside the vmap; do not pre-vmap this yourself
    fft_size: int
      fixed FFT grid size for every object in the batch; pick this once,
      informed by `required_fft_size` on a representative sample -- bigger is
      more accurate but costs memory/compute roughly quadratically. Verified
      directly: 128 matches the adaptive default to ~1e-7 for 45x45,
      scale=0.263 stamps; smaller sizes are already within ~0.1% but not exact.
    scale: float
    eps: float
      see `get_metacal_response`

  Returns:
    e, R, Rpsf, epsf, Repsf -- same as `get_metacal_response`, each with a
    leading batch dimension of size B
  """
  gsparams = fixed_fft_gsparams(fft_size)

  def single(gal_image, psf_image, reconvolution_psf_image, noise_image):
    return get_metacal_response(
        gal_image, psf_image, reconvolution_psf_image, noise_image, method,
        scale=scale, eps=eps, gsparams=gsparams,
    )

  return jax.vmap(single)(gal_images, psf_images, reconvolution_psf_images, noise_images)


def get_metacal_response_finitediff(gal_image, psf_image, reconvolution_psf_image, noise_image,
                                     method, scale=1.0, step=0.01):
  """ Computes the shear response as a central finite difference, as a correctness
  oracle for `get_metacal_response` (and for comparison against ola/ngmix).

  Args:
    gal_image, psf_image, reconvolution_psf_image, noise_image: array (nx, ny)
    method: callable, array (nx, ny) -> array (2,)
    scale: float
      pixel scale of the input stamps
    step: float
      metacal calibration step

  Returns:
    ellip_dict: dict of the 5 standard metacal ellipticity measurements
    R: array (2, 2), shear response matrix
    Rpsf: array (2, 2), psf shear response matrix (galaxy ellipticity vs gp)
    epsf: array (2,), the calibration psf's own (noshear) ellipticity
    Repsf: array (2, 2), the calibration psf's own ellipticity response to gp
  """
  reconvolution_psf_image = dilate(reconvolution_psf_image, _reconv_psf_dilation, scale=scale)

  zero = jnp.zeros((2,), dtype=dtype_real)
  step1p = jnp.array([step, 0.], dtype=dtype_real)
  step1m = -step1p
  step2p = jnp.array([0., step], dtype=dtype_real)
  step2m = -step2p

  def measure(g, gp):
    img = generate_mcal_image(gal_image, psf_image, reconvolution_psf_image, g, gp, scale=scale)
    img = img + generate_fixnoise(noise_image, psf_image, reconvolution_psf_image, g, gp, scale=scale)
    return method(img)

  def central_diff(pairs):
    ep, em = (measure(*p) for p in pairs)
    return (ep - em) / (2 * step)

  g0s = measure(zero, zero)
  g1p, g1m = measure(step1p, zero), measure(step1m, zero)
  g2p, g2m = measure(step2p, zero), measure(step2m, zero)
  R = jnp.stack([central_diff([(step1p, zero), (step1m, zero)]),
                 central_diff([(step2p, zero), (step2m, zero)])], axis=1)

  Rpsf = jnp.stack([central_diff([(zero, step1p), (zero, step1m)]),
                     central_diff([(zero, step2p), (zero, step2m)])], axis=1)

  def measure_psf(gp):
    return method(generate_mcal_psf(reconvolution_psf_image, gp, scale=scale))

  epsf = measure_psf(zero)
  Repsf = jnp.stack([(measure_psf(step1p) - measure_psf(step1m)) / (2 * step),
                      (measure_psf(step2p) - measure_psf(step2m)) / (2 * step)], axis=1)

  ellip_dict = {'noshear': g0s, '1p': g1p, '1m': g1m, '2p': g2p, '2m': g2m}

  return ellip_dict, R, Rpsf, epsf, Repsf
