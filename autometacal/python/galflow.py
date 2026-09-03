import jax.numpy as jnp
import jax_galsim as galsim

dtype_real = jnp.float32

# matches ola's interpolant choice (interp="lanczos11"), not jax_galsim's own
# default (Quintic). Verified differentiable: jax.jacrev through
# InterpolatedImage(x_interpolant=Lanczos(11)).shear().drawImage() agrees with
# central finite differences to ~0.6% at reasonable step sizes (float32).
DEFAULT_INTERPOLANT = galsim.Lanczos(11)


def fixed_fft_gsparams(fft_size):
  """ GSParams that force jax_galsim onto its static-FFT-size code path.

  jax_galsim's adaptive FFT-size selection (`Image.good_fft_size`,
  `GSObject.drawFFT_makeKImage`) does real Python-level control flow
  (`math.log`, `math.ceil`, `if N*dk/2 > maxk`) on values that depend on the
  object's data/shear -- this is fundamentally incompatible with
  `jax.jit`/`jax.vmap` (verified directly: `ConcretizationTypeError` /
  `TracerBoolConversionError`), which need every array-valued decision to
  stay inside traceable JAX ops. Reading `GSObject.drawFFT_makeKImage`'s
  actual source directly found the documented escape hatch: when
  `gsparams.maximum_fft_size == gsparams.minimum_fft_size`, the whole
  adaptive branch is skipped and `Nk` is set to that fixed value inside
  `jax.ensure_compile_time_eval()` -- fully static, so `jit`/`vmap` work.

  Verified directly against the adaptive default: `fft_size=128` gives
  results matching the adaptive computation to ~1e-7 (both the rendered
  image and the shear-response Jacobian) for a 45x45, scale=0.263 stamp;
  smaller sizes (64, 96) are already within ~0.1% but not exact.

  IMPORTANT: unlike the adaptive path, the static path does *not* check
  whether the object actually fits -- if the true required FFT size exceeds
  `fft_size`, this does not raise, it silently aliases. Use
  `required_fft_size` (real, non-jax GalSim) to check per-object fit and
  reject oversized objects *before* batching, at stamp-generation time.

  Args:
    fft_size: int
      fixed k-space grid size (both max and min) to force

  Returns:
    galsim.GSParams
  """
  return galsim.GSParams(maximum_fft_size=fft_size, minimum_fft_size=fft_size)


def interpolated_image(image, scale=1.0, x_interpolant=None, gsparams=None):
  """ Wrap a single pixel-array stamp as an interpolated GSObject.

  Args:
    image: array (nx, ny)
      pixel stamp
    scale: float
      pixel scale (arbitrary units; default 1.0 = one unit per pixel)
    x_interpolant: jax_galsim.Interpolant, optional
      real-space interpolant; defaults to `DEFAULT_INTERPOLANT` (Lanczos(11),
      matching ola)
    gsparams: galsim.GSParams, optional
      pass `fixed_fft_gsparams(N)` to make this (and anything built from it:
      `.shear()`, `Convolve`, `Deconvolve`, `drawImage`) `jit`/`vmap`-safe;
      default `None` keeps jax_galsim's normal adaptive FFT sizing, fine for
      a single, non-batched call.

  Returns:
    galsim.InterpolatedImage
  """
  if x_interpolant is None:
    x_interpolant = DEFAULT_INTERPOLANT
  im = galsim.Image(jnp.asarray(image, dtype=dtype_real), scale=scale)
  return galsim.InterpolatedImage(im, x_interpolant=x_interpolant, gsparams=gsparams)


def shear(image, g1, g2, scale=1.0, x_interpolant=None, gsparams=None):
  """ Applies a reduced shear g1, g2 to a single image stamp.

  Args:
    image: array (nx, ny)
      pixel stamp
    g1, g2: shear components to apply
    scale: float
      pixel scale of `image` and of the returned stamp
    gsparams: galsim.GSParams, optional
      see `interpolated_image`

  Returns:
    array (nx, ny)
      sheared image stamp
  """
  nx, ny = image.shape
  obj = interpolated_image(image, scale=scale, x_interpolant=x_interpolant, gsparams=gsparams)
  obj = obj.shear(g1=g1, g2=g2)
  return obj.drawImage(nx=nx, ny=ny, scale=scale, method='fft').array


def dilate(image, factor, scale=1.0, x_interpolant=None, gsparams=None):
  """ Dilate a single image stamp by `factor` around its center, preserving flux.

  Args:
    image: array (nx, ny)
      pixel stamp
    factor: float
      linear dilation factor (>=1 grows the profile)
    scale: float
      pixel scale of `image` and of the returned stamp
    gsparams: galsim.GSParams, optional
      see `interpolated_image`

  Returns:
    array (nx, ny)
      dilated image stamp
  """
  nx, ny = image.shape
  obj = interpolated_image(image, scale=scale, x_interpolant=x_interpolant, gsparams=gsparams)
  obj = obj.dilate(factor)
  return obj.drawImage(nx=nx, ny=ny, scale=scale, method='fft').array
