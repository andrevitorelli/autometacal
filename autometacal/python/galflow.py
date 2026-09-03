import math
import warnings

import jax.numpy as jnp
import jax_galsim as galsim

dtype_real = jnp.float32

# jax_galsim's GSObject._determine_wcs (called on every drawImage()) does
# `jax.lax.cond(wcs.scale <= 0, branch_a, branch_b, ...)`, and branch_a
# hardcodes `PixelScale(jnp.float_(nqs))` -- `jnp.float_` is JAX's legacy
# alias for float64. `lax.cond` abstractly evaluates *both* branches (for
# output shape/dtype consistency) even though only one actually runs, so
# this fires on every single drawImage() call regardless of what `scale`/
# `wcs` we pass in -- verified directly via `python -W error::UserWarning`
# stack trace; there is no caller-side parameter that avoids it. Since this
# package deliberately runs float32 throughout (jax_enable_x64 off, for GPU
# throughput -- see CLAUDE.md), this specific, well-understood warning is
# always harmless noise here, not a real precision bug; suppress it
# precisely by message rather than broadly (a real new UserWarning should
# still surface).
warnings.filterwarnings(
    'ignore', message=r'Explicitly requested dtype float64.*', category=UserWarning,
)

# Second, distinct trigger for the complex128 warning `interpolated_image`'s
# _force_stepk/_force_maxk avoids for the maxk/stepk *size-estimation* call:
# `Image.calculate_fft()` also hardcodes `dtype=np.complex128` internally
# when actually computing an object's k-space *data* during rendering
# (drawFFT -> _drawKImage -> InterpolatedImage._kim -> calculate_fft()) --
# unlike the size-estimation case, this one is core, unavoidable work (we
# genuinely need the k-space representation to render), not something a
# caller-side parameter can skip. Verified directly the same way. Same
# reasoning as the float64 case applies: harmless given this package's
# deliberate float32-everywhere choice.
warnings.filterwarnings(
    'ignore', message=r'Explicitly requested dtype complex128.*', category=UserWarning,
)

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

  # Force stepk/maxk instead of letting jax_galsim compute them adaptively
  # from the pixel data. Traced directly (via `python -W error::UserWarning`
  # to get a full stack trace): the adaptive path (`.maxk`/`.stepk`
  # properties, triggered by drawImage's internal `_determine_wcs` call)
  # goes through `InterpolatedImage._getMaxK` -> `._kim` ->
  # `Image.calculate_fft()`, which hardcodes `dtype=np.complex128`
  # regardless of the input array's dtype or any `dtype=` we pass to
  # drawImage -- with jax_enable_x64 off (this package's deliberate
  # float32-for-GPU choice), that's silently downcast to complex64 on
  # every single call, generating a `UserWarning` each time. Forcing static
  # values skips that computation (and its warning) entirely.
  # maxk = Nyquist frequency for this pixel scale (always a safe upper
  # bound); stepk matches InterpolatedImage's own default pad_factor=4.0.
  # Verified directly against the adaptive default: matches to ~0.01%
  # relative (image) / ~1e-7 (jit/vmap'd response Jacobian, see
  # `fixed_fft_gsparams`'s docstring for the batching case this shares a
  # root cause with).
  nx, ny = image.shape
  n = max(nx, ny)
  force_maxk = math.pi / scale
  force_stepk = 2.0 * math.pi / (4.0 * n * scale)

  return galsim.InterpolatedImage(
      im, x_interpolant=x_interpolant, gsparams=gsparams,
      calculate_stepk=False, calculate_maxk=False,
      _force_stepk=force_stepk, _force_maxk=force_maxk,
  )


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
  # method='no_pixel': `image` is already rendered pixel data, so the
  # InterpolatedImage built from it already represents a pixel-convolved
  # profile (that's what makes drawing it with no_pixel reproduce the input).
  # method='fft'/'auto' would convolve by an *additional* Pixel on top of
  # that -- see `generate_mcal_image`'s comment for the full explanation and
  # how this was found.
  return obj.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel').array


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
  # method='no_pixel': see `shear`'s comment above -- `image` already
  # includes the pixel response, so don't convolve by an extra Pixel here.
  return obj.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel').array
