This is an instruction for claude code for this repository.

# Goal

This is a stale repository of an attempt to create a metacalibration code (as in Erin
Sheldon's ngmix) with automatic differentiation. It currently uses TensorFlow. We are
migrating it to JAX.

Our goal is to make autometacal into a package that can be used by a shear measurement
pipeline; autometacal should not be a pipeline in itself. The goal tests are single
galaxy stamps and proof that autodifferentiation metacal works on them (multi-galaxy
batching/vmap performance is explicitly out of scope for v1).

Knowledge base:
- This computer has GalSim, ngmix, ola, and JAX-GalSim(ish) available. See "Key findings" below.
- The original paper: https://arxiv.org/abs/1702.02600
- The follow up: arxiv.org/abs/1702.02601

# Key findings (from research spike, 2026-09-02)

- **`jax-galsim`** (PyPI package `jax-galsim`, repo `GalSim-developers/JAX-GalSim`,
  calendar-versioned, latest tested `2026.7.0`) already implements the GSObject algebra
  `ola`'s `metacal.py` relies on: `InterpolatedImage`, `Lanczos`/`Quintic`/`Cubic`/etc.
  interpolants, `Deconvolve`, `Convolve`, `Shear`/`Transform`, `drawImage`. It depends on
  the real `galsim` package (needs `galsim>=2.8`; this machine's base conda env has 2.7.2,
  which is too old and breaks import — install in an isolated env).
- Verified directly (throwaway venv): `jax.jacobian` through
  `InterpolatedImage → shear → drawImage` gives finite, correct-shaped gradients at nonzero
  shear.
- Verified directly: the full metacal-style chain
  (`InterpolatedImage → Deconvolve(psf) → Convolve → shear → Convolve(reconv_psf) → drawImage`)
  also differentiates cleanly at nonzero shear (tested g1=1e-6, 0.001, 0.01 — finite, stable,
  consistent magnitude).
- **Phase 0 spike result (2026-09-02): the zero-shear NaN is narrow, not a fundamental
  blocker.** Root cause isolated: `jax.jacobian(...)` at exactly `g1=g2=0` returns all-NaN
  **only when the input galaxy+PSF are perfectly circularly symmetric** (e.g. an unsheared
  `Gaussian`/`Moffat`, centered) — this holds whether the stamp is drawn by real GalSim or by
  jax-galsim itself, so it's tied to the object's symmetry, not the drawing backend. For any
  stamp with real asymmetry (an intrinsically sheared galaxy, in this spike `g=(0.02,-0.01)`,
  drawn with real GalSim), the Jacobian is fully finite at literal `g1=g2=0` — both at the raw
  pixel level (0/2025 NaN) and through a downstream ellipticity estimator (0/4 NaN) — and
  matches the values obtained at tiny nonzero shears (1e-3 down to 1e-7) to ~4 significant
  figures, and is in the right ballpark vs. a central finite-difference cross-check
  (`step=0.01`, ola/ngmix-style): diagonal terms matched closely (1.3225 vs 1.3186 autodiff
  vs. FD; 1.28657 vs 1.28657), off-diagonal terms agreed in order of magnitude. Likely
  mechanism: an FFT fast-path for exactly-symmetric (Hermitian) inputs that isn't correctly
  differentiated at the identity transform — not yet root-caused at the jax-galsim source
  level.
  - **Caveat that matters for this repo specifically**: metacal unit tests conventionally use
    idealized *circular* null-test galaxies (zero true shear, symmetric profile) to isolate
    shape-measurement bias — exactly the case that triggers the NaN. So this isn't just a
    theoretical edge case; it's likely to be hit by the kind of goal tests plans.md describes
    unless avoided deliberately.
  - **Mitigation, corrected in Phase 2 (the "stable from `1e-3` to `1e-7`" claim below was
    wrong)**: evaluating at `gs ≈ 1e-6`/`1e-7` does *not* avoid the instability — it just
    trades the literal-zero NaN for silent blow-up. Spot-checked directly (raw pixel
    Jacobian of `generate_mcal_image`, no NaN anywhere but col-norms/max-abs of the
    g2/gp2 columns specifically): stable and converged for `eps` in `[1e-4, 1e-2]`
    (col-norms agree to 3-4 significant figures across that range), then blows up by
    ~1-2 orders of magnitude per decade below `1e-4` (e.g. one off-diagonal response
    term went `-0.005 → +9.0 → -37.2` from `eps=1e-4→1e-6→1e-7` on a test stamp) — this
    matters specifically for the *off-diagonal/cross* response terms; diagonal terms
    stayed misleadingly stable throughout, which is presumably how the original claim
    below was reached without noticing the problem. **`autometacal/python/metacal.py` now
    defaults `eps=1e-3`** (`_response_eps`) — safely inside the stable region, still an
    order of magnitude below metacal's `0.01` calibration step. Verified end-to-end at
    this value: `get_metacal_response`'s `R`/`Rpsf`/`Repsf` diagonal terms agree with
    `get_metacal_response_finitediff` (`step=0.01`) to ~2.5%, no NaN, no blow-up.
  - Secondary/non-blocking action: file a minimal repro upstream to `GalSim-developers/JAX-GalSim`
    (circular Gaussian, deconvolve+shear+reconvolve+drawImage, `jax.jacobian` at `g=0` → NaN)
    since it looks like a genuine correctness bug there.
- Verified directly: `jax.vmap` over *different shear values* breaks with
  `ConcretizationTypeError` because `drawImage`'s automatic FFT-size selection
  (`Image.good_fft_size`) calls plain Python `math.log`/`math.ceil` on traced values. This
  does NOT block per-galaxy zero-shear Jacobians (each galaxy's Jacobian is evaluated at a
  single concrete shear value, g=0) — it only blocks vmapping across a batch of *shear
  values*, which metacal doesn't need. Batching over galaxies (not shears) is unverified but
  out of scope for v1 per the goal above.
- Conclusion: **the original plan item "develop our own Lanczos interpolators" is very
  likely unnecessary** — depend on `jax-galsim` for interpolation/shear/deconvolve instead,
  *provided* the zero-shear NaN can be fixed (Phase 0 below). If it can't be fixed cleanly
  upstream or via a local patch, fall back to a from-scratch JAX Lanczos interpolator for
  just the shear step, still using jax-galsim for everything else.
- `ola`'s `metacal.py` (`/home/andre/github/ola/ola/metacal_package/metacal.py`) is plain
  NumPy + GalSim, **no JAX anywhere in that repo**. Its `get_all_metacal` builds the
  5 images (`noshear/1p/1m/2p/2m` [+ `_psf` variants]) via `galsim.Deconvolve`/`.shear()`/
  `galsim.Convolve`/`drawImage`, and `get_metacal_response` does central finite differences
  on the resulting ellipticities — exactly the pattern plans.md wants replaced with autodiff.
  There is nothing to port from ola's response computation itself, only a structure/API
  shape to mirror: `get_all_metacal` → `measure_ellipticities` → `get_metacal_response`
  becomes, in autometacal, a single autodiff-based `get_metacal_response`.
  - **Correction from Phase 2 (read `ola/ola/metacal_package/metacal.py` directly, in
    full, rather than relying on a paraphrase)**: two structural details matter and were
    missing above. (1) ola's `"_psf"` metacal types shear the *reconvolution* psf model
    (`reconv_psfs[i]`), not the observed input psf — the current (pre-Phase-2) TF
    `generate_mcal_psf` sheared the observed `psf_images` instead, which is what
    autometacal's new `metacal.py` now deviates from TF to match ola on. (2) ola's
    `get_fixnoise` deconvolves the noise by the *input* psf, rotates 90°, shears by the
    *galaxy* shear `g` (never `gp` — fixnoise is skipped entirely for `"_psf"` types),
    rotates back -90°, then reconvolves with the (possibly gp-sheared) reconvolution psf.
    The 90°/-90° sandwich around the shear step is mathematically equivalent (spin-2
    rotation identity, confirmed by direct calculation) to directly applying `shear(-g)`
    to the unrotated deconvolved noise — i.e. ola's fixnoise noise term gets the
    *negated* galaxy shear, not the same-signed shear the TF code used. `metacal.py`'s new
    `generate_fixnoise` ports ola's literal rotate/shear/rotate-back recipe rather than
    the TF code's rotate-the-final-result-by-90° approach, and generalizes it to the joint
    `(g, gp)` autodiff case by reconvolving with the same `gp`-sheared reconvolution psf
    used for the galaxy term (ola never needs this combination since its finite-difference
    types vary `g` and `gp` one at a time).

# Migration plan (phased)

**Phase 0 — De-risk spike (done, see Key findings above).**
Reproduced the deconvolve→shear→reconvolve chain on stamps built the same way as
`tests/test_metacal.py::make_data`, isolated the zero-shear NaN to perfectly circular
inputs, and validated a cheap mitigation (evaluate the Jacobian at `gs≈1e-6` instead of
literal `0.0`) against a central finite-difference cross-check. The spike script lived in a
session-scoped scratchpad and was not committed to the repo — recreate it as a real test in
Phase 5 (that's where its logic belongs long-term: an autodiff-vs-finite-diff response
comparison on a `make_data`-style stamp, run through the eventual `gaussmom` port rather
than the crude ad-hoc moment estimator used in the spike). Remaining Phase 0 follow-up,
non-blocking: file the upstream repro with jax-galsim (circular Gaussian/Moffat, deconvolve
+shear+reconvolve+drawImage, `jax.jacobian` at `g=0` → all-NaN).

**Phase 1 — Replace `autometacal/python/galflow.py`'s hand-rolled FFT pipeline** with thin
wrappers over `jax-galsim`'s `InterpolatedImage`/`Deconvolve`/`Convolve`/`.shear()`/
`drawImage()`, keeping today's array-in/array-out function signatures (`shear`, `dilate`,
etc.) so downstream code changes minimally.

**Phase 2 — Rewrite `autometacal/python/metacal.py`** on top of Phase 1:
`generate_mcal_image`/`generate_mcal_psf`/`generate_fixnoise` become pure JAX;
`get_metacal_response` replaces TF's `tf.GradientTape()` + `batch_jacobian` with
`jax.jacobian` (jacrev — output dim 2 ≪ input dim 4) evaluated at `gs=0`. This is the
plans.md item-4 deliverable: replace the "5 images + finite differences" pattern with a
single autodiff response function. Keep a ported `get_metacal_response_finitediff` as the
correctness oracle to check the autodiff result against (and against `ola`/`ngmix`).

**Phase 3 — Port the ellipticity estimator** (`autometacal/python/gaussmom.py` +
`autometacal/python/tf_ngmix/{gmix,moments,pixels}.py`) to `jax.numpy`. Mechanical/low-risk:
no TF-specific ops beyond shape inference (`get_shape().as_list()` → `.shape`). Defer
`fitting.py` (model-fitting-based ellipticity) unless a currently-passing test exercises it.

**Phase 4 — Audit `autometacal/python/datasets/`** for TF dependencies (likely minimal —
these are GalSim-based stamp generators) and port as needed.

**Phase 5 — Tests & CI**: port `tests/test_metacal.py` / `tests/test_metacal_ngmix.py` to
the JAX path; retarget `tests/test_interpolation_gradients.py` at jax-galsim's interpolants;
retire or rescope `tests/test_tf_ngmix.py` / `tests/test_model_fitting.py` per the Phase 3
decision; update `.github/workflows/main.yml` to install `jax`/`jax-galsim`/`galsim>=2.8`
instead of the TensorFlow stack.

**Phase 6 — Cleanup**: update `setup.py` dependencies, `README.md` install instructions
(drop the custom tensorflow-addons fork steps), refresh `CLAUDE.md` to describe the new
JAX/jax-galsim architecture once the migration lands.

# Status

- [x] Step 1 (check status of GalSim-jax) — done, see Key findings.
- [x] Step 2 (plan the migration) — this document.
- [x] Phase 0 (de-risk spike) — done, see Key findings. Mitigation validated; no upstream
      fix required to proceed.
- [ ] Step 3 (Lanczos interpolators) — superseded, no longer needed: depend on jax-galsim.
- [x] Step 4 / Phase 1 (`galflow.py` rewrite: `shear`/`dilate` now thin wrappers over
      `jax_galsim` object composition — `InterpolatedImage` → `.shear()`/`.dilate()` →
      `drawImage(method='fft')`, single `(nx,ny)` stamps, no batch dim). Verified directly
      (module loaded standalone, package `__init__` chain still broken until Phase 2/3 —
      see note below): output shape/dtype preserved, `shear(0,0)` ≈ identity (max abs
      pixel diff ~0.9%, flux preserved to 1e-4), `dilate` flux-preserving, `jax.jacrev`
      finite (no NaN) at `g=(0,0)` for both an asymmetric *and* a perfectly symmetric test
      stamp through plain `shear()` — confirms the zero-shear NaN gotcha documented above
      is specific to the full deconvolve+shear+reconvolve chain (Phase 2), not `shear()`
      itself. `makekimg`/`makekpsf`/`dtype_complex` dropped (no longer needed — no manual
      FFT deconvolve pipeline in this design). Note: `metacal.py`, `util.py`, `fitting.py`,
      `gaussmom.py`, `tf_ngmix/{gmix,pixels}.py` still import the old TF-era `galflow`
      symbols and TensorFlow itself (not installed in this env) — `import autometacal`
      stays broken until Phase 2/3 land; this is expected mid-migration state, not a Phase
      1 regression.
- [x] Phase 2 (`metacal.py` rewrite). `generate_mcal_image`/`generate_mcal_psf`/
      `generate_fixnoise` rebuilt as `jax_galsim` object-composition graphs (deconvolve →
      shear → reconvolve, drawn once), matching ola's actual structure (see Key findings
      correction above) rather than the pre-migration TF behavior where it differed.
      `get_metacal_response` uses `jax.jacrev` over `gs=[g1,g2,gp1,gp2]`; kept
      `get_metacal_response_finitediff` (step=0.01) as the oracle, both returning the same
      5-tuple `(e/ellip_dict, R, Rpsf, epsf, Repsf)`. **Found and fixed a real bug in the
      Phase-0 mitigation while spot-checking** (see the correction under "the zero-shear
      NaN is narrow" above): `eps=1e-6` doesn't avoid the instability, it just swaps the
      literal-zero NaN for a silent blow-up in the off-diagonal response terms;
      `_response_eps` is now `1e-3`, empirically the stable region. Verified end-to-end
      with a standalone test stamp (asymmetric double-Gaussian galaxy, Gaussian PSF, tiny
      noise) and a crude unweighted-moments `method` stub (real `gaussmom` port is Phase
      3): no NaN anywhere, `R`/`Rpsf`/`Repsf` diagonal terms agree with the finite-diff
      oracle to ~2.5%, off-diagonal terms small and consistent (O(1e-3)) in both. Tested
      by fabricating stub `autometacal`/`autometacal.python` namespace packages in
      `sys.modules` so `galflow.py`/`metacal.py` could be loaded and exercised without the
      rest of the (still TF-broken) package — `import autometacal` itself stays broken
      until Phase 3 (gaussmom/tf_ngmix) and Phase 4 (datasets/util) land.
- [x] Phase 3 (port `gaussmom.py` + `tf_ngmix/{gmix,moments,pixels}.py` to `jax.numpy`).
      Mechanical as expected, single-stamp (no batch dim, matching Phase 1/2). Renamed
      `gmix_eval_pixel_tf` → `gmix_eval_pixel` and dropped its vestigial extra singleton
      broadcast dims (traced through by hand: they were unused generality from a batched
      ngmix layout never actually exercised here — same numerical result, less noise).
      `g1g2_to_e1e2`'s Python `if g==0.0` branch rewritten with `jnp.where` (harmless today
      since it's always called with literal-constant zeros, but a plain Python conditional
      is a landmine the moment anyone reuses `create_gmix` with a traced/nonzero shear —
      e.g. the deferred `fitting.py`). Threaded the `weights` kwarg through to
      `make_pixels` instead of the original's hardcoded `tf.ones(...)` — on inspection
      this turned out **not** to be a live bug: `get_moments` never actually reads the
      per-pixel weight column for the pure-moments calculation (only the Gaussian weight
      kernel matters there), so this is a no-op on current outputs, just now honest about
      what the `weights` kwarg does (kept for whatever later code, e.g. model fitting,
      might actually consume that column). Verified: (1) a synthetic elongated Gaussian
      (sigma_x=5, sigma_y=3) gives `e1≈0.213>0, e2≈0.007≈0` as expected; (2) full
      `galflow`+`metacal`+`tf_ngmix`+`gaussmom` pipeline end-to-end through
      `get_metacal_response`, using the *real* `get_moment_ellipticities` as `method`
      (replacing Phase 2's crude test stub) — no NaN, `R`/`Rpsf`/`Repsf` diagonal terms
      agree with the finite-diff oracle to ~2.4%, consistent with Phase 2's crude-method
      numbers. `fitting.py` still deferred (no passing test exercises it).
- [x] Phase 4 (audit `datasets/` + `util.py`). **The plan's "likely minimal TF leakage"
      assumption for `datasets/` was wrong** (grep confirmed this back in Phase 1, revisited
      properly here): `galgen.py`/`cfis.py`/`simple.py` aren't lightly TF-touched, they're
      built entirely on `tensorflow_datasets`'s `GeneratorBasedBuilder`/`BuilderConfig`
      scaffold (versioning, splits, `tfds.features.Tensor`) — a real TF-ecosystem framework
      dependency with no JAX equivalent worth porting to, wrapped around otherwise-plain
      GalSim/numpy/scipy stamp-generation logic. Asked the user how to handle it (options:
      strip TFDS and keep plain generator functions / retire entirely / leave untouched);
      **decision: retire `datasets/` entirely** — deleted the subpackage. Justification:
      only consumer was `tests/test_tf_ngmix.py` (already marked for retirement in Phase
      3/5), `setup.py` already had `tensorflow-datasets` commented out as a dependency, and
      no goal test needs it (`tests/test_metacal.py`'s `make_data` builds stamps directly
      with real GalSim, no `datasets/` dependency). `util.py`'s `noiseless_real_mcal_image`
      also retired (deleted) — only consumer was the package `__init__.py`'s re-export, and
      it's now fully redundant with a 3-line `jax_galsim` composition using Phase 1/2
      primitives (deconvolve by psf, shear, reconvolve by the *same* psf — no unique logic
      left to preserve). **Also fixed a real blocker Phase 1-3 left in place**: the package
      `__init__.py` was still eagerly importing `fitting.py` (untouched, still
      TensorFlow-only, deliberately deferred), which meant `import autometacal` would have
      stayed broken even after Phases 1-3 for a reason unrelated to any of that work.
      Dropped the eager `fitting` import (and the now-deleted `datasets`/`util` imports)
      from `autometacal/python/__init__.py` — `fitting.py` itself is untouched and still
      importable directly (`import autometacal.python.fitting`) once/if it's ported later,
      it's just no longer on the package's default import path. **Verified: `import
      autometacal` now actually works** (first time since the migration started — Phases
      1-3 could only be tested via a standalone-module-loading workaround). Re-ran the full
      Phase 3 integration test through the real import (not the workaround) and got
      identical numbers. `pytest --collect-only`: 1 module collects clean
      (`test_metacal.py`); the 6 still-TF test/experiment files that error at collection
      (`test_tf_ngmix.py`, `test_model_fitting.py`, `test_interpolation_gradients.py`,
      `test_metacal_ngmix.py`, and 3 `experiment/` scripts) are expected pre-Phase-5 state,
      not a Phase 4 regression — `test_tf_ngmix.py`/`test_model_fitting.py` are exactly the
      two files the plan already flagged for retirement/rescoping in Phase 5.
- [x] Interpolant/precision follow-up (between Phase 4 and 5, prompted by a user question
      about differentiability). **Verified `jax_galsim`'s `Quintic` and `Lanczos(11)` are
      both genuinely differentiable with correct gradients** — cross-checked `jax.jacrev`
      against central finite differences (not just "doesn't crash/NaN"): float32 agrees to
      ~0.6% at `h=1e-2`, breaking down by `h=1e-4` (see below); float64 agrees to
      ~1e-4-1e-5 relative error down to `h=1e-4`, still reasonable at `h=1e-6`, only
      breaking down at `h=1e-8`. **`galflow.py`'s default interpolant switched from
      jax_galsim's own default (`Quintic`) to `DEFAULT_INTERPOLANT = galsim.Lanczos(11)`**
      (module-level constant, used whenever `x_interpolant=None`) — matches `ola`'s actual
      choice (`interp="lanczos11"`), per user decision. Re-verified the full
      `get_metacal_response` pipeline end-to-end with the new default: no NaN, `R` diagonal
      still agrees with the finite-diff oracle to ~2.4% (unchanged from the Quintic
      numbers).
      - **Likely explains the Phase 2 `eps<1e-4` blow-up**: float32's precision floor. FD
        step sizes at or below `~1e-4` become meaningless in float32 (the pipeline's output
        literally stops changing — verified directly: `f(g0+h)==f(g0)` bit-for-bit at
        `h=1e-5` in one test), so the earlier "response Jacobian blows up below eps~1e-4"
        finding is almost certainly this same float32 precision-floor effect, not a
        jax-galsim correctness bug. float64 pushes that floor down to `~1e-6`+.
      - **The separate literal-`g=0`-NaN-for-symmetric-input bug is *not* a precision
        artifact** — confirmed it still NaNs even under float64. That's a real structural
        issue (singularity or FFT fast-path bug for exactly-Hermitian input), unaffected
        by dtype; the `eps` mitigation in `get_metacal_response` is still needed for that
        case regardless of precision.
      - **Decision: keep `dtype_real = jnp.float32` as the default** (user: this code
        targets GPUs, where float64 throughput is much worse than float32 on most
        consumer/typical hardware) — `_response_eps = 1e-3` in `metacal.py` stays as the
        practical floor rather than being tightened now that float64 was shown to allow
        smaller eps. Revisit only if a future use case specifically needs float64
        precision and can tolerate its cost.
- [x] Phase 5 (tests & CI). All 5 old test files replaced/retired:
      - `test_metacal.py` — rewritten single-stamp: compares `generate_mcal_image`
        (noshear/1p/2p) against ngmix's own `ngmix.metacal.get_all_metacal` on the same
        observation. Cross-implementation (different default interpolant — ours
        `Lanczos(11)` vs ngmix's own default `lanczos15` — plus float32 vs ngmix's
        float64), so tolerance is set relative to peak flux (5%) rather than a tiny
        absolute atol; still tight enough to catch a real regression.
      - `test_metacal_ngmix.py` — rewritten as **the goal test**: single realistic stamp
        (asymmetric galaxy+PSF, avoiding the documented zero-shear-NaN case), checks no
        NaN anywhere, `get_metacal_response` (autodiff) agrees with
        `get_metacal_response_finitediff` to 10%, and both agree with ngmix's own
        independent metacal response (via `ngmix.metacal.get_all_metacal` +
        `ngmix.gaussmom.GaussMom`, central-differenced) to 20%. Actual measured
        values for reference: autodiff `R[0,0]=0.326`, finite-diff `R[0,0]=0.319` (2.4%
        apart, consistent with everything since Phase 2), ngmix `R11=0.342` (~5-7% off
        ours — expected given independent interpolant/pipeline) — tolerances are
        generous but not toothless; a genuinely broken response (wrong sign, off by 2x,
        NaN) would still fail. Dropped the original's 1000-trial Monte Carlo m/c bias
        campaign entirely — at ~5s/call unjitted (see below) that's hours, and
        `plans.md`'s own goal is "single galaxy stamps," not a statistical bias study.
      - `test_interpolation_gradients.py` — retargeted at `jax_galsim` (from
        `tensorflow_addons.image.resampler`+`numdifftools`, and `scipy.misc.face`, which
        no longer exists in current scipy — good thing this got caught here). Verifies
        `jax.jacrev` through `galflow.shear` matches finite differences for both
        `Lanczos(11)` and `Quintic`, plus a near-(but not at)-zero-shear case. Step size
        deliberately `1e-2`, per the float32-precision-floor finding from the
        interpolant/precision follow-up above.
      - `test_tf_ngmix.py` → **rescoped to `test_gaussmom.py`**: compares
        `get_moment_ellipticities` against `ngmix.gaussmom.GaussMom` directly (20 random
        stamps), tight tolerance (rtol=1e-3) since both implement the same weighted-moments
        algorithm — passes, confirming the Phase 3 port is numerically faithful. (The old
        file referenced `autometacal.datasets.galaxies.make_data`, which never existed
        anywhere in the repo — already dead code pre-migration.)
      - `test_model_fitting.py` — retired (deleted). `fitting.py` stays deferred/untouched
        per Phase 3; this test also depended on the defunct external `galflow` pip package
        (`import galflow as gf`, not this repo's `autometacal.python.galflow`), unrelated
        to this migration.
      - Added `pytest.ini` (`testpaths = tests`) — bare `pytest` (what CI runs) was
        otherwise also collecting `experiment/*.py` (still-TF scratch scripts, e.g.
        `test_runners.py`, `pujol_test.py` match pytest's default `test_*`/`*_test.py`
        patterns) and failing at import. Without this fix the CI workflow below would
        still fail outright regardless of how correct the real tests are.
      - `.github/workflows/main.yml` updated: installs `galsim>=2.8`, `ngmix`,
        `jax`/`jaxlib`/`jax-galsim` instead of the TensorFlow/tensorflow-addons/GalFlow
        stack; bumped `pyver` matrix from `3.8` to `3.12` (matches this dev environment;
        `jax-galsim` itself requires `>=3.11`).
      - Full suite: `pytest` (bare, matching CI) → **6 passed** in ~44s, no errors, no
        skips. First time the whole test suite has been green since the migration started.
- [x] Phase 6 (cleanup). `setup.py`: description updated ("Metacalibration by automatic
      differentiation, in JAX"), `install_requires` set to
      `['jax', 'jaxlib', 'jax-galsim', 'galsim>=2.8']` (was commented out entirely before).
      `README.md`: dropped the TensorFlow/GalFlow/custom-tensorflow-addons-fork install
      instructions, replaced with the actual `pip install jax jaxlib jax-galsim
      "galsim>=2.8"` (+ `ngmix` for tests) steps, matching the CI workflow. `CLAUDE.md`:
      fully rewritten to describe the finished JAX/jax_galsim architecture — updated
      commands, the real module layout (`galflow.py`/`metacal.py`/`gaussmom.py`+
      `tf_ngmix/`/deferred `fitting.py`, `datasets/`+`util.py` gone), the `_response_eps`
      gotcha and why it exists, and a per-file description of what each test in `tests/`
      actually checks (so this doesn't need re-deriving from scratch next time). Full
      `pytest` re-run after all doc/config edits: still 6 passed, ~46s, no regressions.
      **Migration (all 6 phases + the interpolant/precision follow-up) is now complete.**
