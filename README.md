# autometacal

[![CI](https://github.com/CosmoStat/autometacal/actions/workflows/main.yml/badge.svg)](https://github.com/CosmoStat/autometacal/actions/workflows/main.yml)

Metacalibration and shape measurement by automatic differentiation

Project led by [@andrevitorelli](https://github.com/andrevitorelli)


## Requirements

This project relies on [jax-galsim](https://github.com/GalSim-developers/JAX-GalSim) (which
itself depends on [GalSim](https://github.com/GalSim-developers/GalSim) `>=2.8`) for
image interpolation, shearing and (de)convolution, and on [JAX](https://github.com/jax-ml/jax)
for automatic differentiation:
```bash
$ pip install jax jaxlib jax-galsim "galsim>=2.8"
```

To run the tests, [ngmix](https://github.com/esheldon/ngmix) is also required, as several
tests cross-check autometacal's results against it:
```bash
$ pip install ngmix
```
