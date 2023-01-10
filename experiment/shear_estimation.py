import numpy as np

def metacal_shear(e,R):
  return np.linalg.inv(R) @ e

def metacal_shear_psf(e,R,ep, Rp):
  return np.linalg.inv(R) @ (e - Rp @ ep)

def bootstrap(x,resample):
  return x[resample].mean(axis=0)

def evaluator(ellips, R, ellips_psf, Repsf, Rpsf):
  nboot=200 #bootstraps make it look professional
  size=ellips.shape[0]
  resample = np.random.randint(0,size,[nboot,size])
  
  e = bootstrap(ellips,resample).reshape(-1,2,1)
  ep = bootstrap(ellips_psf, resample).reshape(-1,2,1)
  
  
  Rep = bootstrap(Repsf, resample)
  R = bootstrap(R,resample)
  Rp = bootstrap(Rpsf,resample)
  
  #Response-calibrated shears
  shears = metacal_shear(e,R)
  shear = shears.mean(axis=0)
  shear_err = shears.std(axis=0)
  
  epsf = metacal_shear(ep.mean(axis=0),Rep.mean(axis=0))
  #PSF-corrected shears
  shearcorrs = metacal_shear_psf(e,R,epsf, Rp)
  shearcorr = shearcorrs.mean(axis=0)
  shearcorr_err = shearcorrs.std(axis=0)
  
  return shear[...,0], shear_err[...,0],  shearcorr[...,0], shearcorr_err[...,0]