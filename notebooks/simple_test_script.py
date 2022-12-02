#os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#utilities
from multiprocessing import Pool
from scipy.stats import linregress
import tqdm

#this is us
import autometacal as amc
import tensorflow_datasets as tfds
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#...vs them!
import ngmix
import galsim

rng = np.random.RandomState(31415)

#stamp properties
stamp_size = 51
batch_size = 30
scale = 0.263
window_fwhm = 1.2

#computation settings
ncpus = 8

#test settings
g_psf =  np.array([0.01,0.0],dtype='float32')
noise_level = 1e-6
fixnoise = True
shapenoise = False

psf_noise_level = noise_level/1000
psf_fwhm = 0.7


#datasetless observer



def ngmix_booter_moments(rng,**kwargs):
  '''create ngmix bootstraper'''
  options = {
    'psf': 'gauss',
    'types': ['noshear', 
              '1p', '1m', '2p', '2m'#, '1p_psf', '1m_psf', '2p_psf', '2m_psf'
              ],
    'scale': scale,
        
  }

  weight_fwhm = window_fwhm
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

  # these "runners" run the measurement code on observations
  psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
  runner = ngmix.runners.Runner(fitter=fitter)
 
  boot = ngmix.metacal.MetacalBootstrapper(
      runner=runner,
      psf_runner=psf_runner,
      rng=rng,
      psf=options['psf'],
      types=options['types'],
    fixnoise = fixnoise
  )
  return boot

boot = ngmix_booter_moments(rng)

def ngmix_step(obs):
  '''runs 1 ngmix measurement in 1 observation'''
  resdict, _ = boot.go(obs)
  return resdict


def get_metacal_response_ngmix_moments(resdict,shear_type = 'e',PSF=False):
  '''gets the shear response for ngmix results'''
  step=0.01
 
  #noshear
  g0s = np.array([resdict['noshear'][shear_type][0], resdict['noshear'][shear_type][1]])
  
  #shear
  g1p = np.array([resdict['1p'][shear_type][0], resdict['1p'][shear_type][1]])
  g1m = np.array([resdict['1m'][shear_type][0], resdict['1m'][shear_type][1]])
  g2p = np.array([resdict['2p'][shear_type][0], resdict['2p'][shear_type][1]])
  g2m = np.array([resdict['2m'][shear_type][0], resdict['2m'][shear_type][1]])    
  
  R11 = (g1p[0]-g1m[0])/(2*step)
  R21 = (g1p[1]-g1m[1])/(2*step) 
  R12 = (g2p[0]-g2m[0])/(2*step)
  R22 = (g2p[1]-g2m[1])/(2*step)
  
  R = np.array(
    [[R11,R12],
     [R21,R22]])
  
  if PSF:
      #PSF
      g1p_psf = np.array([resdict['1p_psf'][shear_type][0], resdict['1p_psf'][shear_type][1]])
      g1m_psf = np.array([resdict['1m_psf'][shear_type][0], resdict['1m_psf'][shear_type][1]])
      g2p_psf = np.array([resdict['2p_psf'][shear_type][0], resdict['2p_psf'][shear_type][1]])
      g2m_psf = np.array([resdict['2m_psf'][shear_type][0], resdict['2m_psf'][shear_type][1]])    

      R11_psf = (g1p_psf[0]-g1m_psf[0])/(2*step)
      R21_psf = (g1p_psf[1]-g1m_psf[1])/(2*step) 
      R12_psf = (g2p_psf[0]-g2m_psf[0])/(2*step)
      R22_psf = (g2p_psf[1]-g2m_psf[1])/(2*step)  

      ellip_dict = {
        'noshear':g0s,
        '1p':g1p,
        '1m':g1m,
        '2p':g2p,
        '2m':g2m,
        '1p_psf':g1p_psf,
        '1m_psf':g1m_psf,
        '2p_psf':g2p_psf,
        '2m_psf':g2m_psf,    
      } 
      Rpsf = np.array(
        [[R11_psf,R12_psf],
         [R21_psf,R22_psf]])
      return ellip_dict, R, Rpsf
  else:
      ellip_dict = {
        'noshear':g0s,
        '1p':g1p,
        '1m':g1m,
        '2p':g2p,
        '2m':g2m,  
      } 
      
      return ellip_dict, R, np.eye(2)