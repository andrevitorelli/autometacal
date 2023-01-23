import numpy as np
import tensorflow as tf
import galsim

def pujol(ellip_method,gal_generator,psf_generator,gal_noise_img,psf_noise_img,**kwargs):
  options = {  
  'step': 0.02,
  'psf_fwhm' : 0.7,
  'method' : 'auto',
  'stamp_size' : 51,
  'scale' : 0.263
   
  }
  options.update(kwargs)
  
  #make a real psf
  psf = psf_generator(psf_fwhm=options['psf_fwhm'])
  
  #get some galaxy model
  gal0 = gal_generator()
  
  #shear model with step in each component
  #shear g1
  gal1 = gal0.shear(g1=options['step'],g2=0)
  #shear g2
  gal2 = gal0.shear(g1=0,g2=options['step'])
  
  
  gal0 = galsim.Convolve([gal0,psf])
  gal1 = galsim.Convolve([gal1,psf])
  gal2 = galsim.Convolve([gal2,psf])
  
  #get array from images
  gal0im = gal0.drawImage(nx=options['stamp_size'],ny=options['stamp_size'],scale=options['scale'],method=options['method']).array + gal_noise_img
  gal1im = gal1.drawImage(nx=options['stamp_size'],ny=options['stamp_size'],scale=options['scale'],method=options['method']).array + gal_noise_img 
  gal2im = gal2.drawImage(nx=options['stamp_size'],ny=options['stamp_size'],scale=options['scale'],method=options['method']).array + gal_noise_img
  
  #make an estimation of the psf
  psf_obs = psf.drawImage(nx=options['stamp_size'],ny=options['stamp_size'],scale=options['scale'],method=options['method']).array + psf_noise_img

  #get estimated ellipticities
  print(gal0im.shape)
  g0p = ellip_method(gal0im).numpy()[0]
  g1p = ellip_method(gal1im).numpy()[0]
  g2p = ellip_method(gal2im).numpy()[0]
  

  #return residual R
  R11 = (g1p[0]-g0p[0])/(options['step'])
  R21 = (g1p[1]-g0p[1])/(options['step']) 
  R12 = (g2p[0]-g0p[0])/(options['step'])
  R22 = (g2p[1]-g0p[1])/(options['step'])
  
  Residual = np.array(
    [[R11,R12],
     [R21,R22]])
  
  return Residual