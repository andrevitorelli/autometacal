import tensorflow as tf
from autometacal.python.galflow import shear, dilate

def makekimg(img,dtypes='complex64'):
  im_shift = tf.signal.ifftshift(img,axes=[1,2]) # The ifftshift is to remove the phase for centered objects
  im_complex = tf.cast(im_shift, dtypes)
  im_fft = tf.signal.fft2d(im_complex)
  imk = tf.signal.fftshift(im_fft, axes=[1,2])#the fftshift is to put the 0 frequency at the center of the k image
  return imk

def makekpsf(psf,dtypes='complex64'):
  psf_complex = tf.cast(psf, dtype=dtypes)
  psf_fft = tf.signal.fft2d(psf_complex)
  psf_fft_abs = tf.abs(psf_fft)
  psf_fft_abs_complex = tf.cast(psf_fft_abs,dtype=dtypes)
  kpsf = tf.signal.fftshift(psf_fft_abs_complex,axes=[1,2])
  
  return kpsf

def generate_mcal_image(gal_images,
                        psf_images,
                        reconvolution_psf_images,
                        g, gp,
                        padfactor=3):
  """ Generate a metacalibrated image given input and target PSFs.
  
  Args: 
    gal_images: tf.Tensor or np.array
      (batch_size, N, N ) image of galaxies
    psf_images: tf.Tensor or np.array
      (batch_size, N, N ) image of psf model
    reconvolution_psf_image: tf.Tensor
      (N, N ) tensor of reconvolution psf model
    g: tf.Tensor or np.array
    [batch_size, 2] input shear
  Returns:
    img: tf.Tensor
      tf tensor containing image of galaxy after deconvolution by psf_deconv, 
      shearing by g, and reconvolution with reconvolution_psf_image.
  
  """
  #cast stuff as float32 tensors
  batch_size, nx, ny = gal_images.get_shape().as_list() 
  g = tf.convert_to_tensor(g, dtype=tf.float32)  
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)  
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_images, dtype=tf.float32)
  
  #dilate reconvolution psf
  dilate_fact = 1. + 2.*tf.reduce_sum(g**2,axis=1)
  reconvolution_psf_image = dilate(reconvolution_psf_image[...,tf.newaxis],dilate_fact)[...,0]
  
  #pad images
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
  gal_images = tf.pad(gal_images,paddings)
  psf_images = tf.pad(psf_images,paddings)
  reconvolution_psf_images = tf.pad(reconvolution_psf_image,paddings)
  
  #Convert galaxy images to k space
  imk = makekimg(gal_images,dtypes='complex64')#the fftshift is to put the 0 frequency at the center of the k image
  
  #Convert psf images to k space  
  kpsf = makekpsf(psf_images)

  #Convert reconvolution psf image to k space 
  krpsf = makekpsf(reconvolution_psf_images)

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype='complex64')
  mask = tf.expand_dims(mask, axis=0)
  
  # Deconvolve image from input PSF
  im_deconv = imk /(kpsf+1e-10) * mask

  # Apply shear to the  deconv image
  im_sheared = shear(tf.expand_dims(im_deconv,-1), g[...,0], g[...,1])[...,0]  

  # Apply shear to the  kpsf image
  krpsf_sheared = shear(tf.expand_dims(krpsf,-1), gp[...,0], gp[...,1])[...,0]    
  
  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(im_sheared * krpsf_sheared * mask ))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))
  return img[:,fact*nx:-fact*nx,fact*ny:-fact*ny]

def generate_fixnoise(noise,psf_images,reconvolution_psf_image,g,gp):
  noise = tf.convert_to_tensor(noise,dtype=tf.float32)
  shearednoise = generate_mcal_image(noise,
                                   psf_images,
                                   reconvolution_psf_image,g,gp )
  rotshearednoise = tf.image.rot90(shearednoise[...,tf.newaxis],k=-1)[...,0]
  return rotshearednoise

def get_metacal_response(gal_images,
                         psf_images,
                         reconvolution_psf_image,
                         noise,
                         method):
  """
  Convenience function to compute the shear response
  """  
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)
  batch_size, _ , _ = gal_images.get_shape().as_list()
  gs = tf.zeros([batch_size,4])
  epsf = method(reconvolution_psf_image) #doesn't work - biased
  with tf.GradientTape() as tape:
    tape.watch(gs)
    # Measure ellipticity under metacal
    mcal_image = generate_mcal_image(gal_images,
                                   psf_images,
                                   reconvolution_psf_image,
                                   gs[:,0:2],gs[:,2:4])
    
    mcal_image += generate_fixnoise(noise,
                                    psf_images,
                                    reconvolution_psf_image,gs[:,0:2],gs[:,2:4])
    
    e = method(mcal_image)

  Rs = tape.batch_jacobian(e, gs)
  R, Rpsf = Rs[...,0:2], Rs[...,2:4]
  return e, epsf, R, Rpsf


def get_metacal_response_finitediff(gal_image,psf_image,reconv_psf_image,noise,step,step_psf,method):
  """
  Gets shear response as a finite difference operation, 
  instead of automatic differentiation.
  """

  batch_size, _ , _ = gal_image.get_shape().as_list()
  step_batch = tf.constant(step,shape=(batch_size,1),dtype=tf.float32)
  
  noshear = tf.zeros([batch_size,2])
  step1p = tf.pad(step_batch,[[0,0],[0,1]])
  step1m = tf.pad(-step_batch,[[0,0],[0,1]])
  step2p = tf.pad(step_batch,[[0,0],[1,0]])
  step2m = tf.pad(-step_batch,[[0,0],[1,0]])  
  
  #noshear
  img0s = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    noshear,noshear
  )  
  
  #############SHEAR RESPONSE
  #1p
  img1p = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    step1p,noshear
  )+ generate_fixnoise(
     noise,
     psf_image,)
  
  #1m
  img1m = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    [[-step,0]],noshear
  )+ generate_fixnoise(
     noise,
     psf_image,)
  
  #2p
  img2p = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    [[0,step]],noshear
  )+ generate_fixnoise(
     noise,
     psf_image,)
  
  #2m
  img2m = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    [[0,-step]],noshear
  )+ generate_fixnoise(
     noise,
     psf_image,)
  
  g0s = method(img0s)
  g1p = method(img1p)
  g1m = method(img1m)
  g2p = method(img2p)
  g2m = method(img2m)
  
  R11 = (g1p[:,0]-g1m[:,0])/(2*step)
  R21 = (g1p[:,1]-g1m[:,1])/(2*step) 
  R12 = (g2p[:,0]-g2m[:,0])/(2*step)
  R22 = (g2p[:,1]-g2m[:,1])/(2*step)
  
  R = tf.transpose(tf.convert_to_tensor(
    [[R11,R21],
     [R12,R22]],dtype=tf.float32)
  ) 
  
  ####################PSF RESPONSE  
  #1p_psf
  img1p_psf = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    noshear,step1p
  )+ generate_fixnoise(
    noise,
    psf_image,
    noshear,step1p  
  )

  #1m_psf
  img1m_psf = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    noshear,step1m
  )+ generate_fixnoise(
    noise,
    psf_image,
    noshear,step1m
  )
  #2p_psf
  img2p_psf = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    noshear,step2p
  )+ generate_fixnoise(
    noise,
    psf_image,
    noshear,step2p
  )
  #2m_psf
  img2m_psf = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    noshear,step2m
  )+ generate_fixnoise(
      noise,
    psf_image,
    noshear,step2m
  )
  
  g1p_psf = method(img1p_psf)
  g1m_psf = method(img1m_psf)
  g2p_psf = method(img2p_psf)
  g2m_psf = method(img2m_psf)
 
  Rpsf11 = (g1p_psf[:,0]-g1m_psf[:,0])/(2*step_psf)
  Rpsf21 = (g1p_psf[:,1]-g1m_psf[:,1])/(2*step_psf) 
  Rpsf12 = (g2p_psf[:,0]-g2m_psf[:,0])/(2*step_psf)
  Rpsf22 = (g2p_psf[:,1]-g2m_psf[:,1])/(2*step_psf)
 
  R = tf.transpose(tf.convert_to_tensor(
    [[R11,R21],
     [R12,R22]],dtype=tf.float32)
  )
  
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

  return ellip_dict, R, Rpsf