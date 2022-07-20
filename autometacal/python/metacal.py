# This file will contain the tools needed to generate a metacal image
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from autometacal.python.galflow import shear, dilate, makekimg, makekpsf, dtype_complex, dtype_real 

def generate_mcal_image(gal_images,
                        psf_images,
                        reconvolution_psf_images,
                        g,
                        padfactor=5):
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
  g = tf.convert_to_tensor(g, dtype=dtype_real)  
  gal_images = tf.convert_to_tensor(gal_images, dtype=dtype_real)  
  psf_images = tf.convert_to_tensor(psf_images, dtype=dtype_real)
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_images, dtype=dtype_real)
  
  #dilate reconvolution psf
  dilate_fact = 1.0 + 2.*tf.math.sqrt(tf.reduce_sum(g**2,axis=1))
  reconvolution_psf_image = dilate(reconvolution_psf_image[...,tf.newaxis],dilate_fact)[...,0]
  
  #pad images
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
  gal_images = tf.pad(gal_images,paddings)
  psf_images = tf.pad(psf_images,paddings)
  reconvolution_psf_images = tf.pad(reconvolution_psf_image,paddings)
  
  #Convert galaxy images to k space
  imk = makekimg(gal_images,dtype=dtype_complex)#the fftshift is to put the 0 frequency at the center of the k image
  
  #Convert psf images to k space  
  kpsf = makekimg(psf_images,dtype=dtype_complex)

  #Convert reconvolution psf image to k space 
  krpsf = makekimg(reconvolution_psf_images, dtype=dtype_complex)

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype=dtype_complex)
  mask = tf.expand_dims(mask, axis=0)

  # Deconvolve image from input PSF
  im_deconv = imk * 1./(kpsf*mask+1e-10)

  # Apply shear
  im_sheared = shear(tf.expand_dims(im_deconv,-1), g[...,0], g[...,1])[...,0]

  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(im_sheared * krpsf * mask))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))

  return img[:,fact*nx:-fact*nx,fact*ny:-fact*ny]

def get_metacal_response(gal_images,
                         psf_images,
                         reconvolution_psf_image,
                         method):
  """
  Convenience function to compute the shear response
  """  
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_image, dtype=tf.float32)
  batch_size, _ , _ = gal_images.get_shape().as_list()
  g = tf.zeros([batch_size, 2])
  with tf.GradientTape() as tape:
    tape.watch(g)
    # Measure ellipticity under metacal
    e = method(generate_mcal_image(gal_images,
                                   psf_images,
                                   reconvolution_psf_image,
                                   g))
    
  # Compute response matrix

  R = tape.batch_jacobian(e, g)
  return e, R


def get_metacal_response(gal_images,
                         psf_images,
                         reconvolution_psf_image,
                         method):
  """
  Convenience function to compute the shear response
  """  
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_image, dtype=tf.float32)
  batch_size, _ , _ = gal_images.get_shape().as_list()
  g = tf.zeros([batch_size, 2])
  with tf.GradientTape() as tape:
    tape.watch(g)
    # Measure ellipticity under metacal
    e = method(generate_mcal_image(gal_images,
                                   psf_images,
                                   reconvolution_psf_image,
                                   g))
    
  # Compute response matrix

  R = tape.batch_jacobian(e, g)
  return e, R


def get_metacal_response_finitediff(gal_image,
                                    psf_image,
                                    reconv_psf_image,
                                    step,
                                    method):
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
    
  img0s = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    noshear
  ) 
  
  shears1p = step1p
  img1p = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears1p
  )
  
  shears1m = step1m 
  img1m = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears1m
  ) 
  
  shears2p = step2p 
  img2p = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears2p
  )
  
  shears2m = step2m 
  img2m = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears2m
  ) 
  
  g0s = method(img0s)
  g1p = method(img1p)
  g1m = method(img1m)
  g2p = method(img2p)
  g2m = method(img2m)
  
  R11 = (g1p[:,0]-g1m[:,0])/(2*step)
  R21 = (g1p[:,1]-g1m[:,1])/(2*step) 
  R12 = (g2p[:,0]-g2m[:,0])/(2*step)
  R22 = (g2p[:,1]-g2m[:,1])/(2*step)
 
  #N. B.:The matrix is correct. 
  #The transposition will swap R12 with R21 across a batch correctly.
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
  } 
  
  return ellip_dict, R