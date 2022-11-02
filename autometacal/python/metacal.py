import tensorflow as tf
from autometacal.python.galflow import shear, dilate, makekimg, makekpsf, dtype_complex, dtype_real

padfactors = 5

def generate_mcal_image(gal_images,
                        psf_images,
                        reconvolution_psf_images,
                        g, gp,
                        padfactor=padfactors):
  """ Generate a metacalibration image given input and target PSFs.
  
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
    
  #pad images
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
  gal_images = tf.pad(gal_images,paddings)
  psf_images = tf.pad(psf_images,paddings)
  reconvolution_psf_images = tf.pad(reconvolution_psf_image,paddings)
  
  #Convert galaxy images to k space
  imk = makekimg(gal_images,dtype=dtype_complex)#the fftshift is to put the 0 frequency at the center of the k image
  
  #Convert psf images to k space  
  kpsf = makekpsf(psf_images,dtype=dtype_complex)

  #Convert reconvolution psf image to k space 
  krpsf = makekpsf(reconvolution_psf_images, dtype=dtype_complex)

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype=dtype_complex)
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



def generate_mcal_psf(psf_images, gp, padfactor=padfactors):
  """ Generate a metacalibration psf image """

  #cast stuff as float32 tensors
  batch_size, nx, ny = psf_images.get_shape().as_list() 
  gp = tf.convert_to_tensor(gp, dtype=dtype_real)  
  psf_images = tf.convert_to_tensor(psf_images, dtype=dtype_real)
    
  #pad images
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
  psf_images = tf.pad(psf_images,paddings)
  
  #Convert psf images to k space  
  kpsf = makekpsf(psf_images,dtype=dtype_complex)

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype=dtype_complex)
  mask = tf.expand_dims(mask, axis=0)
  
  # Apply shear to the  kpsf image
  krpsf_sheared = shear(tf.expand_dims(kpsf,-1), gp[...,0], gp[...,1])[...,0]    
  
  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(krpsf_sheared * mask ))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))
  return img[:,fact*nx:-fact*nx,fact*ny:-fact*ny]

def generate_fixnoise(noise,psf_images,reconvolution_psf_image,g,gp):
  """generate fixnoise image by applying same method and rotating by 90deg"""
  noise = tf.convert_to_tensor(noise,dtype=dtype_real)
  shearednoise = generate_mcal_image(
    noise, psf_images, reconvolution_psf_image, g, gp)
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
  #check/cast as tensors
  gal_images = tf.convert_to_tensor(gal_images, dtype=dtype_real)
  psf_images = tf.convert_to_tensor(psf_images, dtype=dtype_real)
  batch_size, _ , _ = gal_images.get_shape().as_list()
  #create shear tensor: 0:2 are shears, 2:4 are PSF distortions
  gs = tf.zeros([batch_size,4])
  
  #measurement and callibration of psf ellipticities
  with tf.GradientTape() as tape:
    gp=gs[:,2:4]
    tape.watch(gp)
    mcal_psf_image = generate_mcal_psf(
      psf_images,
      gp
    )
    epsf = method(mcal_psf_image)
   
  Repsf = tape.batch_jacobian(epsf,gp)
  
  #measurement and callibration of galaxy stamps
  with tf.GradientTape() as tape:
    tape.watch(gs)
    reconvolution_psf_image = dilate(reconvolution_psf_image[...,tf.newaxis],1.001)[...,0]
    mcal_image = generate_mcal_image(gal_images,
                                     psf_images,
                                     reconvolution_psf_image,
                                     gs[:,0:2],gs[:,2:4])
    
    mcal_image += generate_fixnoise(noise,
                                    psf_images,
                                    reconvolution_psf_image,
                                    gs[:,0:2],gs[:,2:4])
    e = method(mcal_image)

  Rs = tape.batch_jacobian(e, gs)
  R, Rpsf = Rs[...,0:2], Rs[...,2:4]
  return e, R, Rpsf, epsf, Repsf




















def get_metacal_response_finitediff(gal_image,psf_image,reconvolution_psf,noise,step,step_psf,method):
  """
  Gets shear response as a finite difference operation, 
  instead of automatic differentiation.
  """

  batch_size, _ , _ = gal_image.get_shape().as_list()
  
  #create shear batches to match transformations
  step_batch = tf.constant(step,shape=(batch_size,1),dtype=dtype_real)
    
  noshear = tf.zeros([batch_size,2],dtype=dtype_real)
  step1p = tf.pad(step_batch,[[0,0],[0,1]])
  step1m = tf.pad(-step_batch,[[0,0],[0,1]])
  step2p = tf.pad(step_batch,[[0,0],[1,0]])
  step2m = tf.pad(-step_batch,[[0,0],[1,0]])  
    
  #full mcal image generator
  def generate_mcal_finitediff(gal,psf,rpsf,noise,gs,gp):
    #rpsf = dilate(rpsf[...,tf.newaxis],1.+2.*tf.norm(gs,axis=1))[...,0]
    
    mcal_image = generate_mcal_image(
      gal, psf, rpsf, gs, gp
    ) + generate_fixnoise(
      noise, psf, rpsf, gs, gp)
    
    return mcal_image
  
  #noshear
  reconvolution_psf_image = dilate(reconvolution_psf[...,tf.newaxis],1.+2.*tf.norm(step1p,axis=1))[...,0]
  img0s = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,noshear)
  g0s = method(img0s)
  
  #shear response
  reconvolution_psf_image = dilate(reconvolution_psf[...,tf.newaxis],1.+2.*tf.norm(step1p,axis=1))[...,0]
  img1p = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step1p,noshear)
  img1m = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step1m,noshear)
  img2p = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step2p,noshear)
  img2m = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step2m,noshear)
  
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
     [R12,R22]],dtype=dtype_real)
  ) 
  
  #psf response
  reconvolution_psf_image = dilate(reconvolution_psf[...,tf.newaxis],1.+2.*tf.norm(noshear,axis=1))[...,0]
  img1p_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step1p)
  img1m_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step1m)
  img2p_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step2p)
  img2m_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step2m)

  g1p_psf = method(img1p_psf)
  g1m_psf = method(img1m_psf)
  g2p_psf = method(img2p_psf)
  g2m_psf = method(img2m_psf)
 
  Rpsf11 = (g1p_psf[:,0]-g1m_psf[:,0])/(2*step_psf)
  Rpsf21 = (g1p_psf[:,1]-g1m_psf[:,1])/(2*step_psf) 
  Rpsf12 = (g2p_psf[:,0]-g2m_psf[:,0])/(2*step_psf)
  Rpsf22 = (g2p_psf[:,1]-g2m_psf[:,1])/(2*step_psf)
 
  Rpsf = tf.transpose(tf.convert_to_tensor(
    [[Rpsf11,Rpsf21],
     [Rpsf12,Rpsf22]],dtype=dtype_real)
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