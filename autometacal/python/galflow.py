_INTERPOLATOR = "bernsteinquintic"
import tensorflow as tf
from tensorflow_addons.image import resampler
 
def shear(img,g1,g2,interpolation=_INTERPOLATOR):
  """
   Applies a shear g1, g2 on an image. 
   
   Args: 
     img: tf.tensor [batch,nx,ny,channels] float32
       batch of images
     g1, g2: shears to be applied
     
  Returns:
    sheared: tf.tensor [batch,nx,ny,channels]
      sheared image

  """
  
  _ , nx, ny, _ = img.get_shape().as_list()
  g1 = tf.convert_to_tensor(g1, dtype=tf.float32)
  g2 = tf.convert_to_tensor(g2, dtype=tf.float32)
  gsqr = g1**2 + g2**2
  
  # Building a batched jacobian
  jac = tf.stack([ 1. + g1, g2,
                g2, 1. - g1], axis=1) / tf.expand_dims(tf.sqrt(1.- gsqr),1)
  jac = tf.reshape(jac, [-1,2,2]) 

  # Inverting these jacobians to follow the TF definition
  if img.dtype == tf.complex64:
    transform_matrix = tf.transpose(jac,[0,2,1])
  else:
    transform_matrix = tf.linalg.inv(jac)
  
  #define a grid at pixel positions
  warp = tf.stack(tf.meshgrid(tf.linspace(0.,tf.cast(nx,tf.float32)-1.,nx), 
                              tf.linspace(0.,tf.cast(ny,tf.float32)-1.,ny)),axis=-1)[..., tf.newaxis]

  #get center
  center = tf.convert_to_tensor([[nx/2],[ny/2]],dtype=tf.float32)
  
  #displace center to origin
  warp = warp - center
  
  #if fourier, no half pixel shift needed
  warp -= int(img.dtype != tf.complex64)*.5

  #apply shear
  warp = tf.matmul(transform_matrix[:, tf.newaxis, tf.newaxis, ...], warp)[...,0]

  #return center
  warp = warp + center[...,0] 
 
  #if fourier, no half pixel shift needed
  warp -= int(img.dtype != tf.complex64)*.5
      
  #apply resampler
  if img.dtype == tf.complex64:
    a = resampler(tf.math.real(img),warp,interpolation)
    b = resampler(tf.math.imag(img),warp,interpolation)
    sheared = tf.complex(a,b)
  else:
    sheared = resampler(img,warp,interpolation)
  return sheared


def dilate(img,factor,interpolator=_INTERPOLATOR):
  """ Dilate images by some factor, preserving the center. 
  
  Args:
    img: tf tensor containing [batch_size, nx, ny, channels] images
    factor: dilation factor (factor >= 1)
  
  Returns:
    dilated: tf tensor containing [batch_size, nx, ny, channels] images dilated by factor around the centre
  """
  img = tf.convert_to_tensor(img,dtype=tf.float32)
  batch_size, nx, ny, _ = img.get_shape()

  #x
  sampling_x = tf.linspace(0.,tf.cast(nx,tf.float32)-1.,nx)[tf.newaxis]
  centred_sampling_x = sampling_x - nx//2
  batched_sampling_x = tf.repeat(centred_sampling_x,batch_size,axis=0)
  rescale_sampling_x = tf.transpose(batched_sampling_x) / factor
  reshift_sampling_x = tf.transpose(rescale_sampling_x)+nx//2
  #y
  sampling_y = tf.linspace(0.,tf.cast(ny,tf.float32)-1.,ny)[tf.newaxis]
  centred_sampling_y = sampling_y - ny//2
  batched_sampling_y = tf.repeat(centred_sampling_y,batch_size,axis=0)
  rescale_sampling_y = tf.transpose(batched_sampling_y) / factor
  reshift_sampling_y = tf.transpose(rescale_sampling_y)+ny//2

  meshx = tf.transpose(tf.repeat([reshift_sampling_x],nx,axis=0),[1,0,2])
  meshy = tf.transpose(tf.transpose(tf.repeat([reshift_sampling_y],ny,axis=0)),[1,0,2])
  warp = tf.transpose(tf.stack([meshx,meshy]),[1,2,3,0])

  dilated= resampler(img,warp,interpolator)
  
  return tf.transpose(tf.transpose(dilated) /factor**2)