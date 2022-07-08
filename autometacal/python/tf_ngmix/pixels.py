"""
Observation implementations from ngmix ported to tensorflow

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""

import tensorflow as tf
from autometacal.python.galflow import dtype_real, dtype_int

def make_pixels(images, weights, centre, pixel_scale):
  
  batch_size, img_x_size, img_y_size = images.get_shape().as_list()
  
  #image shape info
  img_size = img_x_size * img_y_size
  
  #apply jacobian (currently constant!)
  centre_x = tf.cast(centre[0],dtype=dtype_real)
  centre_y = tf.cast(centre[1],dtype=dtype_real)
  grid = tf.cast(tf.meshgrid(tf.range(img_x_size,dtype=dtype_int),tf.range(img_y_size,dtype=dtype_int)), dtype_real)
  X = (grid[0]-centre_x)*pixel_scale
  Y = (grid[1]-centre_y)*pixel_scale
  Xs=tf.tile(X[tf.newaxis],[batch_size,1,1])
  Ys=tf.tile(Y[tf.newaxis],[batch_size,1,1])
  pixel_scale = tf.cast(pixel_scale,dtype=dtype_real)
  batch_size = tf.cast(batch_size,dtype=dtype_int)
  img_size = tf.cast(img_size,dtype=dtype_int)
  #fill pixels
  pixels = tf.stack([tf.reshape(Xs,[batch_size,-1]),
            tf.reshape(Ys,[batch_size,-1]),
            tf.fill([batch_size,img_size],pixel_scale*pixel_scale), 
            tf.reshape(images,[batch_size,-1]),
            tf.reshape(weights,[batch_size,-1])],axis=-1)
  
  return pixels