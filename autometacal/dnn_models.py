import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tf.keras.preprocessing as kpre


def Ribli19(imsize=50, n_target=2 ,n_channels=1, nf=64, reg = 5e-5,
          padding='same', gpu='0'):
  """CNN model creator for predicting ellipticity in galaxy stamps with many channels.
  Author: Derszo Ribli

  Parameters
  ----------
    imsize : size of the galaxy image stamp in pixels

    n_target : Number of parameters required - by default, ellipticity can be described as a 2-spinor (g1, g2)
    but more fully as a 2-spinor, a 2-vector (center position) and a scalar for the stamp size.


    n_channels : number of different channels used as inputs. Channels can correspond to images in different bands, or a PSF model.

    nf: number of convolution filters for each 2D convolution block.

    reg: regularization parameter for the l2 regulator

    padding: padding for the convolution filters

    gpu: number of gpus in the gpu array


  Returns
  -------
  Model: The keras model.
  """

#input
inp = kl.Input((imsize, imsize,n_channels))

# conv block 1
  x = kl.Conv2D(nf, (3, 3), padding=padding,kernel_regularizer=kr.l2(reg))(inp)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(nf, (3, 3), padding=padding,kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 2
  x = kl.Conv2D(2*nf, (3, 3), padding=padding,kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(2*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 3
  x = kl.Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(2*nf, (1, 1), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))
  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 4
  x = kl.Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(4*nf, (1, 1), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))
  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 5
  x = kl.Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(8*nf, (1, 1), padding=padding,  kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))
  #end of conv

  #bottleneck + top
  x = kl.GlobalAveragePooling2D()(x)
  x = kl.Dense(n_target, name = 'final_dense_n%d_ngpu%d' % (n_target, len(gpu.split(','))))(x)

  # make model
  model = km.Model(inputs=inp, outputs=x)

return model

def adapt_efficient_net(top_dropout_rate = 0.4, train_full_model= False) -> Model:
  """This code uses adapts the most up-to-date version of EfficientNet with NoisyStudent weights to a regression
  problem. Most of this code is adapted from the official keras documentation.
  Auth: Markus Rosenfeld

  Returns
  -------
  Model
      The keras model.
  """

  #TODO: rescaling code

  inputs = layers.Input(shape=(224, 224, 3))  # input shapes of the images should always be 224x224x3 with EfficientNetB0
  # use the downloaded and converted newest EfficientNet wheights


  model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="efficientnetb0_notop.h5")
  # Freeze the pretrained weights
  if not train_full_model:
    model.trainable = False

  # Rebuild top
  x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
  outputs = layers.Dense(1, name="pred")(x)

  # Compile
  model = keras.Model(inputs, outputs, name="EfficientNet")

return model
