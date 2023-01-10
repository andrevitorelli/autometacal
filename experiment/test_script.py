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
import test_runners

test_runners.run_field()