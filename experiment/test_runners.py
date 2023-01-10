from numpy import array, linspace
from data_generator import observer

def run_field(shear_true,g_psf,noise_level,batches):
  '''run a constant-shear field with amc and ngmix'''
  e_ngmix_list = []
  R_ngmix_list = []
  Rpsf_ngmix_list = []
  e_amc_list = []
  R_amc_list = []
  Rpsf_amc_list = []
  epsf_amc_list = []
  Repsf_amc_list = []
  
  for i in range(batches):
    
    #make observations
    gal_images, psf_images, noise_images , obslist = observer(batch_size,shear_true,g_psf,noise_level)
    
    #run ngmix
    pool = Pool(ncpus)
    dlist = pool.map(ngmix_step,obslist)
    pool.close()

    #REXtractor (REsults eXtractor! :))
    results = [get_metacal_response_ngmix_moments(resdict) for resdict in dlist]
    
    #stack results ngmix
    e_ngmix_list += [result[0]['noshear'] for result in results]
    R_ngmix_list += [result[1] for result in results]
    Rpsf_ngmix_list += [result[2] for result in results]

    #run AutoMetaCal
    e_amc, R_amc, Rpsf_amc, epsf_amc, Repsf_amc  = response(
      gal_images, 
      psf_images,
      tf.convert_to_tensor(noise_images,dtype=tf.float32))
    
    #stack results amc
    e_amc_list += [e_amc]
    epsf_amc_list += [epsf_amc]
    R_amc_list +=[R_amc]
    Rpsf_amc_list += [Rpsf_amc]
    Repsf_amc_list += [Repsf_amc]
  
  #gather multiple ngmix batches
  e_ngmix = array(e_ngmix_list)
  R_ngmix = array(R_ngmix_list)
  epsf_ngmix = array([0*g_psf for n in range(len(e_ngmix))])
  Repsf_ngmix = array([eye(2) for n in range(len(e_ngmix))])
  Rpsf_ngmix = array(Rpsf_ngmix_list)
  
  #evaluate ngmix
  shear_ngmix, shear_err_ngmix, shearcorr_ngmix, shearcorr_err_ngmix= evaluator(
    e_ngmix, 
    R_ngmix,
    epsf_ngmix,
    Repsf_ngmix,
    Rpsf_ngmix
  )  
  
  #gather multiple amc batches
  e_amc = tf.concat(e_amc_list,axis=0)
  R_amc = tf.concat(R_amc_list,axis=0)
  epsf_amc =tf.concat(epsf_amc_list,axis=0)
  Rpsf_amc = tf.concat(Rpsf_amc_list,axis=0)
  Repsf_amc = tf.concat(Repsf_amc_list,axis=0)
  
  #evaluate amc
  shear_amc, shear_amc_err, shearcorr_amc, shearcorr_amc_err= evaluator(
    e_amc.numpy(), 
    R_amc.numpy(), 
    epsf_amc.numpy(),
    Repsf_amc.numpy(), 
    Rpsf_amc.numpy()
  ) 
  
  result_ngmix = {
    'e':e_ngmix,
    'R':R_ngmix,
    'Rpsf':Rpsf_ngmix, 
    'shear':shear_ngmix, 
    'shear_err':shear_err_ngmix, 
    'shearcorr':shearcorr_ngmix, 
    'shearcorr_err':shearcorr_err_ngmix, 
  }
  
  result_amc = {
    'e':e_amc,
    'R':R_amc,
    'Rpsf':Rpsf_amc, 
    'shear':shear_amc, 
    'shear_err':shear_amc_err, 
    'shearcorr':shearcorr_amc, 
    'shearcorr_err':shearcorr_amc_err, 
  }
 
  result = {
    'ngmix' : result_ngmix,
    'amc' : result_amc
  }
  
  return result


def test_loop(noise_level,fieldbatches=5):
  shear_range = linspace(-.01,.01,4)

  shear_ngmix_list = []
  shear_err_ngmix_list = []
  shearcorr_ngmix_list = []
  shearcorr_err_ngmix_list = []
  R_ngmix_list = []
  Rpsf_ngmix_list = []

  shear_amc_list = []
  shear_err_amc_list = []
  shearcorr_amc_list = []
  shearcorr_err_amc_list = []
  R_amc_list = []
  Rpsf_amc_list = []
    
  for shear in shear_range:
    shear_true = array([shear,0.],dtype='float32')
    result = run_field(shear_true,g_psf,noise_level,fieldbatches)
    #ngmix
    shear_ngmix_list += [result['ngmix']['shear'] ] 
    shear_err_ngmix_list += [result['ngmix']['shear_err']]
    shearcorr_ngmix_list += [result['ngmix']['shearcorr']]
    shearcorr_err_ngmix_list += [result['ngmix']['shearcorr_err']]
    R_ngmix_list += [result['ngmix']['R']]
    Rpsf_ngmix_list += [result['ngmix']['Rpsf']]
    #amc   
    shear_amc_list += [result['amc']['shear']]
    shear_err_amc_list += [result['amc']['shear_err']]
    shearcorr_amc_list += [result['amc']['shearcorr']]
    shearcorr_err_amc_list += [result['amc']['shearcorr_err']]
    R_amc_list += [result['amc']['R']]
    Rpsf_amc_list += [result['amc']['Rpsf']]
  
  #g1
  #ngmix
  X1 = shear_range#[:,0]
  Y1_ngmix =  array(shearcorr_ngmix_list)[:,0]
  Y1err_ngmix = array(shearcorr_err_ngmix_list)[:,0]
  g1_scatter_ngmix = Y1err_ngmix.mean()

  #biases
  ngmix_bias1 = linregress(X1,Y1_ngmix)
  m1_ngmix = ngmix_bias1.slope - 1
  c1_ngmix = ngmix_bias1.intercept

  #errors
  m1_ngmix_err = ngmix_bias1.stderr
  c1_ngmix_err = ngmix_bias1.intercept_stderr
  
  #amc
  Y1_amc =  array(shearcorr_amc_list)[:,0] 
  Y1err_amc = array(shearcorr_err_amc_list)[:,0]
  g1_scatter_amc = Y1err_amc.mean()

  #biases
  amc_bias1 = linregress(X1,Y1_amc)
  m1_amc = amc_bias1.slope - 1
  c1_amc = amc_bias1.intercept

  #errors
  m1_amc_err = amc_bias1.stderr
  c1_amc_err = amc_bias1.intercept_stderr
  
  #g2
  X2 = shear_range#[:,1]
  Y2_ngmix =  array(shearcorr_ngmix_list)[:,1]
  Y2err_ngmix = array(shearcorr_err_ngmix_list)[:,1]
  g2_scatter_ngmix = Y2err_ngmix.mean()

  #biases
  ngmix_bias2 = linregress(X2,Y2_ngmix)
  m2_ngmix = ngmix_bias2.slope - 1
  c2_ngmix = ngmix_bias2.intercept

  #errors
  m2_ngmix_err = ngmix_bias.stderr
  c2_ngmix_err = ngmix_bias.intercept_stderr
  
  Y2_amc =  array(shearcorr_amc_list)[:,1] 
  Y2err_amc = array(shearcorr_err_amc_list)[:,0]

  #biases
  amc_bias2 = linregress(X2,Y2_amc)
  m2_amc = amc_bias2.slope-1
  c2_amc = amc_bias2.intercept

  #errors
  m2_amc_err = amc_bias.stderr
  c2_amc_err = amc_bias.intercept_stderr
  
  result = {
    'm1_amc' :[m1_amc,m1_amc_err],
    'c1_amc' :[c1_amc,c1_amc_err], 
    'm1_ngmix' :[m1_ngmix,m1_ngmix_err],
    'c1_ngmix' :[c1_ngmix,c1_ngmix_err],
    'g1_scatter_amc' : g1_scatter_amc,
    'g1_scatter_ngmix' : g1_scatter_ngmix,
    'm2_amc' :[m2_amc,m2_amc_err],
    'c2_amc' :[c2_amc,c2_amc_err],
    'm2_ngmix' :[m2_ngmix,m2_ngmix_err], 
    'c2_ngmix' :[c2_ngmix,c2_ngmix_err],
    #'g2_scatter_amc' : g2_scatter_amc,
    #'g2_scatter_ngmix' : g2_scatter_ngmix,
  }
  
  return result