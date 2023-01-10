
import os, subprocess,sys
import numpy as np
import params as par

scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

sys.path.append(os.path.join(code_dir, 'scripts', 'multispecies'))

beta_p_list = np.array([0.0, 1e-2])

for i in range(beta_p_list.shape[0]):
  
  params = {}
  r = {}
  r['code_path'] = code_dir
  r['compute_sys'] = 'frontera'
  #r['extra_modules'] = '\nsource /work/07544/ghafouri/longhorn/gits/claire_glia.sh\n'
  r['extra_modules'] = '\nsource /work/07544/ghafouri/frontera/gits/env_glia.sh\n'
  r['extra_modules'] += '\nconda activate gen\n'
  scratch = os.getenv("SCRATCH")
  case = 1
  params['beta_p'] = beta_p_list[i]
  #params['output_dir'] = scratch +"/results/syn_results/case%d_%.2e_nonhealthy/"%(case, params['beta_p'])
  params['output_dir'] = scratch +"/results/syn_results/case%d_%.2e/"%(case, params['beta_p'])
  params['log_dir'] = params['output_dir']
  params['pat_dir'] = scratch + "/results/syndata/case%d/C1_me_test_vel/"%case
  params['d1_path'] = scratch + "/results/syndata/case%d/C1_me_test_vel/seg_ms_rec_final.nc"%case
  #params['a_seg_path'] = os.path.join(params['pat_dir'], 'seg_t0_nonhealthy.nc')
  params['a_seg_path'] = os.path.join(params['pat_dir'], 'seg_t0.nc')
  params['d0_path'] = scratch + '/results/syndata/case%d/C1_me_test_vel/c0_true_syn.nc'%case
  params['Q_neg'] = '/work2/07544/ghafouri/ls6/gits/gliomas/U_Big_obs_weighted_nonnormalized.npy'
  params['extra_modules'] = 'source /work/07544/ghafouri/frontera/gits/env_glia.sh'

  params['n'] = 160 
  params['sigma_cma'] = 0.5
  params['popsize'] = 8
  params['tolfunrel'] = 1e-8
  params['tolfun'] = 1e-3
  params['solver'] = 'multi_species'
  params['model'] = 6

  params['rho_lb'] = 5.0 
  params['rho_ub'] = 25.0

  params['k_lb'] = 0.001
  params['k_ub'] = 0.1

  params['alpha_0_lb'] = 0.1 
  params['alpha_0_ub'] = 10.0

  params['beta_0_lb'] = 0.1
  params['beta_0_ub'] = 15.0

  params['death_rate_lb'] = 1.0
  params['death_rate_ub'] = 20.0

  params['ox_source_lb'] = 1.0
  params['ox_source_ub'] = 8.0

  params['ox_consumption_lb'] = 1.0
  params['ox_consumption_ub'] = 20.0

  params['ox_hypoxia_lb'] = 0.001
  params['ox_hypoxia_ub'] = 0.8

  params['ox_inv_lb'] = 0.2
  params['ox_inv_ub'] = 1.0

  params['invasive_thres_lb'] = 0.001
  params['invasive_thres_ub'] = 0.3


  params['gamma'] = 0
  params['k'] = (params['k_lb'] + params['k_ub']) /2
  params['rho'] = (params['rho_lb'] + params['rho_ub']) / 2
  params['beta_0'] = (params['beta_0_lb'] + params['beta_0_ub']) /2
  params['alpha_0'] = (params['alpha_0_lb'] + params['alpha_0_ub']) /2
  params['death_rate'] = (params['death_rate_lb'] + params['death_rate_ub']) /2
  params['ox_source'] = (params['ox_source_lb'] + params['ox_source_ub']) /2
  params['ox_consumption'] = (params['ox_consumption_lb'] + params['ox_consumption_ub']) /2
  params['ox_hypoxia'] = (params['ox_hypoxia_lb'] + params['ox_hypoxia_ub']) /2
  params['ox_inv'] = (params['ox_inv_lb'] + params['ox_inv_ub']) /2
  params['invasive_thres'] = (params['invasive_thres_lb'] + params['invasive_thres_ub']) /2
  '''
  params['k'] = 0.04 
  params['rho'] = 17.0 
  params['beta_0'] = 3.0 
  params['alpha_0'] = 1.0
  params['death_rate'] = 12.0
  params['ox_source'] = 3.0 
  params['ox_consumption'] = 10.0
  params['ox_hypoxia'] = 0.2
  params['ox_inv'] = 0.6
  params['invasive_thres'] = 0.02
  '''

  r['wtime_h'] = 12
  par.submit(params, r, True, True)
