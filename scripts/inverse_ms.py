
import os, subprocess,sys
import numpy as np
import params as par

scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

sys.path.append(os.path.join(code_dir, 'scripts', 'multispecies'))

r = {}

params = {}
r['code_path'] = code_dir
r['compute_sys'] = 'longhorn'
r['extra_modules'] = '\nsource /work/07544/ghafouri/longhorn/gits/claire_glia.sh\n'
r['extra_modules'] = '\nconda activate gen\n'

params['pat_dir'] = "/scratch/07544/ghafouri/results/syndata/case2/C1_me_test_vel/"
params['output_dir'] = "/scratch/07544/ghafouri/results/syn_results/case2_5/"
params['a_seg_path'] = '/scratch/07544/ghafouri/data/adni-nc/160/50052_seg_aff2jakob_ants_160.nc'
params['d0_path'] = '/scratch/07544/ghafouri/results/syndata/case2/C1_me_test_vel/c0_true_syn.nc'
params['Q_neg'] = '/work/07544/ghafouri/longhorn/gits/gliomas_serial/U_Big_obs2.npy' 

params['beta_p'] = 5e-3

params['n'] = 160 
params['sigma_cma'] = 0.4 
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

r['wtime_h'] = 16
par.submit(params, r, True, True)
