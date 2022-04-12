
import os, subprocess,sys
import numpy as np

scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

sys.path.append(os.path.join(code_dir, 'scripts', 'multispecies'))
from run_inverse_ms import run_multispecies_inversion as run


pat_dir = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/160/' 
res_dir = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/results/res_syndata_160_14_rho/'


if not os.path.exists(res_dir):
  os.mkdir(res_dir)


params_in = {}

params_in['rho'] = (7.0, 4.0, 9.0)
params_in['k'] = (0.01, 0.005, 0.2)
params_in['gamma'] = (1E5, 3E4, 1.2E5)
params_in['ox_hypoxia'] = (0.5, 0.3, 0.7)
params_in['death_rate'] = (0.2, 0.1, 1.0)
params_in['alpha_0'] = (0.05, 0.01, 1.0)
params_in['ox_consumption'] = (6.0, 1.0, 20.0)
params_in['ox_source'] = (40.0, 0.0, 75.0)
params_in['beta_0'] = (0.06, 0.01, 0.1)
params_in['sigma_b'] = (0.9, 0.5, 1.0)
params_in['ox_inv'] = (0.7, 0.65, 1.0)
params_in['invasive_thres'] = (0.01, 0.001, 0.1)

sigma = 0.3

list_params = ['k', 'rho', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'sigma_b', 'ox_inv', 'invasive_thres']
#list_inv_params = ['k', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0']
#list_inv_params = ['k', 'rho', 'gamma', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']
#list_inv_params = ['k', 'rho', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']
list_inv_params = ['rho', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']

init_vec = [] 
lb_vec = []
ub_vec = []


for i in range(len(list_params)):
  init_vec.append(str(params_in[list_params[i]][0]))
  lb_vec.append(str(params_in[list_params[i]][1]))
  ub_vec.append(str(params_in[list_params[i]][2]))


str_inits = ' '.join(init_vec)
str_lbs = ' '.join(lb_vec)
str_ubs = ' '.join(ub_vec)
str_inv_params = ' '.join(list_inv_params)
str_params = ' '.join(list_params)


job_name = os.path.join(res_dir, 'job.sh')

with open(job_name, 'w') as f:
  job_header = "#!/bin/bash\n\n"
  job_header += "#SBATCH -J tuinv\n"
  job_header += "#SBATCH -o "+os.path.join(res_dir, 'log.txt')+"\n"
  job_header += "#SBATCH -p rtx\n"
  job_header += "#SBATCH -N 1 \n"
  job_header += "#SBATCH -n 1 \n"
  job_header += "#SBATCH -t 48:00:00\n\n"
  
  f.write(job_header)
  f.write("source ~/.bashrc\n\n")
  f.write("source /work2/07544/ghafouri/frontera/gits/env_glia.sh\n\n")
  
  f.write("conda activate mriseg\n\n")
  #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" \n" 
  cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms5.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" \n"
  f.write(cmd)
   
  
