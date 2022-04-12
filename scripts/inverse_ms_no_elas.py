
import os, subprocess,sys
import numpy as np

scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

sys.path.append(os.path.join(code_dir, 'scripts', 'multispecies'))
from run_inverse_ms_no_elas import run_multispecies_inversion as run


syn = 'case4'

pat_dir = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/'+syn+'/160/' 
pat_fwd_dir = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/results/syn_results/'+syn+'_fwd/' 
res_dir = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/results/syn_results/'+syn+'_inv/true_elas/'


if not os.path.exists(res_dir):
  os.mkdir(res_dir)


true_params = {}

if syn == 'case1':
  true_params['rho'] = 14.0 
  true_params['k'] =  0.4
  true_params['gamma'] = 1E5 
  true_params['death_rate'] = 0.9 
  true_params['alpha_0'] =  0.8
  true_params['ox_consumption'] = 3.0
  true_params['ox_source'] = 120.0
  true_params['beta_0'] = 0.6
  true_params['ox_inv'] =  0.9
  true_params['ox_hypoxia'] = 0.7 
  true_params['invasive_thres'] = 0.01
elif syn == 'case2' :
  true_params['rho'] = 14.0 
  true_params['k'] =  0.3
  true_params['gamma'] = 1.1E5 
  true_params['death_rate'] = 0.8
  true_params['alpha_0'] =  0.6
  true_params['ox_consumption'] = 3.0
  true_params['ox_source'] = 120.0
  true_params['beta_0'] = 0.8
  true_params['ox_inv'] =  0.8
  true_params['ox_hypoxia'] = 0.6 
  true_params['invasive_thres'] = 0.005
elif syn == 'case3' :
  true_params['rho'] = 16.0 
  true_params['k'] =  0.4
  true_params['gamma'] = 1.05E5 
  true_params['death_rate'] = 0.9
  true_params['alpha_0'] =  0.4
  true_params['ox_consumption'] = 6.0
  true_params['ox_source'] = 105.0
  true_params['beta_0'] = 0.8
  true_params['ox_inv'] =  0.8
  true_params['ox_hypoxia'] = 0.8
  true_params['invasive_thres'] = 0.02
else:
  true_params['rho'] = 15.0 
  true_params['k'] =  0.5
  true_params['gamma'] = 1.1E5 
  true_params['death_rate'] = 1.0
  true_params['alpha_0'] =  0.4
  true_params['ox_consumption'] = 2.0
  true_params['ox_source'] = 130.0
  true_params['beta_0'] = 0.7
  true_params['ox_inv'] =  0.7
  true_params['ox_hypoxia'] = 0.6 
  true_params['invasive_thres'] = 0.005
  



params_in = {}
'''
#params_in['rho'] = (14.0, 4.0, 20.0)
params_in['rho'] = (true_params['rho'], 4.0, 20.0)
params_in['k'] = (0.4, 0.01, 0.8)
params_in['gamma'] = (0, 0, 1.3E5)
#params_in['ox_hypoxia'] = (0.5, 0.3, 0.7)
params_in['ox_hypoxia'] = (true_params['ox_hypoxia'], 0.3, 1.0)
params_in['ox_hypoxia'] = (0.5, 0.3, 0.7)
params_in['death_rate'] = (0.6, 0.1, 1.1)
params_in['alpha_0'] = (0.5, 0.01, 1.0)
params_in['ox_consumption'] = (6.0, 1.0, 20.0)
params_in['ox_source'] = (70.0, 30.0, 130.0)
params_in['beta_0'] = (0.5, 0.01, 1.0)
#params_in['ox_inv'] = (0.7, 0.5, 1.0)
params_in['ox_inv'] = (true_params['ox_inv'], 0.3, 1.0)
params_in['invasive_thres'] = (true_params['invasive_thres'], 0.001, 0.2)
#params_in['given_velocities'] = 1
'''

#params_in['rho'] = (14.0, 4.0, 20.0)
params_in['rho'] = (true_params['rho'], 4.0, 20.0)
params_in['k'] = (true_params['k'], 0.01, 0.8)
params_in['gamma'] = (true_params['gamma'], 3E4, 1.3E5)
params_in['ox_hypoxia'] = (true_params['ox_hypoxia'], 0.3, 1.0)
params_in['death_rate'] = (true_params['death_rate'], 0.1, 1.1)
params_in['alpha_0'] = (true_params['alpha_0'], 0.01, 1.0)
params_in['ox_consumption'] = (true_params['ox_consumption'], 1.0, 20.0)
params_in['ox_source'] = (true_params['ox_source'], 30.0, 130.0)
params_in['beta_0'] = (true_params['beta_0'], 0.01, 1.0)
params_in['ox_inv'] = (true_params['ox_inv'], 0.3, 1.0)
params_in['invasive_thres'] = (true_params['invasive_thres'], 0.001, 0.2)




v_dir = os.path.join(pat_fwd_dir,'vel')








sigma = 0.01




list_params = ['k', 'rho', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv', 'invasive_thres']
#list_inv_params = ['k', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0']
#list_inv_params = ['k', 'rho', 'gamma', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']
#list_inv_params = ['k', 'rho', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']
#list_inv_params = ['k', 'rho', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv']
#list_inv_params = ['rho', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv']
list_inv_params = ['k', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv']
#list_inv_params = ['k', 'rho', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0']

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
  cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" \n"
  f.write(cmd)
   
  
