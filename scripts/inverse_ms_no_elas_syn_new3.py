
import os, subprocess,sys
import numpy as np

scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

sys.path.append(os.path.join(code_dir, 'scripts', 'multispecies'))
from run_inverse_ms_no_elas import run_multispecies_inversion as run


for i in range(1,9):

  syn = 'case%d'%i
  resolution = 160
  scratch = os.getenv('SCRATCH')
  pat_dir = os.path.join(scratch, 'results/syndata/'+syn+'/C1_me/') 
  #pat_fwd_dir = os.path.join(scratch, 'results/syn_results/me_inv_160/'+syn+'/fwd_me/') 
  pat_fwd_dir = os.path.join(scratch, 'results/syn_results/me_inv_160/'+syn+'/fwd_me/') 
  res_dir = os.path.join(scratch, 'results/syn_results/ms_inv_160/'+syn+'_4/')
  #res_dir = os.path.join(scratch, 'results/syn_results/tmp/'+'')

  v_dir = pat_fwd_dir


  if not os.path.exists(res_dir):
    os.mkdir(res_dir)


  true_params = {}

  true_params['gamma'] = 0.0


  params_in = {}
  params_in['k'] = (0.4, 0.01, 0.8)
  params_in['rho'] = (14.0, 1.0, 18.0)
  #params_in['rho'] = (true_params['rho'], 4.0, 20.0)
  params_in['gamma'] = (0, 0, 1.3E5)
  #params_in['ox_hypoxia'] = (0.5, 0.3, 0.7)
  #params_in['ox_hypoxia'] = (true_params['ox_hypoxia'], 0.3, 1.0)
  params_in['ox_hypoxia'] = (0.5, 0.1, 0.7)
  params_in['death_rate'] = (3.0, 0.1, 10.0)
  params_in['alpha_0'] = (0.5, 0.01, 4.0)
  params_in['ox_consumption'] = (10.0, 1.0, 40.0)
  params_in['ox_source'] = (3.0, 0.5, 20.0)
  params_in['beta_0'] = (0.5, 0.01, 4.0)
  params_in['ox_inv'] = (0.7, 0.2, 1.0)
  #params_in['ox_inv'] = (true_params['ox_inv'], 0.3, 1.0)
  #params_in['invasive_thres'] = (true_params['invasive_thres'], 0.001, 0.2)
  #params_in['invasive_thres'] = (0.005, 5e-5, 0.02)
  params_in['invasive_thres'] = (-2, -5, -0.5)
  params_in['given_velocities'] = 1
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

  '''











  sigma = 0.4




  list_params = ['k', 'rho', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv', 'invasive_thres']
  #list_inv_params = ['k', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0']
  #list_inv_params = ['k', 'rho', 'gamma', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']
  #list_inv_params = ['k', 'rho', 'death_rate', 'alpha_0', 'ox_consumption', 'beta_0']
  #list_inv_params = ['k', 'rho', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv']
  #list_inv_params = ['rho', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv']
  #list_inv_params = ['k', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv']
  list_inv_params = ['k', 'rho', 'ox_hypoxia','death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv', 'invasive_thres']

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
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" \n"
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_32.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_64.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_32_obs.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_16_obs.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_16_obs_nocoeff3.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'test.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
    f.write(cmd)
     
    
