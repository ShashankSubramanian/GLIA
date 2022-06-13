
import os, subprocess,sys
import numpy as np

scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

sys.path.append(os.path.join(code_dir, 'scripts', 'multispecies'))
from run_inverse_ms_no_elas import run_multispecies_inversion as run

bias_list =[1e-4] 

for i in range(1,9):
   
  for (j, bias) in enumerate(bias_list):
    print(i, j)  
    syn = 'case%d'%i
    resolution = 160
    scratch = os.getenv('SCRATCH')
    pat_dir = os.path.join(scratch, 'results/syndata/'+syn+'/C1_me/') 
    #pat_fwd_dir = os.path.join(scratch, 'results/syn_results/me_inv_160/'+syn+'/fwd_me/') 
    pat_fwd_dir = os.path.join(scratch, 'results/syn_results/me_inv_160/'+syn+'/fwd_me/') 
    res_dir = os.path.join(scratch, 'results/syn_results/ms_inv_160/'+syn+'/bias%d'%(j+1))
    #res_dir = os.path.join(scratch, 'results/syn_results/tmp/'+'')

    #v_dir = pat_fwd_dir
    v_dir = os.path.join(res_dir, 'fwd_ms/')


    if not os.path.exists(res_dir):
      os.makedirs(res_dir)


    true_params = {}

    true_params['gamma'] = 0.0

    params_in = {}
    params_in['gamma'] = (0, 0, 1.3E5)
    recon_file = os.path.join(res_dir, 'recon_info.dat')
    if os.path.exists(recon_file):
      with open(recon_file, 'r') as f:
        lines = f.readlines()
        l = lines[1].split(" ")
        params_in['k'] = (float(l[0]), 0.01, 0.8)
        params_in['rho'] = (float(l[1]), 1.0, 17.0)
        params_in['ox_hypoxia'] = (float(l[2]), 0.1, 0.7)
        params_in['death_rate'] = (float(l[3]), 0.1, 30.0)
        params_in['alpha_0'] = (float(l[4]), 0.01, 6.0)
        params_in['ox_consumption'] = (float(l[5]), 1.0, 20.0)
        params_in['ox_source'] = (float(l[6]), 0.5, 20.0)
        params_in['beta_0'] = (float(l[7]), 0.01, 6.0)
        params_in['ox_inv'] = (float(l[8]), 0.2, 1.0)

    else:
      params_in['k'] = (0.4, 0.01, 0.8)
      params_in['rho'] = (10.0, 1.0, 17.0)
      params_in['ox_hypoxia'] = (0.5, 0.1, 0.7)
      params_in['death_rate'] = (7.0, 0.1, 30.0)
      params_in['alpha_0'] = (1.0, 0.01, 6.0)
      params_in['ox_consumption'] = (5.0, 1.0, 20.0)
      params_in['ox_source'] = (3.0, 0.5, 20.0)
      params_in['beta_0'] = (1.0, 0.01, 6.0)
      params_in['ox_inv'] = (0.7, 0.2, 1.0)
    print(params_in)
    params_in['invasive_thres'] = (-3, -3.001, -2.999)
    params_in['given_velocities'] = 1

    sigma = 0.2

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


    for l in range(len(list_params)):
      init_vec.append(str(params_in[list_params[l]][0]))
      lb_vec.append(str(params_in[list_params[l]][1]))
      ub_vec.append(str(params_in[list_params[l]][2]))


    str_inits = ' '.join(init_vec)
    str_lbs = ' '.join(lb_vec)
    str_ubs = ' '.join(ub_vec)
    str_inv_params = ' '.join(list_inv_params)
    str_params = ' '.join(list_params)


    job_name = os.path.join(res_dir, 'job.sh')

    with open(job_name, 'w') as f:
      job_header = "#!/bin/bash\n\n"
      job_header += "#SBATCH -J ms_inv\n"
      job_header += "#SBATCH -o "+os.path.join(res_dir, 'log.txt')+"\n"
      job_header += "#SBATCH -e "+os.path.join(res_dir, 'err.txt')+"\n"
      job_header += "#SBATCH -p v100\n"
      job_header += "#SBATCH -N 1 \n"
      job_header += "#SBATCH -n 1 \n"
      job_header += "#SBATCH -t 10:00:00\n\n"
      
      f.write(job_header)
      f.write("source ~/.bashrc\n\n")
      f.write("source /work2/07544/ghafouri/longhorn/gits/claire_glia.sh\n\n") 
      #f.write("source /work2/07544/ghafouri/frontera/gits/env_glia.sh\n\n")
      
      f.write("conda activate gen\n\n")
      #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_16_obs_bias4.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" -b "+str(bias)+" \n"
      cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_opt.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" -b "+str(bias)+" \n"
      #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'run_inverse_ms_no_elas_16_obs_bias6.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" -b "+str(bias)+" \n"
      #cmd = "python -u "+os.path.join(code_dir, 'scripts', 'multispecies', 'test.py')+" -p "+os.path.join(pat_dir)+" -r "+os.path.join(res_dir)+" -lb "+str_lbs+" -ub "+str_ubs+ " -i "+str_inits+" -inv "+str_inv_params+" -total "+str_params+" -sigma "+str(sigma)+" -vel "+v_dir+" -n "+str(resolution)+" \n"
      f.write(cmd)
       
      
