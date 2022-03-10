import os,sys
from run_multispecies import create_tusolver_config
import argparse
import subprocess 


scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(os.path.join(code_dir, 'scripts', 'utils'))

from file_io import readNetCDF, createNetCDF

import numpy as np

import cma









def run_multispecies_inversion(pat_dir, res_dir, init_vec, lb_vec, ub_vec, is_syn=False):
 
  list_vars = ['rho', 'k', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'sigma_b', 'ox_inv', 'invasive_thres']
  '''
  lb_vec = []
  ub_vec = []
  init_vec = []
  for i in range(len(list_vars)):
    lb_vec.append(lb_params[list_vars[i]])
    ub_vec.append(ub_params[list_vars[i]])
    init_vec.append(init_params[list_vars[i]])
  
  '''
  init_vec = np.array(init_vec)
  lb_vec = np.array(lb_vec)
  ub_vec = np.array(ub_vec)

  cma_init = (init_vec - lb_vec)/(ub_vec - lb_vec) 
  fun = lambda x : create_cma_output(x, pat_dir, res_dir, lb_vec, ub_vec)

  es = cma.CMAEvolutionStrategy(cma_init, 0.5, {'bounds': [0, 1]}) 
  
  es.optimize(fun)
  res = es.result()
  print(res)




def create_cma_output(x, pat_dir, res_dir, lb_vec, ub_vec):
 
  x = x * (ub_vec - lb_vec) + lb_vec;
  print("Testing ")
  print(x)
  
  forward_params = {}
  forward_params['rho'] = x[0]
  forward_params['k'] = x[1] 
  forward_params['gamma'] = x[2] 
  forward_params['ox_hypoxia'] = x[3] 
  forward_params['death_rate'] = x[4] 
  forward_params['alpha_0'] = x[5] 
  forward_params['ox_consumption'] = x[6] 
  forward_params['ox_source'] = x[7] 
  forward_params['beta_0'] = x[8] 
  forward_params['sigma_b'] = x[9] 
  forward_params['ox_inv'] = x[10] 
  forward_params['invasive_thres'] = x[11] 
  
  en_t1 = os.path.join(pat_dir, 'en_t1.nc')
  ed_t1 = os.path.join(pat_dir, 'ed_t1.nc')
  nec_t1 = os.path.join(pat_dir, 'nec_t1.nc')

  res_forward_dir = os.path.join(res_dir, 'forward_1')
  if not os.path.exists(res_forward_dir):
    os.mkdir(res_forward_dir)
  
  create_tusolver_config(pat_dir, res_forward_dir, forward_params)
  config_path = os.path.join(res_forward_dir, 'solver_config.txt') 
  log_path = os.path.join(res_forward_dir, 'solver_log.txt') 
  tusolver_path = os.path.join(code_dir, 'build', 'last', 'tusolver')
  cmd = 'CUDA_VISIBLE_DEVICES=0 ibrun '+tusolver_path+' -config '+config_path+' > '+log_path+'\n\n'
  cmd += 'wait\n\n'
  excmd(cmd)

  en_rec_path = os.path.join(res_forward_dir, 'en_rec_final.nc')
  ed_rec_path = os.path.join(res_forward_dir, 'ed_rec_final.nc')
  nec_rec_path = os.path.join(res_forward_dir, 'nec_rec_final.nc')

  dat_en_true = readNetCDF(en_t1) 
  dat_ed_true = readNetCDF(ed_t1) 
  dat_nec_true = readNetCDF(nec_t1) 
  
  dat_en_rec = readNetCDF(en_rec_path)
  dat_ed_rec = readNetCDF(ed_rec_path)
  dat_nec_rec = readNetCDF(nec_rec_path)
  
  px_en = np.sum((dat_en_true>0))
  px_nec = np.sum((dat_nec_true>0))
  px_ed = np.sum((dat_ed_true>0))
 
  tot = px_en + px_nec + px_ed

  w_en = tot/px_en
  w_nec = tot/px_nec
  w_ed = tot/px_ed

  diff_en = w_en * np.linalg.norm((dat_en_true - dat_en_rec).flatten())**2
  diff_nec = w_nec * np.linalg.norm((dat_nec_true - dat_nec_rec).flatten())**2
  diff_ed = w_ed * np.linalg.norm((dat_ed_true - dat_ed_rec).flatten())**2
  
  J = diff_en + diff_nec + diff_ed

  print('Objective function (En + Nec + Ed = J) : %.4f + %.4f + $.4f = %.4f '%(diff_en, diff_nec, diff_ed, J))

  return J



def excmd(cmd, skip=False):
  print(cmd)
  if not skip:
    os.system(cmd)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Multi spcies inversion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to the patient data', required = True)
  r_args.add_argument('-r', '--result_dir', type = str, help = 'path to the results dir', required = True)
  r_args.add_argument('-lb', '--lb_bounds', nargs="+", type=float, help = 'list of lower bounds', required = True)
  r_args.add_argument('-ub', '--ub_bounds', nargs="+", type=float, help = 'list of upper bounds', required = True)
  r_args.add_argument('-i', '--init_vals', nargs="+", type=float, help = 'list of initial guess', required = True)
  args = parser.parse_args();

  pat_dir = args.patient_dir
  res_dir = args.result_dir
  if not os.path.exists(res_dir):
    os.mkdir(res_dir)

  lb_vec = args.lb_bounds
  ub_vec = args.ub_bounds
  init_vec = args.init_vals
 
  
  run_multispecies_inversion(pat_dir, res_dir, init_vec, lb_vec, ub_vec)    
   
  




