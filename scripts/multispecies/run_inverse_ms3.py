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





def run_multispecies_inversion(pat_dir, res_dir, init_vec, lb_vec, ub_vec, inv_params, list_vars, is_syn=False):
 
  #list_vars = ['rho', 'k', 'gamma', 'ox_hypoxia', 'death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'sigma_b', 'ox_inv', 'invasive_thres']
  '''
  lb_vec = []
  ub_vec = []
  init_vec = []
  for i in range(len(list_vars)):
    lb_vec.append(lb_params[list_vars[i]])
    ub_vec.append(ub_params[list_vars[i]])
    init_vec.append(init_params[list_vars[i]])
  
  '''
  
  
  cma_init = []
  cma_lb = []
  cma_ub = []
  for (i, param) in enumerate(list_vars):
    if param in inv_params:
      cma_lb.append(lb_vec[i])
      cma_ub.append(ub_vec[i])
      cma_init.append(init_vec[i])
      
      
  
  cma_init = np.array(cma_init)
  cma_lb = np.array(cma_lb)
  cma_ub = np.array(cma_ub)

  cma_init = (cma_init - cma_lb)/(cma_ub - cma_lb) 
  fun = lambda x : create_cma_output(x, pat_dir, res_dir, cma_lb, cma_ub, init_vec, inv_params, list_vars)

  es = cma.CMAEvolutionStrategy(cma_init, 0.3, {'bounds': [0, 1]})
  opts = cma.CMAOptions()
  opts.set('tolfunc',1e-2) 
  opts.set('popsize', 12)
  
  while not es.stop():
    solutions = es.ask()
    es.tell(solutions, fun(solutions))
    es.disp()
  
  res = es.result()
  print(res)




def create_cma_output(solutions, pat_dir, res_dir, lb_vec, ub_vec, init_vec, inv_params, list_vars):
 
  print("Testing ")
  
  
  en_t1 = os.path.join(pat_dir, 'en_t1.nc')
  ed_t1 = os.path.join(pat_dir, 'ed_t1.nc')
  nec_t1 = os.path.join(pat_dir, 'nec_t1.nc')
  seg_t1 = os.path.join(pat_dir, 'seg_t1.nc')

  dat_en_true = readNetCDF(en_t1) 
  dat_ed_true = readNetCDF(ed_t1) 
  dat_nec_true = readNetCDF(nec_t1) 
  dat_seg_true = readNetCDF(seg_t1) 
  dat_vt_true = dat_seg_true.copy()
  dat_vt_true[dat_vt_true != 7] = 0
  dat_vt_true[dat_vt_true == 7] = 1

  num_devices = 4
  J_vec = []
  num_eval = int(len(solutions))
  num_batch = int(np.ceil(num_eval / num_devices))

  for i in range(num_batch):
    cmd = ''
    for j in range(num_devices):
      if i * num_devices + j >= num_eval:
        break
    
      res_forward_dir = os.path.join(res_dir, 'forward_%d'%j)
      if not os.path.exists(res_forward_dir):
        os.mkdir(res_forward_dir)
      
      x = solutions[i * num_devices + j]
      x = x * (ub_vec - lb_vec) + lb_vec;
       
      forward_params = {}
      str_disp = "Forward params (%d) : "%j
      for (k, param) in enumerate(inv_params):
        forward_params[param] = x[k]
        str_disp += param + " = " + str(x[k]) + ", "
      for (k, param) in enumerate(list_vars):
        if param not in inv_params:
          forward_params[param] = init_vec[k]
      print(str_disp)
      '''
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
      '''
      create_tusolver_config(pat_dir, res_forward_dir, forward_params)
      config_path = os.path.join(res_forward_dir, 'solver_config.txt') 
      tmp = i * num_devices + j 
      log_path = os.path.join(res_forward_dir, 'solver_log_%d.txt'%tmp) 
      tusolver_path = os.path.join(code_dir, 'build', 'last', 'tusolver')
      #print(solutions[i * num_devices + j])
       
      cmd += 'CUDA_VISIBLE_DEVICES=%d ibrun '%j+tusolver_path+' -config '+config_path+' > '+log_path+' &\n\n'
      cmd += '\n\n'
    
    cmd += "wait\n\n"
    excmd(cmd)
    #subprocess.Popen(['wait'])
    #os.wait()
    for j in range(num_devices):
      if i * num_devices + j >= num_eval:
        break
      res_forward_dir = os.path.join(res_dir, 'forward_%d'%j)
      en_rec_path = os.path.join(res_forward_dir, 'en_rec_final.nc')
      ed_rec_path = os.path.join(res_forward_dir, 'ed_rec_final.nc')
      nec_rec_path = os.path.join(res_forward_dir, 'nec_rec_final.nc')
      seg_rec_path = os.path.join(res_forward_dir, 'seg_rec_final.nc')
        
      dat_en_rec = readNetCDF(en_rec_path)
      dat_ed_rec = readNetCDF(ed_rec_path)
      dat_nec_rec = readNetCDF(nec_rec_path)
      dat_seg_rec = readNetCDF(seg_rec_path)
      dat_vt_rec = dat_seg_rec.copy()
      dat_vt_rec[dat_vt_rec != 7] = 0 
      dat_vt_rec[dat_vt_rec == 7] = 1 
 
      px_en = np.linalg.norm((dat_en_true).flatten())
      px_nec = np.linalg.norm((dat_nec_true).flatten())
      px_ed = np.sum((dat_ed_true>0))
      px_vt = np.sum((dat_vt_true > 0))
      
      tot = px_en + px_nec + px_ed
      
      w_en = 1/px_en
      w_nec = 1/px_nec
      #w_ed = 1/px_ed
      w_ed = 1/px_ed
      w_vt = 1/px_vt

      diff_en = w_en * np.linalg.norm((dat_en_true - dat_en_rec).flatten())**2
      diff_nec = w_nec * np.linalg.norm((dat_nec_true - dat_nec_rec).flatten())**2
      diff_ed = w_ed * np.linalg.norm((dat_ed_true - dat_ed_rec).flatten())**2
      diff_vt =  w_vt * np.linalg.norm((dat_vt_true - dat_vt_rec).flatten())**2
      J = diff_en + diff_nec + diff_ed + diff_vt
      J_vec.append(J) 
      out = "Objective function (En + Nec + Ed + Misfit[VT]= J) : %.4f + %.4f + %.4f + %.4f = %.4f "%(diff_en, diff_nec, diff_ed, diff_vt, J)
      print(out)
      
      os.remove(en_rec_path)
      os.remove(ed_rec_path)
      os.remove(nec_rec_path)
      
  return J_vec



def excmd(cmd, skip=False):
  #print(cmd)
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
  r_args.add_argument('-inv', '--inv_params', nargs="+", type=str, help = 'list of initial guess', required = True)
  r_args.add_argument('-total', '--total_params', nargs="+", type=str, help = 'list of initial guess', required = True)
  args = parser.parse_args();

  pat_dir = args.patient_dir
  res_dir = args.result_dir
  if not os.path.exists(res_dir):
    os.mkdir(res_dir)

  lb_vec = args.lb_bounds
  ub_vec = args.ub_bounds
  init_vec = args.init_vals
  inv_params = args.inv_params 
  list_vars = args.total_params
   
  run_multispecies_inversion(pat_dir, res_dir, init_vec, lb_vec, ub_vec, inv_params, list_vars)    
   
  




