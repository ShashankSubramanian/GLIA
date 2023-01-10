import os,sys
from run_multispecies_no_elas import create_tusolver_config
import argparse
import subprocess 


scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(os.path.join(code_dir, 'scripts', 'utils'))

from file_io import readNetCDF, createNetCDF
from image_tools import smoothBinaryMap
import configparser

import numpy as np

import cma





def run_multispecies_inversion(params):
 
  
  cma_init = []
  cma_lb = []
  cma_ub = []
  
  list_vars = ['k', 'rho', 'beta_0', 'alpha_0', 'death_rate', 'ox_source', 'ox_consumption', 'ox_hypoxia', 'ox_inv', 'invasive_thres'] 

  for p in list_vars:
    cma_init.append(params[p])
    cma_lb.append(params[p+'_lb'])
    cma_ub.append(params[p+'_ub'])
     
      
  cma_init = np.array(cma_init)
  cma_lb = np.array(cma_lb)
  cma_ub = np.array(cma_ub)

  cma_init = (cma_init - cma_lb)/(cma_ub - cma_lb) 
  fun = lambda x : create_cma_output(x, cma_lb, cma_ub, cma_init, list_vars, params)

  opts = cma.CMAOptions()
  #opts.set('tolfunc',1e-2) 
  #opts.set('popsize', 12)
  es = cma.CMAEvolutionStrategy(cma_init, params['sigma_cma'], {'bounds': [0, 1], 'popsize':params['popsize'], 'tolfun': params['tolfun'], 'tolfunrel':params['tolfunrel'], 'seed' : np.nan})
  
  while not es.stop():
    solutions = es.ask()
    es.tell(solutions, fun(solutions))
    es.disp()
 
  res = es.result_pretty()
  print(res)

  out = res.xbest
  out_inv = out * (cma_ub - cma_lb) + cma_lb
  print("before noremalize ", out_inv)

  
def create_cma_output(solutions, lb_vec, ub_vec, init_vec, list_vars, params):
 
  print("Testing ")
  
  seg_t1 = params['d1_path']
  
  dat_seg_true = readNetCDF(seg_t1)

  dat_en_true = dat_seg_true.copy()
  dat_ed_true = dat_seg_true.copy()
  dat_nec_true =  dat_seg_true.copy()
  dat_wm_true = dat_seg_true.copy()
  dat_gm_true = dat_seg_true.copy()

  dat_en_true[dat_en_true != 4] = 0  
  dat_en_true[dat_en_true == 4] = 1

  dat_ed_true[dat_ed_true != 2] = 0  
  dat_ed_true[dat_ed_true == 2] = 1
  
  dat_nec_true[dat_nec_true != 1] = 0  
  dat_nec_true[dat_nec_true == 1] = 1

  dat_vt_true = dat_seg_true.copy()
  dat_vt_true[dat_vt_true != 7] = 0
  dat_vt_true[dat_vt_true == 7] = 1
  
  dat_en_true = smoothBinaryMap(dat_en_true, 1)  
  dat_ed_true = smoothBinaryMap(dat_ed_true, 1)  
  dat_nec_true = smoothBinaryMap(dat_nec_true, 1)  

 
  num_devices = 4
  num_eval = int(len(solutions))
  J_vec = np.zeros(num_eval)
  num_batch = int(np.ceil(num_eval / num_devices))

  num_var = len(list_vars)
  

  for i in range(num_batch):
    cmd = ''
    for j in range(num_devices):
      if i * num_devices + j >= num_eval:
        break
       
      res_forward_dir = os.path.join(params['output_dir'], 'forward_%d'%j)
      if not os.path.exists(res_forward_dir):
        os.mkdir(res_forward_dir)
      
      x = solutions[i * num_devices + j]
      x = x * (ub_vec - lb_vec) + lb_vec;

      forward_params = {}
      str_disp = "Forward params (%d) : "%j
      
      for (k, p) in enumerate(list_vars):
        forward_params[p] = x[k]
        str_disp += p + " = " + str(x[k]) + ", "
      forward_params['gamma'] = 0  
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
       
      if forward_params['ox_inv'] < forward_params['ox_hypoxia']:
        J_vec[i * num_devices + j] = 10.0
        continue  
      


      create_tusolver_config(params['pat_dir'], res_forward_dir, forward_params, True, params['d0_path'], params['a_seg_path'], params['n'])
      
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

    

    w_en = 0.5
    w_nec = 0.5
    w_ed = 0.5
    w_vt = 0.5
    for j in range(num_devices):
      if i * num_devices + j >= num_eval:
        break
      if J_vec[i * num_devices + j] != 0:
        out = "Objective function (En + Nec + Ed + Misfit[VT] = J) : = %.4f "%(J_vec[i * num_devices + j])
        print(out)
        continue
      x = solutions[i * num_devices + j]
      x = x * (ub_vec - lb_vec) + lb_vec;

      res_forward_dir = os.path.join(params['output_dir'], 'forward_%d'%j)
      en_rec_path = os.path.join(res_forward_dir, 'en_rec_final.nc')
      nec_rec_path = os.path.join(res_forward_dir, 'nec_rec_final.nc')
      i_rec_path = os.path.join(res_forward_dir, 'i_rec_final.nc')
      wm_rec_path = os.path.join(res_forward_dir, 'wm_rec_final.nc')
      gm_rec_path = os.path.join(res_forward_dir, 'gm_rec_final.nc')
      seg_rec_path = os.path.join(res_forward_dir, 'seg_rec_final.nc')
      bg_rec_path = os.path.join(res_forward_dir, 'bg.nc') 
      vt_rec_path = os.path.join(res_forward_dir, 'vt_rec_final.nc') 
      csf_rec_path = os.path.join(res_forward_dir, 'csf_rec_final.nc') 
      
   
      dat_en_rec = readNetCDF(en_rec_path)
      dat_nec_rec = readNetCDF(nec_rec_path)
      dat_i_rec = readNetCDF(i_rec_path)
      dat_wm_rec = readNetCDF(wm_rec_path)
      dat_gm_rec = readNetCDF(gm_rec_path)
      seg_rec = readNetCDF(seg_rec_path)
      vt_csf = readNetCDF(bg_rec_path) + readNetCDF(vt_rec_path) + readNetCDF(csf_rec_path)
        
      if np.amax(dat_i_rec) < x[-1]:
        J_vec[i * num_devices + j] = 10.0
        continue 


     
      Op, On, Ol = obs_operator(dat_en_rec, dat_nec_rec, dat_i_rec, dat_wm_rec, dat_gm_rec, vt_csf, x[-1], params['HS_seg'], params['HS_ed'])

      Op = smoothBinaryMap(Op, 1) 
      On = smoothBinaryMap(On, 1) 
      Ol = smoothBinaryMap(Ol, 1) 
 
      dx = 2 * np.pi / params['n']       
      '''
      fname = os.path.join(res_forward_dir, 'Ol.nc')
      createNetCDF(fname, Ol.shape, Ol)
      fname = os.path.join(res_forward_dir, 'On.nc')
      createNetCDF(fname, Ol.shape, On)
      fname = os.path.join(res_forward_dir, 'Op.nc')
      createNetCDF(fname, Ol.shape, Op)

       
      fname = os.path.join(res_forward_dir, 'dat_ed_true.nc')
      createNetCDF(fname, Ol.shape, dat_ed_true)
      fname = os.path.join(res_forward_dir, 'dat_p_true.nc')
      createNetCDF(fname, Ol.shape, dat_en_true)
      fname = os.path.join(res_forward_dir, 'dat_n_true.nc')
      createNetCDF(fname, Ol.shape, dat_nec_true)

      ''' 

      diff_en = w_en * np.linalg.norm((dat_en_true - Op).flatten())**2 * dx**3
      diff_nec = w_nec * np.linalg.norm((dat_nec_true - On).flatten())**2 * dx**3
      diff_ed = w_ed * np.linalg.norm((dat_ed_true - Ol).flatten())**2 * dx**3
      #diff_vt =  w_vt * np.linalg.norm((dat_vt_true - dat_vt_rec).flatten())**2 * dx**3
      
      Q_neg = np.load(params['q_neg'])
      bias = params['beta_p'] * 0.5 * np.linalg.norm(Q_neg.T * x)
      
      #J = diff_en + diff_nec + diff_ed + diff_vt + bias 
      J = diff_en + diff_nec + diff_ed + bias 
      
      J_vec[i * num_devices + j] = J 
      #out = "Objective function (En + Nec + Ed + + Bias = J) : %.4f + %.4f + %.4f + %.4f + %.4f = %.4f "%(diff_en, diff_nec, diff_ed, diff_vt, bias, J)
      out = "Objective function (En + Nec + Ed + + Bias = J) : %.4f + %.4f + %.4f + %.4f = %.4f "%(diff_en, diff_nec, diff_ed, bias, J)
      print(out)
      ''' 
      os.remove(en_rec_path)
      os.remove(nec_rec_path)
      os.remove(i_rec_path)
      '''
  return J_vec



def excmd(cmd, skip=False):
  #print(cmd)
  if not skip:
    os.system(cmd)


def read_config(config_path):
  
  config = configparser.ConfigParser()
  config.read(config_path)
  params = dict(config['DEFAULT'].items())
  
  for k,v in params.items():
    if isfloat(v):
      params[k] = float(v)
  return params 


def isfloat(x):
   
  try :
    a = float(x)
  except (TypeError, ValueError):
    return False
  else:
    return True

def obs_operator(p_vec, n_vec, i_vec, wm_vec, gm_vec, vt_csf, i_th, s_gen, s_ed):
  
  c_vec = p_vec + n_vec + i_vec 
 
  Oc = np.ones(c_vec.shape)
  Oc[c_vec < wm_vec] = 0.0
  Oc[c_vec < gm_vec] = 0.0
  Oc[c_vec < vt_csf] = 0.0
  
  Op = Oc.copy()
  Op[p_vec < n_vec] = 0.0 
  Op[p_vec < i_vec] = 0.0 

  On = Oc.copy()
  On[n_vec < p_vec] = 0.0 
  On[n_vec < i_vec] = 0.0 

  Ol = 1 - Op - On
  Ol[c_vec < vt_csf] = 0.0
  Ol[i_vec < i_th] = 0.0


  '''
  Oc = HS(c_vec, wm_vec, s_gen) * HS(c_vec, gm_vec, s_gen) * HS(c_vec, vt_csf, s_gen)
  On = HS(n_vec, p_vec, s_gen) * HS(n_vec, i_vec, s_gen) * Oc
  Op = HS(p_vec, n_vec, s_gen) * HS(p_vec, i_vec, s_gen) * Oc
  Ol = HS(i_vec, i_th, s_ed) * (1 - Op - On) * HS(c_vec, vt_csf, s_gen)
  '''
  return Op, On, Ol

def HS(vec_0, vec_1, n):

    out = (1 / (1 + np.exp(-(n)*(vec_0 - vec_1))))

    return out
 



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Multi spcies inversion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-config', '--ms_solver_config', type = str, help = 'path to the patient data', required = True)
  args = parser.parse_args();

  config_path = args.ms_solver_config

  params = read_config(config_path)
  params['HS_seg'] = 1024
  params['HS_ed'] = 1024
  run_multispecies_inversion(params) 
   
  




