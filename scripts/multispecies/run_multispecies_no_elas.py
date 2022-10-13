import os, sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(os.path.join(code_dir, 'scripts'))
import params as par
import subprocess

###############
r = {}
p = {}
submit_job = True;
use_gpu = True;
###############





def create_tusolver_config(pat_dir, res_dir, forward_params, is_syn, d0_path, a_seg_path, n):

  r = {}
  p = {}
  submit_job = False;
  use_gpu = True;
  p['n'] = n
  p['output_dir'] = res_dir +'/'
  p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
  '''
  if n == 256:
    p['a_seg_path'] = os.path.join(pat_dir, 'seg_t0.nc')
  else:
    p['a_seg_path'] = os.path.join(pat_dir, 'seg_t0_nx%d.nc'%n)
  '''
  
  
  #p['p_seg_path'] = os.path.join(pat_dir, 'seg_t1.nc')
  p['a_wm_path'] = "" 
  p['a_gm_path'] = ""
  p['a_csf_path'] = ""
  p['a_vt_path'] = "" 
  p['a_seg_path'] = a_seg_path
  #p['d0_path'] = os.path.join(pat_dir, 'c0_true_syn_nx160.nc')
  p['d0_path'] = d0_path
  p['d1_path'] = ""
  
  p['solver'] = 'multi_species'
  p['model'] = 5
  p['verbosity'] = 3
  p['syn_flag'] = 0
  p['user_cms'] = []
  
  p['rho_data'] = forward_params['rho'] 
  p['k_data'] = forward_params['k'] 
  p['gamma_data'] = forward_params['gamma']
  p['ox_hypoxia_data'] = forward_params['ox_hypoxia'] 
  p['death_rate_data'] = forward_params['death_rate'] 
  p['alpha_0_data'] = forward_params['alpha_0']
  p['ox_consumption_data'] = forward_params['ox_consumption']
  p['ox_source_data'] = forward_params['ox_source']
  p['beta_0_data'] = forward_params['beta_0']
  p['sigma_b_data'] = 1.0 
  p['ox_inv_data'] = forward_params['ox_inv']
  p['invasive_thres_data'] = forward_params['invasive_thres']
  p['write_multispecies_output'] = 1 
  p['write_output'] = 1
  p['nt_data'] = 80
  p['dt_data'] = 0.0125
  p['k_gm_wm'] = 0.2
  p['HS_shape_factor'] = 16
  p['r_gm_wm'] = 0
  p['ratio_i0_c0'] = 0.0
  p['time_history_off'] = 1
  '''
  if forward_params['gamma'] == 0:
    p['given_velocities'] = 1
    p['velocity_prefix'] = v_dir
  else:
  '''
  p['given_velocities'] = 0
    

  r['code_path'] = code_dir;
  r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
  r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['wtime_h'] = 2
  par.submit(p, r, submit_job, use_gpu);
  
  
   
def create_tusolver_config_temp(pat_dir, res_dir, forward_params, is_syn, d0_path, a_seg_path, n):

  r = {}
  p = {}
  submit_job = False;
  use_gpu = True;
  p['n'] = n
  p['output_dir'] = res_dir +'/'
  p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
  p['a_seg_path'] = a_seg_path
  
  #p['p_seg_path'] = os.path.join(pat_dir, 'seg_t1.nc')
  p['a_wm_path'] = "" 
  p['a_gm_path'] = ""
  p['a_csf_path'] = ""
  p['a_vt_path'] = "" 
  #p['d0_path'] = os.path.join(pat_dir, 'c0_true_syn_nx160.nc')
  p['d0_path'] = d0_path
  p['d1_path'] = ""
  
  p['solver'] = 'multi_species'
  p['model'] = 5
  p['verbosity'] = 3
  p['syn_flag'] = 0
  p['user_cms'] = []
  
  p['rho_data'] = forward_params['rho'] 
  p['k_data'] = forward_params['k'] 
  p['gamma_data'] = forward_params['gamma']
  p['ox_hypoxia_data'] = forward_params['ox_hypoxia'] 
  p['death_rate_data'] = forward_params['death_rate'] 
  p['alpha_0_data'] = forward_params['alpha_0']
  p['ox_consumption_data'] = forward_params['ox_consumption']
  p['ox_source_data'] = forward_params['ox_source']
  p['beta_0_data'] = forward_params['beta_0']
  p['ox_inv_data'] = forward_params['ox_inv']
  p['invasive_thres_data'] = forward_params['invasive_thres']
  p['write_multispecies_output'] = 1 
  p['write_output'] = 1
  p['nt_data'] = 80
  p['dt_data'] = 0.0125
  p['k_gm_wm'] = 0.2
  p['r_gm_wm'] = 0
  p['ratio_i0_c0'] = 0.0
  p['time_history_off'] = 1
  p['given_velocities'] = 0
  '''
  if forward_params['gamma'] == 0:
    p['given_velocities'] = 1
    p['velocity_prefix'] = v_dir
  else:
  '''

  r['code_path'] = code_dir;
  r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
  r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['wtime_h'] = 2
  par.submit(p, r, submit_job, use_gpu);
  
  
   
