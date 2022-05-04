import os, sys
import params as par
import subprocess

###############
submit_job = True;
use_gpu = True;
###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters



pat_list = []
for i in range(1,9):
  pat_list.append('case'+str(i))
pat_dir = '/scratch1/07544/ghafouri/results/syndata'
#res_dir = '/scratch1/07544/ghafouri/results/syn_results/true_p_true_m/ms_inv_160/case1/'
res_dir = '/scratch1/07544/ghafouri/results/syn_results/true_p_true_m/ms_inv_160/'
fwd_me_dir = '/scratch1/07544/ghafouri/results/syn_results/true_p_true_m/me_inv_160/'

for pat in pat_list:

  pat_mod = pat + '/bias2'
  r = {}
  p = {}
  p['n'] = 160                           # grid resolution in each dimension
 
  p['output_dir'] = os.path.join(res_dir, pat_mod, 'fwd_ms/');            # results path
  
  p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
  p['a_seg_path'] = os.path.join(pat_dir, pat, 'C1_me', 'seg_t0_nx160.nc')
  p['a_gm_path'] = ""
  p['a_wm_path'] = ""
  p['a_csf_path'] = ""
  p['a_vt_path'] = ""
  p['d1_path'] = ""
  p['d0_path'] = os.path.join(fwd_me_dir, pat, 'fwd_me', 'c0_true_syn.nc')
  p['mri_path'] = ""
  p['solver'] = 'multi_species'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
  p['model'] = 5                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
  p['verbosity'] = 3                    # various levels of output density
  p['syn_flag'] = 0                     # create synthetic data

  #cmd = 'python multispecies/extract_params.py -results_path '+os.path.join(res_dir, pat)
  #print(cmd)
  #os.system(cmd) 
  recon_file=os.path.join(res_dir, pat_mod, 'recon_info.dat')

  with open(recon_file, 'r') as f:
    lines = f.readlines()
    if len(lines) > 1:
      l = lines[1].split(" ")
      
      p['k_data'] = l[0]
      p['rho_data'] = l[1]
      p['ox_hypoxia_data'] = l[2]
      p['death_rate_data'] = l[3]
      p['alpha_0_data'] = l[4]
      p['ox_consumption_data'] = l[5]
      p['ox_source_data'] = l[6]
      p['beta_0_data'] = l[7]
      p['ox_inv_data'] = l[8]
      p['invasive_thres_data'] = l[9]
    ####### tumor params for synthetic data
    #if case == 1:
    #p['gamma_data'] = 7E4
    p['gamma_data'] = 0
  p['prediction'] = 0
  
  p['smoothing_factor_data_t0'] = 0
  p['given_velocities'] = 1
  p['velocity_prefix'] = os.path.join(fwd_me_dir, pat, 'fwd_me/')
  #p['sigma_factor'] = 3
  #p['sigma_spacing'] = 2
  p['write_output'] = 1
  p['k_gm_wm'] = 0.2                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
  p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)
  p['ratio_i0_c0'] = 0.0

  p['nt_data'] = 80
  p['dt_data'] = 0.0125
  p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

  ############### === define run configuration if job submit is needed; else run from results folder directly
  r['code_path'] = code_dir;
  r['compute_sys'] = 'frontera'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
  r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['wtime_h'] = 2
  r['queue'] = 'rtx'
  ###############=== write config to write_path and submit job

  par.submit(p, r, submit_job, use_gpu);




