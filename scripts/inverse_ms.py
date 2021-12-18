import os, sys
import params as par
import subprocess

###############
r = {}
p = {}
submit_job = True;
use_gpu = True;
###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['n'] = 256                           # grid resolution in each dimension
p['output_dir'] = os.path.join(code_dir, 'results/inverse_ms/');                          # results path
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['patient_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path'] = os.path.join(code_dir, 'results/seg_t[60].nc')
p['p_seg_path'] = os.path.join(code_dir, 'results/seg_t[60].nc')
p['a_gm_path'] = ""
p['a_wm_path'] = ""
p['a_csf_path'] = ""
p['a_vt_path'] = ""
p['d1_path'] = os.path.join(code_dir, 'results/forward_ms5/c_t[100].nc');
p['d0_path'] = os.path.join(code_dir, 'results/c_t[60].nc')
p['mri_path'] = ""



p['smoothing_factor_data'] = 0
p['smoothing_factor_data_t0'] = 0
#p['mri_path'] = os.path.join(code_dir, 'data/51566_t1_aff2jakob.nc')
p['solver'] = 'inverse_multi_species'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 6                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species 6: inversion mass-effect
p['verbosity'] = 3                    # various levels of output density
p['syn_flag'] = 0                     # create synthetic data
p['user_cms'] = [(132,69,148,0.8),(132,64,148,0.5),(132,69,144,0.3)]      # location of tumor seed (can be multiple; [(x1,y1,z1,activation1),(x2,y2,z2,activation2)])


p['regularization'] = "L2"

####### tumor params for synthetic data
p['init_gamma'] = 2.0E4
p['init_rho'] = 8.0
p['init_k'] = 0.02
p['init_ox_hypoxia'] = 0.2
p['init_death_rate'] = 0.6
p['init_alpha_0'] = 0.2
p['init_ox_consumption'] = 5.0
p['init_ox_source'] = 40.0
p['init_beta_0'] = 0.1


p['gamma_lb'] = 0
p['gamma_ub'] = 13E4
p['rho_lb'] = 0 
p['rho_ub'] = 13.0
p['kappa_lb'] = 0.005 
p['kappa_ub'] = 0.4
p['ox_hypoxia_lb'] = 0.001 
p['ox_hypoxia_ub'] = 1.0
p['death_rate_lb'] = 0.001
p['death_rate_ub'] = 1.0
p['alpha_0_lb'] = 0.001
p['alpha_0_ub'] = 1.0
p['ox_consumption_lb'] = 0.1
p['ox_consumption_ub'] = 20.0
p['ox_source_lb'] = 1.0
p['ox_source_ub'] = 70.0
p['beta_0_lb'] = 0.001
p['beta_0_ub'] = 1.0

p['sigma_gamma'] = 3E4
p['sigma_rho'] = 3.0
p['sigma_k'] = 0.1
p['sigma_ox_hypoxia'] = 0.25
p['sigma_death_rate'] = 0.25
p['sigma_alpha_0'] = 0.25
p['sigma_ox_consumption'] = 5.0
p['sigma_ox_source'] = 14.0
p['sigma_beta_0'] = 0.25

p['prediction'] = 0
#p['write_output'] = 0
p['k_gm_wm'] = 0.0                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)

p['nt_inv'] = 100
p['dt_inv'] = 0.01
p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration if job submit is needed; else run from results folder directly
r['code_path'] = code_dir;
r['compute_sys'] = 'maverick2'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)

###############=== write config to write_path and submit job
par.submit(p, r, submit_job, use_gpu);




