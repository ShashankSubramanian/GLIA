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
p['output_dir'] = os.path.join(code_dir, 'results/forward_ms5/');                          # results path
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path'] = os.path.join(code_dir, 'results/seg_t[60].nc')
p['a_gm_path'] = ""
p['a_wm_path'] = ""
p['a_csf_path'] = ""
p['a_vt_path'] = ""
p['d1_path'] = ""
p['d0_path'] = os.path.join(code_dir, 'results/c_t[60].nc')
#p['mri_path'] = os.path.join(code_dir, 'data/51566_t1_aff2jakob.nc')
p['mri_path'] = ""
p['solver'] = 'forward'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 5                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                    # various levels of output density
p['syn_flag'] = 1                     # create synthetic data
p['user_cms'] = [(132,69,148,0.8),(132,64,148,0.5),(132,69,144,0.3)]      # location of tumor seed (can be multiple; [(x1,y1,z1,activation1),(x2,y2,z2,activation2)])

####### tumor params for synthetic data
p['rho_data'] = 6
p['k_data'] = 0.1
p['gamma_data'] = 5.0E4
p['ox_hypoxia_data'] = 0.65
p['death_rate_data'] = 0.4
p['alpha_0_data'] = 0.4
p['ox_consumption_data'] = 8.0
p['ox_source_data'] = 55.0
p['beta_0_data'] = 0.3
p['prediction'] = 1
#p['write_output'] = 0
p['k_gm_wm'] = 0.2                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)

p['nt_data'] = 100
p['dt_data'] = 0.01
p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration if job submit is needed; else run from results folder directly
r['code_path'] = code_dir;
r['compute_sys'] = 'maverick2'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)

###############=== write config to write_path and submit job
par.submit(p, r, submit_job, use_gpu);




