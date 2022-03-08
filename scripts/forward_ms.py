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
#p['output_dir'] = os.path.join(code_dir, 'results/init_forward_ms11/');            # results path
p['output_dir'] = os.path.join(code_dir, 'results/syn_ic22/');            # results path
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
#p['a_seg_path'] = os.path.join(code_dir, 'results/ic1/seg_t[0].nii.gz')
p['a_seg_path'] = os.path.join(code_dir, 'data/51566_seg_aff2jakob_ants.nii.gz')
p['a_gm_path'] = ""
p['a_wm_path'] = ""
p['a_csf_path'] = ""
p['a_vt_path'] = ""
p['d1_path'] = ""
#p['d0_path'] = os.path.join(code_dir, 'results/c_t[60].nc')
p['d0_path'] = ""
#p['d0_path'] = os.path.join(code_dir, 'results/ic1/c_t[0].nii.gz')

#p['mri_path'] = os.path.join(code_dir, 'data/51566_t1_aff2jakob.nc')
p['mri_path'] = ""
p['solver'] = 'multi_species'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 5                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                    # various levels of output density
p['syn_flag'] = 1                     # create synthetic data
p['user_cms'] = [(129,78,154,0.5),(129,78,150,0.5),(129,74,155,0.5),(129,79,152,0.3),(129,76,154,0.3),(129,77,152,0.2),(129,75,152,0.1),(129,73,153,0.1)]

####### tumor params for synthetic data
p['rho_data'] = 6.0
p['k_data'] = 0.2
p['gamma_data'] = 6.5E4
p['ox_hypoxia_data'] = 0.6
p['death_rate_data'] = 1.0
p['alpha_0_data'] = 0.2
p['ox_consumption_data'] = 3.0
p['ox_source_data'] = 75.0
p['beta_0_data'] = 0.02
p['sigma_b_data'] = 0.9
p['ox_inv_data'] = 0.7
p['invasive_thres_data'] = 0.01
p['prediction'] = 1
#p['sigma_factor'] = 3
#p['sigma_spacing'] = 2
#p['write_output'] = 0
p['k_gm_wm'] = 0                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)
p['ratio_i0_c0'] = 0.0

p['nt_data'] = 200
p['dt_data'] = 0.01
p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration if job submit is needed; else run from results folder directly
r['code_path'] = code_dir;
r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
r['wtime_h'] = 2
###############=== write config to write_path and submit job
par.submit(p, r, submit_job, use_gpu);




