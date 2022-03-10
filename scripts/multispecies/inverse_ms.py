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
p['output_dir'] = os.path.join(code_dir, 'results/inverse_ms12_sc3/');                          # results path
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['patient_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path'] = os.path.join(code_dir, 'results/syn_ic25/seg_t[0].nii.gz')
p['p_seg_path'] = os.path.join(code_dir, 'results/syn_ic25/seg_t[0].nii.gz')
p['a_gm_path'] = ""
p['a_wm_path'] = ""
p['a_csf_path'] = ""
p['a_vt_path'] = ""
p['d1_path'] = os.path.join(code_dir, 'results/syn_ic25/c_t[200].nii.gz');
p['d1_nec_path'] = os.path.join(code_dir, 'results/syn_ic25/n_t[200].nii.gz');
p['d1_ed_path'] = os.path.join(code_dir, 'results/syn_ic25/ed_t[200].nii.gz');
p['d1_en_path'] = os.path.join(code_dir, 'results/syn_ic25/p_t[200].nii.gz');
p['d0_path'] = os.path.join(code_dir, 'results/syn_ic25/c_t[0].nii.gz')
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
p['init_gamma'] = 6.5E4
p['init_rho'] = 8.0
p['init_k'] = 0.1
p['init_ox_hypoxia'] = 0.4
p['init_death_rate'] = 0.9
p['init_alpha_0'] = 0.2
p['init_ox_consumption'] = 4.0
p['init_ox_source'] = 55.0
p['init_beta_0'] = 0.02
p['init_sigma_b'] = 0.8
p['init_ox_inv'] = 0.5
p['init_invasive_thres'] = 0.02
p['obs_threshold_1'] = 0.005

#p['gamma_lb'] = 3E4
#p['gamma_ub'] = 8.0E4
p['gamma_lb'] = p['init_gamma'] * 0.9999
p['gamma_ub'] = p['init_gamma'] * 1.0001
p['rho_lb'] = 4.0 
p['rho_ub'] = 12.0
p['kappa_lb'] = 0.005 
p['kappa_ub'] = 0.3
#p['kappa_lb'] = p['init_k'] * 0.9999
#p['kappa_ub'] = p['init_k'] * 1.0001
p['ox_hypoxia_lb'] = 0.001 
p['ox_hypoxia_ub'] = 1.0
p['death_rate_lb'] = 0.001
p['death_rate_ub'] = 2.0
#p['alpha_0_lb'] = 0.001
#p['alpha_0_ub'] = 1.0
p['alpha_0_lb'] = p['init_alpha_0'] * 0.9999
p['alpha_0_ub'] = p['init_alpha_0'] * 1.0001
p['ox_consumption_lb'] = 0.1
p['ox_consumption_ub'] = 6.0
p['ox_source_lb'] = 40.0
p['ox_source_ub'] = 80.0
#p['beta_0_lb'] = 0.001
#p['beta_0_ub'] = 1.0
p['beta_0_lb'] = p['init_beta_0'] * 0.9999
p['beta_0_ub'] = p['init_beta_0'] * 1.0001
p['sigma_b_lb'] = 0.001 
p['sigma_b_ub'] = 1.0 
p['ox_inv_lb'] = 0.001
p['ox_inv_ub'] = 1.0
p['invasive_thres_lb'] = 0.001
p['invasive_thres_ub'] = 1.0

p['ratio_i0_c0'] = 0.0
p['sigma_gamma'] = 3E4
p['sigma_rho'] = 4.0
p['sigma_k'] = 0.2
p['sigma_ox_hypoxia'] = 0.4
p['sigma_death_rate'] = 0.6
p['sigma_alpha_0'] = 0.3
p['sigma_ox_consumption'] = 2.0
p['sigma_ox_source'] = 20.0
p['sigma_beta_0'] = 0.2
p['sigma_thres'] = 0.2
p['sigma_ox_inv'] = 0.2
p['sigma_invasive_thres'] = 0.3

p['prediction'] = 0
#p['write_output'] = 0
p['k_gm_wm'] = 0.0                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)

p['nt_inv'] = 200
p['dt_inv'] = 0.01
p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration if job submit is needed; else run from results folder directly
r['code_path'] = code_dir;
r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
r['wtime_h'] = 48
###############=== write config to write_path and submit job
par.submit(p, r, submit_job, use_gpu);




