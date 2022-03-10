import os, sys
import params as par
import subprocess
import numpy as np
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
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
#p['a_seg_path'] = os.path.join(code_dir, 'results/init_forward_ms2/seg_t[0].nii.gz')
p['a_seg_path'] = os.path.join(code_dir, 'data/51566_seg_aff2jakob_ants.nii.gz')
p['a_gm_path'] = ""
p['a_wm_path'] = ""
p['a_csf_path'] = ""
p['a_vt_path'] = ""
p['d1_path'] = ""
#p['d0_path'] = os.path.join(code_dir, 'results/c_t[60].nc')
p['d0_path'] = ""
#p['d0_path'] = os.path.join(code_dir, 'results/init_forward_ms2/c_t[0].nii.gz')

#p['mri_path'] = os.path.join(code_dir, 'data/51566_t1_aff2jakob.nc')
p['mri_path'] = ""
p['solver'] = 'multi_species'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 5                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                    # various levels of output density
p['syn_flag'] = 1                    # create synthetic data
p['user_cms'] = [(129,81,151,0.7),(129,81,147,0.7),(129,77,151,0.7),(129,81,149,0.4),(129,79,151,0.4)]      # location of tumor seed (can be multiple; [(x1,y1,z1,activation1),(x2,y2,z2,activation2)])

####### tumor params for synthetic data

plist = [[8, 8, 8], 
         [0.05, 0.1, 0.15],
         [2E4, 4E4, 6E4],
         [0.05, 0.15, 0.25],
         [0.01, 0.02, 0.03],
         [0.35, 0.65, 0.5],
         [0.65, 0.75, 0.85],
         [40, 55, 70],
         [4, 8, 12],
         [0.8, 1.0, 1.2],
         [0.8, 0.9, 1.0]]
plist = np.array(plist)
print(plist[:, 1])
parray = np.array(plist[:,1])

counter = -1
for i in range(11):
  for j in range(3):
    counter += 1
    p['output_dir'] = os.path.join(code_dir, 'results/forward_ms%d/'%counter);
    r['log_dir'] = os.path.join(code_dir, 'results/forward_ms%d/'%counter);
    parray[i] = plist[i][j]
    p['rho_data'] = parray[0]
    p['k_data'] = parray[1]
    p['gamma_data'] = parray[2]
    p['alpha_0_data'] = parray[3]
    p['beta_0_data'] = parray[4]
    p['ox_hypoxia_data'] = parray[5]
    p['ox_inv_data'] = parray[6]
    p['ox_source_data'] = parray[7]
    p['ox_consumption_data'] = parray[8]
    p['death_rate_data'] = parray[9]
    p['sigma_b_data'] = parray[10]
    p['prediction'] = 1
    p['k_gm_wm'] = 0                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
    p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)
    p['ratio_i0_c0'] = 0.0
    p['nt_data'] = 40
    p['dt_data'] = 0.025
    p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)
    ############### === define run configuration if job submit is needed; else run from results folder directly
    r['code_path'] = code_dir;
    r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
    r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
    r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
    ###############=== write config to write_path and submit job
    par.submit(p, r, submit_job, use_gpu);




