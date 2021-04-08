import os, sys
import params as par
import subprocess

###############
r = {}
p = {}
submit_job = False;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['n'] = 256                           # grid resolution in each dimension
p['output_dir'] = os.path.join(code_dir, 'results/check_mf/');                          # results path
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path'] = '/scratch/05027/shas1693/pglistr_tumor/brain_data/atlas/atlas-2.nc'
p['mri_path'] = '/scratch/05027/shas1693/pglistr_tumor/brain_data/atlas/atlas-2-t1.nc'
#p['a_seg_path'] = '/scratch/05027/shas1693/adni-nc/256/50052_seg_aff2jakob_ants_256.nc'
#p['mri_path'] = '/scratch/05027/shas1693/adni-nc/256/50052_t1_aff2jakob_256.nc'
p['solver'] = 'forward'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 4                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                    # various levels of output density
p['syn_flag'] = 1                     # create synthetic data
p['user_cms'] = [(136,88,116,1),(123,149,85,0.05)]      # location of tumor seed (can be multiple; (x,y,z,activation))
p['rho_data'] = 10                     # tumor parameters for synthetic data
p['k_data'] = 0.02
p['gamma_data'] = 10E4
p['k_gm_wm'] = 0                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)
p['nt_data'] = 30
p['dt_data'] = 0.04
p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration if job submit is needed; else run from results folder directly
r['code_path'] = code_dir;
r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn
r['mpi_tasks'] = 1                    # mpi tasks (other job params like nodes, waittime are defaulted from params.py; overwrite here if needed)

###############=== write config to write_path and submit job
par.submit(p, r, submit_job);
