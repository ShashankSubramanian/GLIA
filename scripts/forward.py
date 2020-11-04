import os, sys
import params as par
import subprocess

###############
r = {}
p = {}
submit_job = True;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['output_dir'] = os.path.join(code_dir, 'results/forward/');   # results path
p['a_gm_path'] = code_dir + "/brain_data/t16/256/t16_gm.nc"     # atlas paths
p['a_wm_path'] = code_dir + "/brain_data/t16/256/t16_wm.nc"
p['a_csf_path'] = code_dir + "/brain_data/t16/256/t16_csf.nc"
p['a_vt_path'] = code_dir + "/brain_data/t16/256/t16_vt.nc"
p['mri_path'] = code_dir + "/brain_data/t16/t1.nc"
p['solver'] = 'forward'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 2                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                  # various levels of output density
p['syn_flag'] = 1                   # create synthetic data
p['user_cms'] = [(137,169,96,1)]    # arbitrary number of TILs (x,y,z,scale) with activation scale
p['rho_data'] = 10                  # tumor parameters for synthetic data
p['k_data'] = 0.025
p['gamma_data'] = 12E4
p['nt_data'] = 25
p['dt_data'] = 0.04
p['time_history_off'] = 1           # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration
r['code_path'] = code_dir;
r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
par.submit(p, r, submit_job);
