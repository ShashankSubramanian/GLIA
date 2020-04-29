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
p['a_gm_path'] = code_dir + "/brain_data/256/gray_matter.nc"     # atlas paths
p['a_wm_path'] = code_dir + "/brain_data/256/white_matter.nc"
p['a_csf_path'] = code_dir + "/brain_data/256/csf.nc"
p['a_kf_path'] = code_dir + "/brain_data/256/"                # kf matrices should be in the format of kf11.nc, kf12.nc ...
p['a_vt_path'] = "" #code_dir + "/brain_data/t16/256/t16_vt.nc"
p['mri_path'] = "" #code_dir + "/brain_data/t16/t1.nc"
p['solver'] = 'forward'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 1                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                  # various levels of output density
p['syn_flag'] = 1                   # create synthetic data
p['user_cms'] = [(90,60,132,1)]    # arbitrary number of TILs (x,y,z,scale) with activation scale
p['rho_data'] = 8                  # tumor parameters for synthetic data
p['k_data'] = 0.01
p['kf_data'] = 0.3
p['gamma_data'] = 12E4
p['nt_data'] = 50
p['dt_data'] = 0.02
p['time_history_off'] = 1           # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration
r['code_path'] = code_dir
r['compute_sys'] = 'rebels'         # TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
par.submit(p, r, submit_job)
