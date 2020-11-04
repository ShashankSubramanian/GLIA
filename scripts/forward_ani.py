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
data_dir = "/scratch/ghafouri/data/128/"
out_dir = "/scratch/ghafouri/results/syn/forward_noise_42/"
kfs = [3.42627]
ks = [0.12134]
Ts = [1.0]
rho = 12.6708




############### === define parameters

for kf in kfs:
  for k in ks:
    for T in Ts:
        p['n'] = 128
	name = 'forward-' +'k' + str(k) + '-kf' + str(kf) + '-r' + str(rho) + '-T' + str(T) + '/'
	p['output_dir'] = out_dir + name #os.path.join(code_dir,  name);   # results path
	p['d0_path'] = ""
	p['d1_path'] = ""
	p['prediction'] = 0
	p['a_gm_path'] = data_dir + "gray_matter.nc"     # atlas paths
	p['a_wm_path'] = data_dir + "white_matter.nc"
	p['a_vt_path'] = data_dir + "csf.nc"
	p['a_kf_path'] = '/scratch/ghafouri/data/ADNI_PROCESSED_256/128/'                 # kf matrices should be in the format of kf11.nc, kf12.nc ...
	p['a_csf_path'] = "" #code_dir + "/brain_data/t16/256/t16_vt.nc"
	p['mri_path'] = "" #code_dir + "/brain_data/t16/t1.nc"
	p['solver'] = 'forward'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
	p['model'] = 2                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
	p['verbosity'] = 3                  # various levels of output density
	p['syn_flag'] = 1                   # create synthetic data
	p['user_cms'] = [(118,130,154,1),(118,130,98,1)]    # arbitrary number of TILs (x,y,z,scale) with activation scale
	p['obs_threshold_0'] = 0.0
	p['obs_threshold_1'] = 0.0
	p['k_gm_wm'] = 1.0
	p['rho_data'] = rho                  # tumor parameters for synthetic data
	p['k_data'] = k
	p['kf_data'] = kf
	p['gamma_data'] = 12E4
	p['dt_data'] = 0.01
	p['nt_data'] = int(T/p['dt_data'])
	p['time_history_off'] = 1           # 1: do not allocate time history (only works with forward solver or FD inversion)

	############### === define run configuration
	r['code_path'] = code_dir
	r['compute_sys'] = 'rebels'         # TACC systems are: maverick2, frontera, stampede2, longhorn

	###############=== write config to write_path and submit job
	par.submit(p, r, submit_job)
