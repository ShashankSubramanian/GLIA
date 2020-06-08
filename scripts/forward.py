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
p['output_dir'] = os.path.join(code_dir, 'results/atlas-2-tu-case4/');   # results path
#p['a_gm_path'] = code_dir + "/brain_data/t16/256/t16_gm.nc"     # atlas paths
#p['a_wm_path'] = code_dir + "/brain_data/t16/256/t16_wm.nc"
#p['a_csf_path'] = code_dir + "/brain_data/t16/256/t16_csf.nc"
#p['a_vt_path'] = code_dir + "/brain_data/t16/256/t16_vt.nc"
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"              # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path'] = code_dir + "/brain_data/atlas/atlas-2.nc"                # paths to atlas material properties
p['mri_path'] = code_dir + "/brain_data/atlas/atlas-2-t1.nc"
p['solver'] = 'forward'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
p['model'] = 4                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 3                  # various levels of output density
p['syn_flag'] = 1                   # create synthetic data
#p['user_cms'] = [(133,78,148,1),(144,70,143,0.2)]    #atlas3 arbitrary number of TILs (x,y,z,scale) with activation scale
#p['user_cms'] = [(135,83,145,1),(143,71,151,0.2)]   #atlas2 arbitrary number of TILs (x,y,z,scale) with activation scale
#p['user_cms'] = [(138,48,117,1)]   #atlas2-awayfrom vt arbitrary number of TILs (x,y,z,scale) with activation scale
p['user_cms'] = [(69,110,80,1)]   #atlas2-awayfrom vt arbitrary number of TILs (x,y,z,scale) with activation scale
##p['user_cms'] = [(137,169,96,1),(130,160,112,0.1)]    # arbitrary number of TILs (x,y,z,scale) with activation scale
p['rho_data'] = 10                  # tumor parameters for synthetic data
p['k_data'] = 0.01
p['gamma_data'] = 10E4
p['k_gm_wm']            = 0.25                  # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm']            = 1                  # rho ratio gm/wm (if zero, rho=0 in gm)
p['nt_data'] = 25
p['dt_data'] = 0.04
p['time_history_off'] = 1           # 1: do not allocate time history (only works with forward solver or FD inversion)

############### === define run configuration
r['code_path'] = code_dir;
r['compute_sys'] = 'longhorn'         # TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
par.submit(p, r, submit_job);
