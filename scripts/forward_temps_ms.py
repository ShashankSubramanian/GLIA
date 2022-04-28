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
p['n'] = 160                           # grid resolution in each dimension
inv_pat_dir = "/scratch1/07544/ghafouri/results/syn_results/inv_p_temp_m/"
atlas_dir = "/scratch1/07544/ghafouri/data/adni-nc/"
list_pats=[]

for i in os.listdir(inv_pat_dir):
  if 'case' in i:
    list_pats.append(i)

for pat in list_pats:

  list_path = os.path.join(inv_pat_dir, pat, 'atlas-list.txt')
  with open(list_path) as f:
    lines = f.readlines()
  for j in lines:
    at = j.strip('\n')
  
    r = {}
    p = {}
    p['n'] = 160                           # grid resolution in each dimension
    scratch = os.getenv('SCRATCH')
    #p['output_dir'] = os.path.join(scratch, 'results/syn_results/me_inv_160/case%d/fwd_me/'%syn);                          # results path
    #p['output_dir'] = os.path.join(scratch, 'results/syn_results/true_p_true_m/me_inv_160/case%d/fwd_me/'%syn);                          # results path
    p['output_dir'] = os.path.join(inv_pat_dir, pat, 'fwd_me', at+'/');                          # results path
    p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
    p['a_seg_path'] = os.path.join(atlas_dir, '160', at+'_seg_aff2jakob_ants_160.nc')
    p['mri_path'] = "" #os.path.join(code_dir, 'testdata/atlas_t1.nc')
    p['solver'] = 'forward'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
    p['model'] = 4                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
    p['verbosity'] = 1                    # various levels of output density
    p['syn_flag'] = 0                     # create synthetic data
    #p['d0_path'] = os.path.join(code_dir, 'syndata/case%d/160/c0_true_syn_nx160.nc'%syn)
    p['d0_path'] = os.path.join(inv_pat_dir, pat, 'tu/160', at, 'c0_input.nc')
     
    recon_file = os.path.join(inv_pat_dir, pat, 'tu/160', at, 'reconstruction_info.dat')
    with open(recon_file, 'r') as f:
      lines = f.readlines()
      if len(lines) > 1:
        l = lines[1].split(" ")
        rho       = float(l[0])
        kappa     = float(l[1])
        gamma = float(l[2])


    p['rho_data'] = rho                
    p['k_data'] = kappa 
    p['gamma_data'] = gamma 
    p['k_gm_wm'] = 0.2                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
    p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)
    p['nt_data'] = 50
    p['dt_data'] = 0.02
    p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

    ############### === define run configuration if job submit is needed; else run from results folder directly
    r['code_path'] = code_dir;
    r['compute_sys'] = 'frontera'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
    r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
    r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
    p['write_all_velocities'] = 1
    ###############=== write config to write_path and submit job
    par.submit(p, r, submit_job, use_gpu);
