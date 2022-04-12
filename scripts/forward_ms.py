import os, sys
import params as par
import subprocess

###############
submit_job = True;
use_gpu = True;
###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters

for i in range(1,5):

  #if i <= 2: continue
  #if i != 1: continue
  r = {}
  p = {}
  p['n'] = 256                           # grid resolution in each dimension
  case = i
  
  #p['n'] = 160                           # grid resolution in each dimension
  #p['output_dir'] = os.path.join(code_dir, 'results/init_forward_ms11/');            # results path

  #p['output_dir'] = os.path.join(code_dir, 'syndata/case%d/fwd_256/'%case);            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case%d/fwd_256_tc/'%case);            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case%d/fwd_256_tc=en/'%case);            # results path
  p['output_dir'] = os.path.join('/scratch1/07544/ghafouri/results', 'syndata/case%d/C1_fwd/'%case);            # results path
  #p['output_dir'] = os.path.join('/scratch1/07544/ghafouri/results', 'syndata/case%d/C1_fwd_rec/'%case);            # results path
  #p['output_dir'] = os.path.join('/scratch1/07544/ghafouri/results', 'syndata/case%d/C1_me_fwd_rec/'%case);            # results path
  #p['output_dir'] = os.path.join('/scratch1/07544/ghafouri/results', 'syndata/case%d/C1_me_fwd/'%case);            # results path
    
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case1/fwd/');            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case2/fwd/');            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case3/256/');            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case3/fwd_256/');            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case4/256/');            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/case4/fwd_256/');            # results path
  #p['output_dir'] = os.path.join(code_dir, 'syndata/test/');            # results path
  p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8]"                                           # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
  #p['a_seg_path'] = os.path.join(code_dir, 'results/ic1/seg_t[0].nii.gz')
  #p['a_seg_path'] = os.path.join(code_dir, 'syndata/160/seg_rec_final.nc')
  if case == 1:
    p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/256/50052_seg_aff2jakob_ants.nc'
  #p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/160/50052_seg_aff2jakob_ants_160.nc'
  # case 2
  if case == 2:
    p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/256/50441_seg_aff2jakob_ants.nc'
  #p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/160/50463_seg_aff2jakob_ants_160.nc'
  # case 3
  if case ==3:
    p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/256/50463_seg_aff2jakob_ants.nc'
  #p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/160/50463_seg_aff2jakob_ants_160.nc'
  # case 4
  if case == 4:
    p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/256/50698_seg_aff2jakob_ants.nc'
  #p['a_seg_path'] = '/scratch1/07544/ghafouri/data/adni-nc/160/50698_seg_aff2jakob_ants_160.nc'


   
  p['a_gm_path'] = ""
  p['a_wm_path'] = ""
  p['a_csf_path'] = ""
  p['a_vt_path'] = ""
  p['d1_path'] = ""
  #p['d0_path'] = os.path.join(code_dir, 'results/c_t[60].nc')
  #p['d0_path'] = os.path.join(code_dir, 'syndata/160/c0Recon_transported.nc')
  #p['d0_path'] = "" 
  #p['d0_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/case1/160/c0_true_syn_nx160.nc'
  #p['d0_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/case2/160/c0_true_syn_nx160.nc'
  #p['d0_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/case3/160/c0_true_syn_nx160.nc'
  #p['d0_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/case4/256/case%d_c0Recon_transported.nc'%case
  #p['d0_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/case4/256/case%d_c0Recon_transported_tc=en.nc'%case
  #p['d0_path'] = '/scratch1/07544/ghafouri/results/syndata/case%d/256/case%d_c0Recon_transported_tc=nec.nc'%(case, case)

  #p['d0_path'] = "" 
  #p['d0_path'] = "/scratch1/07544/ghafouri/results/syn_results/C1/til_inv/case%d/inversion/nx256/obs-1.0/c0_rec.nc"%case
  #p['d0_path'] = "/scratch1/07544/ghafouri/results/syn_results/me_inv/case%d/reg/case%d/case%d_c0Recon_transported.nc"%(case,case,case)
  p['d0_path'] = "" 
  #p['d0_path'] = os.path.join(code_dir, 'results/ic1/c_t[0].nii.gz')

  #p['mri_path'] = os.path.join(code_dir, 'data/51566_t1_aff2jakob.nc')
  p['mri_path'] = ""
  p['solver'] = 'multi_species'               # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
  p['model'] = 5                        # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
  p['verbosity'] = 3                    # various levels of output density
  p['syn_flag'] = 1                     # create synthetic data
  #p['user_cms'] = [(129,78,154,0.5),(129,78,150,0.5),(129,74,155,0.5),(129,79,152,0.3),(129,76,154,0.3),(129,77,152,0.2),(129,75,152,0.1),(129,73,153,0.1)]
  if case == 1:
    #p['user_cms'] = [(129,83,156,1.0),(132,82,158,0.6),(128,89,154,0.4)]
    #p['user_cms'] = [(129,83,156,1.0),(132,82,158,0.6)]
    p['user_cms'] = [(129,83,156,1.0)]
  elif case == 2:
    #p['user_cms'] = [(129,83,98,1.0),(126,81,97,0.6),(121,87,95,0.4)]
    #p['user_cms'] = [(129,83,98,1.0),(126,81,97,0.6)]
    p['user_cms'] = [(129,83,98,1.0)]

  elif case == 3:
    #p['user_cms'] = [(129,113,96,1.0),(136,105,96,0.6),(131,115,102,0.4)]
    #p['user_cms'] = [(129,113,96,1.0),(136,105,96,0.6)]
    p['user_cms'] = [(129,113,96,1.0)]

  elif case ==4:
    #p['user_cms'] = [(129,126,161,1.0),(122,122,156,0.6),(132,124,159,0.4)]
    #p['user_cms'] = [(129,126,161,1.0),(122,122,156,0.6)]
    p['user_cms'] = [(129,126,161,1.0)]

  ####### tumor params for synthetic data
  p['invasive_thres_data'] = 0.001
  #if case == 1:
  p['rho_data'] = 14.0
  p['k_data'] = 0.03
  #p['gamma_data'] = 7E4
  p['gamma_data'] = 0
  p['ox_hypoxia_data'] = 0.6
  p['death_rate_data'] = 5.0
  p['alpha_0_data'] = 0.4
  p['ox_consumption_data'] = 20.0
  p['ox_source_data'] = 10.0
  p['beta_0_data'] = 1.2
  p['sigma_b_data'] = 0.9
  p['ox_inv_data'] = 0.8
  p['prediction'] = 0
  '''
  if case == 2:

    p['rho_data'] = 14.0
    p['k_data'] = 0.07
    p['gamma_data'] = 9E4
    #p['gamma_data'] = 0
    p['ox_hypoxia_data'] = 0.5
    p['death_rate_data'] = 10.0
    p['alpha_0_data'] = 0.3
    p['ox_consumption_data'] = 20.0
    p['ox_source_data'] = 5.0
    p['beta_0_data'] = 0.4
    p['sigma_b_data'] = 0.9
    p['ox_inv_data'] = 0.8
    #p['invasive_thres_data'] = 0.01
    p['prediction'] = 0

  if case == 3:
    p['rho_data'] = 15.0
    p['k_data'] = 0.05
    #p['gamma_data'] = 8.5E4
    p['gamma_data'] = 0
    p['ox_hypoxia_data'] = 0.4
    p['death_rate_data'] = 22.0
    p['alpha_0_data'] = 0.3
    p['ox_consumption_data'] = 30.0
    p['ox_source_data'] = 5.0
    p['beta_0_data'] = 1.0
    p['sigma_b_data'] = 0.9
    p['ox_inv_data'] = 0.6
    #p['invasive_thres_data'] = 0.01
    p['prediction'] = 0

  if case == 4:
    p['rho_data'] = 13.0
    p['k_data'] = 0.03
    p['gamma_data'] = 1.0E5
    #p['gamma_data'] = 0
    p['ox_hypoxia_data'] = 0.3
    p['death_rate_data'] = 21.0
    p['alpha_0_data'] = 0.02
    p['ox_consumption_data'] = 25.0
    p['ox_source_data'] = 10.0
    p['beta_0_data'] = 0.9
    p['sigma_b_data'] = 0.9
    p['ox_inv_data'] = 0.7
    #p['invasive_thres_data'] = 0.01
    p['prediction'] = 0

  '''


  #p['sigma_factor'] = 3
  #p['sigma_spacing'] = 2
  p['write_output'] = 1
  p['k_gm_wm'] = 0.2                      # kappa ratio gm/wm (if zero, kappa=0 in gm)
  p['r_gm_wm'] = 0                      # rho ratio gm/wm (if zero, rho=0 in gm)
  p['ratio_i0_c0'] = 0.0

  p['nt_data'] = 100
  p['dt_data'] = 0.01
  p['time_history_off'] = 1             # 1: do not allocate time history (only works with forward solver or FD inversion)

  ############### === define run configuration if job submit is needed; else run from results folder directly
  r['code_path'] = code_dir;
  r['compute_sys'] = 'frontera'         # TACC systems are: maverick2, frontera, stampede2, longhorn; cbica for upenn system
  r['mpi_tasks'] = 1                    # mpi tasks (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['nodes'] = 1                        # number of nodes  (other job params like waittime are defaulted from params.py; overwrite here if needed)
  r['wtime_h'] = 2
  ###############=== write config to write_path and submit job

  par.submit(p, r, submit_job, use_gpu);




