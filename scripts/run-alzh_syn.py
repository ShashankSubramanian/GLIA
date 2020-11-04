import os, sys
import params as par
import subprocess

################
r = {}
p = {}
submit_job = False

################
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

################ ==== define parameters
case_directs = [ 'd_nc-80' ] #, 'd_nc-20', 'd_nc-40', 'd_nc-60', 'd_nc-80']
nlevels = ['']#, 'sp0.1', 'sp0.5', 'sp1.0', 'sp1.5', 'sp2.0']

Tp = {'0'  : {'t1.0' : 1.0,  't1.2' : 1.2,  't1.5' : 1.5},
      '20' : {'t1.0' : 0.44, 't1.2' : 0.64, 't1.5' : 0.94},
      '40' : {'t1.0' : 0.28, 't1.2' : 0.48, 't1.5' : 0.78},
      '60' : {'t1.0' : 0.17, 't1.2' : 0.37, 't1.5' : 0.67},
      '80' : {'t1.0' : 0.10, 't1.2' : 0.3,  't1.5' : 0.6}
 }

scale_k = 1E-1
prefix = "0368Y01"
data_dir = '/scratch/ghafouri/data/128/'

#for case_dir in case_directs:
 #   for nlevel in nlevels:
case = 'forward-k0.1-kf2.5-r12-T1.0'
case_b = case.split('forward-')[-1]

out_dir = '/scratch/ghafouri/results/syn/' + 'inverse_noise_42_obs0_' + case_b + '/'
case_dir = '/scratch/ghafouri/results/syn/' + case
submit_job = True
p['n'] = 128
p['output_dir'] 	= out_dir #os.path.join(code_dir, 'results/inverse_rd_syn_m2_s2/') + case + '/'
p['d1_path']		= '/scratch/ghafouri/results/syn/128_data_noise/' + 'c1_noise_42.nc'
#p['d1_path'] 		= case_dir + '/' + 'c1_true_syn_before_observation' + '.nc'    # tumor data path
p['d0_path'] 		= case_dir + '/' + 'c0_true_syn' + '.nc'    # tumor data path
p['a_wm_path']		= data_dir + 'white_matter.nc'      # healthy patient path  
p['a_gm_path']		= data_dir + 'gray_matter.nc'      # healthy patient path  
p['a_vt_path']		= data_dir + 'csf.nc'      # healthy patient path  
p['a_csf_path'] 	= "" #code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_csf.nc'
p['a_kf_path']		= '/scratch/ghafouri/data/ADNI_PROCESSED_256/128/' #'/scratch/ghafouri/data/ADNI_PROCESSED_256/128/'
    
p['p_wm_path'] 		= ""#code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_wm.nc'		# patient brain data path 
p['p_gm_path']              = "" #code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_gm.nc'
p['p_csf_path']		= "" #code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_csf.nc' 
p['p_vt_path'] 		= "" 
p['velocity_x1']	= "" #code_dir + '/syn_data1/reg/' + 'velocity-field-x1.nc' 
p['velocity_x2']	= "" #code_dir + '/syn_data1/reg/' + 'velocity-field-x2.nc'
p['velocity_x3']	= "" #code_dir + '/syn_data1/reg/' + 'velocity-field-x3.nc'

p['solver'] 		= 'reaction_diffusion'			# modes : sparse_til , nonsparse_til , reaction_diffusion , mass_effect , multi_species , forward , test 

p['model']		= 2
p['verbosity']		= 3
p['two_time_points']	= 1
p['prediction']         = 0
p['regularization']	= "L2"
p['beta_p']		= 1E-4

p['syn_flag'] 		= 0
p['k_data']		= 0.05
p['kf_data']		= 2.0
p['rho_data']		= 12
p['dt_data']		= 0.01
p['nt_data']		= 100
p['user_cms']		= [(116,134,102,1),(116,134,154,1)]




p['dt_inv'] 		= 0.01
p['nt_inv'] 		= 100
p['init_rho'] 		= 6
p['init_k'] 		= 0.01
p['init_kf']		= 0.01
p['init_gamma']		= 12E4
p['kappa_lb']		= 1E-4/scale_k
p['kappa_ub']		= 1/scale_k
p['kappaf_lb']		= 1e-3
p['kappaf_ub']		= 5
p['number_diffusions']  = 2
p['rho_lb']		= 4
p['rho_ub']		= 15
p['k_gm_wm']		= 1.0
#p['pred_times']		= [1.0, 1.2, 1.5]

p['obs_threshold_0']	= 0.0
p['obs_threshold_1']	= 0.0    
p['obs_threshold_rel']	= 1
p['time_history_off']	= 1
p['sparsity_level']	= 5

p['smoothing_factor']	= 1.0
p['smoothing_factor_data'] = 0
p['smoothing_factor_data_t0'] = 0

p['ls_max_func_evals']  = 50
p['threshold_data_driven'] = 1E-5

r['compute_sys'] 	= 'rebels'
r['code_path']		= code_dir

par.submit(p, r, submit_job)
