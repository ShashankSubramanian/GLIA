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

data_dir = '/scratch/ghafouri/data/ADNI-data/AD_processed/'

################ ==== define parameters
#case_directs = ['CASE_035_S_4114']
'''
TF = {
       '022_S_6013' :  {'t1' : 0.40,  't2' : 0.1, 't02' : 0.2},
       '023_S_1190' :  {'t1' : 0.47,  't2' : 0.1, 't02' : 0.2},
       '127_S_4301' :  {'t1' : 0.47,  't2' : 0.63, 't02' : 1.63},
       '127_S_2234' :  {'t1' : 0.66,  't2' : 0.63, 't02' : 1.63},
       '012_S_6073' :  {'t1' : 0.35,  't2' : 0.63, 't02' : 1.63},
       '033_S_4179' :  {'t1' : 0.37,  't2' : 0.63, 't02' : 1.63},
       '941_S_4036' :  {'t1' : 0.46,  't2' : 0.63, 't02' : 1.63},
       '032_S_5289' :  {'t1' : 0.74,  't2' : 0.63, 't02' : 1.63},
       '035_S_4114' :  {'t1' : 0.42,  't2' : 0.63, 't02' : 1.63}}
'''
CASES = ['003_S_6264', '022_S_6013', '022_S_6796']
#CASES=['022_S_6796']
CASES=[CASES[2]]
#CASES = [C[5]]
#, 'CASE_023_S_1190', 'CASE_033_S_4179', 'CASE_032_S_5289','CASE_035_S_4114']
#CASES = ['CASE_022_S_6013']
#,'CASE_127_S_2234']

adv_dict = {'003_S_6264' : True, '022_S_6013' : True, '022_S_6796' : False, '114_S_6347' : False, '012_S_6073' : False, '033_S_4179' : False, '941_S_4036' : True,'032_S_5289' : False, '035_S_4114' : True }

prefix = '/coreg_tau_unires_t'

T_dict = {'003_S_6264': 0.30, '022_S_6013':0.38, '022_S_6796':0.36, '114_S_6347': 0.24}
for case in CASES:

    submit_job = True
    case_dir = data_dir + case + '/data_processed/'
    
    p['output_dir'] 		= '/scratch/ghafouri/results/real/inverse_rd_' + case + '/th0/'
    p['d0_path'] 		= case_dir + prefix + str(0) + '_normalized_masked.nc'     # tumor data path
    p['d1_path'] 		= case_dir + prefix + str(1) + '_normalized_masked.nc'

    p['a_wm_path']		= case_dir + 'wm.nc'      # healthy patient path  
    p['a_gm_path'] 		= case_dir + 'gm.nc'
    p['a_csf_path'] 		= '' 
    p['a_vt_path'] 		= case_dir + 'csf.nc'
    p['a_kf_path']		= case_dir 
    
    
    p['velocity_x1']		= "" #code_dir + '/' + case_dir + '/' + 'reg-1-0/velocity-field-x1.nc'
    p['velocity_x2']		= "" #code_dir + '/' + case_dir + '/' + 'reg-1-0/velocity-field-x2.nc'
    p['velocity_x3']		= "" #code_dir + '/' + case_dir + '/' + 'reg-1-0/velocity-field-x3.nc'
    if adv_dict[case]:
      p['velocity_x1'] = '/scratch/ghafouri/data/ADNI-data/AD_processed/'+case + '/regT-0/regT-0velocity-field-x1.nc'
      p['velocity_x2'] = '/scratch/ghafouri/data/ADNI-data/AD_processed/'+case + '/regT-0/regT-0velocity-field-x2.nc'
      p['velocity_x3'] = '/scratch/ghafouri/data/ADNI-data/AD_processed/'+case + '/regT-0/regT-0velocity-field-x3.nc'

    #p['p_wm_path'] 		= data_dir + prefix_p + '_seg_wm.nc'		# patient brain data path 
    #p['p_gm_path']              = data_dir + prefix_p + '_seg_gm.nc'
    #p['p_csf_path']		= ''
    #p['p_vt_path'] 		= data_dir + prefix_p + '_seg_csf.nc'
    p['solver'] 		= 'reaction_diffusion'			# modes : sparse_til , nonsparse_til , reaction_diffusion , mass_effect , multi_species , forward , test 
    
    p['verbosity']		= 3
    p['model']			= 2
    p['regularization']		= "L2"
    p['beta_p']			= 1E-4
    p['dt_inv'] 		= 0.01
    p['nt_inv'] 		= int(T_dict[case]/p['dt_inv'])
    
    p['two_time_points']	= 1    
    p['init_rho'] 		= 0.1
    p['init_k'] 		= 1E-2
    p['init_kf']		= 1E-2
    p['init_gamma']		= 12E4
    p['obs_threshold_rel']	= 1
    p['obs_threshold_0']	= 0.0
    p['obs_threshold_1']	= 0.0
    
    p['syn_flag'] 		= 0 
    p['prediction'] 		= 0 
    
    p['time_history_off']	= 1
    p['sparsity_level']		= 5
    p['k_gm_wm']		= 1.0 
    p['kappa_lb'] 		= 1E-5
    p['kappa_ub'] 		= 1
    p['kappaf_lb']		= 1E-5
    p['kappaf_ub']		= 5
    p['number_diffusions']	= 1
    
    p['rho_lb']			= 1E-4
    p['rho_ub']			= 30
    p['ls_max_func_evals']      = 50
    p['threshold_data_driven']  = 1E-4
    
    p['smoothing_factor']	= 1.0    
    p['smoothing_factor_data'] = 0
    p['smoothing_factor_data_t0'] = 0
    
    r['compute_sys'] 		= 'rebels'
    r['code_path']		= code_dir
    
    par.submit(p, r, submit_job)
