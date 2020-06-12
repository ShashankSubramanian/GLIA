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
case_directs = ['CASE_035_S_4114']
TF = {
      '035_S_4114' : {'t1' : 0.42, 't2' : 0.63, 't02' : 1.63}, 
     }
prefix = 'time_point_0'
prefix_p = 'time_point_1'
scale_k = 1E-1


for case_dir in case_directs:


    submit_job = True
    case = str(case_dir.split('CASE_')[-1])
    data_dir = code_dir + '/' + case_dir + '/data/'

    p['output_dir'] 		= os.path.join(code_dir, 'results/inverse_RD_final/') + case_dir + '/th06/'
    p['d0_path'] 		= data_dir +  'time_point_0_tau_th06.nc'     # tumor data path
    p['d1_path'] 		= data_dir +  'time_point_1_tau_th06.nc'

    p['a_wm_path']		= data_dir + prefix + '_seg_wm.nc'      # healthy patient path  
    p['a_gm_path'] 		= data_dir + prefix + '_seg_gm.nc'
    p['a_csf_path'] 		= '' 
    p['a_vt_path'] 		= data_dir + prefix + '_seg_csf.nc'
    p['a_kf_path']		= ''
    
    p['velocity_x1']		= "" #code_dir + '/' + case_dir + '/' + 'reg-1-0/velocity-field-x1.nc'
    p['velocity_x2']		= "" #code_dir + '/' + case_dir + '/' + 'reg-1-0/velocity-field-x2.nc'
    p['velocity_x3']		= "" #code_dir + '/' + case_dir + '/' + 'reg-1-0/velocity-field-x3.nc'
    
    p['p_wm_path'] 		= data_dir + prefix_p + '_seg_wm.nc'		# patient brain data path 
    p['p_gm_path']              = data_dir + prefix_p + '_seg_gm.nc'
    p['p_csf_path']		= ''
    p['p_vt_path'] 		= data_dir + prefix_p + '_seg_csf.nc'
    p['solver'] 		= 'reaction_diffusion'			# modes : sparse_til , nonsparse_til , reaction_diffusion , mass_effect , multi_species , forward , test 
    
    p['verbosity']		= 1
    p['model']			= 2
    p['regularization']		= "L2"
    p['beta_p']			= 1E-4
    p['dt_inv'] 		= 0.01
    p['nt_inv'] 		= int(TF[case]['t1']/p['dt_inv'])
    
    p['two_time_points']	= 1    
    p['init_rho'] 		= 4
    p['init_k'] 		= 1E-2/scale_k
    p['init_gamma']		= 12E4
    p['obs_threshold_rel']	= 1
    p['obs_threshold_0']	= 0.6
    p['obs_threshold_1']	= 0.6
    
    p['syn_flag'] 		= 0 
    p['prediction'] 		= 0 
    
    p['time_history_off']	= 1
    p['sparsity_level']		= 5
    
    p['kappa_lb'] 		= 1E-4/scale_k
    p['kappa_ub'] 		= 1/scale_k
    p['rho_lb']			= 1E-4
    p['rho_ub']			= 15
    p['ls_max_func_evals']      = 20
    p['threshold_data_driven']  = 1E-4 
    
    p['smoothing_factor']	= 1.5    
    p['smoothing_factor_data'] = 0
    p['smoothing_factor_data_t0'] = 0
    
    r['compute_sys'] 		= 'rebels'
    r['code_path']		= code_dir
    
    par.submit(p, r, submit_job)
