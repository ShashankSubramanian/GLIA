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
case_directs = ['data'] # 'd_nc-20', 'd_nc-40', 'd_nc-60', 'd_nc-80']


for case_dir in case_directs:


    submit_job = True
    

    p['output_dir'] 		= os.path.join(code_dir, 'results/inverse_real/') + case_dir + '/'
    p['d0_path'] 		= code_dir + '/real_data/' + case_dir + '/time_point_0_tau.nc'     # tumor data path
    p['d1_path'] 		= code_dir + '/real_data/' + case_dir + '/time_point_1_tau.nc'

    p['a_wm_path']		= code_dir + '/real_data/' + case_dir + 'time_point_0_wm.nc'      # healthy patient path  
    p['a_gm_path'] 		= code_dir + '/real_data/' + case_dir + 'time_point_0_gm.nc'
    p['a_csf_path'] 		= code_dir + '/real_data/' + case_dir + 
    p['a_vt_path'] 		= code_dir + '/real_data/' + 'reg-1-0/'
    p['a_kf_path']		= ""
    
    p['p_wm_path'] 		= code_dir + '/real_data/' + case_dir + 'time_point_0_wm.nc'		# patient brain data path 
    p['p_gm_path']              = code_dir + '/real_data/' + case_dir + 'time_point_0_gm.nc'
    p['p_csf_path']		= code_dir + '/real_data/' + case_dir + 'time_point_0_csf.nc'
    p['p_vt_path'] 		= code_dir + '/real_data/' + 'reg-2-1/'
    p['solver'] 		= 'sparse_til'			# modes : sparse_til , nonsparse_til , reaction_diffusion , mass_effect , multi_species , forward , test 
    p['verbosity']		= 1
    p['syn_flag'] 		= 1 
    p['user_cms'] 		= [(105,29,108,1),(104,43,108,1),(110,133,108,1),(149,133,108,1)]
   # p['pred_times']		= [Tp[tpoint]['t1.0'], Tp[tpoint]['t1.2'], Tp[tpoint]['t1.5']]
    p['velocity_x1']		= ''
    p['velocity_x2']		= ''
    p['velocity_x3']		= ''
    p['two_time_points']	= 1    
    p['obs_threshold_0']	= 0.6
    p['obs_threshold_1']	= 0.6

    p['init_rho'] 		= 4
    p['init_k'] 		= 1E-2
    p['init_gamma']		= 12E4
    p['dt_inv'] 		= 0.01
    p['nt_inv'] 		= 100
    p['time_history_off']	= 0
    p['sparsity_level']		= 10
    p['beta_p']			= 1E-4
    p['model']			= 2
    
    p['k_data']			= 0.1
    p['kf_data']		= 0.1
    p['rho_data']		= 8


    r['compute_sys'] 		= 'rebels'
    r['code_path']		= code_dir
    
    par.submit(p, r, submit_job)
