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
case_directs = ['data/'] # 'd_nc-20', 'd_nc-40', 'd_nc-60', 'd_nc-80']


for case_dir in case_directs:
   # tpoint = case_dir.split('d_nc-')[-1]
   # Tp = {'0'  : {'t1.0' : 1.0,  't1.2' : 1.2,  't1.5' : 1.5},
      #    '20' : {'t1.0' : 0.44, 't1.2' : 0.64, 't1.5' : 0.94},
      #    '40' : {'t1.0' : 0.28, 't1.2' : 0.48, 't1.5' : 0.78},
      #    '60' : {'t1.0' : 0.17, 't1.2' : 0.37, 't1.5' : 0.67},
      #    '80' : {'t1.0' : 0.10, 't1.2' : 0.3,  't1.5' : 0.6}
      #   }

   # Ts = {'0' : 100, '20' : 44,   '40' : 28,   '60' : 17,   '80' : 10}
   # Ta = {'0' : -1,  '20' : 0.56, '40' : 0.72, '60' : 0.83, '80' : 0.90}

    submit_job = True
    

    p['output_dir'] 		= os.path.join(code_dir, 'results/inverse/') + case_dir + '/'
    p['d0_path'] 		= code_dir + '/syn_data1/' + '/tc/c0True.nc'     # tumor data path
    p['d1_path'] 		= code_dir + '/syn_data1/' + '/tc/dataBeforeObservation.nc'

    p['a_wm_path']		= code_dir + '/syn_data1/' + case_dir + '0368Y01_wm.nc'      # healthy patient path  
    p['a_gm_path'] 		= code_dir + '/syn_data1/' + case_dir + '0368Y01_gm.nc'  
    p['a_csf_path'] 		= code_dir + '/syn_data1/' + case_dir + '0368Y01_csf.nc'  
    p['a_vt_path'] 		= ""
    p['a_kf_path']		= code_dir + '/brain_data/256/'
    
    p['p_wm_path'] 		= code_dir + '/syn_data1/' + case_dir + '0368Y01_wm.nc'		# patient brain data path 
    p['p_gm_path']              = code_dir + '/syn_data1/' + case_dir + '0368Y01_gm.nc'
    p['p_csf_path']		= code_dir + '/syn_data1/' + case_dir + '0368Y01_csf.nc' 
    p['p_vt_path'] 		= "" 
    p['solver'] 		= 'sparse_til'			# modes : sparse_til , nonsparse_til , reaction_diffusion , mass_effect , multi_species , forward , test 
    p['verbosity']		= 1
    p['syn_flag'] 		= 1 
    p['user_cms'] 		= [(105,29,108,1)]
   # p['pred_times']		= [Tp[tpoint]['t1.0'], Tp[tpoint]['t1.2'], Tp[tpoint]['t1.5']]
    p['velocity_x1']		= ''
    p['velocity_x2']		= ''
    p['velocity_x3']		= ''
    p['obs_threshold_0']	= 0.1
    p['obs_threshold_1']	= 0.1    
    p['two_time_points_']	= 1
    p['init_rho'] 		= 6
    p['init_k'] 		= 1E-1
    p['init_kf']		= 1.8E-1
    p['init_gamma']		= 12E4
    p['dt_inv'] 		= 0.01
    p['nt_inv'] 		= 100
    p['time_history_off']	= 1
    p['sparsity_level']		= 10
    p['beta_p']			= 1E-4
    p['model']			= 2
    
    p['k_data']			= 0.18
    p['kf_data']		= 0.18
    p['rho_data']		= 8


    r['compute_sys'] 		= 'rebels'
    r['code_path']		= code_dir
    
    par.submit(p, r, submit_job)
