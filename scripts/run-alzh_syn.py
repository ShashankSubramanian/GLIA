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
nlevels = ['sp2.0']#, 'sp0.1', 'sp0.5', 'sp1.0', 'sp1.5', 'sp2.0']

Tp = {'0'  : {'t1.0' : 1.0,  't1.2' : 1.2,  't1.5' : 1.5},
      '20' : {'t1.0' : 0.44, 't1.2' : 0.64, 't1.5' : 0.94},
      '40' : {'t1.0' : 0.28, 't1.2' : 0.48, 't1.5' : 0.78},
      '60' : {'t1.0' : 0.17, 't1.2' : 0.37, 't1.5' : 0.67},
      '80' : {'t1.0' : 0.10, 't1.2' : 0.3,  't1.5' : 0.6}
 }

scale_k = 1E-1
prefix = "0368Y01"

for case_dir in case_directs:
    for nlevel in nlevels:
	 
	tpoint = case_dir.split('d_nc-')[-1]
	submit_job = True
         
	p['output_dir'] 	= os.path.join(code_dir, 'results/inverse_rd_final/') + case_dir + '/' + str(nlevel.split('sp')[-1])  +  '/'
	p['d0_path'] 		= code_dir + '/syn_data1/tc/' + case_dir + '/' + 'data_t0_noise-' + str(nlevel.split('sp')[-1]) + '.nc'    # tumor data path
	p['d1_path'] 		= code_dir + '/syn_data1/tc/' + case_dir + '/' + 'data_t1_noise-' + str(nlevel.split('sp')[-1]) + '.nc'    # tumor data path
	p['a_wm_path']		= code_dir + '/syn_data1/data/' + prefix + '_seg_wm.nc'      # healthy patient path  
	p['a_gm_path'] 		= code_dir + '/syn_data1/data/' + prefix  + '_seg_gm.nc'  
	p['a_vt_path'] 		= code_dir + '/syn_data1/data/' + prefix + '_seg_csf.nc' 
	p['a_csf_path'] 	= "" #code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_csf.nc'
	p['a_kf_path']		= code_dir + '/brain_data/256/'
	    
	p['p_wm_path'] 		= ""#code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_wm.nc'		# patient brain data path 
	p['p_gm_path']              = "" #code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_gm.nc'
	p['p_csf_path']		= "" #code_dir + '/syn_data1/' + case_dir + '0368Y01_seg_csf.nc' 
	p['p_vt_path'] 		= "" 
	p['velocity_x1']	= "" #code_dir + '/syn_data1/reg/' + 'velocity-field-x1.nc' 
	p['velocity_x2']	= "" #code_dir + '/syn_data1/reg/' + 'velocity-field-x2.nc'
	p['velocity_x3']	= "" #code_dir + '/syn_data1/reg/' + 'velocity-field-x3.nc'
	
	p['solver'] 		= 'reaction_diffusion'			# modes : sparse_til , nonsparse_til , reaction_diffusion , mass_effect , multi_species , forward , test 
	
	p['model']		= 2
	p['verbosity']		= 1
	p['syn_flag'] 		= 0
	p['two_time_points']	= 1
        p['prediction']         = 1
	p['regularization']	= "L2"
	p['beta_p']		= 1E-4
	
	p['dt_inv'] 		= 0.01
	p['nt_inv'] 		= int(Tp[tpoint]['t1.0']/p['dt_inv'])
	p['init_rho'] 		= 6
	p['init_k'] 		= 1E-2/scale_k
	p['init_kf']		= p['init_k']
	p['init_gamma']		= 12E4
	p['kappa_lb']		= 1E-4/scale_k
        p['kappa_ub']		= 1/scale_k
	p['rho_lb']		= 4
	p['rho_ub']		= 15
        
        p['pred_times']		= [Tp[tpoint]['t1.0'], Tp[tpoint]['t1.2'], Tp[tpoint]['t1.5']]
	
        p['obs_threshold_0']	= 0.0
	p['obs_threshold_1']	= 0.0    
	p['obs_threshold_rel']	= 1
	p['time_history_off']	= 1
	p['sparsity_level']	= 5
	
	p['smoothing_factor']	= 1.5
        p['smoothing_factor_data'] = 0
        p['smoothing_factor_data_t0'] = 0
	
	p['ls_max_func_evals']  = 20
	p['threshold_data_driven'] = 1E-4
	
	r['compute_sys'] 	= 'rebels'
	r['code_path']		= code_dir

	par.submit(p, r, submit_job)
