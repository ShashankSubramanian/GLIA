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

############### === define parameters
p['output_dir'] 		= os.path.join(code_dir, 'results/inverse-masseffect-c0/');   		# results path
p['d1_path'] 			= code_dir + '/results/t16-case6/c_final.nc'			# tumor data path
p['d0_path']            = code_dir + '/results/reg-t16-case6/atlas-2/c0Recon_transported.nc'                   # path to initial condition for tumor
p['a_gm_path']   		= code_dir + "/brain_data/t16/256/t16_gm.nc"            # atlas paths
p['a_wm_path'] 			= code_dir + "/brain_data/t16/256/t16_wm.nc"
p['a_csf_path'] 		= code_dir + "/brain_data/t16/256/t16_csf.nc"
p['a_vt_path'] 			= code_dir + "/brain_data/t16/256/t16_vt.nc"
p['mri_path'] 			= code_dir + "/brain_data/t16/t1.nc"
p['p_wm_path'] 			= code_dir + '/results/t16-case6/wm_final.nc'            # patient brain data path
p['p_gm_path'] 			= code_dir + '/results/t16-case6/gm_final.nc'
p['p_csf_path'] 		= code_dir + '/results/t16-case6/csf_final.nc'
p['p_vt_path'] 			= code_dir + '/results/t16-case6/vt_final.nc'
p['solver'] 			= 'mass_effect'              # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effect, multi_species, forward, test
p['model'] 				= 4                       	# 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['regularization']     = "L2"                      # L2, L1
p['verbosity'] 			= 1                  		# various levels of output density
p['syn_flag'] 			= 0                  	    # create synthetic data
#p['gaussian_cm_path']   = code_dir + '/results/t16-case6/phi-mesh-forward.txt'                        # path to file with Gaussian support centers, default: none, (generate Gaussians based on target data)
#p['pvec_path']          = code_dir + '/results/t16-case6/p-rec-forward.txt'                        # path to initial guess p vector (if none, use zero)
p['init_rho'] 			= 6                  		# initial guess rho (reaction in wm)
p['init_k'] 			= 0.005                    		# initial guess kappa (diffusivity in wm)
p['init_gamma'] 		= 1E4              		# initial guess (forcing factor for mass effect)
p['nt_inv'] 			= 25                    	# number time steps for inversion
p['dt_inv'] 			= 0.04                  	# time step size for inversion
p['time_history_off'] 	= 0          				# 1: do not allocate time history (only works with forward solver or FD inversion)
p['beta_p'] 			= 0E-4                  	# regularization parameter
p['opttol_grad'] 		= 1E-5             			# relative gradient tolerance
p['newton_maxit'] 		= 50              			# number of iterations for optimizer
p['kappa_lb'] 			= 0.005                   		# lower bound kappa
p['kappa_ub'] 			= 0.05                 		# upper bound kappa
p['rho_lb'] 			= 0                     	# lower bound rho
p['rho_ub'] 			= 15                    	# upper bound rho
p['gamma_lb']           = 0                         # lower bound gamma
p['gamma_ub'] 			= 15E4                		# upper bound gamma
p['lbfgs_vectors'] 		= 5        			        # number of vectors for lbfgs update
p['lbfgs_scale_type']   = "scalar"                  # initial hessian approximation
p['lbfgs_scale_hist']   = 5                         # used vecs for initial hessian approx
p['ls_max_func_evals']  = 20                        # number of max line-search attempts
p['prediction']         = 0                         # enable prediction

############### === define run configuration
r['code_path'] 			= code_dir;
r['compute_sys'] 		= 'longhorn'         			# TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
par.submit(p, r, submit_job);
