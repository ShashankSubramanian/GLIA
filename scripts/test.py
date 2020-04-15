import os, sys
import params as par
import subprocess

############### TEST-SUITE: DO NOT CHANGE ###############

###############
r = {}
p = {}
submit_job = False;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['n'] = 64                         # grid resolution
p['solver'] = 'test-inverse'        # modes: test-forward
p['output_dir'] = os.path.join(code_dir, 'results/' + p['solver'] + '/');   # results path
p['a_gm_path'] = code_dir + "/testdata/gm.nc"     # atlas paths
p['a_wm_path'] = code_dir + "/testdata/wm.nc"
p['a_csf_path'] = code_dir + "/testdata/csf.nc"
p['a_vt_path'] = code_dir + "/testdata/vt.nc"
p['mri_path'] = code_dir + "/testdata/mri.nc"
p['model'] = 1                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] = 1                  # various levels of output density
p['syn_flag'] = 1                   # create synthetic data
p['user_cms'] = [(137,169,96,1)]    # arbitrary number of TILs (x,y,z,scale) with activation scale
p['rho_data'] = 8                  # tumor parameters for synthetic data
p['k_data'] = 0.01
p['gamma_data'] = 12E4
p['nt_data'] = 25
p['dt_data'] = 0.04
p['init_rho'] 			= 15                   		# initial guess rho (reaction in wm)
p['init_k'] 			= 0                    		# initial guess kappa (diffusivity in wm)
p['init_gamma'] 		= 6E4              		    # initial guess (forcing factor for mass effect)
p['nt_inv'] 			= 25                    	# number time steps for inversion
p['dt_inv'] 			= 0.04                  	# time step size for inversion
p['time_history_off'] 	= 0          				# 1: do not allocate time history (only works with forward solver or FD inversion)
p['sparsity_level'] 	= 5             			# target sparsity of recovered IC in sparse_til solver
p['beta_p'] 			= 0E-4                  	# regularization parameter
p['opttol_grad'] 		= 1E-5             			# relative gradient tolerance
p['newton_maxit'] 		= 2              			# number of iterations for optimizer
p['kappa_lb'] 			= 0                   		# lower bound kappa
p['kappa_ub'] 			= 0.1                 		# upper bound kappa
p['rho_lb'] 			= 0                     	# lower bound rho
p['rho_ub'] 			= 15                    	# upper bound rho
p['gamma_ub'] 			= 15                  		# upper bound gamma
p['lbfgs_vectors'] 		= 50				        # number of vectors for lbfgs update
p['ls_max_func_evals']  = 10                        # number of max line-search attempts

############### === define run configuration
r['code_path'] = code_dir;
r['compute_sys'] = 'rebels'         # TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
par.submit(p, r, submit_job);