import os, sys
import params as par
import subprocess

###############
r = {}
p = {}
submit_job = False;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['output_dir'] 		= os.path.join(code_dir, 'results/check/');   		# results path
p['d1_path'] 			  = code_dir + '/results/syn/atlas-2-tu-case3/c_final.nc'			# tumor data path
p['d0_path'] 			  = code_dir + '/results/syn/atlas-2-tu-case3/c0_true_syn.nc'			# tumor data path
p['atlas_labels']   = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"              # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path']     = code_dir + "/brain_data/atlas/atlas-2.nc"                # paths to atlas material properties
p['mri_path'] 			= code_dir + "/brain_data/atlas/atlas-2-t1.nc"
p['p_wm_path'] 			= code_dir + '/results/syn/atlas-2-tu-case3/wm_final.nc'            # patient brain data path
p['p_gm_path'] 			= code_dir + '/results/syn/atlas-2-tu-case3/gm_final.nc'
p['p_csf_path'] 		= code_dir + '/results/syn/atlas-2-tu-case3/csf_final.nc'
p['p_vt_path'] 			= code_dir + '/results/syn/atlas-2-tu-case3/vt_final.nc'
p['solver'] 			  = 'mass_effect'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effect, multi_species, forward, test
p['model'] 				  = 4                       	# 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['regularization'] = "L2"                      # L2, L1
p['verbosity'] 			= 1                  		# various levels of output density
p['syn_flag'] 			= 0                  	    # create synthetic data
#p['gaussian_cm_path']   = code_dir + '/results/atlas-2-tu-case3/phi-mesh-forward.txt'                       
#p['pvec_path']          = code_dir + '/results/atlas-2-tu-case3/p-rec-forward.txt'                        # path to initial guess p vector (if none, use zero)
p['k_gm_wm']        = 0.25                  # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm']        = 1                  # rho ratio gm/wm (if zero, rho=0 in gm)
p['init_rho'] 			= 6                  		# initial guess rho (reaction in wm)
p['init_k'] 			  = 0.005                    		# initial guess kappa (diffusivity in wm)
p['init_gamma'] 		= 1E4              		# initial guess (forcing factor for mass effect)
p['nt_inv'] 			  = 25                    	# number time steps for inversion
p['dt_inv'] 			  = 0.04                  	# time step size for inversion
p['time_history_off'] 	= 0          				# 1: do not allocate time history (only works with forward solver or FD inversion)
p['beta_p'] 			  = 0E-4                  	# regularization parameter
p['opttol_grad'] 		= 1E-5             			# relative gradient tolerance
p['newton_maxit']   = 50              			# number of iterations for optimizer
p['kappa_lb'] 			= 0.005                   		# lower bound kappa
p['kappa_ub'] 			= 0.05                 		# upper bound kappa
p['rho_lb'] 			  = 0                     	# lower bound rho
p['rho_ub'] 			  = 12                    	# upper bound rho
p['gamma_lb']       = 0                         # lower bound gamma
p['gamma_ub'] 			= 12E4                		# upper bound gamma
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