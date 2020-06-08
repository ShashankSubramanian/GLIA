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

brats_str = "Brats18_CBICA_"
names     = ["ABO", "AMH", "ALU", "AAP"]
#names     = ["ABO", "AMH", "ALU"]
#names     = ["AAP"]
##patient_name = "ABO"
for patient_name in names:
  pat         = brats_str + patient_name + "_1"
  real_path   = code_dir + '/brain_data/real_data/' + pat	+ '/data/'
  reg         = code_dir + '/results/' + pat + '/reg/'
  for atlas in os.listdir(reg):
    if atlas[0] == "5": ### adni atlases
      reg_path  = reg + atlas + '/'
############### === define parameters
      p['output_dir'] 		    = os.path.join(code_dir, 'results/' + pat + '/tu/' + atlas + '-noreg/');  	# results path
      p['d1_path'] 			      = real_path + pat + '_tu_aff2jakob.nc'
      p['d0_path']            = real_path + pat + '_c0Recon_aff2jakob.nc'              # path to initial condition for tumor
#      p['d0_path']            = reg_path + '/c0Recon_transported.nc'                # path to initial condition for tumor
      p['a_gm_path']          = reg_path + atlas + '_gm.nc' 
      p['a_wm_path']          = reg_path + atlas + '_wm.nc' 
      p['a_csf_path']         = reg_path + atlas + '_csf.nc' 
      p['a_vt_path']          = reg_path + atlas + '_vt.nc' 
      p['mri_path'] 			    = code_dir + "/../data/adni-nc/" + atlas + "_t1_aff2jakob.nc"
      p['p_wm_path'] 			    = real_path + pat + '_wm_aff2jakob.nc'           # patient brain data path
      p['p_gm_path'] 			    = real_path + pat + '_gm_aff2jakob.nc'           # patient brain data path
      p['p_csf_path']         = real_path + pat + '_csf_aff2jakob.nc'           # patient brain data path
      p['p_vt_path'] 			    = real_path + pat + '_vt_aff2jakob.nc'           # patient brain data path
      p['obs_mask_path']  	  = real_path + pat + '_obs_aff2jakob.nc'           # patient brain data path
      p['solver'] 			      = 'mass_effect'
      p['model'] 				      = 4                       	# mass effect model
      p['regularization']     = "L2"                      # L2, L1
      p['verbosity'] 			    = 1                  		# various levels of output density
      p['syn_flag'] 			    = 0                  	    # create synthetic data
#p['gaussian_cm_path']   = code_dir + '/results/t16-case6/phi-mesh-forward.txt'                        # path to file with Gaussian support centers, default: none, (generate Gaussians based on target data)
#p['pvec_path']          = code_dir + '/results/t16-case6/p-rec-forward.txt'                        # path to initial guess p vector (if none, use zero)
      p['init_rho'] 			    = 6                  		# initial guess rho (reaction in wm)
      p['init_k'] 			      = 0.005                    		# initial guess kappa (diffusivity in wm)
      p['init_gamma'] 		    = 1E4              		# initial guess (forcing factor for mass effect)
      p['nt_inv'] 			      = 25                    	# number time steps for inversion
      p['dt_inv'] 			      = 0.04                  	# time step size for inversion
      p['k_gm_wm']            = 0.2                  # kappa ratio gm/wm (if zero, kappa=0 in gm)
      p['r_gm_wm']            = 1                  # rho ratio gm/wm (if zero, rho=0 in gm)
      p['time_history_off'] 	= 0          				# 1: do not allocate time history (only works with forward solver or FD inversion)
      p['beta_p'] 			      = 0E-4                  	# regularization parameter
      p['opttol_grad'] 		    = 1E-5             			# relative gradient tolerance
      p['newton_maxit'] 		  = 50              			# number of iterations for optimizer
      p['kappa_lb'] 			    = 0.005                   		# lower bound kappa
      p['kappa_ub'] 			    = 0.05                 		# upper bound kappa
      p['rho_lb'] 			      = 0                     	# lower bound rho
      p['rho_ub'] 			      = 12                    	# upper bound rho
      p['gamma_lb']           = 0                         # lower bound gamma
      p['gamma_ub'] 			    = 12E4                		# upper bound gamma
      p['lbfgs_vectors'] 		  = 5        			        # number of vectors for lbfgs update
      p['lbfgs_scale_type']   = "scalar"                  # initial hessian approximation
      p['lbfgs_scale_hist']   = 5                         # used vecs for initial hessian approx
      p['ls_max_func_evals']  = 20                        # number of max line-search attempts
      p['prediction']         = 0                         # enable prediction

############### === define run configuration
      r['code_path'] 			    = code_dir;
      r['compute_sys'] 		    = 'longhorn'         			# TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
      par.submit(p, r, submit_job);
