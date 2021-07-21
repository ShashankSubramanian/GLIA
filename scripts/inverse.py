import os, sys
import params as par
import subprocess

###############
r = {}
p = {}
submit_job = False;
use_gpu = True;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['n'] = 64                           # grid resolution in each dimension
p['output_dir'] = os.path.join(code_dir, 'results/inverse_til/');                          # results path
p['atlas_labels'] = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"                               # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
p['a_seg_path'] = os.path.join(code_dir, 'testdata/patient.nc')
p['d1_path'] = os.path.join(code_dir, 'testdata/patient_tumor.nc')
p['smoothing_factor_data'] = 0          # 0: no smoothing, otherwise kernel width
p['solver'] = 'sparse_til'              # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effect, multi_species, forward, test
p['model'] 	= 1                       	# 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['multilevel'] = 0                     # rescale p activations according to Gaussian width on each level
p['syn_flag'] = 0                       # create synthetic data if 1
p['verbosity'] = 1                  		# various levels of output density
p['init_rho'] = 8                   		# initial guess rho (reaction in wm)
p['init_k'] 	= 0                    		# initial guess kappa (diffusivity in wm)
p['nt_inv']	= 40                    	  # number time steps for inversion
p['dt_inv'] = 0.025                  	  # time step size for inversion
p['time_history_off']	= 0          			# 1: do not allocate time history (only works with forward solver or FD inversion)
p['sparsity_level']	= 1             		# target sparsity of recovered IC in sparse_til solver
p['beta_p'] = 0E-4                  	  # regularization parameter
p['opttol_grad'] = 1E-4             	  # relative gradient tolerance
p['newton_maxit']	= 50              		# number of iterations for optimizer
p['kappa_lb']	= 1E-4                   	# lower bound kappa
p['kappa_ub'] = 0.1                 		# upper bound kappa
p['rho_lb'] = 2                     	  # lower bound rho
p['rho_ub'] = 15                    	  # upper bound rho
p['lbfgs_vectors'] = 50				          # number of vectors for lbfgs update

############### === define run configuration
r['code_path'] 			= code_dir;
r['compute_sys'] 		= 'cbica'    		# TACC systems are: maverick2, frontera, stampede2, longhorn; UPENN system : cbica

###############=== write config to write_path and submit job
par.submit(p, r, submit_job, use_gpu);

### -tao_blmvm_mat_lmvm_num_vecs 50 -tao_blmvm_mat_lmvm_scale_type diagonal -tumor_tao_ls_max_funcs 10
