import os, sys
import params as par
import subprocess
import random

###############
r = {}
p = {}
submit_job = False;
use_gpu = True;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
### randomize the initial guess between parameter values that represent a patient population
init_rho                = random.uniform(5,10)
init_k                  = random.uniform(0.005,0.05)
init_gamma              = random.uniform(1E4,1E5)

p['n'] = 64                                                                                       # grid resolution in each dimension
p['output_dir']     = os.path.join(code_dir, 'results/inverse_masseffect/');                      # results path
p['d0_path']        = os.path.join(code_dir, 'testdata/patient_tumor_IC.nc')
p['d1_path']        = os.path.join(code_dir, 'testdata/patient_tumor.nc')
p['atlas_labels']   = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"                                   
p['a_seg_path']     = os.path.join(code_dir, 'testdata/atlas.nc')
p['mri_path']       = os.path.join(code_dir, 'testdata/atlas_t1.nc')
p['patient_labels'] = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"
p['p_seg_path']     = os.path.join(code_dir, 'testdata/patient.nc')
p['p_vt_path']      = os.path.join(code_dir, 'testdata/patient_vt.nc')
p['smoothing_factor_data'] = 0                  # 0: no smoothing, otherwise kernel width
p['smoothing_factor_data_t0'] = 0               # 0: no smoothing, otherwise kernel width
p['solver']         = 'mass_effect'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effect, multi_species, forward, test
p['model']          = 4                         # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['regularization'] = "L2"                      # L2, L1
p['verbosity']      = 1                         # various levels of output density
p['syn_flag']       = 0                         # create synthetic data
p['k_gm_wm']        = 0                         # kappa ratio gm/wm (if zero, kappa=0 in gm)
p['r_gm_wm']        = 0                         # rho ratio gm/wm (if zero, rho=0 in gm)
p['init_rho']       = init_rho                  # initial guess rho (reaction in wm)
p['init_k']         = init_k                    # initial guess kappa (diffusivity in wm)
p['init_gamma']     = init_gamma                # initial guess (forcing factor for mass effect)
p['nt_inv']         = 25                        # number time steps for inversion
p['dt_inv']         = 0.04                      # time step size for inversion
p['time_history_off']   = 0                     # 1: do not allocate time history (only works with forward solver or FD inversion)
p['beta_p']         = 0E-4                      # regularization parameter
p['opttol_grad']    = 1E-5                      # relative gradient tolerance
p['newton_maxit']   = 50                        # number of iterations for optimizer
p['kappa_lb']       = 0.005                     # lower bound kappa
p['kappa_ub']       = 0.05                      # upper bound kappa
p['rho_lb']         = 0                         # lower bound rho
p['rho_ub']         = 13                        # upper bound rho
p['gamma_lb']       = 0                         # lower bound gamma
p['gamma_ub']       = 13E4                      # upper bound gamma
p['lbfgs_vectors']  = 5                         # number of vectors for lbfgs update
p['lbfgs_scale_type']   = "scalar"              # initial hessian approximation
p['lbfgs_scale_hist']   = 5                     # used vecs for initial hessian approx
p['ls_max_func_evals']  = 20                    # number of max line-search attempts
p['prediction']         = 0                     # enable prediction

############### === define run configuration
r['code_path']      = code_dir;
r['compute_sys']    = 'cbica'                # TACC systems are: maverick2, frontera, stampede2, longhorn; UPENN system : cbica

###############=== write config to write_path and submit job
par.submit(p, r, submit_job, use_gpu);
