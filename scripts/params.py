""" This script defines all parameters for the tumor solver
    and generates a config file, wich can be read in by the
    c++ binary.
    The script can be called with a dict 'set_params' to override
    defaults defined here.
"""
import os


 ### ________________________________________________________________________ ___
 ### //////////////////////////////////////////////////////////////////////// ###
def write_config(set_params, run):
    """ 'set_params' is a dictionary which can be populated
         with user values for compulsory tumor parameters.
         'run' is a dictionary containig compute system and
         job information.
    """

    #############################################################################
    #############################################################################
    #### Tumoer Solver Parameters  ###
    #### ------------------------- ###

    p = {}
    r = {}

    r['code_path'] = os.getcwd()
    r['mpi_tasks'] = 1

    for key, value in run.items():
        r[key] = value;

    # ------------------------------ DO NOT TOUCH ------------------------------ #
    # any changes to the defaults defined below, are to be defined in the dicts  #
    # 'set_params', and 'run', which are passed to this routine. Do not modify   #
    # the defaults                                                               #
    # ------------------------------ DO NOT TOUCH ------------------------------ #


    ### grid
    p['n'] = 256                        # grid resolution in each dimension
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### inversion scheme
    p['solver'] = 'forward'             # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
    p['invert_diff'] = 1                # enable diffusion inversion
    p['invert_reac'] = 1                # enable reaction inversion
    p['multilevel'] = 0                 # rescale p activations according to Gaussian width on each level
    p['inject_solution'] = 0            # use coarse level solution as warm-start
    p['pre_reacdiff_solve'] = 0         # reaction/diffusion solver before sparse til solve
    p['verbosity'] = 3                  # various levels of output density
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### optimizer
    p['newton_solver'] = "QN"           # GN, QN
    p['line_search'] = "armijo"         # 'armijo' : armijo conditions; 'mt' : more-thuene (wolfe conditions)
    p['ce_loss'] = 0                    # cross-entropy or L2 loss
    p['regularization'] = "L2"          # L2, L1
    p['beta_p'] = 1E-4                  # regularization parameter
    p['opttol_grad'] = 1E-6             # relative gradient tolerance
    p['newton_maxit'] = 50              # number of iterations for optimizer
    p['krylov_maxit'] = 1               # if GN: number krylov iterations
    p['gist_maxit'] = 2                 # number iterations for L1 CoSaMp solver
    p['kappa_lb'] = 0.0001              # lower bound kappa
    p['kappa_ub'] = 1                 # upper bound kappa
    p['kappaf_lb'] = 0.0001              # lower bound kappa
    p['kappaf_ub'] = 1                 # upper bound kappa
    p['rho_lb'] = 0                     # lower bound rho
    p['rho_ub'] = 15                    # upper bound rho
    p['gamma_lb'] = 0                   # lower bound gamma
    p['gamma_ub'] = 15E4                # upper bound gamma
    p['lbfgs_vectors'] = 10             # number of vectors for lbfgs update
    p['lbfgs_scale_type'] = "diagonal"  # initial hessian approximation
    p['lbfgs_scale_hist'] = 5           # used vecs for initial hessian approx
    p['ls_max_func_evals'] = 10         # number of max line-search attempts
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### forward solver
    p['accuracy_order'] = 2             # time order accuracy
    p['ip_order']       = 3             # interpolation accuracy for semi-langrangian
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### tumor params
    p['model'] = 1                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
    p['init_rho'] = 10                   # initial guess rho (reaction in wm)
    p['init_k'] = 1E-2                  # initial guess kappa (diffusivity in wm)
    p['init_kf'] = 0
    p['init_gamma'] = 12E4              # initial guess (forcing factor for mass effect)
    p['nt_inv'] = 40                    # number time steps
    p['dt_inv'] = 0.025                 # time step size
    p['k_gm_wm'] = 0.0                  # kappa ratio gm/wm (if zero, kappa=0 in gm)
    p['r_gm_wm'] = 0.0                  # rho ratio gm/wm (if zero, rho=0 in gm)
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### data
    p['smoothing_factor'] = 1           # kernel width for smoothing of data and material properties
    p['smoothing_factor_data'] = 1      # 0: no smoothing, otherwise kernel width (for data at t=1 and material properties)
    p['smoothing_factor_data_t0'] = 1   # 0: no smoothing, otherwise kernel width (for data at t=0)
    p['obs_threshold_1'] = -0.99        # threshold for data d(1): points above threshold are observed
    p['obs_threshold_0'] = -0.99        # threshold for data d(0): points above threshold are observed
    p['obs_threshold_rel'] = 0          # 0: threshold numbers are absolute cell density numbers; 1: relative (percentage of max cell density)
    p['obs_lambda'] = -1                # if > 0: creates observation mask OBS = 1[TC] + lambda*1[B/WT] from segmentation file (only works if segmentation is read)
    p['two_time_points'] = 0           # 0: only data at t=1 is provided, 1: data for both t=1 and t=0 is provided
    p['atlas_labels'] = ""              # example (brats): '[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### initial condition
    p['sparsity_level'] = 5             # target sparsity of recovered IC in sparse_til solver
    p['thresh_component_weight'] = 1E-3 # threshold for weight of data component (to be considered for sparsity comp.)
    p['gaussian_selection_mode'] = 1    # 0: grid-based; 1: based on target data
    p['number_gaussians'] = 1           # only used if selection mode = 0
    p['sigma_factor'] = 1               # kernel width of Gaussians
    p['sigma_spacing'] = 2              # spacing of Gaussians
    p['threshold_data_driven'] = 0.1    # threshold of cell density in data considered for Gaussian support
    p['gaussian_volume_fraction'] = 0.9 # place Gaussian only if volume fraction of Gaussian lays within tumor
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### prediction
    p['prediction'] = 1                 # enable prediction
    p['pred_times'] = [1.0, 1.2, 1.5]   # times for prediction
    p['dt_pred'] = 0.01                 # time step size for prediction
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### synthetic data
    p['syn_flag'] = 1                   # create synthetic data
    p['user_cms'] = [(137,169,96,1)]    # arbitrary number of TILs (x,y,z,scale) with activation scale
    p['rho_data'] = 10                  # tumor parameters for synthetic data
    p['k_data'] = 0.025
    p['kf_data'] = 0
    p['gamma_data'] = 12E4
    p['nt_data'] = 25
    p['dt_data'] = 0.04
    p['testcase'] = 0                   # 0: brain single focal synthetic
                                        # 1: No-brain constant coefficients
                                        # 2: No-brain sinusoidal coefficients
                                        # 3: brain multifocal synthetic tumor with nearby ground truths
                                        # 4: brain multifocal synthetic tumor with far away ground truths
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### paths
    p['output_dir'] = r['code_path'] + '/results/';
    p['d1_path'] = r['code_path'] + '/brain_data/' + str(p['n']) +'/cpl/c1p.nc'
    p['d1_path'] = r['code_path'] + '/brain_data/' + str(p['n']) +'/cpl/c1p.nc'
    p['output_dir'] = r['code_path'] + '/results/';
    p['d1_path'] = r['code_path'] + '/brain_data/' + str(p['n']) +'/cpl/c1p.nc'
    p['d0_path'] = ""                   # path to initial condition for tumor
    p['a_seg_path'] = ""                # paths to atlas material properties
    p['a_wm_path'] = r['code_path'] + '/brain_data/' + str(p['n']) +'/white_matter.nc'
    p['a_gm_path'] = r['code_path'] + '/brain_data/' + str(p['n']) +'/gray_matter.nc'
    p['a_csf_path'] = ""
    p['a_vt_path'] = r['code_path'] + '/brain_data/' + str(p['n']) +'/csf.nc'
    p['a_kf_path'] = r['code_path'] + '/brain_data/' + str(p['n']) + '/w/'

    p['p_seg_path'] = ""                # [optional] paths to patient material properties for mass effect
    p['p_wm_path'] = ""
    p['p_gm_path'] = ""
    p['p_csf_path'] = ""
    p['p_vt_path'] = ""
    p['mri_path'] = ""                  # [optional] path to mri
    p['obs_mask_path'] = ""             # [optional] path to custom obs mask, default: none
    p['support_data_path'] = ""         # [optional] path to data for support, default: none, (target data for inversion is used)
    p['gaussian_cm_path'] = ""          # [optional] path to file with Gaussian support centers, default: none, (generate Gaussians based on target data)
    p['pvec_path'] = ""                 # [optional] path to initial guess p vector (if none, use zero)
    p['data_comp_path'] = ""            # [optional] path to label image of connected components of target data
    p['data_comp_data_path'] = ""       # [optional] path to .dat file for connected components of target data
    p['velocity_x1'] = ""               # [optional] path to velocity for meterial transport
    p['velocity_x2'] = ""
    p['velocity_x3'] = ""
    # ------------------------------ DO NOT TOUCH ------------------------------ #
    ### performance
    p['time_history_off'] = 1           # 1: do not allocate time history (only works with forward solver or FD inversion)
    p['store_phi']        = 0           # 1: store every Gaussian as 3d image
    p['store_adjoint']    = 1           # 1: store adjoint time history
    p['write_output']     = 1           # 1: write .nc and .nii.gz output

    #############################################################################
    #############################################################################

    # override defaults
    for key, value in set_params.items():
        if key in p:
            p[key] = value;
            print(" Setting {} = {}".format(key, value))
        else:
            print(" Error: key {} not supported.".format(key))
    for key, value in p.items():
        if key not in set_params:
            print(" Using default {} = {}".format(key, value))
    # create dir
    if not os.path.exists(p['output_dir']):
        os.makedirs(p['output_dir'])


    # write config file
    with open(os.path.join(p['output_dir'], "solver_config.txt"), "w") as f:
        f.write("#\n");
        f.write("#### Tumor Solver Configuration File ###\n");
        f.write("#### ------------------------------- ###\n");
        f.write("\n");
        f.write("\n");
        f.write("### grid" + "\n");
        f.write("n=" + str(p['n']) + "\n");

        f.write("\n");
        f.write("### inversion scheme" + "\n");
        f.write("solver=" + str(p['solver']) + "\n");
        f.write("invert_diff=" + str(p['invert_diff']) + "\n");
        f.write("invert_reac=" + str(p['invert_reac']) + "\n");
        f.write("multilevel=" + str(p['multilevel']) + "\n");
        f.write("inject_solution=" + str(p['inject_solution']) + "\n");
        f.write("pre_reacdiff_solve=" + str(p['pre_reacdiff_solve']) + "\n");
        f.write("verbosity=" + str(p['verbosity']) + "\n");

        f.write("\n");
        f.write("### optimizer" + "\n");
        f.write("newton_solver=" + str(p['newton_solver']) + "\n");
        f.write("line_search=" + str(p['line_search']) + "\n");
        f.write("ce_loss=" + str(p['ce_loss']) + "\n");
        f.write("regularization=" + str(p['regularization']) + "\n");
        f.write("beta_p=" + str(p['beta_p']) + "\n");
        f.write("opttol_grad=" + str(p['opttol_grad']) + "\n");
        f.write("newton_maxit=" + str(p['newton_maxit']) + "\n");
        f.write("krylov_maxit=" + str(p['krylov_maxit']) + "\n");
        f.write("gist_maxit=" + str(p['gist_maxit']) + "\n");
        f.write("kappa_lb=" + str(p['kappa_lb']) + "\n");
        f.write("kappa_ub=" + str(p['kappa_ub']) + "\n");
        f.write("rho_lb=" + str(p['rho_lb']) + "\n");
        f.write("rho_ub=" + str(p['rho_ub']) + "\n");
        f.write("gamma_lb=" + str(p['gamma_lb']) + "\n");
        f.write("gamma_ub=" + str(p['gamma_ub']) + "\n");
        f.write("lbfgs_vectors=" + str(p['lbfgs_vectors']) + "\n");
        f.write("lbfgs_scale_type=" + str(p['lbfgs_scale_type']) + "\n");
        f.write("lbfgs_scale_hist=" + str(p['lbfgs_scale_hist']) + "\n");
        f.write("ls_max_func_evals=" + str(p['ls_max_func_evals']) + "\n");

        f.write("\n");
        f.write("### forward solver" + "\n");
        f.write("accuracy_order=" + str(p['accuracy_order']) + "\n");
        f.write("ip_order=" + str(p['ip_order']) + "\n");

        f.write("\n");
        f.write("### tumor params" + "\n");
        f.write("model=" + str(p['model']) + "\n");
        f.write("init_rho=" + str(p['init_rho']) + "\n");
        f.write("init_k=" + str(p['init_k']) + "\n");
        f.write("init_kf=" + str(p['init_kf']) + "\n");
        f.write("init_gamma=" + str(p['init_gamma']) + "\n");
        f.write("nt_inv=" + str(p['nt_inv']) + "\n");
        f.write("dt_inv=" + str(p['dt_inv']) + "\n");
        f.write("k_gm_wm=" + str(p['k_gm_wm']) + "\n");
        f.write("r_gm_wm=" + str(p['r_gm_wm']) + "\n");

        f.write("\n");
        f.write("### data" + "\n");
        f.write("smoothing_factor=" + str(p['smoothing_factor']) + "\n");
        f.write("smoothing_factor_data=" + str(p['smoothing_factor_data']) + "\n");
        f.write("obs_threshold_1=" + str(p['obs_threshold_1']) + "\n");
        f.write("obs_threshold_0=" + str(p['obs_threshold_0']) + "\n");
        f.write("obs_threshold_rel=" + str(p['obs_threshold_rel']) + "\n");
        f.write("obs_lambda=" + str(p['obs_lambda']) + "\n");
        f.write("two_time_points=" + str(p['two_time_points']) + "\n");
        f.write("atlas_labels=" + str(p['atlas_labels']) + "\n");

        f.write("\n");
        f.write("### initial condition" + "\n");
        f.write("sparsity_level=" + str(p['sparsity_level']) + "\n");
        f.write("thresh_component_weight=" + str(p['thresh_component_weight']) + "\n");
        f.write("gaussian_selection_mode=" + str(p['gaussian_selection_mode']) + "\n");
        f.write("number_gaussians=" + str(p['number_gaussians']) + "\n");
        f.write("sigma_factor=" + str(p['sigma_factor']) + "\n");
        f.write("sigma_spacing=" + str(p['sigma_spacing']) + "\n");
        f.write("threshold_data_driven=" + str(p['threshold_data_driven']) + "\n");
        f.write("gaussian_volume_fraction=" + str(p['gaussian_volume_fraction']) + "\n");

        f.write("\n");
        f.write("### prediction" + "\n");
        f.write("prediction=" + str(p['prediction']) + "\n");
        f.write("pred_times=" + str(p['pred_times']) + "\n");
        f.write("dt_pred=" + str(p['dt_pred']) + "\n");

        f.write("\n");
        f.write("### synthetic data" + "\n");
        f.write("syn_flag=" + str(p['syn_flag']) + "\n");
        f.write("user_cms=" + str(p['user_cms']) + "\n");
        f.write("rho_data=" + str(p['rho_data']) + "\n");
        f.write("k_data=" + str(p['k_data']) + "\n");
        f.write("kf_data=" + str(p['kf_data']) + "\n");
        f.write("gamma_data=" + str(p['gamma_data']) + "\n");
        f.write("nt_data=" + str(p['nt_data']) + "\n");
        f.write("dt_data=" + str(p['dt_data']) + "\n");
        f.write("testcase=" + str(p['testcase']) + "\n");

        f.write("\n");
        f.write("### paths" + "\n");
        f.write("output_dir=" + str(p['output_dir']) + "\n");
        f.write("d1_path=" + str(p['d1_path']) + "\n");
        f.write("d0_path=" + str(p['d0_path']) + "\n");
        f.write("a_seg_path=" + str(p['a_seg_path']) + "\n");
        f.write("a_wm_path=" + str(p['a_wm_path']) + "\n");
        f.write("a_gm_path=" + str(p['a_gm_path']) + "\n");
        f.write("a_csf_path=" + str(p['a_csf_path']) + "\n");
        f.write("a_vt_path=" + str(p['a_vt_path']) + "\n");
        f.write("a_kf_path=" + str(p['a_kf_path']) + "\n");
        f.write("p_seg_path=" + str(p['p_seg_path']) + "\n");
        f.write("p_wm_path=" + str(p['p_wm_path']) + "\n");
        f.write("p_gm_path=" + str(p['p_gm_path']) + "\n");
        f.write("p_csf_path=" + str(p['p_csf_path']) + "\n");
        f.write("p_vt_path=" + str(p['p_vt_path']) + "\n");
        f.write("mri_path=" + str(p['mri_path']) + "\n");
        f.write("obs_mask_path=" + str(p['obs_mask_path']) + "\n");
        f.write("support_data_path=" + str(p['support_data_path']) + "\n");
        f.write("gaussian_cm_path=" + str(p['gaussian_cm_path']) + "\n");
        f.write("pvec_path=" + str(p['pvec_path']) + "\n");
        f.write("data_comp_path=" + str(p['data_comp_path']) + "\n");
        f.write("data_comp_data_path=" + str(p['data_comp_data_path']) + "\n");
        f.write("velocity_x1=" + str(p['velocity_x1']) + "\n");
        f.write("velocity_x2=" + str(p['velocity_x2']) + "\n");
        f.write("velocity_x3=" + str(p['velocity_x3']) + "\n");

        f.write("\n");
        f.write("### performance" + "\n");
        f.write("time_history_off=" + str(p['time_history_off']) + "\n");
        f.write("store_phi=" + str(p['store_phi']) + "\n");
        f.write("store_adjoint=" + str(p['store_adjoint']) + "\n");
        f.write("write_output=" + str(p['write_output']) + "\n");

    ibman = ""
    if 'ibrun_man' in r and r['ibrun_man']:
        ibman = r['ibrun_man']
    cmd = ""
    if r['compute_sys'] == 'hazelhen':
        ppn = 24;
        if r['mpi_tasks'] < 24:
            ppn = r['mpi_tasks'];
        cmd = cmd + "aprun -n " + str(r['mpi_tasks']) + " -N " + str(ppn) + " ";
    elif r['compute_sys'] in ['stampede2', 'frontera', 'maverick2', 'longhorn']:
        cmd = cmd + "ibrun " + ibman;
    else:
        cmd = cmd + "mpirun ";
    run_str = cmd + r['code_path'] + "/build/last/tusolver "
    run_str += " -config " + p['output_dir'] + "/solver_config.txt"
#    for key, value in p.items():
#        run_str += " -" + str(key) + " " + str(value) + " ";

    return run_str



### ________________________________________________________________________ ___
### //////////////////////////////////////////////////////////////////////// ###
def write_jobscript_header(tu_params, run_params):
    """ writes the header for the job script to string.
    """
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    code_dir = scripts_path + '/../'
    if "output_dir" in tu_params:
        out_dir = tu_params['output_dir']
    else:
        out_dir = scripts_path

    if 'code_path' not in run_params:
        run_params['code_path'] = code_dir
    if 'compute_sys' not in run_params:
        run_params['compute_sys'] = 'rebels'
    if 'queue' not in run_params:
        if run_params['compute_sys'] == 'rebels':
            run_params['queue'] = 'rebels'
        elif run_params['compute_sys'] == 'stampede2':
            run_params['queue'] = 'skx-normal'
        elif run_params['compute_sys'] == 'longhorn':
            run_params['queue'] = 'v100'
        elif run_params['compute_sys'] == 'maverick2':
            run_params['queue'] = 'gtx'
        elif run_params['compute_sys'] == 'frontera':
                run_params['queue'] = 'normal'
        else:
            run_params['queue'] = 'normal'
    if 'nodes' not in run_params:
        if run_params['compute_sys'] == 'rebels':
            run_params['nodes'] = 1
        elif run_params['compute_sys'] == 'stampede2':
            run_params['nodes'] = 3
        elif run_params['compute_sys'] == 'longhorn':
            run_params['nodes'] = 1
        elif run_params['compute_sys'] == 'maverick2':
            run_params['nodes'] = 1
        elif run_params['compute_sys'] == 'frontera':
            run_params['nodes'] = 1
        else:
            run_params['nodes'] = 1
    if 'mpi_taks' not in run_params:
        if run_params['compute_sys'] == 'rebels':
            run_params['mpi_taks'] = 20
        elif run_params['compute_sys'] == 'stampede2':
            run_params['mpi_taks'] = 32
        elif run_params['compute_sys'] == 'longhorn':
            run_params['mpi_taks'] = 1
        elif run_params['compute_sys'] == 'maverick2':
            run_params['mpi_taks'] = 1
        elif run_params['compute_sys'] == 'frontera':
            run_params['mpi_taks'] = 32
        else:
            run_params['mpi_taks'] = 1

    if 'wtime_h' not in run_params:
        run_params['wtime_h'] = 48
    if 'wtime_m' not in run_params:
        run_params['wtime_m'] = 0
    if 'log_dir' not in run_params:
        run_params['log_dir'] = tu_params['output_dir']
    fname = out_dir + '/job.sh'
    job_header = "" #open(fname, 'w+')
    job_header += "#!/bin/bash\n"
    job_header += "#SBATCH -J tuinv\n"
    job_header += "#SBATCH -o " + run_params['log_dir'] + "/log\n"
    job_header += "#SBATCH -p " + str(run_params['queue']) + "\n"
    job_header += "#SBATCH -N " + str(run_params['nodes']) + "\n"
    job_header += "#SBATCH -n " + str(run_params['mpi_taks']) + "\n"
    #job_header += "#SBATCH --exclude=luke"
    job_header += "#SBATCH -t "+str(run_params['wtime_h'])+":"+str(run_params['wtime_m'])+":00\n\n"
    job_header += "source ~/.bashrc\n"
    job_header += "export OMP_NUM_THREADS=1\n\n"
    if 'extra_modules' in run_params:
      job_header += str(run_params['extra_modules']) + "\n"

    return job_header

### ________________________________________________________________________ ___
### //////////////////////////////////////////////////////////////////////// ###
def submit(tu_params, run_params, submit_job = True):
    """ creates config file and submits job """
    import subprocess
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    code_dir = scripts_path + '/../'
    if "output_dir" in tu_params:
        out_dir = tu_params['output_dir']
    else:
        out_dir = scripts_path

    fname = scripts_path + '/job.sh'
    job_file = open(fname, 'w+')

    job_header = write_jobscript_header(tu_params, run_params)
    run_str = write_config(tu_params, run_params)

    job_file.write(job_header)
    job_file.write(run_str)
    job_file.close()
    ### submit jobfile
    if submit_job:
        subprocess.call(['sbatch', fname])



 ### ________________________________________________________________________ ___
 ### //////////////////////////////////////////////////////////////////////// ###
if __name__=='__main__':
    import subprocess
    scripts_path = os.path.dirname(os.path.realpath(__file__))

    submit_job = False;

    code_dir = scripts_path + '/../'
    params = {}
    run = {}
    params['output_dir'] = os.path.join(code_dir, 'config/')
    run['code_path'] = code_dir

    submit(params, run, submit_job)



## ===============================================================================================+
## ===============================================================================================+
## ===============================================================================================+
## ===============================================================================================+
