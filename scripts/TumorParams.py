### Run parameters for inverse tumor solve
### This script defines a method which sets all tumor input parameters
### and returns the run command for the inverse tumor solve
### run.py runs the code with default parameters defined here if
### not overwritten
import os

def getTumorRunCmd(params):
    ### params is a dictionary which can be populated with user values for compulsory tumor parameters
    ### params can contain the following keys:
    ### N, data_path, results_path, gm_path, wm_path, csf_path
    ### Other parameters are default and changed only if necessary

    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################

    ### tumor_dir is the path to pglistr
    if not 'code_path' in params:
        print('Path to pglistr not provided: Setting as current working directory\n')
        params['code_path'] = os.getcwd()
    tumor_dir = params['code_path']

    ### TUMOR PARAMETERS SET BEGIN

    ### No of discretization points (Assumed uniform)
    N = 128
    ### Path to all output results (Directories are created automatically)
    results_path = tumor_dir + '/results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    ### Input data
    ### Path to data
    data_path = tumor_dir + '/brain_data/' + str(N) +'/cpl/c1p.nc'
    ### Atlas
    ### Path to gm
    gm_path = tumor_dir + '/brain_data/' + str(N) +'/gray_matter.nc'
    ### Path to csf
    csf_path = tumor_dir + '/brain_data/' + str(N) +'/csf.nc'
    ### Path to wm
    wm_path = tumor_dir + '/brain_data/' + str(N) +'/white_matter.nc'
    ### Path to custom obs mask, default: none
    obs_mask_path = ""
    ### Path to data for support, default: none, (target data for inversion is used)
    support_data_path = ""
    ### Path to file with Gaussian support centers, default: none, (generate Gaussians based on target data)
    gaussian_cm_path = ""
    ### Path to initial guess p vector, default: none, (use zero initial guess)
    pvec_path = ""
    ### Path to label image of connected components of target data
    data_comp_path = ""
    ### Path to .dat file for connected components of target data
    data_comp_dat_path = ""


    verbosity = 1
    ### Other user parameters which typically stay as default: Change if needed
    ### Flag to create synthetic data
    create_synthetic = 1
    ### Inversion tumor parameters  -- Tumor is inverted with these parameters: Use k_inv=0 if diffusivity is being inverted
    rho_inv = 15
    k_inv = 0.0
    nt_inv = 20
    dt_inv = 0.05

    ### tumor regularization type -- L1, L1c, L2, L2b  : L1c is cosamp
    reg_type = "L1c"
    ### Model type: 1: RD, 2: RD + pos, 3: RD + full objective, 4: Mass effect
    model = 1
    ### Synthetic data parameters  -- Tumor is grown with these parameters
    rho_data = 12
    k_data = 0.05
    nt_data = 100
    dt_data = 0.01

    ### Testcase: 0: brain single focal synthetic
    ###              1: No-brain constant coefficients
    ###              2: No-brain sinusoidal coefficients
    ###              3: brain multifocal synthetic tumor with nearby ground truths
    ##               4: brain multifocal synthetic tumor with far away ground truths
    tumor_testcase = 0

    multilevel = 0;

    ### k_gm_wm ratio
    k_gm_wm = 0.0
    ### r_gm_wm ratio
    r_gm_wm = 0.0
    ### Smoothing factor: Number of voxels to smooth material properties and basis functions
    smooth_f = 1.5
    ### Interpolation flag   -- Flag to solve an interpolation problem (find parameterization of the data) only
    interp_flag = 0
    ### Solve for reaction/diffusin flag -- Flag to solve only for reaction diffusion, assumes c(0) to be read in
    solve_rho_k = 0;
    ### Prediction flag -- Flag to predict tumor at a later time
    predict_flag = 1
    ### Forward flag -- Flag to run only forward solve
    forward_flag = 0
    ### Diffusivity inversion flag  -- Flag to invert for diffusivity/diffusion coefficient
    diffusivity_flag = 1
    ### Reaction inversion flag -- Flag to invert for reaction coefficient
    reaction_flag = 1
    ### Radial basis flag: 1 - data driven, 0 - grid-based (bounding box)  (Use data-driven for all tests)
    basis_type = 1
    ### Lambda continuation flag -- Flag for parameter continuation in L1 optimization (Keep turned on)
    lam_cont = 1
    ### Tumor L2 regularization
    beta = 1e-4
    ### No of radial basis functions (Only used if basis_type is grid-based)
    np = 64
    ### Factor (integer only) which controls the variance of the basis function for synthetic data (\sigma  =  fac * 2 * pi / meshsize)
    fac = 2
    ### Spacing factor between radial basis functions (Keep as 2 to have a well-conditioned matrix for the radial basis functions)
    space = 2
    ### Gaussian volume fraction -- Fraction of Gaussian that has to be tumorous to switch on the basis function at any grid point
    gvf = 0.99
    ### Threshold of data tumor concentration above which Gaussians are switched on
    data_thres = 0.1
    ### Observation detection threshold
    obs_thres = 0.0
    ### Noise scaling for low freq noise: 0.05, 0.25, 0.5
    noise_scale = 0.0
    ### Target sparsity we expect for our initial tumor condition -- used in GIST
    target_spars = 0.99
    ### Sparsity level we expect for our initial tumor condition -- used in CoSaMp
    sparsity_lvl = 10
    ### Factor (integer only) which controls the variance of the basis function for tumor inversion (\sigma  =  fac * 2 * pi / meshsize)
    dd_fac = 2
    ### Solver type: QN - Quasi newton, GN - Gauss newton
    solvertype = "QN"
    ### Line-search type: armijo - armijo line-search, mt - more-thuene line search (wolfe conditions)
    linesearchtype = "mt"
    ### Newton max iterations
    newton_maxit = 50
    ### GIST max iterations (for L1 solver)
    gist_maxit = 2
    ### Krylov max iterations
    max_krylov_iter = 30
    ### Relative gradient tolerance
    grad_tol = 1E-5
    ### Forward solver time order of accuracy
    accuracy_order = 2
    ### number of line-search attempts
    ls_max_func_evals = 10
    ## lower bound on kappa
    lower_bound_kappa = 1E-3
    ## upper bound on kappa
    upper_bound_kappa = 1

    ### TUMOR PARAMETERS SET END

    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################


    ### Error checking and run script string generation
    # ---
    error_flag = 0
    if 'N' in params:
        N = params['N']
    else:
        print ('Default N = {} used'.format(N))
    # ---
    if 'grad_tol' in params:
        grad_tol = params['grad_tol']
    else:
        print ('Default grad_tol = {} used'.format(grad_tol))
    # ---
    if 'newton_maxit' in params:
        newton_maxit = params['newton_maxit']
    else:
        print ('Default newton_maxit = {} used'.format(newton_maxit))
    # ---
    if 'gist_maxit' in params:
        gist_maxit = params['gist_maxit']
    else:
        print ('Default gist_maxit = {} used'.format(gist_maxit))
    # ---
    if 'data_thres' in params:
        data_thres = params['data_thres']
    else:
        print ('Default data_thres = {} used'.format(data_thres))
    # ---
    if 'rho_inv' in params:
        rho_inv = params['rho_inv']
    else:
        print ('Default rho = {} used'.format(rho_inv))
    # ---
    if 'ls_max_func_evals' in params:
        ls_max_func_evals = params['ls_max_func_evals']
    else:
        print ('Default ls_max_func_evals = {} used'.format(ls_max_func_evals))
    # ---
    if 'gvf' in params:
        gvf = params['gvf']
    else:
        print ('Default gvf = {} used'.format(gvf))
    # ---
    if 'predict_flag' in params:
        predict_flag = params['predict_flag']
    else:
        print ('Default predict_flag = {} used'.format(predict_flag))
    # ---
    if 'k_inv' in params:
        k_inv = params['k_inv']
    else:
        print ('Default k = {} used'.format(k_inv))
    # ---
    if 'beta' in params:
        beta = params['beta']
    else:
        print ('Default beta = {} used'.format(beta))
    # ---
    if 'linesearchtype' in params:
        linesearchtype = params['linesearchtype']
    else:
        print ('Using default line-search type = {} used'.format(linesearchtype))
    # ---
    if 'multilevel' in params:
        multilevel = params['multilevel']
    # ---
    if 'dd_fac' in params:
        dd_fac = params['dd_fac']
    else:
        print ('Default dd_fac = {} used'.format(dd_fac))
    # ---
    if 'upper_bound_kappa' in params:
        upper_bound_kappa = params['upper_bound_kappa']
        print('Setting upper bound {} on kappa'.format(upper_bound_kappa))
    else:
        print('Default upper bound {} on kappa used'.format(upper_bound_kappa))
    # ---
    if 'lower_bound_kappa' in params:
        lower_bound_kappa = params['lower_bound_kappa']
        print('Setting lower bound {} on kappa'.format(lower_bound_kappa))
    else:
        print('Default lower bound {} on kappa used'.format(lower_bound_kappa))
    # ---
    if 'sparsity_lvl' in params:
        sparsity_lvl = params['sparsity_lvl']
    else:
        print ('Default sparsity_lvl = {} used'.format(sparsity_lvl))
    # ---
    if 'results_path' in params:
        results_path = params['results_path']
        if not os.path.exists(results_path):
            print ('Results path does not exist, making the required folders and sub-folders...\n')
            os.makedirs(results_path)
    # ---
    if 'data_path' in params:
        data_path = params['data_path']
        print('Tumor data path = {}'.format(data_path))
    else:
        if not os.path.exists(data_path):
            if not create_synthetic:
                print('Default data path does not exist and no input path provided!\n')
                error_flag = 1
        else:
            print ('Default datapath = {} used'.format(data_path))
    # ---
    if 'gm_path' in params:
        gm_path = params['gm_path']
        print('Gray matter path = {}'.format(gm_path))
    else:
        if not os.path.exists(gm_path):
            print('Default atlas gray matter path does not exist and no input path provided!\n')
            error_flag = 1
        else:
            print ('Default atlas gray matter path = {} used'.format(gm_path))
    # ---
    if 'wm_path' in params:
        wm_path = params['wm_path']
        print('White matter path = {}'.format(wm_path))
    else:
        if not os.path.exists(wm_path):
            print('Default atlas white matter path does not exist and no input path provided!\n')
            error_flag = 1
        else:
            print ('Default atlas white matter path = {} used'.format(wm_path))
    # ---
    if "diffusivity_inversion" in params:
        diffusivity_flag = params['diffusivity_inversion'];
    # ---
    if 'csf_path' in params:
        csf_path = params['csf_path']
        print('CSF path = {}'.format(csf_path))
    else:
        if not os.path.exists(csf_path):
            print('Default atlas csf path does not exist and no input path provided!\n')
            error_flag = 1
        else:
            print ('Default atlas csf path = {} used'.format(csf_path))
    # ---
    if 'obs_mask_path' in params:
        obs_mask_path = params['obs_mask_path']
        print('OBS mask path = {}'.format(obs_mask_path))
    else:
        print('No custom observation mask given.')
    # ---
    if 'support_data_path' in params:
        support_data_path = params['support_data_path']
        print('support_data path = {}'.format(support_data_path))
    else:
        print('Using target tumor data for Gaussian support selection.')
    # ---
    if 'gaussian_cm_path' in params:
        gaussian_cm_path = params['gaussian_cm_path']
        print('path to file with Gaussian centers = {}'.format(gaussian_cm_path))
    else:
        print('Generating Gaussian support from target data.')
    # ---
    if 'pvec_path' in params:
        pvec_path = params['pvec_path']
        print('p vector initial guess path = {}'.format(pvec_path))
    else:
        print('Using zero initial guess for p vector.')
    # ---
    if 'data_comp_path' in params:
        data_comp_path = params['data_comp_path']
        print('path to file with target data comp label image = {}'.format(data_comp_path))
    else:
        print('No label image for components of target data set.')
    # ---
    if 'data_comp_dat_path' in params:
        data_comp_dat_path = params['data_comp_dat_path']
        print('path to file with target data comp  = {}'.format(data_comp_dat_path))
    else:
        print('No .dat file for components of target data given.')
    # ---
    if "solve_rho_k" in params:
        solve_rho_k = params['solve_rho_k']
        print('solving for rho and k only (c(0) must be set via p and Gaussian centers)')


    ibman = ""
    if 'ibrun_man' in params and params['ibrun_man']:
        ibman = " -n " + str(params['mpi_pernode']) + " -o 0 "
    cmd = ""
    if params['compute_sys'] == 'hazelhen':
        ppn = 24;
        if params['mpi_pernode'] < 24:
            ppn = params['mpi_pernode'];
        cmd = cmd + "aprun -n " + str(params['mpi_pernode']) + " -N " + str(ppn) + " ";
    elif params['compute_sys'] == 'stampede2':
        cmd = cmd + "ibrun " + ibman;
    else:
        cmd = cmd + "mpirun ";
    run_str = cmd + tumor_dir + "/build/brats19/inverse -nx " + str(N) + " -ny " + str(N) + " -nz " + str(N) + " -beta " + str(beta) + \
    " -multilevel " + str(multilevel) + \
    " -rho_inversion " + str(rho_inv) + " -k_inversion " + str(k_inv) + " -nt_inversion " + str(nt_inv) + " -dt_inversion " + str(dt_inv) + \
    " -rho_data " + str(rho_data) + " -k_data " + str(k_data) + " -nt_data " + str(nt_data) + " -dt_data " + str(dt_data) + \
    " -regularization " + reg_type + " -interpolation " + str(interp_flag) + " -diffusivity_inversion " + str(diffusivity_flag) + " -reaction_inversion " + str(reaction_flag) + \
    " -basis_type " + str(basis_type) + " -number_gaussians " + str(np) + " -sigma_factor " + str(fac) + " -sigma_spacing " + str(space) + \
    " -testcase " + str(tumor_testcase) + \
    " -solve_rho_k " + str(solve_rho_k) + \
    " -gaussian_volume_fraction " + str(gvf) +  \
    " -lambda_continuation " + str(lam_cont) +  \
    " -target_sparsity " + str(target_spars) +  \
    " -sparsity_level " + str(sparsity_lvl) +  \
    " -threshold_data_driven " + str(data_thres) +  \
    " -sigma_data_driven " + str(dd_fac) + \
    " -output_dir " + results_path + \
    " -newton_solver " + solvertype + \
    " -line_search "   + linesearchtype + \
    " -newton_maxit " + str(newton_maxit) + \
    " -gist_maxit " + str(gist_maxit) + \
    " -krylov_maxit " + str(max_krylov_iter) + \
    " -rel_grad_tol " + str(grad_tol) + \
    " -syn_flag " + str(create_synthetic) + \
    " -data_path " + data_path + \
    " -gm_path " + gm_path + \
    " -wm_path " + wm_path + \
    " -csf_path " + csf_path + \
    " -obs_mask_path " + obs_mask_path + \
    " -support_data_path " + support_data_path + \
    " -gaussian_cm_path " + gaussian_cm_path + \
    " -pvec_path " + pvec_path + \
    " -data_comp_path " + data_comp_path + \
    " -data_comp_dat_path " + data_comp_dat_path + \
    " -model " + str(model) + \
    " -smooth " + str(smooth_f) + \
    " -observation_threshold " + str(obs_thres) + \
    " -k_gm_wm " + str(k_gm_wm) + \
    " -r_gm_wm " + str(r_gm_wm) + \
    " -low_freq_noise " + str(noise_scale) + \
    " -prediction " + str(predict_flag) + \
    " -forward " + str(forward_flag) + \
    " -order " + str(accuracy_order) + \
    " -verbosity " + str(verbosity) + \
    " -kappa_lb " + str(lower_bound_kappa) + \
    " -kappa_ub " + str(upper_bound_kappa) + \
    " -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5 -tumor_tao_ls_monitor  -tumor_tao_ls_max_funcs " + str(ls_max_func_evals) + " "

    return run_str, error_flag
