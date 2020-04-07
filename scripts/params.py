""" This script defines all parameters for the tumor solver
    and generates a config file, hwich can be read in by the
    c++ binary.
    The script can be called with a dict 'set_params' to override
    defaults defined here.
"""
import os

def write_config(set_params, run):
    """ 'set_params' is a dictionary which can be populated
         with user values for compulsory tumor parameters.
         'run' is a dictionary containig compute system and
         job information.
    """

    #############################################################################
    #############################################################################
    #
    #### Tumoer Solver Parameters  ###
    #### ------------------------- ###

    p = {}
    r = {}

    r['code_path'] = os.getcwd()
    r['mpi_tasks'] = 1

    for key, value in run.items():
        r[key] = value;

    ### grid
    p['nx'] = 256                         # grid resolution in each dimension
    p['ny'] = 256
    p['nz'] = 256

    ### inversion scheme
    p['solver'] = 'sparse_til'          # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effec, multi_species, forward, test
    p['invert_diff'] = 1                # enable diffusion inversion
    p['invert_reac'] = 1                # enable reaction inversion
    p['multilevel'] = 1                 # rescale p activations according to Gaussian width on each level
    p['inject_solution'] = 1            # use coarse level solution as warm-start
    p['pre_reacdiff_solve'] = 1         # reaction/diffusion solver before sparse til solve
    p['verbosity'] = 1                  # various levels of output density

    ### optimizer
    p['newton_solver'] = "QN"           # GN, QN
    p['line_search'] = "armijo"         # 'armijo' : armijo conditions; 'mt' : more-thuene (wolfe conditions)
    p['ce_loss'] = 0                    # cross-entropy or L2 loss
    p['regularization'] = "L1c"         # L2, L1c,
    p['beta_p'] = 1E-4                  # regularization parameter
    p['rel_grad_tol'] = 1E-5            # relative gradient tolerance
    p['newton_maxit'] = 50              # number of iterations for optimizer
    p['krylov_maxit'] = 1               # if GN: number krylov iterations
    p['gist_maxit'] = 2                 # number iterations for L1 CoSaMp solver
    p['kappa_lb'] = 0.0001              # lower bound kappa
    p['kappa_ub'] = 1.0                 # upper bound kappa
    p['rho_lb'] = 0                     # lower bound rho
    p['rho_ub'] = 15                    # upper bound rho
    p['lbfgs_vectors'] = 10             # number of vectors for lbfgs update
    p['lbfgs_scale_type'] = "diagonal"  # initial hessian approximation
    p['lbfgs_scale_hist'] = 5           # used vecs for initial hessian approx
    p['ls_max_func_evals'] = 10         # number of max line-search attempts

    ### forward solver
    p['accuracy_order'] = 2             # time order accuracy
    p['ip_order']       = 3             # interpolation accuracy for semi-langrangian

    ### tumor params
    p['model'] = 1                      # 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
    p['init_rho'] = 8                   # initial guess rho (reaction in wm)
    p['init_k'] = 1E-2                  # initial guess kappa (diffusivity in wm)
    p['init_gamma'] = 12E4              # initial guess (forcing factor for mass effect)
    p['nt_inv'] = 40                    # number time steps
    p['dt_inv'] = 0.025                 # time step size
    p['k_gm_wm'] = 0.0                  # kappa ratio gm/wm (if zero, kappa=0 in gm)
    p['r_gm_wm'] = 0.0                  # rho ratio gm/wm (if zero, rho=0 in gm)

    ### data
    p['smoothing_factor'] = 1           # kernel width for smoothing of data and material properties
    p['smooth_data'] = 1                # 0: no smoothing
    p['obs_threshold_1'] = -0.99        # threshold for data d(1): points above threshold are observed
    p['obs_threshold_0'] = -0.99        # threshold for data d(0): points above threshold are observed
    p['obs_threshold_rel'] = 0          # 0: threshold numbers are absolute cell density numbers; 1: relative (percentage of max cell density)

    ### initial condition
    p['sparsity_level'] = 5             # target sparsity of recovered IC in sparse_til solver
    p['gaussian_selection_mode'] = 1    # 0: grid-based; 1: based on target data
    p['number_gaussians'] = 64          # only used if selection mode = 0
    p['sigma_factor'] = 1               # kernel width of Gaussians
    p['sigma_spacing'] = 2              # spacing of Gaussians
    p['threshold_data_driven'] = 0.1    # threshold of cell density in data considered for Gaussian support
    p['gaussian_volume_fraction'] = 0.9 # place Gaussian only if volume fraction of Gaussian lays within tumor

    ### prediction
    p['prediction'] = 1                 # enable prediction
    p['pred_times'] = [1.0, 1.2, 1.5]   # times for prediction
    p['dt_pred'] = 0.01                 # time step size for prediction

    ### synthetic data
    p['syn_flag'] = 0                   # create synthetic data
    p['user_cms'] = [(112,136,144,1)]   # arbitrary number of TILs (x,y,z,scale) with activation scale
    p['rho_data'] = 12                  # tumor parameters for synthetic data
    p['k_data'] = 0.05
    p['gamma_data'] = 12E4
    p['nt_data'] = 100
    p['dt_data'] = 0.01
    p['testcase'] = 0                   # 0: brain single focal synthetic
                                        # 1: No-brain constant coefficients
                                        # 2: No-brain sinusoidal coefficients
                                        # 3: brain multifocal synthetic tumor with nearby ground truths
                                        # 4: brain multifocal synthetic tumor with far away ground truths
    ### paths
    p['output_dir'] = r['code_path'] + '/results/';
    p['d1_path'] = r['code_path'] + '/brain_data/' + str(p['nx']) +'/cpl/c1p.nc'
    p['d0_path'] = ""                   # path to initial condition for tumor
    p['a_seg'] = ""                     # paths to atlas material properties
    p['a_wm_path'] = r['code_path'] + '/brain_data/' + str(p['nx']) +'/white_matter.nc'
    p['a_gm_path'] = r['code_path'] + '/brain_data/' + str(p['nx']) +'/gray_matter.nc'
    p['a_csf_path'] = r['code_path'] + '/brain_data/' + str(p['nx']) +'/csf.nc'
    p['a_glm_path'] = r['code_path'] + '/brain_data/' + str(p['nx']) +'/glial_matter.nc'
    p['a_ve_path'] = ""
    p['p_seg'] = ""                     # paths to patient material properties for mass effect
    p['p_wm_path'] = ""
    p['p_gm_path'] = ""
    p['p_csf_path'] = ""
    p['p_glm_path'] = ""
    p['p_ve_path'] = ""
    p['mri_path'] = ""                  # path to mri
    p['obs_mask_path'] = ""             # path to custom obs mask, default: none
    p['support_data_path'] = ""         # path to data for support, default: none, (target data for inversion is used)
    p['gaussian_cm_path'] = ""          # path to file with Gaussian support centers, default: none, (generate Gaussians based on target data)
    p['pvec_path'] = ""                 # path to initial guess p vector (if none, use zero)
    p['data_comp_path'] = ""            # path to label image of connected components of target data
    p['data_comp_data_path'] = ""       # path to .dat file for connected components of target data


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
    with open(os.path.join(p['output_dir'], "sover_config.txt"), "w") as f:
        f.write("#\n");
        f.write("#### Tumor Solver Configuration File ###\n");
        f.write("#### ------------------------------- ###\n");
        f.write("\n");
        f.write("\n");
        f.write("### grid" + "\n");
        f.write("nx=" + str(p['nx']) + "\n");
        f.write("ny=" + str(p['ny']) + "\n");
        f.write("nz=" + str(p['nz']) + "\n");

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
        f.write("rel_grad_tol=" + str(p['rel_grad_tol']) + "\n");
        f.write("newton_maxit=" + str(p['newton_maxit']) + "\n");
        f.write("krylov_maxit=" + str(p['krylov_maxit']) + "\n");
        f.write("gist_maxit=" + str(p['gist_maxit']) + "\n");
        f.write("kappa_lb=" + str(p['kappa_lb']) + "\n");
        f.write("kappa_ub=" + str(p['kappa_ub']) + "\n");
        f.write("rho_lb=" + str(p['rho_lb']) + "\n");
        f.write("rho_ub=" + str(p['rho_ub']) + "\n");
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
        f.write("init_gamma=" + str(p['init_gamma']) + "\n");
        f.write("nt_inv=" + str(p['nt_inv']) + "\n");
        f.write("dt_inv=" + str(p['dt_inv']) + "\n");
        f.write("k_gm_wm=" + str(p['k_gm_wm']) + "\n");
        f.write("r_gm_wm=" + str(p['r_gm_wm']) + "\n");

        f.write("\n");
        f.write("### data" + "\n");
        f.write("smoothing_factor=" + str(p['smoothing_factor']) + "\n");
        f.write("smooth_data=" + str(p['smooth_data']) + "\n");
        f.write("obs_threshold_1=" + str(p['obs_threshold_1']) + "\n");
        f.write("obs_threshold_0=" + str(p['obs_threshold_0']) + "\n");
        f.write("obs_threshold_rel=" + str(p['obs_threshold_rel']) + "\n");

        f.write("\n");
        f.write("### initial condition" + "\n");
        f.write("sparsity_level=" + str(p['sparsity_level']) + "\n");
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
        f.write("gamma_data=" + str(p['gamma_data']) + "\n");
        f.write("nt_data=" + str(p['nt_data']) + "\n");
        f.write("dt_data=" + str(p['dt_data']) + "\n");
        f.write("testcase=" + str(p['testcase']) + "\n");

        f.write("\n");
        f.write("### paths" + "\n");
        f.write("output_dir=" + str(p['output_dir']) + "\n");
        f.write("d1_path=" + str(p['d1_path']) + "\n");
        f.write("d0_path=" + str(p['d0_path']) + "\n");
        f.write("a_seg=" + str(p['a_seg']) + "\n");
        f.write("a_wm_path=" + str(p['a_wm_path']) + "\n");
        f.write("a_gm_path=" + str(p['a_gm_path']) + "\n");
        f.write("a_csf_path=" + str(p['a_csf_path']) + "\n");
        f.write("a_glm_path=" + str(p['a_glm_path']) + "\n");
        f.write("a_ve_path=" + str(p['a_ve_path']) + "\n");
        f.write("p_seg=" + str(p['p_seg']) + "\n");
        f.write("p_wm_path=" + str(p['p_wm_path']) + "\n");
        f.write("p_csf_path=" + str(p['p_csf_path']) + "\n");
        f.write("p_glm_path=" + str(p['p_glm_path']) + "\n");
        f.write("p_ve_path=" + str(p['p_ve_path']) + "\n");
        f.write("mri_path=" + str(p['mri_path']) + "\n");
        f.write("obs_mask_path=" + str(p['obs_mask_path']) + "\n");
        f.write("support_data_path=" + str(p['support_data_path']) + "\n");
        f.write("gaussian_cm_path=" + str(p['gaussian_cm_path']) + "\n");
        f.write("pvec_path=" + str(p['pvec_path']) + "\n");
        f.write("data_comp_path=" + str(p['data_comp_path']) + "\n");
        f.write("data_comp_data_path=" + str(p['data_comp_data_path']) + "\n");

    ibman = ""
    if 'ibrun_man' in r and r['ibrun_man']:
        ibman = " -n " + str(r['mpi_tasks']) + " -o 0 "
    cmd = ""
    if r['compute_sys'] == 'hazelhen':
        ppn = 24;
        if r['mpi_tasks'] < 24:
            ppn = r['mpi_tasks'];
        cmd = cmd + "aprun -n " + str(r['mpi_tasks']) + " -N " + str(ppn) + " ";
    elif r['compute_sys'] in ['stampede2', 'frontera', 'maverick2']:
        cmd = cmd + "ibrun " + ibman;
    else:
        cmd = cmd + "mpirun ";
    run_str = cmd + r['code_path'] + "/build/last/inverse -nx "
    for key, value in p.items():
        run_str += " -" + str(key) + " " + str(value) + " ";

    return run_str


if __name__=='__main__':
    import subprocess
    scripts_path = os.path.dirname(os.path.realpath(__file__))

    submit = False;

    code_dir = scripts_path + '/../'
    params = {}
    run = {}

    # params['output_dir'] = os.path.join(code_dir, 'results/check/');
    params['output_dir'] = os.path.join(code_dir, 'config/');
    run['code_path'] = code_dir
    run['compute_sys'] = 'frontera'

    if run['compute_sys'] == 'rebels':
        run['queue'] = 'rebels'
        run['nodes'] = 1
        run['mpi_taks'] = 20
    elif run['compute_sys'] == 'stampede2':
        run['queue'] = 'skx-normal'
        run['nodes'] = 3
        run['mpi_taks'] = 64
    elif run['compute_sys'] == 'frontera':
        run['queue'] = 'v100'
        run['nodes'] = 1
        run['mpi_taks'] = 1
    elif run['compute_sys'] == 'maverick2':
        run['queue'] = 'gtx'
        run['nodes'] = 1
        run['mpi_taks'] = 1
    elif run['compute_sys'] == 'hazelhen':
        run['nodes'] = 3
        run['mpi_taks'] = 64
    else:
        run['queue'] = 'normal'
        run['nodes'] = 1
        run['mpi_taks'] = 1

    run_str = write_config(params, run)

    # if not err:
    if True:
        print('No errors, submitting jobfile\n')
        fname = scripts_path + '/job.sh'
        submit_file = open(fname, 'w+')
        if run['compute_sys'] == 'hazelhen':
            submit_file.write("#!/bin/bash\n" + \
            "#PBS -N ITP\n" + \
            "#PBS -l nodes="+str(run['nodes'])+":ppn=24 \n" + \
            "#PBS -l walltime=01:00:00 \n" + \
            "#PBS -m e\n" + \
            "#PBS -M kscheufele@austin.utexas.edu\n\n" + \
            "source /zhome/academic/HLRS/ipv/ipvscheu/env_intel.sh\n" + \
            "export OMP_NUM_THREADS=1\n")
        else:
            submit_file.write ("#!/bin/bash\n" + \
            "#SBATCH -J tuinv\n" + \
            "#SBATCH -o " + params['output_dir'] + "/log\n" + \
            "#SBATCH -p " + str(run['queue']) + "\n" + \
            "#SBATCH -N " + str(run['nodes']) + "\n" + \
            "#SBATCH -n " + str(run['mpi_taks']) + "\n" + \
            "#SBATCH -t 06:00:00\n" + \
            "source ~/.bashrc\n" + \
            "export OMP_NUM_THREADS=1\n")
        submit_file.write(run_str)
        submit_file.close()

        ### submit jobfile
        if submit:
            if run['compute_sys'] == 'hazelhen':
                subprocess.call(['qsub', fname])
            else:
                subprocess.call(['sbatch', fname])
    else:
        print('Errors, no job submitted\n')



    # + str(N) + " -ny " + str(N) + " -nz " + str(N) + " -beta " + str(beta) + \
    # " -multilevel " + str(multilevel) + \
    # " -inject_solution " + str(inject_solution) + \
    # " -pre_reacdiff_solve " + str(pre_reacdiff_solve) + \
    # " -rho_inversion " + str(rho_inv) + " -k_inversion " + str(k_inv) + " -nt_inversion " + str(nt_inv) + " -dt_inversion " + str(dt_inv) + \
    # " -rho_data " + str(rho_data) + " -k_data " + str(k_data) + " -nt_data " + str(nt_data) + " -dt_data " + str(dt_data) + \
    # " -regularization " + reg_type + " -interpolation " + str(interp_flag) + " -diffusivity_inversion " + str(diffusivity_flag) + " -reaction_inversion " + str(reaction_flag) + \
    # " -basis_type " + str(basis_type) + " -number_gaussians " + str(np) + " -sigma_factor " + str(fac) + " -sigma_spacing " + str(space) + \
    # " -testcase " + str(tumor_testcase) + \
    # " -solve_rho_k " + str(solve_rho_k) + \
    # " -gaussian_volume_fraction " + str(gvf) +  \
    # " -lambda_continuation " + str(lam_cont) +  \
    # " -target_sparsity " + str(target_spars) +  \
    # " -sparsity_level " + str(sparsity_lvl) +  \
    # " -threshold_data_driven " + str(data_thres) +  \
    # " -sigma_data_driven " + str(dd_fac) + \
    # " -output_dir " + output_dir + \
    # " -newton_solver " + solvertype + \
    # " -line_search "   + linesearchtype + \
    # " -newton_maxit " + str(newton_maxit) + \
    # " -gist_maxit " + str(gist_maxit) + \
    # " -krylov_maxit " + str(max_krylov_iter) + \
    # " -rel_grad_tol " + str(grad_tol) + \
    # " -syn_flag " + str(create_synthetic) + \
    # " -data_path " + p['d1_path'] + \
    # " -gm_path " + p['gm_path'] + \
    # " -wm_path " + p['wm_path'] + \
    # " -csf_path " + p['csf_path'] + \
    # " -glm_path " + p['glm_path'] + \
    # " -p_gm_path " + p_gm_path + \
    # " -p_wm_path " + p_wm_path + \
    # " -p_csf_path " + p_csf_path + \
    # " -p_glm_path " + p_glm_path + \
    # " -z_cm " + str(z_cm) + \
    # " -y_cm " + str(y_cm) + \
    # " -x_cm " + str(x_cm) + \
    # " -obs_mask_path " + obs_mask_path + \
    # " -support_data_path " + support_data_path + \
    # " -gaussian_cm_path " + gaussian_cm_path + \
    # " -pvec_path " + pvec_path + \
    # " -data_comp_path " + data_comp_path + \
    # " -data_comp_dat_path " + data_comp_dat_path + \
    # " -init_tumor_path " + init_tumor_path + \
    # " -model " + str(model) + \
    # " -smooth " + str(smooth_f) + \
    # " -observation_threshold " + str(obs_thres) + \
    # " -k_gm_wm " + str(k_gm_wm) + \
    # " -r_gm_wm " + str(r_gm_wm) + \
    # " -low_freq_noise " + str(noise_scale) + \
    # " -prediction " + str(predict_flag) + \
    # " -forward " + str(forward_flag) + \
    # " -ip_order " + str(ip_order) + \
    # " -ce_loss " + str(ce_loss) + \
    # " -order " + str(accuracy_order) + \
    # " -verbosity " + str(verbosity) + \
    # " -forcing_factor " + str(forcing_factor) + \
    # " -kappa_lb " + str(lower_bound_kappa) + \
    # " -kappa_ub " + str(upper_bound_kappa) + \
    # " -mri_path " + mri_path + \
    # " -tao_blmvm_mat_lmvm_num_vecs 50 -tao_blmvm_mat_lmvm_scale_type diagonal" + \
    # " -tumor_tao_ls_max_funcs " + str(p['ls_max_func_evals']) + " "

    # -tao_test_hessian -tao_test_hessian_view
    # " -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5 " + \
    # " -tao_bqnls_mat_lmvm_num_vecs 50 -tao_bqnls_mat_lmvm_scale_type diagonal " + \
    # " -tao_blmvm_mat_lmvm_num_vecs 50 -tao_blmvm_mat_lmvm_scale_type diagonal " + \