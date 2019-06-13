#include "Utils.h"
#include "TumorSolverInterface.h"

static char help[] = "Inverse Driver \
\n Testcase 1 - Constant reaction and diffusion coefficient \
\n Testcase 2 - Sinusoidal reaction and diffusion coefficient";


struct HealthyProbMaps { //Stores prob maps for healthy atlas and healthy tissues of patient -- Only used if model is full objective
    Vec gm, wm, csf, glm, bg;
    Vec gm_data, wm_data, csf_data, glm_data, bg_data;
    Vec xi_gm, xi_wm, xi_csf, xi_glm, xi_bg;

    HealthyProbMaps (Vec gm_in, Vec wm_in, Vec csf_in, Vec glm_in, Vec bg_in) {
        gm = gm_in; wm = wm_in; csf = csf_in; glm = glm_in; bg = bg_in;
        VecDuplicate (gm, &gm_data); VecDuplicate (gm, &xi_gm);
        VecDuplicate (gm, &wm_data); VecDuplicate (gm, &xi_wm);
        VecDuplicate (gm, &csf_data); VecDuplicate (gm, &xi_csf);
        VecDuplicate (gm, &bg_data); VecDuplicate (gm, &xi_bg);
    }

    ~HealthyProbMaps () {
        PetscErrorCode ierr = 0;
        ierr = VecDestroy (&gm_data);
        ierr = VecDestroy (&wm_data);
        ierr = VecDestroy (&csf_data);
        ierr = VecDestroy (&bg_data);
        ierr = VecDestroy (&xi_gm);
        ierr = VecDestroy (&xi_wm);
        ierr = VecDestroy (&xi_csf);
        ierr = VecDestroy (&xi_bg);
    }
};

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode generateSinusoidalData (Vec &d, std::shared_ptr<NMisc> n_misc);
PetscErrorCode computeError (double &error_norm, double &error_norm_c0, Vec p_rec, Vec data, Vec data_obs, Vec c_0, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode readData (Vec &data, Vec &support_data, Vec &data_components, Vec &c_0, Vec &p_rec, std::shared_ptr<NMisc> n_misc, char*, char *, char *);
PetscErrorCode readAtlas (Vec &wm, Vec &gm, Vec &glm, Vec &csf, Vec &bg, std::shared_ptr<NMisc> n_misc, char*, char*, char*, char*);
PetscErrorCode readObsFilter (Vec &obs_mask, std::shared_ptr<NMisc> n_misc, char*);
PetscErrorCode createMFData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode setDistMeasuresFullObj (std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<HealthyProbMaps> h_maps, Vec);
PetscErrorCode computeSegmentation(std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc);
PetscErrorCode applyLowFreqNoise (Vec data, std::shared_ptr<NMisc> n_misc);



int main (int argc, char** argv) {
 /* ACCFFT, PETSC setup begin */
    PetscErrorCode ierr;
    PetscInitialize (&argc, &argv, (char*) 0, help);
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);


    // Input parameters (controlled from run script)
    int n[3];
    n[0] = 64;
    n[1] = 64;
    n[2] = 64;
    int testcase = 0;
    double beta_user = -1.0;
    double rho_inv= -1.0;
    double k_inv = 0.0;
    int nt_inv = 0.0;
    double dt_inv = 0.0;
    double rho_data = -1.0;
    double k_data = -1.0;
    int nt_data = 0.0;
    double dt_data = 0.0;
    double data_comp_weights[10];
    int ncomp = 10;
    char reg[10];
    char results_dir[400];
    char data_path[400];
    char support_data_path[400];
    char gm_path[400];
    char wm_path[400];
    char glm_path[400];
    char csf_path[400];
    char obs_mask_path[400];
    char data_comp_path[400];
    char data_comp_dat_path[400];
    char p_vec_path[400];
    char gaussian_cm_path[400];
    int np_user = 0;
    int interp_flag = 0;
    int diffusivity_flag = 0;
    int reaction_flag = 0;
    int solve_rho_k_only_flag = 0;
    int checkpointing_flag = 1;
    // int invert_obs_mask = 0;
    int basis_type = 0;
    double sigma = -1.0;
    double spacing_factor = -1.0;
    double gvf = -1.0;
    double target_spars = -1.0;
    int sparsity_level = -1.0;
    int lam_cont = 0;
    int verbosity_in = 1;
    double sigma_dd = -1.0;
    double data_thres = -1.0;
    double obs_thres = -1.0;

    double k_gm_wm = -1.0;
    double r_gm_wm = -1.0;

    char newton_solver[10];
    int newton_maxit = -1;
    int gist_maxit = -1;
    int krylov_maxit = -1;

    int syn_flag = -1;
    int model = -1;

    int fwd_flag = 0;

    int flag_cosamp = 0;

    double sm = -1;

    double opttolgrad = -1.0;

    double low_freq_noise_scale = -1.0;

    int predict_flag = 0;
    int order_of_accuracy = -1;

    PetscBool strflg;
    PetscOptionsBegin (PETSC_COMM_WORLD, NULL, "Tumor Inversion Options", "");
    PetscOptionsInt ("-nx", "NX", "", n[0], &n[0], NULL);
    PetscOptionsInt ("-ny", "NY", "", n[1], &n[1], NULL);
    PetscOptionsInt ("-nz", "NZ", "", n[2], &n[2], NULL);
    PetscOptionsInt ("-model", "model", "", model, &model, NULL);
    PetscOptionsInt ("-number_gaussians", "Tumor np", "", np_user, &np_user, NULL);
    PetscOptionsInt ("-testcase", "Test Cases", "", testcase, &testcase, NULL);
    PetscOptionsReal ("-beta", "Tumor Regularization", "", beta_user, &beta_user, NULL);
    PetscOptionsReal ("-rho_inversion", "Tumor inversion reaction coefficient", "", rho_inv, &rho_inv, NULL);
    PetscOptionsReal ("-k_inversion", "Tumor inversion reaction coefficient", "", k_inv, &k_inv, NULL);
    PetscOptionsInt ("-nt_inversion", "Tumor inversion reaction coefficient", "", nt_inv, &nt_inv, NULL);
    PetscOptionsReal ("-dt_inversion", "Tumor inversion reaction coefficient", "", dt_inv, &dt_inv, NULL);
    PetscOptionsReal ("-rho_data", "Tumor inversion reaction coefficient", "", rho_data, &rho_data, NULL);
    PetscOptionsReal ("-k_data", "Tumor inversion reaction coefficient", "", k_data, &k_data, NULL);
    PetscOptionsInt ("-nt_data", "Tumor inversion reaction coefficient", "", nt_data, &nt_data, NULL);
    PetscOptionsReal ("-dt_data", "Tumor inversion reaction coefficient", "", dt_data, &dt_data, NULL);
    PetscStrcpy (reg, "L2b"); //default reg
    PetscOptionsString ("-regularization", "Tumor regularization", "", reg, reg, 10, NULL);
    PetscStrcpy (results_dir, "./results/check/"); //default
    PetscOptionsString ("-output_dir", "Path to results directory", "", results_dir, results_dir, 400, NULL);
    PetscOptionsInt ("-interpolation", "Interpolation flag", "", interp_flag, &interp_flag, NULL);
    PetscOptionsInt ("-diffusivity_inversion", "Diffusivity inversion flag", "", diffusivity_flag, &diffusivity_flag, NULL);
    PetscOptionsInt ("-reaction_inversion", "Reaction coefficient inversion flag", "", reaction_flag, &reaction_flag, NULL);
    PetscOptionsInt ("-basis_type", "Radial basis type", "", basis_type, &basis_type, NULL);
    PetscOptionsReal ("-sigma_factor", "Standard deviation factor of grid-based radial basis", "", sigma, &sigma, NULL);
    PetscOptionsReal ("-sigma_spacing", "Standard deviation spacing factor of grid-based radial basis", "", spacing_factor, &spacing_factor, NULL);
    PetscOptionsReal ("-gaussian_volume_fraction", "Gaussian volume fraction of data-driven radial basis", "", gvf, &gvf, NULL);
    PetscOptionsReal ("-target_sparsity", "Target sparsity of continuation", "", target_spars, &target_spars, NULL);
    PetscOptionsInt ("-lambda_continuation", "Lambda continuation", "", lam_cont, &lam_cont, NULL);
    PetscOptionsReal ("-sigma_data_driven", "Sigma for data-driven Gaussians", "", sigma_dd, &sigma_dd, NULL);
    PetscOptionsReal ("-threshold_data_driven", "Data threshold for data-driven Gaussians", "", data_thres, &data_thres, NULL);
    PetscOptionsReal ("-observation_threshold", "Observation detection threshold", "", obs_thres, &obs_thres, NULL);
    PetscOptionsReal ("-k_gm_wm", "WM to GM ratio for diffusivity", "", k_gm_wm, &k_gm_wm, NULL);
    PetscOptionsReal ("-r_gm_wm", "WM to GM ratio for reaction", "", r_gm_wm, &r_gm_wm, NULL);
    PetscOptionsReal ("-smooth", "Smoothing factor", "", sm, &sm, NULL);
    PetscOptionsReal ("-low_freq_noise", "Noise level for low frequency noise addition", "", low_freq_noise_scale, &low_freq_noise_scale, NULL);
    PetscStrcpy (newton_solver, "QN");
    PetscOptionsString ("-newton_solver", "Newton solver type", "", newton_solver, newton_solver, 10, NULL);
    PetscOptionsInt ("-newton_maxit", "Newton max iterations", "", newton_maxit, &newton_maxit, NULL);
    PetscOptionsInt ("-gist_maxit", "GIST max iterations", "", gist_maxit, &gist_maxit, NULL);
    PetscOptionsInt ("-krylov_maxit", "Krylov max iterations", "", krylov_maxit, &krylov_maxit, NULL);
    PetscOptionsReal ("-rel_grad_tol", "Relative gradient tolerance for L2 solves", "", opttolgrad, &opttolgrad, NULL);
    PetscOptionsInt ("-syn_flag", "Flag for synthetic data generation", "", syn_flag, &syn_flag, NULL);
    PetscOptionsInt ("-sparsity_level", "Sparsity level guess for tumor initial condition", "", sparsity_level, &sparsity_level, NULL);
    PetscOptionsInt ("-prediction", "Flag to predict future tumor growth", "", predict_flag, &predict_flag, NULL);
    PetscOptionsInt ("-forward", "Flag to do only the forward solve using data generation parameters", "", fwd_flag, &fwd_flag, NULL);
    PetscOptionsInt ("-order", "Order of accuracy of PDE solver", "", order_of_accuracy, &order_of_accuracy, NULL);

    PetscOptionsString ("-data_path", "Path to data", "", data_path, data_path, 400, NULL);
    PetscOptionsString ("-support_data_path", "Path to data used to generate Gaussian support", "", support_data_path, support_data_path, 400, NULL);
    PetscOptionsString ("-gm_path", "Path to GM", "", gm_path, gm_path, 400, NULL);
    PetscOptionsString ("-wm_path", "Path to WM", "", wm_path, wm_path, 400, NULL);
    PetscOptionsString ("-csf_path", "Path to CSF", "", csf_path, csf_path, 400, NULL);
    PetscOptionsString ("-glm_path", "Path to GLM", "", glm_path, glm_path, 400, NULL);
    PetscOptionsString ("-obs_mask_path", "Path to observation mask", "", obs_mask_path, obs_mask_path, 400, NULL);
    PetscOptionsString ("-pvec_path", "Path to initial guess p vector", "", p_vec_path, p_vec_path, 400, NULL);
    PetscOptionsString ("-gaussian_cm_path", "Path to file with Gaussian centers", "", gaussian_cm_path, gaussian_cm_path, 400, NULL);
    PetscOptionsInt    ("-verbosity", "solver verbosity (1-4)", "", verbosity_in, &verbosity_in, NULL);
    PetscOptionsInt    ("-checkpointing_flag", "solver writes checkpoints for p vector and corresponding Gaussian centers", "", checkpointing_flag, &checkpointing_flag, NULL);
    PetscOptionsInt    ("-solve_rho_k", "Flag to do only the inversion for reaction and diffusion coefficient, keeping the initial condition c(0) fixed (needs to be read in)", "", solve_rho_k_only_flag, &solve_rho_k_only_flag, NULL);

    // bool flag = PETSC_FALSE;
    // PetscOptionsGetRealArray(NULL,NULL, "-data_comp_weights", data_comp_weights, &ncomp, &flag);
    PetscOptionsString ("-data_comp_path", "Path to label img of data components", "", data_comp_path, data_comp_path, 400, NULL);
    PetscOptionsString ("-data_comp_dat_path", "Path to .dat file of data components", "", data_comp_dat_path, data_comp_dat_path, 400, NULL);


    PetscOptionsEnd ();


    PCOUT << " ----- Grid Size: " << n[0] << "x" << n[1] << "x" << n[2] << " ---- " << std::endl;
    switch (testcase) {
        case CONSTCOEF: {
            PCOUT << " ----- Test Case 1: No brain, Constant reaction and diffusion ---- " << std::endl;
            break;
        }
        case SINECOEF: {
            PCOUT << " ----- Test Case 2: No brain, Sinusoidal reaction and diffusion ---- " << std::endl;
            break;
        }
        case BRAIN: {
            PCOUT << " ----- Full brain test ---- " << std::endl;
            break;
        }
        case BRAINNEARMF: {
            PCOUT << " ----- Full brain test with multifocal nearby synthetic tumors ---- " << std::endl;
            break;
        }
        case BRAINFARMF: {
            PCOUT << " ----- Full brain test with multifocal faroff synthetic tumors ---- " << std::endl;
            break;
        }
        default: break;
    }

    accfft_init();
    MPI_Comm c_comm;
    int c_dims[2] = { 0 };
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
    int isize[3], osize[3], istart[3], ostart[3];
    int64_t alloc_max = accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    double *c_0 = (double*) accfft_alloc (alloc_max);
    Complex *c_hat = (Complex*) accfft_alloc (alloc_max);
    accfft_plan *plan = accfft_plan_dft_3d_r2c (n, c_0, (double*) c_hat, c_comm, ACCFFT_MEASURE);
    accfft_free (c_0);
    accfft_free (c_hat);
/* ACCFFT, PETSC setup end */
/* --------------------------------------------------------------------------------------------------------------*/

{
    EventRegistry::initialize ();
    Event e1 ("solve-tumor-inverse-tao");
    //Generate synthetic data
    //Synthetic parameters: Overwrite n_misc
    Vec c_0, data, support_data, p_rec, wm, gm, glm, csf, bg, obs_mask, data_components;

    PetscErrorCode ierr = 0;
    double rho_temp, k_temp, dt_temp, nt_temp;
    bool overwrite_model = true; //don't change -- controlled from the run script
    bool read_support_data   = (support_data_path != NULL && strlen(support_data_path) > 0); // path set?
    bool use_custom_obs_mask = (obs_mask_path != NULL && strlen(obs_mask_path) > 0);         // path set?
    bool use_data_comps      = (data_comp_path != NULL && strlen(data_comp_path) > 0);       // path set?
    bool warmstart_p         = (p_vec_path != NULL && strlen(p_vec_path) > 0);               // path set?
    if (warmstart_p  && not (gaussian_cm_path != NULL && strlen(gaussian_cm_path) > 0)){
      PCOUT << " ERROR: if initial guess for p is used, Gaussian centers need to be specified. " << std::endl;
      exit(-1);
    }
    double rho = rho_data;
    double k = k_data;
    double dt = dt_data;
    int nt = nt_data;

    // Other vars
    double prec_norm;
    double *prec_ptr;
    double l2_rel_error = 0.0;
    double error_norm_c0 = 0.0;

    std::vector<double> out_params;
    int n_gist = 0, n_newton;
    std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, osize, istart, ostart, plan, c_comm, c_dims, testcase);   //This class contains all required parameters


    // Read input parameters (controlled from run script)
    if (beta_user >= 0) {    //user has provided tumor reg
        n_misc->beta_ = beta_user;
        n_misc->beta_changed_ = true;
    }
    PetscStrcmp ("L2b", reg, &strflg);
    if (strflg) {
        n_misc->regularization_norm_ = L2b;
    }
    PetscStrcmp ("L2", reg, &strflg);
    if (strflg) {
        n_misc->regularization_norm_ = L2;
    }
    PetscStrcmp ("L1", reg, &strflg);
    if (strflg) {
        n_misc->regularization_norm_ = L1;
    }
    PetscStrcmp ("L1c", reg, &strflg);
    if (strflg) {
        n_misc->regularization_norm_ = L2;
        flag_cosamp = 1;
    }
    if (diffusivity_flag) {
        n_misc->diffusivity_inversion_ = true;
    }
    if (checkpointing_flag) {
      n_misc->write_p_checkpoint_ = true;
    }
    if (reaction_flag) {
        n_misc->reaction_inversion_ = true;
    }
    if (basis_type) {
        n_misc->bounding_box_ = 0;
        n_misc->np_ = 1;
    } else {
        n_misc->bounding_box_ = 1;
        n_misc->np_ = np_user;
    }
    if (lam_cont) {
        n_misc->lambda_continuation_ = true;
    } else {
        n_misc->lambda_continuation_ = false;
    }
    if (target_spars > -1.0) {
        n_misc->target_sparsity_ = target_spars;
    }
    if (sparsity_level > -1.0) {
        n_misc->sparsity_level_ = sparsity_level;
    }
    if (gvf > -1.0) {
        n_misc->gaussian_vol_frac_ = gvf;
    }
    if (sigma > -1.0) {
        n_misc->phi_sigma_ = sigma * (2.0 * M_PI / n_misc->n_[0]);
    }
    if (spacing_factor > -1.0) {
        n_misc->phi_spacing_factor_ = spacing_factor;
    }
    if (sigma_dd > -1.0) {
        n_misc->phi_sigma_data_driven_ = sigma_dd * 2.0 * M_PI / n_misc->n_[0];
    }

    if (data_thres > -1.0) {
        n_misc->data_threshold_ = data_thres;
    }

    if (obs_thres > -1.0) {
        n_misc->obs_threshold_ = obs_thres;
    }

    if (opttolgrad != -1.0) {
        n_misc->opttolgrad_ = opttolgrad;
    }

    PetscStrcmp ("QN", newton_solver, &strflg);
    if (strflg) {
        n_misc->newton_solver_ = QUASINEWTON;
    }
    PetscStrcmp ("GN", newton_solver, &strflg);
    if (strflg) {
        n_misc->newton_solver_ = GAUSSNEWTON;
    }

    if (newton_maxit != -1.0) {
        n_misc->newton_maxit_ = newton_maxit;
    }

    if (gist_maxit != -1.0) {
        n_misc->gist_maxit_ = gist_maxit;
    }

    if (krylov_maxit != -1.0) {
        n_misc->krylov_maxit_ = krylov_maxit;
    }

    if (model != -1) {
        n_misc->model_ = model;
    }

    if (sm >= 0) {
        n_misc->smoothing_factor_ = sm;
    }

    if (k_gm_wm > -1.0) {
        n_misc->k_gm_wm_ratio_ = k_gm_wm;
    }

    if (r_gm_wm > -1.0) {
        n_misc->r_gm_wm_ratio_ = r_gm_wm;
    }

    if (low_freq_noise_scale > -1.0) {
        n_misc->low_freq_noise_scale_ = low_freq_noise_scale;
    }

    if (order_of_accuracy != -1) {
        n_misc->order_ = order_of_accuracy;
    }


    n_misc->predict_flag_ = predict_flag;
    n_misc->verbosity_ = verbosity_in;
    n_misc->writepath_.str (std::string ());                                       //clear the writepath stringstream
    n_misc->writepath_ << results_dir;

    rho_temp = n_misc->rho_;
    k_temp = n_misc->k_;
    dt_temp = n_misc->dt_;
    nt_temp = n_misc->nt_;

    if (overwrite_model) {
        n_misc->rho_ = rho;
        n_misc->k_ = k;
        n_misc->dt_ = dt;
        n_misc->nt_ = nt;
    }

    std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc, nullptr, nullptr);
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();

    // Set optimization flags of tumor solver from input script
    solver_interface->getInvSolver()->getOptSettings ()->newton_maxit = n_misc->newton_maxit_;
    solver_interface->getInvSolver()->getOptSettings ()->newtonsolver = n_misc->newton_solver_;
    solver_interface->getInvSolver()->getOptSettings ()->gist_maxit = n_misc->gist_maxit_;
    solver_interface->getInvSolver()->getOptSettings ()->krylov_maxit = n_misc->krylov_maxit_;
    solver_interface->getInvSolver()->getOptSettings ()->opttolgrad = n_misc->opttolgrad_;

    solver_interface->getInvSolver()->getOptSettings ()->verbosity = n_misc->verbosity_;

    bool read_atlas = true;   // Set from run script outside
    if (read_atlas) {
        ierr = readAtlas (wm, gm, glm, csf, bg, n_misc, gm_path, wm_path, csf_path, glm_path);
        ierr = tumor->mat_prop_->setValuesCustom (gm, wm, nullptr, csf, bg, n_misc);    //Overwrite Matprop with custom atlas
        ierr = solver_interface->updateTumorCoefficients (nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true);
    }
    std::shared_ptr<HealthyProbMaps> h_maps = std::make_shared<HealthyProbMaps> (gm, wm, csf, nullptr, bg);

    double self_exec_time = -MPI_Wtime ();
    std::array<double, 7> timers = {0};

    if (syn_flag == 1) {
        PCOUT << "Generating Synthetic Data --->" << std::endl;
        if (n_misc->testcase_ == BRAINFARMF || n_misc->testcase_ == BRAINNEARMF) {
            ierr = createMFData (c_0, data, p_rec, solver_interface, n_misc);
        } else {
            ierr = generateSyntheticData (c_0, data, p_rec, solver_interface, n_misc);
        }
        read_support_data = false;
        support_data = data;
    } else {
        ierr = readData (data, support_data, data_components, c_0, p_rec, n_misc, data_path, support_data_path, data_comp_path);
        if(use_data_comps) {
          PCOUT << " SET LABELS " << std::endl;
          tumor->phi_->setLabels (data_components);           
        }
        if(use_custom_obs_mask){
          ierr = readObsFilter(obs_mask, n_misc, obs_mask_path);
          PCOUT << "Use custom observation mask\n";
        }
    }

    Vec data_nonoise;
    ierr = VecDuplicate (data, &data_nonoise);
    ierr = VecCopy (data, data_nonoise);

    if (syn_flag == 1) {
        // add low freq model noise
        ierr = applyLowFreqNoise (data, n_misc);
        Vec temp;
        double noise_err_norm, rel_noise_err_norm;
        ierr = VecDuplicate (data, &temp);      CHKERRQ (ierr);
        ierr = VecSet (temp, 0.);               CHKERRQ (ierr);
        ierr = VecCopy (data_nonoise, temp);    CHKERRQ (ierr);
        ierr = VecAXPY (temp, -1.0, data);      CHKERRQ (ierr);
        ierr = VecNorm (temp, NORM_2, &noise_err_norm);               CHKERRQ (ierr);  // diff btw noise corrupted signal and ground truth
        ierr = VecNorm (data_nonoise, NORM_2, &rel_noise_err_norm);   CHKERRQ (ierr);
        rel_noise_err_norm = noise_err_norm / rel_noise_err_norm;
        PCOUT << "[--------------- Low frequency relative error = " << rel_noise_err_norm << " -------------------]" << std::endl;

        // if (n_misc->writeOutput_)
        //     dataOut (data, n_misc, "dataNoise.nc");

        ierr = VecDestroy (&temp);          CHKERRQ (ierr);


        PCOUT << "Data generated with parameters: rho = " << n_misc->rho_ << " k = " << n_misc->k_ << " dt = " << n_misc->dt_ << " Nt = " << n_misc->nt_ << std::endl;
    } else {
        PCOUT << "Data read\n";
    }

    if (fwd_flag) {
        PCOUT << "Forward solve completed: exiting...\n";
    } else {
        PCOUT << "Inverse solver begin" << std::endl;

        n_misc->rho_ = rho_inv;
        // n_misc->k_ = (n_misc->diffusivity_inversion_) ? 0 : k_inv;
        n_misc->k_ = k_inv; // (n_misc->diffusivity_inversion_) ? 0 : k_inv;
        n_misc->dt_ = dt_inv;
        n_misc->nt_ = nt_inv;

        PCOUT << "Inversion with tumor parameters: rho = " << n_misc->rho_ << " k = " << n_misc->k_ << " dt = " << n_misc->dt_ << " Nt = " << n_misc->nt_ << std::endl;
        PCOUT << "Results in: " << n_misc->writepath_.str().c_str() << std::endl;

        std::string file_concomp(data_comp_dat_path);
        if(use_data_comps){
          readConCompDat(tumor->phi_->component_weights_, tumor->phi_->component_centers_, file_concomp);
        }

        if (!n_misc->bounding_box_) {
            // ierr = tumor->mat_prop_->setValuesCustom (gm, wm, glm, csf, n_misc);    //Overwrite Matprop with custom atlas
            // set the observation operator filter : default filter
            if (use_custom_obs_mask) {
              ierr = tumor->obs_->setCustomFilter (obs_mask);
              PCOUT << "Set custom observation mask" << std::endl;
            } else {
              ierr = tumor->obs_->setDefaultFilter (data);
              PCOUT << "Set default observation mask based on input data and threshold " << tumor->obs_->threshold_  << std::endl;
            }
            // apply observer on ground truth, store observed data in d
            // observation operator is applied before gaussians are set.
            ierr = tumor->obs_->apply (data, data);                                     CHKERRQ (ierr);
            ierr = tumor->obs_->apply (support_data, support_data);                     CHKERRQ (ierr);

            int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
            int nr = (n_misc->reaction_inversion_)    ? n_misc->nr_ : 0;
            // if p vector and Gaussian centers are read in
            if (warmstart_p) {
              PCOUT << "Solver warmstart using p and Gaussian centers" << std::endl;
              std::string file_p(p_vec_path);
              std::string file_cm(gaussian_cm_path);
              ierr = tumor->phi_->setGaussians (file_cm);                               CHKERRQ (ierr);     //Overwrites bounding box phis with custom phis
              ierr = tumor->phi_->setValues (tumor->mat_prop_);                         CHKERRQ (ierr);
              readBIN(&p_rec, n_misc->np_ + nk + nr, file_p);
            } else {
              ierr = tumor->phi_->setGaussians (support_data);                          CHKERRQ (ierr);     //Overwrites bounding box phis with custom phis
              ierr = tumor->phi_->setValues (tumor->mat_prop_);                         CHKERRQ (ierr);
              //re-create p_rec
              ierr = VecDestroy (&p_rec);                                               CHKERRQ (ierr);
              #ifdef SERIAL
                  ierr = VecCreateSeq (PETSC_COMM_SELF, n_misc->np_ + nk, &p_rec);      CHKERRQ (ierr);
              #else
                  ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                          CHKERRQ (ierr);
                  ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                CHKERRQ (ierr);
                  ierr = VecSetFromOptions (p_rec);                                     CHKERRQ (ierr);
              #endif
            }
            ierr = solver_interface->setParams (p_rec, nullptr);
        }

        if (n_misc->model_ == 3) {// Modified objective
            setDistMeasuresFullObj (solver_interface, h_maps, data);
        }

        ierr = tumor->rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, tumor->mat_prop_, n_misc);
        ierr = tumor->k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, tumor->mat_prop_, n_misc);
        if (!warmstart_p) {
            ierr = VecSet (p_rec, 0);                                                       CHKERRQ (ierr);
            ierr = solver_interface->setInitialGuess (0.);
        }


        if (interp_flag) {
            PCOUT << "SOLVING INTERPOLATION WITH DATA" << std::endl;
            ierr = solver_interface->solveInterpolation (data, p_rec, tumor->phi_, n_misc); //interpolates c_0 or data <---- CHECK
            PCOUT << " --------------  INTERPOLATED P -----------------\n";
            if (procid == 0) {
                ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
            }
            PCOUT << " --------------  -------------- -----------------\n";
            ierr = VecSet (p_rec, 0);                                                       CHKERRQ (ierr);
            ierr = solver_interface->setInitialGuess (0.);
            PCOUT << "SOLVING INTERPOLATION WITH IC" << std::endl;
            ierr = solver_interface->solveInterpolation (c_0, p_rec, tumor->phi_, n_misc); //interpolates c_0 or data <---- CHECK
            PCOUT << " --------------  INTERPOLATED P -----------------\n";
            if (procid == 0) {
                ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
            }
            PCOUT << " --------------  -------------- -----------------\n";
            PCOUT << "Interpolation complete; exiting solver...\n";
        } else {
            bool flag_diff = false;
            if (n_misc->regularization_norm_ == L1 && n_misc->diffusivity_inversion_ == true) {
              n_misc->diffusivity_inversion_ = false;
              flag_diff = true;
            }

            if (solve_rho_k_only_flag) {
                if (!warmstart_p) {PCOUT << "Error: c(0) needs to be set, read in p and Gaussians. exiting solver...\n"; exit(1);}
                ierr = solver_interface->solveInverseReacDiff (p_rec, data, nullptr);     // solve tumor inversion only for rho and k, read in c(0)
            } else if (flag_cosamp) {
                ierr = solver_interface->solveInverseCoSaMp (p_rec, data, nullptr);     // solve tumor inversion using cosamp
            } else {
                ierr = solver_interface->solveInverse (p_rec, data, nullptr);           // solve tumor inversion
            }

            //if L1, then solve for sparse components using weigted L2

            if (n_misc->regularization_norm_ == L1) {
                out_params = solver_interface->getSolverOutParams ();
                n_gist = (int) out_params[0];
                n_misc->regularization_norm_ = wL2; //Set W-L2
                if (flag_diff) {
                    n_misc->diffusivity_inversion_ = true;  //if we want diffusivity inversion after L1
                    n_misc->k_ = 0.0;
                }
            }
            else {
                if (!flag_cosamp) {
                    out_params = solver_interface->getSolverOutParams ();
                    n_newton = (int) out_params[0];
                }
            }

            if (n_misc->regularization_norm_ == wL2) {
                ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
                PCOUT << "Reconstructed P Norm: " << prec_norm << std::endl;
                if (n_misc->diffusivity_inversion_) {
                    ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
                    PCOUT << "k1: " << (n_misc->nk_ > 0 ? prec_ptr[n_misc->np_] : 0) << std::endl;
                    PCOUT << "k2: " << (n_misc->nk_ > 1 ? prec_ptr[n_misc->np_ + 1] : 0) << std::endl;
                    PCOUT << "k3: " << (n_misc->nk_ > 2 ? prec_ptr[n_misc->np_ + 2] : 0) << std::endl;
                    ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
                }
                ierr = computeError (l2_rel_error, error_norm_c0, p_rec, data_nonoise, data, c_0, solver_interface, n_misc);
                PCOUT << "\nL2 Error in Reconstruction: " << l2_rel_error << std::endl;
                PCOUT << " --------------  RECONST P -----------------\n";
                if (procid == 0) {
                    ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);                   CHKERRQ (ierr);
                }
                PCOUT << " --------------  -------------- -----------------\n";
                PCOUT << " \n\n --------------- W-L2 solve for sparse components ----------------------\n\n\n";

                ierr = solver_interface->resetTaoSolver ();                                 //Reset tao objects
                ierr = solver_interface->setInitialGuess (p_rec);
                ierr = solver_interface->solveInverse (p_rec, data, nullptr);

                out_params = solver_interface->getSolverOutParams ();
                n_newton = (int) out_params[1];
            }

            ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
            PCOUT << "Reconstructed P Norm: " << prec_norm << std::endl;
            if (n_misc->diffusivity_inversion_) {
                ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
                PCOUT << "k1: " << (n_misc->nk_ > 0 ? prec_ptr[n_misc->np_] : 0) << std::endl;
                PCOUT << "k2: " << (n_misc->nk_ > 1 ? prec_ptr[n_misc->np_ + 1] : 0) << std::endl;
                PCOUT << "k3: " << (n_misc->nk_ > 2 ? prec_ptr[n_misc->np_ + 2] : 0) << std::endl;
                ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
            }
            ierr = computeError (l2_rel_error, error_norm_c0, p_rec, data_nonoise, data, c_0, solver_interface, n_misc);
            PCOUT << "\nL2 Error in Reconstruction: " << l2_rel_error << std::endl;
            PCOUT << " --------------  RECONST P -----------------\n";
            if (procid == 0) {
                ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);                   CHKERRQ (ierr);
            }
            PCOUT << " --------------  -------------- -----------------\n";

            ierr = computeSegmentation(tumor, n_misc);      // Writes segmentation with c0 and c1

            std::stringstream sstm;
            sstm << n_misc->writepath_ .str().c_str() << "reconP.dat";
            std::ofstream ofile (sstm.str().c_str());
            //write reconstructed p into text file
            if (procid == 0) {
                ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
                int np = n_misc->np_;
                int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
                for (int i = 0; i < np + nk; i++)
                    ofile << prec_ptr[i] << std::endl;
                ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
            }
            ofile.close ();
            // write p to bin file
            std::string fname = n_misc->writepath_ .str() + "p-final.bin";
            writeBIN(p_rec, fname);

            if (n_misc->predict_flag_) {
                PCOUT << "Predicting future tumor growth..." << std::endl;
                // predict tumor growth using inverted parameter values
                // set dt and nt to synthetic values to ensure best accuracy
                n_misc->dt_ = dt_data;
                n_misc->nt_ = (int) (1.5 / dt_data);
                // reset time history
                ierr = solver_interface->getPdeOperators()->resizeTimeHistory (n_misc);
                // apply IC to tumor c0
                ierr = tumor->phi_->apply (tumor->c_0_, p_rec);
                // reaction and diffusion coefficient already set correctly at the end of the
                // optimizer
                ierr = solver_interface->getPdeOperators()->solveState (0);  // time histroy is stored in
                                                                             // pde_operators->c_
                // Write out t = 1.2 and t = 1.5 -- hard coded for now. TODO: make it a user parameter(?)
                dataOut (solver_interface->getPdeOperators()->c_[(int) (1.2 / dt_data)], n_misc, "cPrediction_[t=1.2].nc");
                dataOut (solver_interface->getPdeOperators()->c_[(int) (1.5 / dt_data)], n_misc, "cPrediction_[t=1.5].nc");
                dataOut (solver_interface->getPdeOperators()->c_[(int) (1.0 / dt_data)], n_misc, "cPrediction_[t=1.0].nc");

                PCOUT << "Prediction complete for t = 1.2 and t = 1.5\n";
            }
        }

    }


    self_exec_time += MPI_Wtime ();
    accumulateTimers (n_misc->timers_, timers, self_exec_time);
    e1.addTimings (timers);
    e1.stop ();
    EventRegistry::finalize ();
    if (procid == 0) {
        EventRegistry r;
        r.print ();
        r.print ("EventsTimings.log", true);
    }

    ierr = VecDestroy (&c_0);               CHKERRQ (ierr);
    ierr = VecDestroy (&data);              CHKERRQ (ierr);
    ierr = VecDestroy (&p_rec);             CHKERRQ (ierr);
    ierr = VecDestroy (&data_nonoise);      CHKERRQ (ierr);

    if (gm != nullptr) {ierr = VecDestroy (&gm);                 CHKERRQ (ierr);}
    if (wm != nullptr) {ierr = VecDestroy (&wm);                 CHKERRQ (ierr);}
    if (csf != nullptr) {ierr = VecDestroy (&csf);               CHKERRQ (ierr);}
    if (bg != nullptr) {ierr = VecDestroy (&bg);                 CHKERRQ (ierr);}
    if (use_custom_obs_mask) {ierr = VecDestroy (&obs_mask);     CHKERRQ (ierr);}
    if (use_data_comps) {ierr = VecDestroy (&data_components);   CHKERRQ (ierr);}
    if (read_support_data)   {ierr = VecDestroy (&support_data); CHKERRQ (ierr);}


}
/* --------------------------------------------------------------------------------------------------------------*/
    accfft_destroy_plan (plan);
    accfft_cleanup();
    MPI_Comm_free(&c_comm);
    ierr = PetscFinalize ();
    return ierr;
}

PetscErrorCode setDistMeasuresFullObj (std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<HealthyProbMaps> h_maps, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    solver_interface->setDistMeassureSimulationGeoImages (h_maps->wm, h_maps->gm, h_maps->csf, nullptr, h_maps->bg);

    Vec temp;
    ierr = VecDuplicate (data, &temp);                   CHKERRQ (ierr);
    ierr = VecCopy (data, temp);                         CHKERRQ (ierr);
    ierr = VecShift (temp, -1.0);                        CHKERRQ (ierr); // (1 - data) is used to scale data healthy tissues
    // Copy atlas to data healthy prob
    ierr = VecCopy (h_maps->gm, h_maps->gm_data);     CHKERRQ (ierr);
    ierr = VecCopy (h_maps->wm, h_maps->wm_data);     CHKERRQ (ierr);
    ierr = VecCopy (h_maps->csf, h_maps->csf_data);   CHKERRQ (ierr);
    ierr = VecCopy (h_maps->bg, h_maps->bg_data);     CHKERRQ (ierr);
    ierr = VecPointwiseMult (h_maps->gm_data, h_maps->gm_data, temp);       CHKERRQ (ierr);
    ierr = VecPointwiseMult (h_maps->wm_data, h_maps->wm_data, temp);       CHKERRQ (ierr);
    ierr = VecPointwiseMult (h_maps->csf_data, h_maps->csf_data, temp);     CHKERRQ (ierr);
    ierr = VecPointwiseMult (h_maps->bg_data, h_maps->bg_data, temp);       CHKERRQ (ierr);

    solver_interface->setDistMeassureTargetDataImages (h_maps->wm_data, h_maps->gm_data, h_maps->csf_data, nullptr, h_maps->bg_data);
    solver_interface->setDistMeassureDiffImages (h_maps->xi_wm, h_maps->xi_gm, h_maps->xi_csf, nullptr, h_maps->xi_bg);

    ierr = VecDestroy (&temp);                      CHKERRQ (ierr);

    PetscFunctionReturn (0);
}

PetscErrorCode createMFData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);             CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                        CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);              CHKERRQ (ierr);
        ierr = VecSetFromOptions (p_rec);                                   CHKERRQ (ierr);
    #endif

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = VecSetFromOptions (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::array<double, 3> cm;

    n_misc->user_cms_.clear ();

    //near
    if (n_misc->testcase_ == BRAINNEARMF) {
        cm[0] = 2 * M_PI / 128 * 56;//82  //Z
        cm[1] = 2 * M_PI / 128 * 68;//64  //Y
        cm[2] = 2 * M_PI / 128 * 72;//52  //X

        n_misc->user_cms_.push_back (cm[0]);
        n_misc->user_cms_.push_back (cm[1]);
        n_misc->user_cms_.push_back (cm[2]);
        n_misc->user_cms_.push_back (1.);
    }

    // far
    if (n_misc->testcase_ == BRAINFARMF) {
        cm[0] = 2 * M_PI / 128 * 60;//82  //Z
        cm[1] = 2 * M_PI / 128 * 44;//64  //Y
        cm[2] = 2 * M_PI / 128 * 76;//52  //X

        n_misc->user_cms_.push_back (cm[0]);
        n_misc->user_cms_.push_back (cm[1]);
        n_misc->user_cms_.push_back (cm[2]);
        n_misc->user_cms_.push_back (1.);
    }
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->phi_->setGaussians (cm, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = tumor->phi_->setValues (tumor->mat_prop_);
    ierr = tumor->setTrueP (n_misc);
    PCOUT << " --------------  SYNTHETIC TRUE P -----------------\n";
    if (procid == 0) {
        ierr = VecView (tumor->p_true_, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
    }
    PCOUT << " --------------  -------------- -----------------\n";
    ierr = tumor->phi_->apply (c_0, tumor->p_true_);                        CHKERRQ (ierr);

    Vec c_temp;
    ierr = VecDuplicate (c_0, &c_temp);                                     CHKERRQ (ierr);
    ierr = VecSet (c_temp, 0.);                                             CHKERRQ (ierr);

    double scaling = 1.;
    // Second tumor location
    // near
    if (n_misc->testcase_ == BRAINNEARMF) {
        cm[0] = 2 * M_PI / 128 * 64;//82
        cm[1] = 2 * M_PI / 128 * 72;//64
        cm[2] = 2 * M_PI / 128 * 76;//52

        n_misc->user_cms_.push_back (cm[0]);
        n_misc->user_cms_.push_back (cm[1]);
        n_misc->user_cms_.push_back (cm[2]);
        n_misc->user_cms_.push_back (scaling);
    }
    // far
    if (n_misc->testcase_ == BRAINFARMF) {
        cm[0] = 2 * M_PI / 128 * 72;//82
        cm[1] = 2 * M_PI / 128 * 80;//64
        cm[2] = 2 * M_PI / 128 * 76;//52

        n_misc->user_cms_.push_back (cm[0]);
        n_misc->user_cms_.push_back (cm[1]);
        n_misc->user_cms_.push_back (cm[2]);
        n_misc->user_cms_.push_back (scaling);
    }
    ierr = tumor->phi_->setGaussians (cm, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = tumor->phi_->setValues (tumor->mat_prop_);
    // ierr = tumor->setTrueP (n_misc, 0.75);
    ierr = tumor->setTrueP (n_misc, scaling);
    PCOUT << " --------------  SYNTHETIC TRUE P -----------------\n";
    if (procid == 0) {
        ierr = VecView (tumor->p_true_, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
    }
    PCOUT << " --------------  -------------- -----------------\n";
    ierr = tumor->phi_->apply (c_temp, tumor->p_true_);                     CHKERRQ (ierr);

    ierr = VecAXPY (c_0, 1.0, c_temp);                                      CHKERRQ (ierr);
    ierr = VecDestroy (&c_temp);                                            CHKERRQ (ierr);

    double max, min;
    ierr = VecMax (c_0, NULL, &max);                                       CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                       CHKERRQ (ierr);

    // ierr = VecScale (c_0, 1.0 / max);                                      CHKERRQ (ierr);

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    if (n_misc->writeOutput_)
        dataOut (c_0, n_misc, "c0True.nc");

    ierr = VecMax (c_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Data IC Max and Min : " << max << " " << min << std::endl;

    ierr = solver_interface->solveForward (c_t, c_0);   //Observation operator is applied in InvSolve ()

    // ierr = tumor->obs_->apply (c_t, c_t);

    if (n_misc->writeOutput_) {
        dataOut (c_t, n_misc, "data.nc");
    }

    PetscFunctionReturn (0);
}

PetscErrorCode readData (Vec &data, Vec &support_data, Vec &data_components, Vec &c_0, Vec &p_rec, std::shared_ptr<NMisc> n_misc, char *data_path, char* support_data_path, char* data_comp_path) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    bool read_support_data = (support_data_path != NULL && strlen(support_data_path) > 0); // path set?
    ierr = VecCreate (PETSC_COMM_WORLD, &data);                             CHKERRQ (ierr);
    ierr = VecSetSizes (data, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = VecSetFromOptions (data);                                        CHKERRQ (ierr);
    ierr = VecDuplicate (data, &c_0);                                       CHKERRQ (ierr);
    if (read_support_data) {
      ierr = VecDuplicate (data, &support_data);                            CHKERRQ (ierr);
    }

    //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);                 CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
        ierr = VecSetFromOptions (p_rec);                                       CHKERRQ (ierr);
    #endif

    dataIn (data, n_misc, data_path);
    if(read_support_data){
      dataIn (support_data, n_misc, support_data_path);
    } else {
      support_data = data;
    }

    if((data_comp_path != NULL && strlen(data_comp_path) > 0)) {
      ierr = VecCreate (PETSC_COMM_WORLD, &data_components);                       CHKERRQ (ierr);
      ierr = VecSetSizes (data_components, n_misc->n_local_, n_misc->n_global_);   CHKERRQ (ierr);
      ierr = VecSetFromOptions (data_components);                                  CHKERRQ (ierr);
      ierr = VecSet (data_components, 0.);                                         CHKERRQ (ierr);
      dataIn (data_components, n_misc, data_comp_path);
    }

    // Smooth the data
    double sigma_smooth = n_misc->smoothing_factor_ * 2 * M_PI / n_misc->n_[0];
    double *data_ptr;
    ierr = VecGetArray (data, &data_ptr);                                       CHKERRQ (ierr);
    ierr = weierstrassSmoother (data_ptr, data_ptr, n_misc, sigma_smooth);
    ierr = VecRestoreArray (data, &data_ptr);                                   CHKERRQ (ierr);

    // size_t pos;
    // std::ifstream ifile;
    // std::string c0_path (data_path);
    // pos = c0_path.find ("data.nc");
    // c0_path.replace (pos, 9, "c0True.nc");
    // ifile.open (c0_path);

    // if (ifile) {
    //     dataIn (c_0, n_misc, c0_path.c_str());
    // } else {
    ierr = VecSet (c_0, 0.);        CHKERRQ (ierr);
    // }

    PetscFunctionReturn (0);
}

PetscErrorCode readObsFilter (Vec &obs_mask, std::shared_ptr<NMisc> n_misc, char *obs_mask_path) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &obs_mask);                       CHKERRQ (ierr);
    ierr = VecSetSizes (obs_mask, n_misc->n_local_, n_misc->n_global_);   CHKERRQ (ierr);
    ierr = VecSetFromOptions (obs_mask);                                  CHKERRQ (ierr);
    ierr = VecSet (obs_mask, 0.);                                         CHKERRQ (ierr);

    dataIn (obs_mask, n_misc, obs_mask_path);
    // double *obs_mask_ptr;
    // ierr = VecGetArray (obs_mask, &obs_mask_ptr);                         CHKERRQ (ierr);
    // for (int i = 0; i < n_misc->n_local_; i++) {
        // if (inversed) {obs_mask_ptr[i] = (obs_mask_ptr[i] > 0) ? 0.0 : 1.0;}
        // else          {obs_mask_ptr[i] = (obs_mask_ptr[i] > 0) ? 1.0 : 0.0;}
    // }
    // ierr = VecRestoreArray (obs_mask, &obs_mask_ptr);                     CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode readAtlas (Vec &wm, Vec &gm, Vec &glm, Vec &csf, Vec &bg, std::shared_ptr<NMisc> n_misc, char *gm_path, char *wm_path, char *csf_path, char *glm_path) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &gm);                             CHKERRQ (ierr);
    ierr = VecSetSizes (gm, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = VecSetFromOptions (gm);                                        CHKERRQ (ierr);

    ierr = VecDuplicate (gm, &wm);                                        CHKERRQ (ierr);
    ierr = VecDuplicate (gm, &csf);                                       CHKERRQ (ierr);
    ierr = VecDuplicate (gm, &bg);                                        CHKERRQ (ierr);
    ierr = VecSet (bg, 0.);                                               CHKERRQ (ierr);

    dataIn (wm, n_misc, wm_path);
    dataIn (gm, n_misc, gm_path);
    dataIn (csf, n_misc, csf_path);

    double sigma_smooth = n_misc->smoothing_factor_ * 2 * M_PI / n_misc->n_[0];
    double *gm_ptr, *wm_ptr, *csf_ptr, *bg_ptr;
    ierr = VecGetArray (gm, &gm_ptr);                    CHKERRQ (ierr);
    ierr = VecGetArray (wm, &wm_ptr);                    CHKERRQ (ierr);
    ierr = VecGetArray (csf, &csf_ptr);                  CHKERRQ (ierr);
    ierr = VecGetArray (bg, &bg_ptr);                    CHKERRQ (ierr);

    ierr = weierstrassSmoother (gm_ptr, gm_ptr, n_misc, sigma_smooth);
    ierr = weierstrassSmoother (wm_ptr, wm_ptr, n_misc, sigma_smooth);
    ierr = weierstrassSmoother (csf_ptr, csf_ptr, n_misc, sigma_smooth);
    // Set bg prob as 1 - sum
    for (int i = 0; i < n_misc->n_local_; i++) {
        bg_ptr[i] = 1.0 - (gm_ptr[i] + wm_ptr[i] + csf_ptr[i]);
    }

    ierr = VecRestoreArray (gm, &gm_ptr);                    CHKERRQ (ierr);
    ierr = VecRestoreArray (wm, &wm_ptr);                    CHKERRQ (ierr);
    ierr = VecRestoreArray (csf, &csf_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (bg, &bg_ptr);                    CHKERRQ (ierr);

    PetscFunctionReturn (0);
}

PetscErrorCode applyLowFreqNoise (Vec data, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    srand (time(NULL));
    double random_noise;
    double noise_level = n_misc->low_freq_noise_scale_;
    double mag = 0.;
    int64_t x_global, y_global, z_global, x_symm, y_symm, z_symm, global_index;
    double freq = 0;
    double amplitude = 0.;

    int *osize = n_misc->osize_;
    int *ostart = n_misc->ostart_;
    int *isize = n_misc->isize_;
    int *istart = n_misc->istart_;
    MPI_Comm c_comm = n_misc->c_comm_;
    int *n = n_misc->n_;
    accfft_plan *plan = n_misc->plan_;

    // Get the fourier transform of data;
    double *d_ptr;
    ierr = VecGetArray (data, &d_ptr);          CHKERRQ (ierr);

    // // remove small aliasing errors
    // for (int i = 0; i < n_misc->n_local_; i++) {
    //     if (d_ptr[i] < 1E-4) {
    //         d_ptr[i] = 0.;
    //     }
    // }

    int alloc_max = accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    Complex *data_hat;
    double *freq_scaling;
    data_hat = (Complex*) accfft_alloc (alloc_max);
    freq_scaling = (double*) accfft_alloc (alloc_max);
    accfft_execute_r2c (plan, d_ptr, data_hat);
    MPI_Barrier (c_comm);

    double *data_hat_mag;
    double *d;
    data_hat_mag = (double*) accfft_alloc(alloc_max);
    double wx, wy, wz;

    int64_t ptr;
    // Find the amplitude of the signal (data)
    for (int i = 0; i < osize[0]; i++) {
        for (int j = 0; j < osize[1]; j++) {
            for (int k = 0; k < osize[2]; k++) {
                ptr = i * osize[1] * osize[2] + j * osize[2] + k;
                d = data_hat[ptr];
                data_hat_mag[ptr] = std::sqrt(d[0] * d[0] + d[1] * d[1]); // amplitude compute
                if (data_hat_mag[ptr] > amplitude)
                    amplitude = data_hat_mag[ptr];

                // populate freq: By symmetery X(N-k) = X(k)
                // instead of enforcing conjugate symmetery manually, just scale with only unique frequencies

                x_global = i + ostart[0];
                y_global = j + ostart[1];
                z_global = k + ostart[2];

                wx = (x_global > n[0] / 2) ? n[0] - x_global : x_global;
                wy = (y_global > n[1] / 2) ? n[1] - y_global : y_global;
                wz = (z_global > n[2] / 2) ? n[2] - z_global : z_global;

                if (wx == 0 && wy == 0 && wz == 0)
                    freq_scaling[ptr] = 1.;
                else
                    freq_scaling[ptr] = (1.0 / (wx * wx + wy * wy + wz * wz));
            }
        }
    }
    // allreduce to find the amplitude of the freq
    double global_amplitude = 0.;
    MPI_Allreduce (&amplitude, &global_amplitude, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier (c_comm);

    // Now add power law noise
    for (int i = 0; i < osize[0]; i++) {
        for (int j = 0; j < osize[1]; j++) {
            for (int k = 0; k < osize[2]; k++) {
                ptr = i * osize[1] * osize[2] + j * osize[2] + k;
                d = data_hat[ptr];
                mag = data_hat_mag[ptr];
                random_noise = (double)rand() / (double)RAND_MAX;
                data_hat_mag[ptr] += noise_level * random_noise * global_amplitude * freq_scaling[ptr];
                // data_hat_mag[ptr] += noise_level * random_noise * global_amplitude * std::sqrt(freq_scaling[ptr]);
                // change data_hat accordingly -- this will change only the unique freq
                if (mag != 0) { // check for non zero components
                    d[0] *= (1.0 / mag) * data_hat_mag[ptr];
                    d[1] *= (1.0 / mag) * data_hat_mag[ptr];
                }
            }
        }
    }

    MPI_Barrier(c_comm);
    accfft_execute_c2r(plan, data_hat, d_ptr);
    MPI_Barrier(c_comm);

    for (int i = 0; i < n_misc->n_local_; i++)
        d_ptr[i] /= n[0] * n[1] * n[2];

    accfft_free (data_hat);
    accfft_free (freq_scaling);
    accfft_free (data_hat_mag);
    ierr = VecRestoreArray (data, &d_ptr);              CHKERRQ (ierr);

    PetscFunctionReturn (0);
}


PetscErrorCode computeError (double &error_norm, double &error_norm_c0, Vec p_rec, Vec data, Vec data_obs, Vec c_0_true, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Vec c_rec_0, c_rec;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();

    double data_norm;
    ierr = VecDuplicate (data, &c_rec_0);                                   CHKERRQ (ierr);
    ierr = VecDuplicate (data, &c_rec);                                     CHKERRQ (ierr);

    ierr = tumor->phi_->apply (c_rec_0, p_rec);

    double *c0_ptr;

    if (n_misc->model_ == 2) {
        ierr = VecGetArray (c_rec_0, &c0_ptr);                              CHKERRQ (ierr);
        for (int i = 0; i < n_misc->n_local_; i++) {
            c0_ptr[i] = 1 / (1 + exp(-c0_ptr[i] + n_misc->exp_shift_));
        }
        ierr = VecRestoreArray (c_rec_0, &c0_ptr);                          CHKERRQ (ierr);
    }

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_rec_0, n_misc);
    #endif

    if (n_misc->writeOutput_)
        dataOut (c_rec_0, n_misc, "c0Recon.nc");

    ierr = solver_interface->solveForward (c_rec, c_rec_0);

    double max, min;
    ierr = VecMax (c_rec, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_rec, NULL, &min);                                      CHKERRQ (ierr);

    // if (n_misc->writeOutput_)
        // dataOut (c_rec, n_misc, "cReconBeforeObservation.nc");

    Vec obs_c_rec;
    ierr = VecDuplicate (c_rec, &obs_c_rec);                                CHKERRQ (ierr);
    ierr = VecCopy (c_rec, obs_c_rec);                                      CHKERRQ (ierr);

    ierr = tumor->obs_->apply (obs_c_rec, obs_c_rec);   //Apply observation to reconstructed C to compare
                                                //values to data ?
                                                // S: obs should not be applied because we want to see how the model captures the
                                                // true boundaries

    PCOUT << "\nC Reconstructed Max and Min : " << max << " " << min << std::endl;


    if (n_misc->writeOutput_)
        dataOut (c_rec, n_misc, "cRecon.nc");

    ierr = VecMax (c_rec_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_rec_0, NULL, &min);                                      CHKERRQ (ierr);

    ierr = VecAXPY (c_rec_0, -1.0, c_0_true);                               CHKERRQ (ierr);
    ierr = VecNorm (c_0_true, NORM_2, &data_norm);                          CHKERRQ (ierr);
    ierr = VecNorm (c_rec_0, NORM_2, &error_norm_c0);                       CHKERRQ (ierr);

    error_norm_c0 /= data_norm;
    PCOUT << "Error norm in IC: " << error_norm_c0 << std::endl;

    ierr = VecAXPY (c_rec, -1.0, data);                                     CHKERRQ (ierr);
    ierr = VecNorm (data, NORM_2, &data_norm);                              CHKERRQ (ierr);
    ierr = VecNorm (c_rec, NORM_2, &error_norm);                            CHKERRQ (ierr);

    PCOUT << "Data mismatch: " << error_norm << std::endl;

    error_norm /= data_norm;

    double obs_c_norm, obs_data_norm;

    ierr = VecAXPY (obs_c_rec, -1.0, data_obs);                             CHKERRQ (ierr);
    ierr = VecNorm (obs_c_rec, NORM_2, &obs_c_norm);                        CHKERRQ (ierr);
    ierr = VecNorm (data_obs, NORM_2, &obs_data_norm);                      CHKERRQ (ierr);

    obs_c_norm /= obs_data_norm;

    PCOUT << "L2 rel error at observation points: " << obs_c_norm << std::endl;

    ierr = VecDestroy (&obs_c_rec);                                         CHKERRQ (ierr);


    // compute weighted l2 error norm for c0
    // first calculate distance of each gaussian from ground truth
    Vec weights, p_true_w, p_diff_w, temp;
    ierr = VecDuplicate (p_rec, &weights);                                  CHKERRQ (ierr);
    ierr = VecDuplicate (p_rec, &p_true_w);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (p_rec, &p_diff_w);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (p_rec, &temp);                                     CHKERRQ (ierr);
    ierr = VecSet (weights, 0.);                                            CHKERRQ (ierr);
    ierr = VecSet (p_true_w, 0.);                                           CHKERRQ (ierr);
    ierr = VecSet (p_diff_w, 0.);                                           CHKERRQ (ierr);
    ierr = VecSet (temp, 0.);                                               CHKERRQ (ierr);

    double *w_ptr, *p_true_ptr;
    ierr = VecGetArray (weights, &w_ptr);                                   CHKERRQ (ierr);
    ierr = VecGetArray (p_true_w, &p_true_ptr);                             CHKERRQ (ierr);

    std::vector<double> dist (n_misc->user_cms_.size());
    double d = 0.;
    int flg = 0;
    for (int i = 0; i < n_misc->np_; i++) {
        dist.clear();
        for (int j = 0; j < n_misc->user_cms_.size() / 4; j++) {
            d = myDistance (&n_misc->user_cms_[4 * j], &tumor->phi_->centers_[3 * i]);
            dist.push_back (d);  // find the distance btw current gaussian and all the ground truth activations
                                 // note: the ground truth has an additional activation, hence the 4 * j

            // snafu for 64
            // if (d < .2) {
            if (d < 1E-5) {
                // this is one of the ground truth gaussians
                p_true_ptr[i] = n_misc->user_cms_[4 * j + 3];
                // flg = 1;
            }
        }
        w_ptr[i] = *(std::min_element (dist.begin(), dist.end()));      // sets the weight to the distance to the nearest ground truth activation
        w_ptr[i] += n_misc->h_[0];                                      // to avoid division by zero

        // snafu for 64
        // if (flg == 1) break;
    }

    std::stringstream sstm;
    sstm << n_misc->writepath_ .str().c_str() << "trueP.dat";
    std::ofstream ofile;
    ofile.open (sstm.str().c_str());
    if (procid == 0) {
        for (int i = 0; i < n_misc->np_; i++)
            ofile << p_true_ptr[i] << std::endl;
    }
    ofile.flush();
    ofile.close ();

    ierr = VecRestoreArray (weights, &w_ptr);                               CHKERRQ (ierr);
    ierr = VecRestoreArray (p_true_w, &p_true_ptr);                         CHKERRQ (ierr);

    // // snafu: if using ground truth from somewhere else (for example, downsampled from a higher resolution)
    // // interpolate c0 to the best capacity with the current basis. This is as close as we can get to the
    // // ground truth
    // PCOUT << " --------------  -------------- -----------------\n";
    // ierr = VecSet (p_true_w, 0);                                                       CHKERRQ (ierr);
    // ierr = solver_interface->setInitialGuess (0.);
    // PCOUT << "SOLVING INTERPOLATION WITH IC" << std::endl;
    // ierr = solver_interface->solveInterpolation (c_0_true, p_true_w, tumor->phi_, n_misc); //interpolates c_0 o
    // PCOUT << " --------------  INTERPOLATED P FOR IC -----------------\n";
    // if (procid == 0) {
    //     ierr = VecView (p_true_w, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
    // }
    // PCOUT << " --------------  -------------- -----------------\n";


    double p_wL2, p_diff_wL2;
    ierr = VecPointwiseMult (temp, p_true_w, weights);                  CHKERRQ (ierr);
    ierr = VecDot (p_true_w, temp, &p_wL2);                             CHKERRQ (ierr);
    p_wL2 = std::sqrt (p_wL2);
    ierr = VecCopy (p_true_w, p_diff_w);                                CHKERRQ (ierr);
    ierr = VecAXPY (p_diff_w, -1.0, p_rec);                             CHKERRQ (ierr);  // diff in p
    double l1_norm_diff, l1_norm_p;
    ierr = VecNorm (p_diff_w, NORM_1, &l1_norm_diff);   CHKERRQ (ierr);
    ierr = VecNorm (p_true_w, NORM_1, &l1_norm_p);      CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp, p_diff_w, weights);                  CHKERRQ (ierr);
    ierr = VecDot (p_diff_w, temp, &p_diff_wL2);                        CHKERRQ (ierr);
    p_diff_wL2 = std::sqrt (p_diff_wL2);

    double dist_err_c0 = p_diff_wL2 / p_wL2;
    double l1_err = l1_norm_diff / l1_norm_p;

    ierr = VecDestroy (&weights);        CHKERRQ (ierr);
    ierr = VecDestroy (&p_true_w);       CHKERRQ (ierr);
    ierr = VecDestroy (&p_diff_w);       CHKERRQ (ierr);
    ierr = VecDestroy (&temp);           CHKERRQ (ierr);

    double *p_rec_ptr;
    ierr = VecGetArray (p_rec, &p_rec_ptr);     CHKERRQ (ierr);
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    double k1, k2, k3, r1, r2, r3;
    k1 = 0.; k2 = 0.; k3 = 0.;
    if (n_misc->diffusivity_inversion_) {
        k1 = p_rec_ptr[n_misc->np_];
        if (n_misc->nk_ > 1)
          k2 = p_rec_ptr[n_misc->np_ + 1];
        if (n_misc->nk_ > 2)
          k3 = p_rec_ptr[n_misc->np_ + 2];
    }

    // if (n_misc->reaction_inversion_) {
    //     r1 = p_rec_ptr[n_misc->np_ + n_misc->nk_];
    //     r2 = (n_misc->nr_ > 1) ? p_rec_ptr[n_misc->np_ + n_misc->nk_ + 1] : 0;
    //     r3 = (n_misc->nr_ > 2) ? p_rec_ptr[n_misc->np_ + n_misc->nk_ + 2] : 0;
    // }

    PCOUT << "P distance error: " << dist_err_c0 << std::endl;
    PCOUT << "P l1 norm: " << l1_err << std::endl;

    std::stringstream ss_out;
    ss_out << n_misc->writepath_ .str().c_str() << "info.dat";
    std::ofstream opfile;
    opfile.open (ss_out.str().c_str());
    if (procid == 0) {
        opfile << "rho k c1_rel c0_rel c0_dist \n";
        opfile << n_misc->rho_ << " " << k1 << " " << error_norm << " "
               << error_norm_c0 << " " << dist_err_c0 << std::endl;
    }
    opfile.flush();
    opfile.close ();

    ierr = VecRestoreArray (p_rec, &p_rec_ptr);     CHKERRQ (ierr);


    ierr = VecDestroy (&c_rec_0); CHKERRQ (ierr);
    ierr = VecDestroy (&c_rec); CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);                 CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
        ierr = VecSetFromOptions (p_rec);                                       CHKERRQ (ierr);
    #endif

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = VecSetFromOptions (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->setTrueP (n_misc);
    PCOUT << " --------------  SYNTHETIC TRUE P -----------------\n";
    if (procid == 0) {
        ierr = VecView (tumor->p_true_, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
    }
    PCOUT << " --------------  -------------- -----------------\n";
    ierr = tumor->phi_->apply (c_0, tumor->p_true_);

    double *c0_ptr;

    if (n_misc->model_ == 2) {
        ierr = VecGetArray (c_0, &c0_ptr);                                  CHKERRQ (ierr);
        for (int i = 0; i < n_misc->n_local_; i++) {
            c0_ptr[i] = 1 / (1 + exp(-c0_ptr[i] + n_misc->exp_shift_));
        }
        ierr = VecRestoreArray (c_0, &c0_ptr);                              CHKERRQ (ierr);
    }

    double max, min;
    ierr = VecMax (c_0, NULL, &max);                                       CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                       CHKERRQ (ierr);


    // Artificially scale the IC -- Keep off.
    // ierr = VecScale (c_0, 1.0 / max);                                      CHKERRQ (ierr);

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    if (n_misc->writeOutput_)
        dataOut (c_0, n_misc, "c0True.nc");

    ierr = VecMax (c_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Data IC Max and Min : " << max << " " << min << std::endl;

    ierr = solver_interface->solveForward (c_t, c_0);   //Observation operator is applied in InvSolve ()

    ierr = VecMax (c_t, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_t, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Data Max and Min (Before observation) : " << max << " " << min << std::endl;

    // ierr = tumor->obs_->apply (c_t, c_t);

    if (n_misc->writeOutput_) {
        dataOut (c_t, n_misc, "dataBeforeObservation.nc");
    }

    PetscFunctionReturn (0);
}


PetscErrorCode computeSegmentation(std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec max;
    ierr = VecDuplicate(tumor->c_0_, &max);                                 CHKERRQ(ierr);
    ierr = VecSet(max, 0);                                                  CHKERRQ(ierr);

    // compute max of gm, wm, csf, bg, tumor
    std::vector<double> v;
    std::vector<double>::iterator max_component;
    double *bg_ptr, *gm_ptr, *wm_ptr, *csf_ptr, *c_ptr, *max_ptr;
    ierr = VecGetArray(tumor->mat_prop_->bg_, &bg_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray(tumor->mat_prop_->gm_, &gm_ptr);                    CHKERRQ(ierr);
    ierr = VecGetArray(tumor->mat_prop_->wm_, &wm_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray(tumor->mat_prop_->csf_, &csf_ptr);                   CHKERRQ(ierr);
    ierr = VecGetArray(tumor->c_0_, &c_ptr);                                CHKERRQ(ierr);
    ierr = VecGetArray(max, &max_ptr);                                      CHKERRQ(ierr);

    // segmentation for c0
    for (int i = 0; i < n_misc->n_local_; i++) {
        // Kill the material properties in regions of tumor
        bg_ptr[i]  = bg_ptr[i]  * (1 - c_ptr[i]);
        wm_ptr[i]  = wm_ptr[i]  * (1 - c_ptr[i]);
        csf_ptr[i] = csf_ptr[i] * (1 - c_ptr[i]);
        gm_ptr[i]  = gm_ptr[i]  * (1 - c_ptr[i]);

        v.push_back(bg_ptr[i]);
        v.push_back(gm_ptr[i]);
        v.push_back(wm_ptr[i]);
        v.push_back(csf_ptr[i]);
        v.push_back(c_ptr[i]);

        max_component = std::max_element(v.begin(), v.end());
        max_ptr[i] = std::distance(v.begin(), max_component);

        v.clear();
    }


    double sigma_smooth = 1.5 * M_PI / n_misc->n_[0];

    // ierr = weierstrassSmoother (max_ptr, max_ptr, n_misc, sigma_smooth);

    ierr = VecRestoreArray(max, &max_ptr);                                      CHKERRQ(ierr);

    if (n_misc->writeOutput_) {
        dataOut (max, n_misc, "seg0.nc");
    }

    // segmentatino for c1
    ierr = VecSet(max, 0);                                                  CHKERRQ(ierr);
    ierr = VecGetArray(tumor->c_t_, &c_ptr);                                CHKERRQ(ierr);
    ierr = VecGetArray(max, &max_ptr);                                      CHKERRQ(ierr);
    for (int i = 0; i < n_misc->n_local_; i++) {
        // Kill the material properties in regions of tumor
        bg_ptr[i] = bg_ptr[i] * (1 - c_ptr[i]);
        wm_ptr[i] = wm_ptr[i] * (1 - c_ptr[i]);
        csf_ptr[i] = csf_ptr[i] * (1 - c_ptr[i]);
        gm_ptr[i] = gm_ptr[i] * (1 - c_ptr[i]);

        v.push_back(bg_ptr[i]);
        v.push_back(gm_ptr[i]);
        v.push_back(wm_ptr[i]);
        v.push_back(csf_ptr[i]);
        v.push_back(c_ptr[i]);

        max_component = std::max_element(v.begin(), v.end());
        max_ptr[i] = std::distance(v.begin(), max_component);

        v.clear();
    }

    ierr = VecRestoreArray(tumor->mat_prop_->bg_, &bg_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor->mat_prop_->gm_, &gm_ptr);                    CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor->mat_prop_->wm_, &wm_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor->mat_prop_->csf_, &csf_ptr);                   CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor->c_0_, &c_ptr);                                CHKERRQ(ierr);

    // ierr = weierstrassSmoother (max_ptr, max_ptr, n_misc, sigma_smooth);


    ierr = VecRestoreArray(max, &max_ptr);                                      CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor->c_t_, &c_ptr);                                CHKERRQ(ierr);

    if (n_misc->writeOutput_) {
        dataOut (max, n_misc, "seg1.nc");
    }

    ierr = VecDestroy (&max);       CHKERRQ (ierr);


    PetscFunctionReturn(0);
}
