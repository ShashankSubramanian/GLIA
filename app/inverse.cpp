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
        if(gm_data  != nullptr) {ierr = VecDestroy (&gm_data); gm_data = nullptr;}
        if(wm_data  != nullptr) {ierr = VecDestroy (&wm_data); wm_data = nullptr;}
        if(csf_data != nullptr) {ierr = VecDestroy (&csf_data); csf_data = nullptr;}
        if(bg_data  != nullptr) {ierr = VecDestroy (&bg_data); bg_data = nullptr;}
        if(xi_gm    != nullptr) {ierr = VecDestroy (&xi_gm); xi_gm = nullptr;}
        if(xi_wm    != nullptr) {ierr = VecDestroy (&xi_wm); xi_wm = nullptr;}
        if(xi_csf   != nullptr) {ierr = VecDestroy (&xi_csf); xi_csf = nullptr;}
        if(xi_bg    != nullptr) {ierr = VecDestroy (&xi_bg); xi_bg = nullptr;}
    }
};

struct MData {
    Vec gm_, wm_, csf_, glm_;
    std::shared_ptr<NMisc> n_misc_;
    std::shared_ptr<SpectralOperators> spec_ops_;

    MData(Vec data, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) : n_misc_ (n_misc), spec_ops_ (spec_ops) {
        VecDuplicate(data, &gm_);
        VecDuplicate(data, &wm_);
        VecDuplicate(data, &csf_);
        VecDuplicate(data, &glm_);
    }

    PetscErrorCode readData(char *gm_path, char *wm_path, char *csf_path, char *glm_path) {
        PetscErrorCode ierr = 0;
        dataIn(gm_, n_misc_, gm_path);
        dataIn(wm_, n_misc_, wm_path);
        dataIn(csf_, n_misc_, csf_path);
        dataIn(glm_, n_misc_, glm_path);

        // ScalarType sigma_smooth = n_misc_->smoothing_factor_ * 2 * M_PI / n_misc_->n_[0];
        // ierr = spec_ops_->weierstrassSmoother (gm_, gm_, n_misc_, sigma_smooth);
        // ierr = spec_ops_->weierstrassSmoother (wm_, wm_, n_misc_, sigma_smooth);
        // ierr = spec_ops_->weierstrassSmoother (csf_, csf_, n_misc_, sigma_smooth);
        // ierr = spec_ops_->weierstrassSmoother (glm_, glm_, n_misc_, sigma_smooth);

        PetscFunctionReturn(ierr);
    }

    PetscErrorCode readData(Vec gm, Vec wm, Vec csf, Vec glm) {
        PetscErrorCode ierr = 0;
        ierr = VecCopy(gm, gm_);    CHKERRQ (ierr);
        ierr = VecCopy(wm, wm_);    CHKERRQ (ierr);
        ierr = VecCopy(csf, csf_);  CHKERRQ (ierr);
        ierr = VecCopy(glm, glm_);  CHKERRQ (ierr);
        PetscFunctionReturn(ierr);        
    }

    ~MData() {
        VecDestroy(&wm_);
        VecDestroy(&gm_);
        VecDestroy(&csf_);
        VecDestroy(&glm_);
    }
};

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, char*);
PetscErrorCode generateSinusoidalData (Vec &d, std::shared_ptr<NMisc> n_misc);
PetscErrorCode computeError (ScalarType &error_norm, ScalarType &error_norm_c0, Vec p_rec, Vec data, Vec data_obs, Vec c_0, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode readData (Vec &data, Vec &support_data, Vec &data_components, Vec &c_0, Vec &p_rec, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, char *data_path, char* support_data_path, char* data_comp_path);
PetscErrorCode readAtlas (Vec &wm, Vec &gm, Vec &glm, Vec &csf, Vec &bg, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, char*, char*, char*, char*);
PetscErrorCode readObsFilter (Vec &obs_mask, std::shared_ptr<NMisc> n_misc, char*);
PetscErrorCode createMFData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode setDistMeasuresFullObj (std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<HealthyProbMaps> h_maps, Vec);
PetscErrorCode computeSegmentation(std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops);
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
    ScalarType beta_user = -1.0;
    ScalarType rho_inv= -1.0;
    ScalarType k_inv = -1.0;
    int nt_inv = 0.0;
    ScalarType dt_inv = 0.0;
    ScalarType rho_data = -1.0;
    ScalarType k_data = -1.0;
    int nt_data = 0.0;
    ScalarType dt_data = 0.0;
    ScalarType data_comp_weights[10];
    int ncomp = 10;
    char reg[10];
    char results_dir[400];
    char data_path[400];
    char support_data_path[400];
    char gm_path[400];
    char wm_path[400];
    char glm_path[400];
    char csf_path[400];
    char p_gm_path[400];
    char p_wm_path[400];
    char p_glm_path[400];
    char p_csf_path[400];
    char obs_mask_path[400];
    char data_comp_path[400];
    char data_comp_dat_path[400];
    char init_tumor_path[400];
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
    ScalarType sigma = -1.0;
    ScalarType spacing_factor = -1.0;
    ScalarType gvf = -1.0;
    ScalarType target_spars = -1.0;
    int sparsity_level = -1.0;
    int lam_cont = 0;
    int verbosity_in = 1;
    ScalarType sigma_dd = -1.0;
    ScalarType data_thres = -1.0;
    ScalarType obs_thres = -1.0;

    ScalarType k_gm_wm = -1.0;
    ScalarType r_gm_wm = -1.0;

    char newton_solver[10];
    char line_search[10];
    int newton_maxit = -1;
    int gist_maxit = -1;
    int krylov_maxit = -1;

    int syn_flag = -1;
    int multilevel_flag = -1;
    int inject_coarse_solution = -1;
    int model = -1;

    int fwd_flag = 0;

    int flag_cosamp = 0;

    ScalarType sm = -1;
    ScalarType forcing_factor = -1;

    ScalarType opttolgrad = -1.0;

    ScalarType low_freq_noise_scale = -1.0;

    ScalarType klb = -1.0;
    ScalarType kub = -1.0;

    int predict_flag = 0;
    int pre_reacdiff_solve = 0;
    int order_of_accuracy = -1;

    ScalarType z_cm, y_cm, x_cm;

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
    PetscOptionsReal ("-forcing_factor", "Forcing factor for mass-effect forward model", "", forcing_factor, &forcing_factor, NULL);


    PetscOptionsReal ("-z_cm", "Z coordinate of tumor loc", "", z_cm, &z_cm, NULL);
    PetscOptionsReal ("-y_cm", "Y coordinate of tumor loc", "", y_cm, &y_cm, NULL);
    PetscOptionsReal ("-x_cm", "X coordinate of tumor loc", "", x_cm, &x_cm, NULL);

    PetscStrcpy (newton_solver, "QN");
    PetscStrcpy (line_search, "mt");
    PetscOptionsString ("-newton_solver", "Newton solver type", "", newton_solver, newton_solver, 10, NULL);
    PetscOptionsString ("-line_search", "Line search type {mt, armijo}", "", line_search, line_search, 10, NULL);
    PetscOptionsInt ("-newton_maxit", "Newton max iterations", "", newton_maxit, &newton_maxit, NULL);
    PetscOptionsInt ("-gist_maxit", "GIST max iterations", "", gist_maxit, &gist_maxit, NULL);
    PetscOptionsInt ("-krylov_maxit", "Krylov max iterations", "", krylov_maxit, &krylov_maxit, NULL);
    PetscOptionsReal ("-rel_grad_tol", "Relative gradient tolerance for L2 solves", "", opttolgrad, &opttolgrad, NULL);
    PetscOptionsInt ("-syn_flag", "Flag for synthetic data generation", "", syn_flag, &syn_flag, NULL);
    PetscOptionsInt ("-multilevel", "Flag indicating whether or not solver is running in multilevel mode", "", multilevel_flag, &multilevel_flag, NULL);
    PetscOptionsInt ("-inject_solution", "Flag indicating if solution from coarser level should be injected (need to set pvec_path and gaussian_cm_path)", "", inject_coarse_solution, &inject_coarse_solution, NULL);
    PetscOptionsInt ("-sparsity_level", "Sparsity level guess for tumor initial condition", "", sparsity_level, &sparsity_level, NULL);
    PetscOptionsInt ("-prediction", "Flag to predict future tumor growth", "", predict_flag, &predict_flag, NULL);
    PetscOptionsInt ("-pre_reacdiff_solve", "Flag to enable reaction/diffusion solve prior to L1 inversion", "", pre_reacdiff_solve, &pre_reacdiff_solve, NULL);
    PetscOptionsInt ("-forward", "Flag to do only the forward solve using data generation parameters", "", fwd_flag, &fwd_flag, NULL);
    PetscOptionsInt ("-order", "Order of accuracy of PDE solver", "", order_of_accuracy, &order_of_accuracy, NULL);

    PetscOptionsString ("-data_path", "Path to data", "", data_path, data_path, 400, NULL);
    PetscOptionsString ("-support_data_path", "Path to data used to generate Gaussian support", "", support_data_path, support_data_path, 400, NULL);
    PetscOptionsString ("-gm_path", "Path to GM", "", gm_path, gm_path, 400, NULL);
    PetscOptionsString ("-wm_path", "Path to WM", "", wm_path, wm_path, 400, NULL);
    PetscOptionsString ("-csf_path", "Path to CSF", "", csf_path, csf_path, 400, NULL);
    PetscOptionsString ("-glm_path", "Path to GLM/Cortical CSF", "", glm_path, glm_path, 400, NULL);
    PetscOptionsString ("-p_gm_path", "Path to GM", "", p_gm_path, p_gm_path, 400, NULL);
    PetscOptionsString ("-p_wm_path", "Path to WM", "", p_wm_path, p_wm_path, 400, NULL);
    PetscOptionsString ("-p_csf_path", "Path to CSF", "", p_csf_path, p_csf_path, 400, NULL);
    PetscOptionsString ("-p_glm_path", "Path to GLM/Cortical CSF", "", p_glm_path, p_glm_path, 400, NULL);
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
    PetscOptionsString ("-init_tumor_path", "Path to nc file containing tumor initial condition", "", init_tumor_path, init_tumor_path, 400, NULL);

    PetscOptionsReal ("-kappa_lb", "Lower bound for diffusivity", "", klb, &klb, NULL);
    PetscOptionsReal ("-kappa_ub", "Upper bound for diffusivity", "", kub, &kub, NULL);


    PetscOptionsEnd ();


    std::stringstream ss;
    ierr = tuMSGstd ("");                                                     CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    ierr = tuMSG("###                                         TUMOR INVERSION SOLVER                                        ###");CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    ss << "    grid size: " << n[0] << "x" << n[1] << "x" << n[2]; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    switch (testcase) {
        case CONSTCOEF: {
            ss << " ----- test case 1: No brain, Constant reaction and diffusion ---- "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            break;
        }
        case SINECOEF: {
            ss << " ----- test case 2: No brain, Sinusoidal reaction and diffusion ---- "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            break;
        }
        case BRAIN: {
            ss << " ----- full brain test ---- "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            break;
        }
        case BRAINNEARMF: {
            ss << " ----- full brain test with multifocal nearby synthetic tumors ---- "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            break;
        }
        case BRAINFARMF: {
            ss << " ----- full brain test with multifocal faroff synthetic tumors ---- "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            break;
        }
        default: break;
    }

    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    // hack
    #ifdef CUDA
        cudaSetDevice (0);
    #endif

    accfft_init();
    MPI_Comm c_comm;
    int c_dims[2] = { 0 };
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

/* ACCFFT, PETSC setup end */
/* --------------------------------------------------------------------------------------------------------------*/

{
    int isize[3], osize[3], istart[3], ostart[3];

    std::shared_ptr<SpectralOperators> spec_ops;
    #if defined(CUDA) && !defined(MPICUDA)
        spec_ops = std::make_shared<SpectralOperators> (CUFFT);
    #else
        spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
    #endif
    spec_ops->setup (n, isize, istart, osize, ostart, c_comm);
    int64_t alloc_max = spec_ops->alloc_max_;
    fft_plan *plan = spec_ops->plan_;

    EventRegistry::initialize ();
    Event e1 ("solve-tumor-inverse-tao");
    //Generate synthetic data
    //Synthetic parameters: Overwrite n_misc
    Vec c_0, data, support_data, p_rec, wm, gm, glm, csf, bg, obs_mask, data_components;

    PetscErrorCode ierr = 0;
    ScalarType rho_temp, k_temp, dt_temp, nt_temp;
    bool overwrite_model = true; //don't change -- controlled from the run script
    bool read_support_data     = (support_data_path != NULL && strlen(support_data_path) > 0); // path set?
    bool read_support_data_nc  = false;
    bool read_support_data_txt = false;
    std::string f(support_data_path), file, path, ext;
    if(read_support_data) {
      ierr = getFileName(path, file, ext, f);                                   CHKERRQ(ierr);
      read_support_data_nc  = (strcmp(ext.c_str(),".nc") == 0);                                  // file ends with *.nc?
      read_support_data_txt = (strcmp(ext.c_str(),".txt") == 0);                                 // file ends with *.txt?
    }
    bool use_custom_obs_mask = (obs_mask_path != NULL && strlen(obs_mask_path) > 0);             // path set?
    bool use_data_comps      = (data_comp_path != NULL && strlen(data_comp_path) > 0);           // path set?
    bool read_data_comp_file = (data_comp_dat_path != NULL && strlen(data_comp_dat_path) > 0);   // path set?
    bool warmstart_p         = (p_vec_path != NULL && strlen(p_vec_path) > 0);                   // path set?
    if (warmstart_p  && not (gaussian_cm_path != NULL && strlen(gaussian_cm_path) > 0)){
      ss << " ERROR: if initial guess for p is used, Gaussian centers need to be specified. "; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      exit(-1);
    }
    if (inject_coarse_solution && (!warmstart_p || !(gaussian_cm_path != NULL && strlen(gaussian_cm_path) > 0) )){
      ss << " ERROR: if coarse solution should be injected, Gaussian centers and p_i values are required. "; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      exit(-1);
    }
    ScalarType rho = rho_data;
    ScalarType k = k_data;
    ScalarType dt = dt_data;
    int nt = nt_data;

    // Other vars
    ScalarType prec_norm;
    ScalarType *prec_ptr;
    ScalarType l2_rel_error = 0.0;
    ScalarType error_norm_c0 = 0.0;

    std::vector<ScalarType> out_params;
    int n_gist = 0, n_newton;
    std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, osize, istart, ostart, plan, c_comm, c_dims, testcase);   //This class contains all required parameters

    n_misc->user_cm_[0] = 2 * M_PI / 256 * z_cm;
    n_misc->user_cm_[1] = 2 * M_PI / 256 * y_cm;
    n_misc->user_cm_[2] = 2 * M_PI / 256 * x_cm;

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

    if (klb != -1.0) n_misc->k_lb_ = klb;
    if (kub != -1.0) n_misc->k_ub_ = kub;

    PetscStrcmp ("QN", newton_solver, &strflg);
    if (strflg) {
        n_misc->newton_solver_ = QUASINEWTON;
    }
    PetscStrcmp ("GN", newton_solver, &strflg);
    if (strflg) {
        n_misc->newton_solver_ = GAUSSNEWTON;
    }

    PetscStrcmp ("armijo", line_search, &strflg);
    if (strflg) {
        n_misc->linesearch_ = ARMIJO;
    }
    PetscStrcmp ("mt", line_search, &strflg);
    if (strflg) {
        n_misc->linesearch_ = MT;
    }

    if (multilevel_flag != -1.0) {
        n_misc->multilevel_ = multilevel_flag;
        ss << " solver is running in multi-level mode"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
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

    if (forcing_factor != -1) {
        n_misc->forcing_factor_ = forcing_factor;
    }

    n_misc->forward_flag_ = fwd_flag;


    n_misc->predict_flag_ = predict_flag;
    n_misc->pre_reacdiff_solve_ = pre_reacdiff_solve;
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

    ss.str(std::string()); ss.clear();
    if (n_misc->verbosity_ >= 2) {
        if (procid == 0) {
            ss << n_misc->writepath_.str().c_str() << "x_it.dat";
            n_misc->outfile_sol_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
            ss << n_misc->writepath_.str().c_str() << "g_it.dat";
            n_misc->outfile_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
            ss << n_misc->writepath_.str().c_str() << "glob_g_it.dat";
            n_misc->outfile_glob_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
            n_misc->outfile_sol_ << std::setprecision(16)<<std::scientific;
            n_misc->outfile_grad_ << std::setprecision(16)<<std::scientific;
            n_misc->outfile_glob_grad_ << std::setprecision(16)<<std::scientific;
        }
    }

    // some checks
    if (n_misc->model_ >= 4) {
        #if defined(CUDA) && !defined(MPICUDA)
            // single gpu mass effect models
            #ifndef SINGLE
                PCOUT << "This forward model only runs with single precision on the GPU. Exiting solver...\n";
                MPI_Comm_free(&c_comm);
                ierr = PetscFinalize ();
                exit(-1);
            #endif
        #endif
    }

    int fwd_temp;
    fwd_temp = n_misc->forward_flag_; // keep track of whether solver is in forward or inverse mode
    if (syn_flag == 1) {
        // data is being generated -- time history should not be stored
        n_misc->forward_flag_ = 1;
    } else {
        // no synthetic data generation -- this means we are solving the inverse problem so set the correct nt, dt
        n_misc->dt_ = dt_inv;
        n_misc->nt_ = nt_inv;
    }

    std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc, spec_ops, nullptr, nullptr);
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();

    // Set optimization flags of tumor solver from input script
    solver_interface->getInvSolver()->getOptSettings ()->newton_maxit = n_misc->newton_maxit_;
    solver_interface->getInvSolver()->getOptSettings ()->newtonsolver = n_misc->newton_solver_;
    solver_interface->getInvSolver()->getOptSettings ()->linesearch = n_misc->linesearch_;
    solver_interface->getInvSolver()->getOptSettings ()->gist_maxit = n_misc->gist_maxit_;
    solver_interface->getInvSolver()->getOptSettings ()->krylov_maxit = n_misc->krylov_maxit_;
    solver_interface->getInvSolver()->getOptSettings ()->opttolgrad = n_misc->opttolgrad_;

    solver_interface->getInvSolver()->getOptSettings ()->verbosity = n_misc->verbosity_;

    bool read_atlas = true;   // Set from run script outside
    if (read_atlas) {
        ierr = readAtlas (wm, gm, glm, csf, bg, n_misc, spec_ops, gm_path, wm_path, csf_path, glm_path);
        ierr = solver_interface->updateTumorCoefficients (wm, gm, glm, csf, bg);
    }

    std::shared_ptr<HealthyProbMaps> h_maps = nullptr;
    if (n_misc->model_ == 3) {// Modified objective
        h_maps = std::make_shared<HealthyProbMaps> (gm, wm, csf, nullptr, bg);
    }

    #ifdef CUDA
        cudaPrintDeviceMemory ();
    #endif
    double self_exec_time = -MPI_Wtime ();
    std::array<double, 7> timers = {0};

    if (syn_flag == 1) {
        ss << " generating Synthetic Data"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        if (n_misc->testcase_ == BRAINFARMF || n_misc->testcase_ == BRAINNEARMF) {
            ierr = createMFData (c_0, data, p_rec, solver_interface, n_misc);
        } else {
            ierr = generateSyntheticData (c_0, data, p_rec, solver_interface, n_misc, spec_ops,init_tumor_path);
        }
        read_support_data_nc = false;
        support_data = data;
    } else {
        ierr = readData (data, support_data, data_components, c_0, p_rec, n_misc, spec_ops, data_path, support_data_path, data_comp_path);
        if(use_custom_obs_mask){
          ierr = readObsFilter(obs_mask, n_misc, obs_mask_path);
          ss << " use custom observation mask"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        }
    }

    Vec data_nonoise;
    ierr = VecDuplicate (data, &data_nonoise);
    ierr = VecCopy (data, data_nonoise);

    if (syn_flag == 1) {
        // add low freq model noise
        if (n_misc->low_freq_noise_scale_ != 0) {
            ierr = applyLowFreqNoise (data, n_misc);
            Vec temp;
            ScalarType noise_err_norm, rel_noise_err_norm;
            ierr = VecDuplicate (data, &temp);      CHKERRQ (ierr);
            ierr = VecSet (temp, 0.);               CHKERRQ (ierr);
            ierr = VecCopy (data_nonoise, temp);    CHKERRQ (ierr);
            ierr = VecAXPY (temp, -1.0, data);      CHKERRQ (ierr);
            ierr = VecNorm (temp, NORM_2, &noise_err_norm);               CHKERRQ (ierr);  // diff btw noise corrupted signal and ground truth
            ierr = VecNorm (data_nonoise, NORM_2, &rel_noise_err_norm);   CHKERRQ (ierr);
            rel_noise_err_norm = noise_err_norm / rel_noise_err_norm;
            ss << " low frequency relative error = " << rel_noise_err_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            // if (n_misc->writeOutput_)
            //     dataOut (data, n_misc, "dataNoise.nc");

            if (temp != nullptr) {ierr = VecDestroy (&temp);          CHKERRQ (ierr); temp = nullptr;}
        }
        ss << " data generated with parameters: rho = " << n_misc->rho_ << " k = " << n_misc->k_ << " dt = " << n_misc->dt_ << " Nt = " << n_misc->nt_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        if (n_misc->model_ >= 4) {
            ss << " mass-effect forcing factor used = " << n_misc->forcing_factor_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            // write out p and phi so that they can be used if needed
            writeCheckpoint(tumor->p_true_, tumor->phi_, n_misc->writepath_.str(), "forward");
            ss << " ground truth phi and p written to file "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        }
    } else {
        ss << " data read"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }

    n_misc->forward_flag_ = fwd_temp; // reset forward flag so that time-history can be stored now if solver is in inverse-mode

    if (fwd_flag) {
        ss << "forward solve completed: exiting..."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    } else {
        ss << " inverse solver begin"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ierr = tumor->mat_prop_->setAtlas(gm, wm, glm, csf, bg);      CHKERRQ(ierr);
        std::shared_ptr<MData> m_data = std::make_shared<MData> (data, n_misc, spec_ops);
        if (n_misc->model_ == 4) {
            n_misc->invert_mass_effect_ = 1;
            // m_data->readData(tumor->mat_prop_->gm_, tumor->mat_prop_->wm_, tumor->mat_prop_->csf_, tumor->mat_prop_->glm_);  // copies synthetic data to m_data
            m_data->readData(p_gm_path, p_wm_path, p_csf_path, p_glm_path);         // reads patient data
            ierr = solver_interface->setMassEffectData(m_data->gm_, m_data->wm_, m_data->csf_, m_data->glm_);   // sets derivative ops data 
            ierr = solver_interface->updateTumorCoefficients(wm, gm, glm, csf, bg);                            // reset matprop to undeformed 
        }
        n_misc->rho_ = rho_inv;
        // n_misc->k_ = (n_misc->diffusivity_inversion_) ? 0 : k_inv;
        n_misc->k_ = k_inv; // (n_misc->diffusivity_inversion_) ? 0 : k_inv;
        n_misc->dt_ = dt_inv;
        n_misc->nt_ = nt_inv;

        ss << " inversion with tumor parameters: rho = " << n_misc->rho_ << " k = " << n_misc->k_ << " dt = " << n_misc->dt_ << " Nt = " << n_misc->nt_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << " results in: " << n_misc->writepath_.str().c_str(); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        std::string file_concomp(data_comp_dat_path);
        if(read_data_comp_file){
          readConCompDat(tumor->phi_->component_weights_, tumor->phi_->component_centers_, file_concomp);
          int nnc = 0;
          for (auto w : tumor->phi_->component_weights_) if (w >= 1E-3) nnc++;
          ss << " set sparsity level to "<< n_misc->sparsity_level_<< " x n_components (w > 1E-3) + n_components (w < 1E-3) = " << n_misc->sparsity_level_ << " x " << nnc <<
          " + " << (tumor->phi_->component_weights_.size() - nnc) << " = " <<  n_misc->sparsity_level_ * nnc + (tumor->phi_->component_weights_.size() - nnc); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
          n_misc->sparsity_level_ =  n_misc->sparsity_level_ * nnc + (tumor->phi_->component_weights_.size() - nnc) ;
        }

        if (!n_misc->bounding_box_) {
            // ierr = tumor->mat_prop_->setValuesCustom (gm, wm, glm, csf, n_misc);    //Overwrite Matprop with custom atlas
            // set the observation operator filter : default filter
            if (use_custom_obs_mask) {
              ierr = tumor->obs_->setCustomFilter (obs_mask);
              ss << " set custom observation mask"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            } else {
              ierr = tumor->obs_->setDefaultFilter (data);
              ss << " set default observation mask based on input data and threshold " << tumor->obs_->threshold_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            }
            // apply observer on ground truth, store observed data in d
            // observation operator is applied before gaussians are set.
            ierr = tumor->obs_->apply (data, data);                                     CHKERRQ (ierr);
            ierr = tumor->obs_->apply (support_data, support_data);                     CHKERRQ (ierr);

            int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
            int nr = (n_misc->reaction_inversion_)    ? n_misc->nr_ : 0;

            // if p vector and Gaussian centers are read in
            if (warmstart_p && !inject_coarse_solution) {
              ss << " solver warmstart using p and Gaussian centers"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
              std::string file_p(p_vec_path);
              std::string file_cm(gaussian_cm_path);
              ierr = tumor->phi_->setGaussians (file_cm);                               CHKERRQ (ierr);     //Overwrites bounding box phis with custom phis
              ierr = tumor->phi_->setValues (tumor->mat_prop_);                         CHKERRQ (ierr);
              ierr = readPVec(&p_rec, n_misc->np_ + nk + nr, n_misc->np_, file_p);      CHKERRQ (ierr);

              // no solver warmstart
            } else {
              // use Gaussian centers of initial support from .txt file (contains labels for data components)
              if (read_support_data_txt) {
                std::string file_cm(support_data_path);
                ierr = tumor->phi_->setGaussians (file_cm, true);                       CHKERRQ (ierr);     //Overwrites bounding box phis with custom phis
              // use *.nc support data to determine Gaussian support, labels have to be set beforehand
              } else if (read_support_data_nc) {
                if(use_data_comps) {
                  ss << " set labels of connected components of data. "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                  tumor->phi_->setLabels (data_components);
                }
                ierr = tumor->phi_->setGaussians (support_data);                        CHKERRQ (ierr);     //Overwrites bounding box phis with custom phis
              } else if (syn_flag) {
                ierr = tumor->phi_->setGaussians (data);                                CHKERRQ (ierr);     //Overwrites bounding box phis with custom phis
              } else {
                ss << "Error: Expecting user input data -support_data_path *.nc or *.txt. exiting..."; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                exit(1);
              }
              ierr = tumor->phi_->setValues (tumor->mat_prop_);                         CHKERRQ (ierr);
              //re-create p_rec
              if (p_rec != nullptr) {ierr = VecDestroy (&p_rec);        CHKERRQ (ierr); p_rec = nullptr;}
              #ifdef SERIAL
                ierr = VecCreateSeq (PETSC_COMM_SELF, n_misc->np_ + nk, &p_rec);                 CHKERRQ (ierr);
                ierr = setupVec (p_rec, SEQ);                                                CHKERRQ (ierr);
              #else
                ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
                ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
                ierr = setupVec (p_rec);                                                CHKERRQ (ierr);
              #endif
            }

            // injection of coarse solution
            if (inject_coarse_solution) {
              Vec coarse_sol = nullptr;
              int np_save = n_misc->np_;
              int np_coarse = 0;
              std::vector<ScalarType> coarse_sol_centers;
              std::string file_cm(gaussian_cm_path);
              std::string file_p(p_vec_path);
              // read phi mesh save nmisc->np_ since it is overwritten
              ierr = readPhiMesh(coarse_sol_centers, n_misc, file_cm, false);           CHKERRQ (ierr);
              ierr = readPVec(&coarse_sol, n_misc->np_ + nk + nr, n_misc->np_, file_p); CHKERRQ (ierr);
              np_coarse = n_misc->np_;
              n_misc->np_ = np_save; // reset to correct value
              // find coarse centers in centers_ of current Phi
              int xc,yc,zc,xf,yf,zf;
              ScalarType *xf_ptr, *xc_ptr;
              ScalarType hx =  2.0 * M_PI / n_misc->n_[0];
              ScalarType hy =  2.0 * M_PI / n_misc->n_[1];
              ScalarType hz =  2.0 * M_PI / n_misc->n_[2];
              ierr = VecGetArray (p_rec, &xf_ptr);                                       CHKERRQ (ierr);
              ierr = VecGetArray (coarse_sol, &xc_ptr);                                  CHKERRQ (ierr);
              for (int j = 0; j < np_coarse; ++j) {
                for (int i = 0; i < n_misc->np_; ++i) {
                  xc = (int)std::round(coarse_sol_centers[3*j + 0]/hx);
                  yc = (int)std::round(coarse_sol_centers[3*j + 1]/hy);
                  zc = (int)std::round(coarse_sol_centers[3*j + 2]/hz);
                  xf = (int)std::round(tumor->phi_->centers_[3*i + 0]/hx);
                  yf = (int)std::round(tumor->phi_->centers_[3*i + 1]/hy);
                  zf = (int)std::round(tumor->phi_->centers_[3*i + 2]/hz);
                  if(xc == xf && yc == yf && zc == zf) {
                    xf_ptr[i] = 2 * xc_ptr[j];            // set initial guess (times 2 since sigma is halfed in every level)
                    n_misc->support_.push_back(i);        // add to support
                  }
                }
              }
              ierr = VecRestoreArray (p_rec, &xf_ptr);                                   CHKERRQ (ierr);
              ierr = VecRestoreArray (coarse_sol, &xc_ptr);                              CHKERRQ (ierr);
              if (coarse_sol != nullptr) {ierr = VecDestroy(&coarse_sol); CHKERRQ(ierr); coarse_sol = nullptr;}
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
            ss << "Running solver in interpolation mode..."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ierr = solver_interface->getInvSolver()->solveInterpolation (data);
            ss << "Interpolation complete; exiting solver..."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        } else {
            bool flag_diff = false;
            if (n_misc->regularization_norm_ == L1 && n_misc->diffusivity_inversion_ == true) {
              n_misc->diffusivity_inversion_ = false;
              flag_diff = true;
            }

            ScalarType gamma;

            if (n_misc->invert_mass_effect_) {
                // apply phi to get tumor c0: rho, kappa already set before; inv-solver setParams will allocate the correct vector sizes
                ierr = tumor->phi_->apply(tumor->c_0_, p_rec);  
                ierr = solver_interface->solveInverseMassEffect (&gamma, data, nullptr); // solve tumor inversion for only mass-effect gamma = forcing factor
                // Reset mat-props and diffusion and reaction operators, tumor IC does not change
                ierr = tumor->mat_prop_->resetValues ();                       CHKERRQ (ierr);
                ierr = tumor->rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, tumor->mat_prop_, n_misc);
                ierr = tumor->k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, tumor->mat_prop_, n_misc);
            } else if (solve_rho_k_only_flag) {
                if (!warmstart_p) {ss << " Error: c(0) needs to be set, read in p and Gaussians. exiting solver..."; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear(); exit(1);}
                ierr = solver_interface->solveInverseReacDiff (p_rec, data, nullptr);     // solve tumor inversion only for rho and k, read in c(0)
            } else if (flag_cosamp) {
                // ierr = solver_interface->setInitialGuess (p_rec);
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
                ss << " reconstructed p norm: " << prec_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                if (n_misc->diffusivity_inversion_) {
                    ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
                    ss << " k1: " << (n_misc->nk_ > 0 ? prec_ptr[n_misc->np_] : 0);     ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                    ss << " k2: " << (n_misc->nk_ > 1 ? prec_ptr[n_misc->np_ + 1] : 0); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                    ss << " k3: " << (n_misc->nk_ > 2 ? prec_ptr[n_misc->np_ + 2] : 0); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                    ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
                }
                ierr = computeError (l2_rel_error, error_norm_c0, p_rec, data_nonoise, data, c_0, solver_interface, n_misc);
                ierr = tuMSGstd(""); CHKERRQ(ierr);
                ss << " l2-error in reconstruction: " << l2_rel_error; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                ss << " --------------  RECONST P -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                if (procid == 0) {
                    ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);                   CHKERRQ (ierr);
                }
                ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                ierr = tuMSGstd(""); CHKERRQ(ierr);
                ss << " --------------- W-L2 solve for sparse components ----------------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

                ierr = solver_interface->resetTaoSolver ();                                 //Reset tao objects
                ierr = solver_interface->setInitialGuess (p_rec);
                ierr = solver_interface->solveInverse (p_rec, data, nullptr);

                out_params = solver_interface->getSolverOutParams ();
                n_newton = (int) out_params[1];
            }

            ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
            ss << " reconstructed p norm: " << prec_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            if (n_misc->diffusivity_inversion_ && !n_misc->reaction_inversion_) {
                ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
                ss << " k1: " << (n_misc->nk_ > 0 ? prec_ptr[n_misc->np_] : 0);     ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                ss << " k2: " << (n_misc->nk_ > 1 ? prec_ptr[n_misc->np_ + 1] : 0); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                ss << " k3: " << (n_misc->nk_ > 2 ? prec_ptr[n_misc->np_ + 2] : 0); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
                ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
            }
            ierr = computeError (l2_rel_error, error_norm_c0, p_rec, data_nonoise, data, c_0, solver_interface, n_misc);
            ss << " l2-error in reconstruction: " << l2_rel_error; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ss << " --------------  RECONST P -----------------";  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            if (procid == 0) {
                ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);                   CHKERRQ (ierr);
            }
            ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            ierr = computeSegmentation(tumor, n_misc, spec_ops);      // Writes segmentation with c0 and c1

            std::stringstream sstm;
            sstm << n_misc->writepath_ .str().c_str() << "reconP.dat";
            std::ofstream ofile;

            //write reconstructed p into text file
            if (procid == 0) {
                ofile.open (sstm.str().c_str());
                ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
                int np = n_misc->np_;
                int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
                for (int i = 0; i < np + nk; i++)
                    ofile << prec_ptr[i] << std::endl;
                ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
                ofile.close ();
            }

            // write p to bin file
            std::string fname = n_misc->writepath_ .str() + "p-final.bin";
            writeBIN(p_rec, fname);

            if (n_misc->predict_flag_) {
                ss << " predicting future tumor growth..."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
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

                ss << " prediction complete for t = 1.2 and t = 1.5"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            }
        }

    }

    if (procid == 0 && n_misc->verbosity_ >= 2) {
        n_misc->outfile_sol_.close();
        n_misc->outfile_grad_.close();
        n_misc->outfile_glob_grad_.close();
    }
    #ifdef CUDA
        cudaPrintDeviceMemory();
    #endif

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
    if (c_0 != nullptr)          {ierr = VecDestroy (&c_0);               CHKERRQ (ierr); c_0 = nullptr;}
    if (data != nullptr)         {ierr = VecDestroy (&data);              CHKERRQ (ierr); data = nullptr;}
    if (p_rec != nullptr)        {ierr = VecDestroy (&p_rec);             CHKERRQ (ierr); p_rec = nullptr;}
    if (data_nonoise != nullptr) {ierr = VecDestroy (&data_nonoise);      CHKERRQ (ierr); data_nonoise = nullptr;}
    if (gm != nullptr)           {ierr = VecDestroy (&gm);                CHKERRQ (ierr); gm = nullptr;}
    if (wm != nullptr)           {ierr = VecDestroy (&wm);                CHKERRQ (ierr); wm = nullptr;}
    if (csf != nullptr)          {ierr = VecDestroy (&csf);               CHKERRQ (ierr); csf = nullptr;}
    if (bg != nullptr)           {ierr = VecDestroy (&bg);                CHKERRQ (ierr); bg = nullptr;}
    if (use_custom_obs_mask && obs_mask != nullptr)                           {ierr = VecDestroy (&obs_mask);          CHKERRQ (ierr); obs_mask = nullptr;}
    if (use_data_comps && read_support_data_nc && data_components != nullptr) {ierr = VecDestroy (&data_components);   CHKERRQ (ierr); data_components = nullptr;}
    if (read_support_data_nc && support_data != nullptr)                      {ierr = VecDestroy (&support_data);      CHKERRQ (ierr); support_data = nullptr;}
}
/* --------------------------------------------------------------------------------------------------------------*/

    MPI_Comm_free(&c_comm);
    ierr = PetscFinalize ();
    return ierr;
}

PetscErrorCode setDistMeasuresFullObj (std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<HealthyProbMaps> h_maps, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    solver_interface->setDistMeassureSimulationGeoImages (h_maps->wm, h_maps->gm, h_maps->csf, h_maps->bg);

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

    solver_interface->setDistMeassureTargetDataImages (h_maps->wm_data, h_maps->gm_data, h_maps->csf_data, h_maps->bg);
    solver_interface->setDistMeassureDiffImages (h_maps->xi_wm, h_maps->xi_gm, h_maps->xi_csf, h_maps->xi_bg);

    if (temp != nullptr) {ierr = VecDestroy (&temp);       CHKERRQ (ierr); temp = nullptr;}

    PetscFunctionReturn (ierr);
}

PetscErrorCode createMFData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    std::stringstream ss;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);                 CHKERRQ (ierr);
        ierr = setupVec (p_rec, SEQ);                                       CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
        ierr = setupVec (p_rec);                                       CHKERRQ (ierr);
    #endif

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = setupVec (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::array<ScalarType, 3> cm;

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
    ss << " --------------  SYNTHETIC TRUE P -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    if (procid == 0) {
        ierr = VecView (tumor->p_true_, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
    }
    ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ierr = tumor->phi_->apply (c_0, tumor->p_true_);                        CHKERRQ (ierr);

    Vec c_temp;
    ierr = VecDuplicate (c_0, &c_temp);                                     CHKERRQ (ierr);
    ierr = VecSet (c_temp, 0.);                                             CHKERRQ (ierr);

    ScalarType scaling = 1.;
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
    ss << " --------------  SYNTHETIC TRUE P -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    if (procid == 0) {
        ierr = VecView (tumor->p_true_, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
    }
    ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ierr = tumor->phi_->apply (c_temp, tumor->p_true_);                     CHKERRQ (ierr);

    ierr = VecAXPY (c_0, 1.0, c_temp);                                      CHKERRQ (ierr);
    if (c_temp != nullptr) {ierr = VecDestroy (&c_temp);                    CHKERRQ (ierr); c_temp = nullptr;}

    ScalarType max, min;
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

    ss << " c data init cond max and min : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    ierr = solver_interface->solveForward (c_t, c_0);   //Observation operator is applied in InvSolve ()

    ierr = VecMax (c_t, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_t, NULL, &min);                                      CHKERRQ (ierr);

    ss << " c data max and min (before observation) : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // ierr = tumor->obs_->apply (c_t, c_t);

    if (n_misc->writeOutput_) {
        dataOut (c_t, n_misc, "dataBeforeObservation.nc");
    }

    PetscFunctionReturn (ierr);
}


PetscErrorCode readData (Vec &data, Vec &support_data, Vec &data_components, Vec &c_0, Vec &p_rec, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, char *data_path, char* support_data_path, char* data_comp_path) {
    PetscFunctionBegin;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PetscErrorCode ierr = 0;

    bool read_support_data     = (support_data_path != NULL && strlen(support_data_path) > 0); // path set?
    bool read_data_comp_data   = (data_comp_path != NULL && strlen(data_comp_path) > 0);       // path set?
    bool read_support_data_nc  = false;
    std::string f(support_data_path), file, path, ext;
    if(read_support_data) {
      ierr = getFileName(path, file, ext, f);                               CHKERRQ(ierr);
      read_support_data_nc = (strcmp(ext.c_str(),".nc") == 0);                                // file ends with *.nc?
    }

    ierr = VecCreate (PETSC_COMM_WORLD, &data);                             CHKERRQ (ierr);
    ierr = VecSetSizes (data, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = setupVec (data);                                                 CHKERRQ (ierr);
    ierr = VecDuplicate (data, &c_0);                                       CHKERRQ (ierr);
    if (read_support_data_nc) {
      ierr = VecDuplicate (data, &support_data);                            CHKERRQ (ierr);
    }

    //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);                 CHKERRQ (ierr);
        ierr = setupVec (p_rec, SEQ);                                       CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
        ierr = setupVec (p_rec);                                       CHKERRQ (ierr);
    #endif

    dataIn (data, n_misc, data_path);
    if(read_support_data_nc){
      dataIn (support_data, n_misc, support_data_path);
    } else {
      support_data = data;
    }

    if(read_data_comp_data && read_support_data_nc) {
      ierr = VecCreate (PETSC_COMM_WORLD, &data_components);                       CHKERRQ (ierr);
      ierr = VecSetSizes (data_components, n_misc->n_local_, n_misc->n_global_);   CHKERRQ (ierr);
      ierr = VecSetFromOptions (data_components);                                  CHKERRQ (ierr);
      ierr = VecSet (data_components, 0.);                                         CHKERRQ (ierr);
      dataIn (data_components, n_misc, data_comp_path);
    }

    // Smooth the data
    // ScalarType sigma_smooth = n_misc->smoothing_factor_ * 2 * M_PI / n_misc->n_[0];

    // ierr = spec_ops->weierstrassSmoother (data, data, n_misc, sigma_smooth);
    ierr = VecSet (c_0, 0.);        CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}

PetscErrorCode readObsFilter (Vec &obs_mask, std::shared_ptr<NMisc> n_misc, char *obs_mask_path) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &obs_mask);                       CHKERRQ (ierr);
    ierr = VecSetSizes (obs_mask, n_misc->n_local_, n_misc->n_global_);   CHKERRQ (ierr);
    ierr = setupVec (obs_mask);                                  CHKERRQ (ierr);
    ierr = VecSet (obs_mask, 0.);                                         CHKERRQ (ierr);

    dataIn (obs_mask, n_misc, obs_mask_path);
    // ScalarType *obs_mask_ptr;
    // ierr = VecGetArray (obs_mask, &obs_mask_ptr);                         CHKERRQ (ierr);
    // for (int i = 0; i < n_misc->n_local_; i++) {
        // if (inversed) {obs_mask_ptr[i] = (obs_mask_ptr[i] > 0) ? 0.0 : 1.0;}
        // else          {obs_mask_ptr[i] = (obs_mask_ptr[i] > 0) ? 1.0 : 0.0;}
    // }
    // ierr = VecRestoreArray (obs_mask, &obs_mask_ptr);                     CHKERRQ (ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode readAtlas (Vec &wm, Vec &gm, Vec &glm, Vec &csf, Vec &bg, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, char *gm_path, char *wm_path, char *csf_path, char *glm_path) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &gm);                             CHKERRQ (ierr);
    ierr = VecSetSizes (gm, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = setupVec (gm);                                                 CHKERRQ (ierr);

    ierr = VecDuplicate (gm, &wm);                                        CHKERRQ (ierr);
    ierr = VecDuplicate (gm, &csf);                                       CHKERRQ (ierr);
    ierr = VecDuplicate (gm, &glm);                                       CHKERRQ (ierr);
    ierr = VecSet (glm, 0.);                                              CHKERRQ (ierr);

    dataIn (wm, n_misc, wm_path);
    dataIn (gm, n_misc, gm_path);
    dataIn (csf, n_misc, csf_path);

    ScalarType sigma_smooth = n_misc->smoothing_factor_ * 2 * M_PI / n_misc->n_[0];
    ierr = spec_ops->weierstrassSmoother (gm, gm, n_misc, sigma_smooth);
    ierr = spec_ops->weierstrassSmoother (wm, wm, n_misc, sigma_smooth);
    ierr = spec_ops->weierstrassSmoother (csf, csf, n_misc, sigma_smooth);

    if (n_misc->model_ >= 4) {
        // mass-effect included
        dataIn (glm, n_misc, glm_path); 
        ierr = spec_ops->weierstrassSmoother (glm, glm, n_misc, sigma_smooth);
    }

    bg = nullptr;

    PetscFunctionReturn (ierr);
}

PetscErrorCode applyLowFreqNoise (Vec data, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    // TODO: Modify for CUDA (@S)

    // int procid, nprocs;
    // MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    // MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    // srand (time(NULL));
    // ScalarType random_noise;
    // ScalarType noise_level = n_misc->low_freq_noise_scale_;
    // ScalarType mag = 0.;
    // int64_t x_global, y_global, z_global, x_symm, y_symm, z_symm, global_index;
    // ScalarType freq = 0;
    // ScalarType amplitude = 0.;

    // int *osize = n_misc->osize_;
    // int *ostart = n_misc->ostart_;
    // int *isize = n_misc->isize_;
    // int *istart = n_misc->istart_;
    // MPI_Comm c_comm = n_misc->c_comm_;
    // int *n = n_misc->n_;
    // fft_plan *plan = n_misc->plan_;

    // // Get the fourier transform of data;
    // ScalarType *d_ptr;
    // ierr = VecGetArray (data, &d_ptr);          CHKERRQ (ierr);

    // // // remove small aliasing errors
    // // for (int i = 0; i < n_misc->n_local_; i++) {
    // //     if (d_ptr[i] < 1E-4) {
    // //         d_ptr[i] = 0.;
    // //     }
    // // }

    // int alloc_max = accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    // accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    // ComplexType *data_hat;
    // ScalarType *freq_scaling;
    // data_hat = (ComplexType*) accfft_alloc (alloc_max);
    // freq_scaling = (ScalarType*) accfft_alloc (alloc_max);
    // fft_execute_r2c (plan, d_ptr, data_hat);
    // MPI_Barrier (c_comm);

    // ScalarType *data_hat_mag;
    // ScalarType *d;
    // data_hat_mag = (ScalarType*) accfft_alloc(alloc_max);
    // ScalarType wx, wy, wz;

    // int64_t ptr;
    // // Find the amplitude of the signal (data)
    // for (int i = 0; i < osize[0]; i++) {
    //     for (int j = 0; j < osize[1]; j++) {
    //         for (int k = 0; k < osize[2]; k++) {
    //             ptr = i * osize[1] * osize[2] + j * osize[2] + k;
    //             d = data_hat[ptr];
    //             data_hat_mag[ptr] = std::sqrt(d[0] * d[0] + d[1] * d[1]); // amplitude compute
    //             if (data_hat_mag[ptr] > amplitude)
    //                 amplitude = data_hat_mag[ptr];

    //             // populate freq: By symmetery X(N-k) = X(k)
    //             // instead of enforcing conjugate symmetery manually, just scale with only unique frequencies

    //             x_global = i + ostart[0];
    //             y_global = j + ostart[1];
    //             z_global = k + ostart[2];

    //             wx = (x_global > n[0] / 2) ? n[0] - x_global : x_global;
    //             wy = (y_global > n[1] / 2) ? n[1] - y_global : y_global;
    //             wz = (z_global > n[2] / 2) ? n[2] - z_global : z_global;

    //             if (wx == 0 && wy == 0 && wz == 0)
    //                 freq_scaling[ptr] = 1.;
    //             else
    //                 freq_scaling[ptr] = (1.0 / (wx * wx + wy * wy + wz * wz));
    //         }
    //     }
    // }
    // // allreduce to find the amplitude of the freq
    // ScalarType global_amplitude = 0.;
    // MPI_Allreduce (&amplitude, &global_amplitude, 1, MPIType, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Barrier (c_comm);

    // // Now add power law noise
    // for (int i = 0; i < osize[0]; i++) {
    //     for (int j = 0; j < osize[1]; j++) {
    //         for (int k = 0; k < osize[2]; k++) {
    //             ptr = i * osize[1] * osize[2] + j * osize[2] + k;
    //             d = data_hat[ptr];
    //             mag = data_hat_mag[ptr];
    //             random_noise = (ScalarType)rand() / (ScalarType)RAND_MAX;
    //             data_hat_mag[ptr] += noise_level * random_noise * global_amplitude * freq_scaling[ptr];
    //             // data_hat_mag[ptr] += noise_level * random_noise * global_amplitude * std::sqrt(freq_scaling[ptr]);
    //             // change data_hat accordingly -- this will change only the unique freq
    //             if (mag != 0) { // check for non zero components
    //                 d[0] *= (1.0 / mag) * data_hat_mag[ptr];
    //                 d[1] *= (1.0 / mag) * data_hat_mag[ptr];
    //             }
    //         }
    //     }
    // }

    // MPI_Barrier(c_comm);
    // fft_execute_c2r(plan, data_hat, d_ptr);
    // MPI_Barrier(c_comm);

    // for (int i = 0; i < n_misc->n_local_; i++)
    //     d_ptr[i] /= n[0] * n[1] * n[2];

    // fft_free (data_hat);
    // fft_free (freq_scaling);
    // fft_free (data_hat_mag);
    // ierr = VecRestoreArray (data, &d_ptr);              CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}


PetscErrorCode computeError (ScalarType &error_norm, ScalarType &error_norm_c0, Vec p_rec, Vec data, Vec data_obs, Vec c_0_true, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Vec c_rec_0, c_rec;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    std::stringstream ss;

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();

    ScalarType data_norm;
    ierr = VecDuplicate (data, &c_rec_0);                                   CHKERRQ (ierr);
    ierr = VecDuplicate (data, &c_rec);                                     CHKERRQ (ierr);

    ierr = tumor->phi_->apply (c_rec_0, p_rec);

    ScalarType *c0_ptr;

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

    ScalarType max, min;
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

    ss << " c reconstructed max and min : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();


    if (n_misc->writeOutput_)
        dataOut (c_rec, n_misc, "cRecon.nc");

    ierr = VecMax (c_rec_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_rec_0, NULL, &min);                                      CHKERRQ (ierr);

    ierr = VecAXPY (c_rec_0, -1.0, c_0_true);                               CHKERRQ (ierr);
    ierr = VecNorm (c_0_true, NORM_2, &data_norm);                          CHKERRQ (ierr);
    ierr = VecNorm (c_rec_0, NORM_2, &error_norm_c0);                       CHKERRQ (ierr);

    error_norm_c0 /= data_norm;
    ss << " error norm in c(0): " << error_norm_c0; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    ierr = VecAXPY (c_rec, -1.0, data);                                     CHKERRQ (ierr);
    ierr = VecNorm (data, NORM_2, &data_norm);                              CHKERRQ (ierr);
    ierr = VecNorm (c_rec, NORM_2, &error_norm);                            CHKERRQ (ierr);

    ss << " data mismatch: " << error_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    error_norm /= data_norm;

    ScalarType obs_c_norm, obs_data_norm;

    ierr = VecAXPY (obs_c_rec, -1.0, data_obs);                             CHKERRQ (ierr);
    ierr = VecNorm (obs_c_rec, NORM_2, &obs_c_norm);                        CHKERRQ (ierr);
    ierr = VecNorm (data_obs, NORM_2, &obs_data_norm);                      CHKERRQ (ierr);

    obs_c_norm /= obs_data_norm;

    ss << " rel. l2-error at observation points: " << obs_c_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    if(obs_c_rec != nullptr) {ierr = VecDestroy (&obs_c_rec);               CHKERRQ (ierr); obs_c_rec = nullptr;}


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

    ScalarType *w_ptr, *p_true_ptr;
    ierr = VecGetArray (weights, &w_ptr);                                   CHKERRQ (ierr);
    ierr = VecGetArray (p_true_w, &p_true_ptr);                             CHKERRQ (ierr);

    std::vector<ScalarType> dist (n_misc->user_cms_.size());
    ScalarType d = 0.;
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
    if (procid == 0) {
        ofile.open (sstm.str().c_str());
        for (int i = 0; i < n_misc->np_; i++)
            ofile << p_true_ptr[i] << std::endl;
        ofile.flush ();
        ofile.close ();
    }

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


    ScalarType p_wL2, p_diff_wL2;
    ierr = VecPointwiseMult (temp, p_true_w, weights);                  CHKERRQ (ierr);
    ierr = VecDot (p_true_w, temp, &p_wL2);                             CHKERRQ (ierr);
    p_wL2 = std::sqrt (p_wL2);
    ierr = VecCopy (p_true_w, p_diff_w);                                CHKERRQ (ierr);
    ierr = VecAXPY (p_diff_w, -1.0, p_rec);                             CHKERRQ (ierr);  // diff in p
    ScalarType l1_norm_diff, l1_norm_p;
    ScalarType *diff_ptr;
    ierr = VecGetArray (p_diff_w, &diff_ptr);                       CHKERRQ (ierr);
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
    if (n_misc->diffusivity_inversion_) {
        diff_ptr[n_misc->np_] = 0;
        if (nk > 1) diff_ptr[n_misc->np_+1] = 0;
        if (nk > 2) diff_ptr[n_misc->np_+2] = 0;
    }
    ierr = VecRestoreArray (p_diff_w, &diff_ptr);                   CHKERRQ (ierr);

    ierr = VecNorm (p_diff_w, NORM_1, &l1_norm_diff);   CHKERRQ (ierr);
    ierr = VecNorm (p_true_w, NORM_1, &l1_norm_p);      CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp, p_diff_w, weights);                  CHKERRQ (ierr);
    ierr = VecDot (p_diff_w, temp, &p_diff_wL2);                        CHKERRQ (ierr);
    p_diff_wL2 = std::sqrt (p_diff_wL2);

    ScalarType dist_err_c0 = p_diff_wL2 / p_wL2;
    ScalarType l1_err = l1_norm_diff / l1_norm_p;

    if(weights != nullptr)  {ierr = VecDestroy (&weights);        CHKERRQ (ierr); weights = nullptr;}
    if(p_true_w != nullptr) {ierr = VecDestroy (&p_true_w);       CHKERRQ (ierr); p_true_w = nullptr;}
    if(p_diff_w != nullptr) {ierr = VecDestroy (&p_diff_w);       CHKERRQ (ierr); p_diff_w = nullptr;}
    if(temp != nullptr)     {ierr = VecDestroy (&temp);           CHKERRQ (ierr); temp = nullptr;}

    ScalarType *p_rec_ptr;
    ierr = VecGetArray (p_rec, &p_rec_ptr);     CHKERRQ (ierr);

    ScalarType k1, k2, k3, r1, r2, r3;
    k1 = 0.; k2 = 0.; k3 = 0.;
    if (n_misc->diffusivity_inversion_) {
        k1 = p_rec_ptr[n_misc->np_];
        if (n_misc->nk_ > 1)
          k2 = p_rec_ptr[n_misc->np_ + 1];
        if (n_misc->nk_ > 2)
          k3 = p_rec_ptr[n_misc->np_ + 2];
    }

    ss << " p distance error: " << dist_err_c0; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << " p l1 norm: " << l1_err; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    std::stringstream ss_out;
    ss_out << n_misc->writepath_ .str().c_str() << "info.dat";
    std::ofstream opfile;

    if (procid == 0) {
        opfile.open (ss_out.str().c_str());
        opfile << "rho k c1_rel c0_rel c0_dist \n";
        opfile << n_misc->rho_ << " " <<  n_misc->k_ << " " << error_norm << " "
               << error_norm_c0 << " " << dist_err_c0 << std::endl;
        opfile.flush();
        opfile.close ();
    }

    ierr = VecRestoreArray (p_rec, &p_rec_ptr);     CHKERRQ (ierr);


    if(c_rec_0 != nullptr) {ierr = VecDestroy (&c_rec_0); CHKERRQ (ierr); c_rec_0 = nullptr;}
    if (c_rec != nullptr)  {ierr = VecDestroy (&c_rec); CHKERRQ (ierr); c_rec = nullptr;}
    PetscFunctionReturn (ierr);
}

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, char *init_tumor_path) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    std::stringstream ss;

    //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
    ScalarType sigma_smooth = 2.0 * M_PI / n_misc->n_[0];

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);                 CHKERRQ (ierr);
        ierr = setupVec (p_rec, SEQ);                                       CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
        ierr = setupVec (p_rec);                                       CHKERRQ (ierr);
    #endif

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = setupVec (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();

    ScalarType *c0_ptr;
    ScalarType c0_min, c0_max;
    if ((init_tumor_path != NULL) && (init_tumor_path[0] != '\0')) {
        ierr = dataIn (c_0, n_misc, init_tumor_path);                       CHKERRQ (ierr);
        ierr = VecMin (c_0, NULL, &c0_min);                                 CHKERRQ (ierr);
        if (c0_min < 0) {
            ss << " tumor init is aliased with min " << c0_min << "; clipping and smoothing...";
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ierr = vecGetArray (c_0, &c0_ptr);                                              CHKERRQ (ierr);
            #ifdef CUDA
                clipVectorCuda (c0_ptr, n_misc->n_local_);
            #else
                for (int i = 0; i < n_misc->n_local_; i++)
                    c0_ptr[i] = (c0_ptr[i] <= 0.) ? 0. : c0_ptr[i];
            #endif
            ierr = vecRestoreArray (c_0, &c0_ptr);                                          CHKERRQ (ierr);
            // smooth a little bit because sometimes registration outputs have too much aliasing
            ierr = spec_ops->weierstrassSmoother (c_0, c_0, n_misc, sigma_smooth);          CHKERRQ (ierr);
            ierr = VecMax (c_0, NULL, &c0_max);                                             CHKERRQ (ierr);
            ierr = VecScale (c_0, (1.0/c0_max));                                            CHKERRQ (ierr);
        }
    } else {
        ierr = tumor->setTrueP (n_misc);
        ss << " --------------  SYNTHETIC TRUE P -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        if (procid == 0) {
            ierr = VecView (tumor->p_true_, PETSC_VIEWER_STDOUT_SELF);          CHKERRQ (ierr);
        }
        ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ierr = tumor->phi_->apply (c_0, tumor->p_true_);
    }

    if (n_misc->model_ == 2) {
        ierr = VecGetArray (c_0, &c0_ptr);                                  CHKERRQ (ierr);
        for (int i = 0; i < n_misc->n_local_; i++) {
            c0_ptr[i] = 1 / (1 + exp(-c0_ptr[i] + n_misc->exp_shift_));
        }
        ierr = VecRestoreArray (c_0, &c0_ptr);                              CHKERRQ (ierr);
    }

    ScalarType max, min;
    ierr = VecMax (c_0, NULL, &max);                                       CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                       CHKERRQ (ierr);

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    if (n_misc->writeOutput_) {
        ierr = dataOut (c_0, n_misc, "c0True.nc");                        CHKERRQ (ierr);
    }

    ierr = VecMax (c_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                      CHKERRQ (ierr);

    ss << " c data initc cond. max and min : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    if (n_misc->model_ == 5) {
        std::map<std::string, Vec> species;
        ierr = solver_interface->solveForward (c_t, c_0, &species);
    } else {
        ierr = solver_interface->solveForward (c_t, c_0);
    }

    ierr = VecMax (c_t, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_t, NULL, &min);                                      CHKERRQ (ierr);

    ss << " c data max and min (before observation) : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    if (n_misc->writeOutput_) {
        dataOut (c_t, n_misc, "dataBeforeObservation.nc");
    }

    PetscFunctionReturn (ierr);
}


PetscErrorCode computeSegmentation(std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec max;
    ierr = VecDuplicate(tumor->c_0_, &max);                                 CHKERRQ(ierr);
    ierr = VecSet(max, 0);                                                  CHKERRQ(ierr);

    // compute max of gm, wm, csf, bg, tumor
    std::vector<ScalarType> v;
    std::vector<ScalarType>::iterator max_component;
    ScalarType *bg_ptr, *gm_ptr, *wm_ptr, *csf_ptr, *c_ptr, *max_ptr;
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


    ScalarType sigma_smooth = 1.5 * M_PI / n_misc->n_[0];

    // ierr = spec_ops->weierstrassSmoother (max_ptr, max_ptr, n_misc, sigma_smooth);

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
        v.push_back(wm_ptr[i]);
        v.push_back(gm_ptr[i]);
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

    // ierr = spec_ops->weierstrassSmoother (max_ptr, max_ptr, n_misc, sigma_smooth);


    ierr = VecRestoreArray(max, &max_ptr);                                      CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor->c_t_, &c_ptr);                                CHKERRQ(ierr);

    if (n_misc->writeOutput_) {
        dataOut (max, n_misc, "seg1.nc");
    }

    if(max != nullptr) {ierr = VecDestroy (&max);       CHKERRQ (ierr); max = nullptr;}


    PetscFunctionReturn (ierr);
}
