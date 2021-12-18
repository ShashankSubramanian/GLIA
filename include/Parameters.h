
#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <stdlib.h>
#include "TypeDefs.h"
#include "Utils.h"

// global variable
extern bool DISABLE_VERBOSE;


/* #### ------------------------------------------------------------------- #### */
/* #### ========                 Optimizer Settings                ======== #### */
/* #### ------------------------------------------------------------------- #### */

struct OptimizerSettings {
public:
  OptimizerSettings ()
  :
    beta_(1E-4),
    regularization_norm_(0),
    opttolgrad_ (1E-3),
    ftol_ (1E-3),
    ls_minstep_(1E-16),
    gtolbound_ (0.8),
    grtol_ (1E-5),
    gatol_ (1E-4),
    newton_maxit_ (30),
    newton_minit_ (1),
    gist_maxit_ (5),
    krylov_maxit_ (30),
    iterbound_ (200),
    fseqtype_ (SLFS),
    newton_solver_ (QUASINEWTON),
    linesearch_ (MT),
    verbosity_ (0),
    reset_tao_ (false),
    cosamp_stage_ (0),
    ls_max_func_evals(10),
    lbfgs_vectors_(10),
    lbfgs_scale_hist_(5),
    lbfgs_scale_type_("diagonal"),
    lmvm_set_hessian_ (false),
    diffusivity_inversion_(true),
    reaction_inversion_(false),
    flag_reaction_inv_ (false),
    pre_reacdiff_solve_(false),
    rescale_init_cond_(false),
    estimate_rho_init_guess_(false),
    multilevel_(false),
    cross_entropy_loss_(false),
    invert_mass_effect_(true),
    prune_components_(true),
    sigma_inv_(0.5),
    sigma_cma_gamma_(3.0E4),
    sigma_cma_rho_(3),
    sigma_cma_k_(0.1),
    sigma_cma_ox_hypoxia_(0.25),
    sigma_cma_death_rate_(0.25),
    sigma_cma_alpha_0_(0.25),
    sigma_cma_ox_consumption_(5.0),
    sigma_cma_ox_source_(14.0),
    sigma_cma_beta_0_(0.25),
    k_lb_ (1E-3),
    k_ub_ (1),
    gamma_lb_(0),
    gamma_ub_(15),
    rho_lb_(1),
    rho_ub_(15),
    ox_hypoxia_lb_(1E-3),
    ox_hypoxia_ub_(1E-3),
    death_rate_lb_(0.5),
    death_rate_ub_(1.5),
    alpha_0_lb_(1E-3),
    alpha_0_ub_(1.0),
    ox_consumption_lb_(1E-3),
    ox_consumption_ub_(20.0),
    ox_source_lb_(1E-3),
    ox_source_ub_(100.0),
    beta_0_lb_(1E-3),
    beta_0_ub_(1.0),
    k_scale_(1.0),
    rho_scale_(1.0),
    gamma_scale_(1.0),
    ox_hypoxia_scale_(1.0),
    death_rate_scale_(1.0),
    alpha_0_scale_(1.0),
    ox_consumption_scale_(1.0),
    ox_source_scale_(1.0),
    beta_0_scale_(1.0)
  {}
  ScalarType beta_;
  ScalarType regularization_norm_;
  ScalarType opttolgrad_;       /// @brief l2 gradient tolerance for optimization
  ScalarType ftol_;             /// @brief l1 function and solution tolerance
  ScalarType ls_minstep_;       /// @brief minimum step length of linesearch
  ScalarType gtolbound_;        /// @brief minimum reduction of gradient (even if maxiter hit earlier)
  ScalarType grtol_;            /// @brief rtol TAO (relative tolerance for gradient, not used)
  ScalarType gatol_;            /// @brief atol TAO (absolute tolerance for gradient)
  int newton_maxit_;            /// @brief maximum number of allowed newton iterations
  int newton_minit_;            /// @brief minimum number of newton steps
  int gist_maxit_;              /// @brief maximum number of GIST iterations
  int krylov_maxit_;            /// @brief maximum number of allowed krylov iterations
  int iterbound_;               /// @brief if GRADOBJ conv. crit is used, max number newton it
  int fseqtype_;                /// @brief type of forcing sequence (quadratic, superlinear)
  int newton_solver_;           /// @brief type of newton slver (0=GN, 1=QN, 2=GN/QN)
  int linesearch_;              /// @brief type of line-search used (0=armijo, 1=mt)
  int verbosity_;               /// @brief controls verbosity of solver
  int cosamp_stage_;            /// @brief indicates stage of cosamp solver for warmstart
  int ls_max_func_evals;        /// @brief maximum number of allowed ls steps per newton iteration
  int lbfgs_vectors_;           /// @brief number of vectors used in lbfgs update
  int lbfgs_scale_hist_;        /// @brief number of vectors used for initial guess of inverse hessian
  std::string lbfgs_scale_type_;/// @brief initial guess for lbfgs inverse hessian
  bool lmvm_set_hessian_;       /// @brief if true lmvm initial hessian ist set as matvec routine
  bool reset_tao_;              /// @brief if true TAO is destroyed and re-created for every new inversion solve, if not, old structures are kept.
  bool diffusivity_inversion_;  /// @brief if true, we also invert for k_i scalings of material properties to construct isotropic part of diffusion coefficient
  bool reaction_inversion_;     /// @brief if true, we also invert for rho
  bool flag_reaction_inv_;      /// @brief internal flag to enable reaction/diffusion methods in InvSolver
  bool pre_reacdiff_solve_;     /// @brief if true, CoSaMp L1 inversion scheme perfroms reaction/diffusion solve before {p,k} inversion
  bool rescale_init_cond_;      /// @brief if true, TIL is rescaled to 1
  bool estimate_rho_init_guess_;/// @brief if true a rough estimate of rho is computed based on given TIL
  bool multilevel_;             /// @brief if true, INT_Omega phi(x) dx = const across levels
  bool cross_entropy_loss_;     /// @brief cross-entropy is used instead of L2 loss
  bool invert_mass_effect_;     /// @brief if true invert for mass-effect parameters {rho,k,gamma}
  bool prune_components_;       /// @brief prunes L2 solution based on components
  ScalarType sigma_inv_;        /// @brief sigma for CMA inversion of MultiSpecies model
  ScalarType k_lb_;             /// @brief lower bound on kappa - depends on mesh; 1E-3 for 128^3 1E-4 for 256^3
  ScalarType k_ub_;             /// @brief upper bound on kappa
  ScalarType gamma_lb_;         /// @brief lower bound on gamma
  ScalarType gamma_ub_;         /// @brief upper bound on gamma
  ScalarType rho_lb_;           /// @brief lower bound on rho
  ScalarType rho_ub_;           /// @brief upper bound on rho
  ScalarType ox_hypoxia_lb_;
  ScalarType ox_hypoxia_ub_;
  ScalarType death_rate_lb_;
  ScalarType death_rate_ub_;
  ScalarType alpha_0_lb_;
  ScalarType alpha_0_ub_;
  ScalarType ox_consumption_lb_;
  ScalarType ox_consumption_ub_;
  ScalarType ox_source_lb_;
  ScalarType ox_source_ub_;
  ScalarType beta_0_lb_;
  ScalarType beta_0_ub_;
  ScalarType sigma_cma_gamma_;
  ScalarType sigma_cma_rho_;
  ScalarType sigma_cma_k_;
  ScalarType sigma_cma_ox_hypoxia_;
  ScalarType sigma_cma_death_rate_;
  ScalarType sigma_cma_alpha_0_;
  ScalarType sigma_cma_ox_consumption_;
  ScalarType sigma_cma_ox_source_;
  ScalarType sigma_cma_beta_0_;
  ScalarType k_scale_;          /// @brief if FD grad model, scaling of kappa
  ScalarType rho_scale_;        /// @brief if FD grad model, scaling of rho
  ScalarType gamma_scale_;      /// @brief if FD grad model, scaling of gamma
  ScalarType ox_hypoxia_scale_;  /// @brief if MultiSpec model, scalaing of hypoxia threshold
  ScalarType death_rate_scale_;  /// @brief if MultiSpec model, scalaing of deathrate 
  ScalarType alpha_0_scale_;     /// @brief if MultiSpec model, scalaing of trans rate from p to i
  ScalarType ox_consumption_scale_;  /// @brief if MultiSpec model, scalaing of oxygen consumption
  ScalarType ox_source_scale_;       /// @brief if MultiSpec model, scalaing of oxygen source 
  ScalarType beta_0_scale_;          /// @brief if MultiSpec model, scalaing of trans rate from i to p
  std::array<ScalarType, 3> bounds_array_;  
};

struct OptimizerFeedback {
public:
  OptimizerFeedback ()
  :
    nb_newton_it_(-1),
    nb_krylov_it_(-1),
    nb_matvecs_(-1),
    nb_objevals_(-1),
    nb_gradevals_(-1),
    solverstatus_(),
    gradnorm_(0.),
    gradnorm0_(0.),
    j0_(0.),
    jval_(0.),
    converged_(false)
  {}
  void reset() {
    converged_ = false;
    nb_newton_it_ = 0;
    nb_krylov_it_ = 0;
    solverstatus_ = "";
    nb_matvecs_ = 0;
    nb_objevals_ = 0;
    nb_gradevals_ = 0;
  }

  int nb_newton_it_;            /// @brief stores the number of required Newton iterations for the last inverse tumor solve
  int nb_krylov_it_;            /// @brief stores the number of required (accumulated) Krylov iterations for the last inverse tumor solve
  int nb_matvecs_;              /// @brief stores the number of required (accumulated) matvecs per tumor solve
  int nb_objevals_;             /// @brief stores the number of required (accumulated) objective function evaluations per tumor solve
  int nb_gradevals_;            /// @brief stores the number of required (accumulated) gradient evaluations per tumor solve
  std::string solverstatus_;    /// @brief gives information about the termination reason of inverse tumor TAO solver
  ScalarType gradnorm_;         /// @brief final gradient norm
  ScalarType gradnorm0_;        /// @brief norm of initial gradient (with p = intial guess)
  ScalarType j0_;               /// @brief initial objective : needed for L1 convergence tests
  ScalarType jval_;             /// @brief orbjective function value
  bool converged_;              /// @brief true if solver converged within bounds
};


/* #### ------------------------------------------------------------------- #### */
/* #### ========                  Tumor Parameters                 ======== #### */
/* #### ------------------------------------------------------------------- #### */

struct TumorParameters {
public:
  TumorParameters()
  : model_ (1)                            // 1: reaction/diffusion (adjoints), 2: reaction-diffusion (finite-differences)
                                          // 3: modified objective
                                          // 4: mass effect
                                          // 5: multi-species
  , np_ (1)                               // Number of gaussians for bounding box
  , nk_ (1)                               // Number of k_i that we like to invert for (1-3)
  , nr_ (1)                               // number of rho_i that we like to invert for (1-3)
  , nt_(1)                                // Total number of time steps
  , dt_ (0.5)                             // Time step
  , k_ (0E-1)                             // Isotropic diffusion coefficient
  , kf_(0.0)                              // Anisotropic diffusion coefficient
  , rho_ (10)                             // Reaction coefficient
  , k_gm_wm_ratio_ (0.0 / 1.0)            // gm to wm diffusion coeff ratio
  , k_glm_wm_ratio_ (0.0)                 // glm to wm diffusion coeff ratio
  , r_gm_wm_ratio_ (0.0 / 5.0)            // gm to wm reaction coeff ratio
  , r_glm_wm_ratio_ (0.0)                 // glm to wm diffusion coeff ratio
  , invasive_threshold_ (0.01)           // invasive threshold for edema
  , E_csf_ (100)                          // Young's modulus of CSF
  , E_healthy_ (2100)                     // Young's modulus of wm and gm
  , E_tumor_ (8000)                       // Young's modulus of tumor
  , E_bg_ (15000)                         // Young's modulus of background
  , nu_csf_ (0.1)                         // Poisson's ratio of CSF
  , nu_healthy_ (0.4)                     // Poisson's ratio of wm and gm
  , nu_tumor_ (0.45)                      // Poisson's ratio of tumor
  , nu_bg_ (0.45)                         // Poisson's ratio of background
  , screen_low_ (1E-2)                    // low screening coefficient
  , screen_high_ (1E2)                    // high screening
  , forcing_factor_ (1E5)                 // mass effect forcing factor (1E5 for casebrats; 6E4 for SRI atlas)
  , use_tanh_scaling_ (true)              // tanh scaling for mass-effect
  , num_species_ (5)                      // Number of species for the multi-species model - only used if model is 5
  , ox_source_ (1)                        // source of oxygen
  , ox_consumption_ (5)                   // consumption of oxygen
  , alpha_0_ (0.1)                        // conversion btw inv and proliferative
  , beta_0_ (0.01)                        // conversion btw inv and proliferative
  , ox_inv_ (0.7)                         // invasive oxygen conc
  , death_rate_ (4)                       // death rate
  , ox_hypoxia_ (0.65)                     // hypoxia threshold
  , sparsity_level_ (5)                   // Level of sparsity for L1 solves
  , thresh_component_weight_ (1E-3)       // components with weight smaller than this threshold are not considered in sparsity computation
  , support_()                            // // support of compressive sampling guess
  , bounding_box_ (0)                     // Flag to set bounding box for gaussians
  , max_p_location_ (0)                   // Location of maximum gaussian scale concentration - this is used to set bounds for reaction inversion
  , p_scale_ (0.0)                        // Scaling factor for initial guess
  // , p_scale_true_ (1.0)                // Scaling factor for synthetic data generation
  , phi_sigma_ (2 * M_PI / 64)            // Gaussian standard deviation for bounding box
  , phi_sigma_data_driven_ (2 * M_PI / 256) // Sigma for data-driven gaussians
  , phi_spacing_factor_ (1.5)             // Gaussian spacing for bounding box
  , data_threshold_ (0.05)                // Data threshold to set custom gaussians
  , gaussian_vol_frac_ (0.0)              // Volume fraction of gaussians to set custom basis functions
  , user_cm_()
  , obs_threshold_0_ (0.0)                 // Observation threshold for data at time t=0
  , obs_threshold_1_ (0.0)                 // Observation threshold for data at time t=1
  , obs_lambda_(-1)                        // enables custom observation operator to be build from segmentation as OBS = 1[TC] + lambda*1{B/WT]
  , relative_obs_threshold_(false)         // if true, observation threshold is relative to max concentration
  , smoothing_factor_ (1)                  // Smoothing factor for material properties
  , smoothing_factor_data_ (1)             // Smoothing factor for read in data
  , smoothing_factor_data_t0_ (1)             // Smoothing factor for read in data
  , smoothing_factor_atlas_ (1)            // Smoothing factor for read in atlas
  , smoothing_factor_patient_ (1)          // Smoothing factor for read in patient healthy tissues
  , use_c0_(false)                         // use c(0) directly, never use phi*p
  , adv_velocity_set_(false)
  , two_time_points_(false)                // enables objective for two time points
  , interpolation_order_ (3)               // interpolation order for SL
  , order_ (2)                             // Order of accuracy for PDE solves
  , write_output_ (1)                      // Print flag for paraview visualization
  , phi_store_ (false)                     // Flag to store phis
  , adjoint_store_ (false)                  // Flag to store half-step concentrations for adjoint solve to speed up time to solution
  , time_history_off_(false)               // if true, time history is not stored (forward solve, or finite-differences)
  , exp_shift_ (10.0)                      // Parameter for positivity shift
  , penalty_ (1E-4)                        // Parameter for positivity objective function
  , beta_changed_ (false)                  // if true, we overwrite beta with user provided beta: only for tumor inversion standalone
  , transport_mri_ (false)                 // transport T1 image
  , verbosity_ (0)                         // Print flag for optimization routines
  , write_p_checkpoint_(true)              // if true, p vector and corresponding Gaussian centers are written to file at certain checkpoints
  , statistics_()                          // stores statistics of function evaluations, gradient evaluaations, iterations etc.
  , outfile_sol_()
  , outfile_grad_()
  , outfile_glob_grad_()
  , writepath_()
  , readpath_()
  , ext_(".nc")
  , force_phi_compute_ (false)             // forces the computation of gaussians on the fly because of insufficient memory on the device 
  , feature_compute_ (false)      // computes temporal biophysical features in the forward solver
  {
    time_horizon_ = nt_ * dt_;
    // SRI-atlas
    user_cm_[0] = 2 * M_PI / 128 * 64;//63  //Z
    user_cm_[1] = 2 * M_PI / 128 * 38;//82  //Y
    user_cm_[2] = 2 * M_PI / 128 * 56;//48  //X
    // user_cms_.push_back (user_cm_[0]);
    // user_cms_.push_back (user_cm_[1]);
    // user_cms_.push_back (user_cm_[2]);
    // user_cms_.push_back (1.); // this is the default scaling

    for(int i=0; i < 7; ++i)
        timers_[i] = 0;

    #ifdef NIFTIIO
      nifti_ref_image_ = NULL;
    #endif
  }

  ~TumorParameters() {
    #ifdef NIFTIIO
      if (nifti_ref_image_ != NULL)
        nifti_image_free(nifti_ref_image_);
    #endif
  }

  // tumor models
  int model_;

  // inversion
  int np_;
  int nk_;
  int nr_;
  int nt_;
  ScalarType dt_;
  ScalarType time_horizon_;

  // reaction diffusion
  ScalarType k_;
  ScalarType kf_;
  ScalarType rho_;
  ScalarType k_gm_wm_ratio_;
  ScalarType k_glm_wm_ratio_;
  ScalarType r_gm_wm_ratio_;
  ScalarType r_glm_wm_ratio_;

  // mass effect
  ScalarType E_csf_, E_healthy_, E_tumor_, E_bg_;
  ScalarType nu_csf_, nu_healthy_, nu_tumor_, nu_bg_;
  ScalarType screen_low_, screen_high_;
  ScalarType forcing_factor_;
  bool use_tanh_scaling_;

  // multi-species
  int num_species_;
  ScalarType ox_source_, ox_consumption_;
  ScalarType alpha_0_, beta_0_, ox_inv_, death_rate_, ox_hypoxia_;
  ScalarType invasive_threshold_;

  // initial condition (inversion)
  int sparsity_level_;            // should this go to opt?
  double thresh_component_weight_;
  std::vector<int> support_;      // support of cs guess
  // initial condition (parametrization)
  int bounding_box_;
  int max_p_location_;
  ScalarType p_scale_;
  ScalarType phi_sigma_;       // TODO(K) can we merge these two? <-- get rid of it
  ScalarType phi_sigma_data_driven_;
  ScalarType phi_spacing_factor_;
  ScalarType data_threshold_;
  ScalarType gaussian_vol_frac_;
  std::array<ScalarType, 3> user_cm_;

  // data
  ScalarType obs_threshold_0_;
  ScalarType obs_threshold_1_;
  ScalarType obs_lambda_;
  ScalarType smoothing_factor_data_;
  ScalarType smoothing_factor_data_t0_;
  ScalarType smoothing_factor_;
  ScalarType smoothing_factor_atlas_;
  ScalarType smoothing_factor_patient_;
  bool use_c0_;
  bool adv_velocity_set_;       /// @brief if true, velocity for explicit advection of material properties is provided
  bool relative_obs_threshold_;
  bool two_time_points_;

  // forward solvers, accuracy
  int interpolation_order_;
  int order_;

  // performance settings
  bool write_output_;
  bool phi_store_;
  bool adjoint_store_;
  bool time_history_off_;

  // misc
  ScalarType exp_shift_;
  ScalarType penalty_;
  bool beta_changed_;

  // auxiliary
  bool transport_mri_;
  int verbosity_;
  bool write_p_checkpoint_;
  TumorStatistics statistics_;
  std::array<double, 7> timers_;
  std::fstream outfile_sol_;
  std::fstream outfile_grad_;
  std::fstream outfile_glob_grad_;
  bool feature_compute_; // computes temporal biophysical features 

  // io paths
  std::string writepath_;
  std::string readpath_;
  // .nc or nifty output
  std::string ext_; // extension ".nc" or ".nii.gz"

  bool force_phi_compute_;

  #ifdef NIFTIIO
    nifti_image* nifti_ref_image_;
  #endif

};


/* #### ------------------------------------------------------------------- #### */
/* #### ========                Distributed Grid                   ======== #### */
/* #### ------------------------------------------------------------------- #### */

struct Grid {
public:
  Grid (int *n, int *isize, int *osize, int *istart, int *ostart,
        fft_plan *plan, MPI_Comm c_comm, int *c_dims) {
    memcpy (n_, n, 3 * sizeof(int));
    memcpy (c_dims_, c_dims, 2 * sizeof(int));
    memcpy (isize_, isize, 3 * sizeof(int));
    memcpy (osize_, osize, 3 * sizeof(int));
    memcpy (istart_, istart, 3 * sizeof(int));
    memcpy (ostart_, ostart, 3 * sizeof(int));
    plan_ = plan;
    c_comm_ = c_comm;
    accfft_alloc_max_ = plan->alloc_max;
    ng_ = n[0] * n[1] * n[2];
    nl_ = isize[0] * isize[1] * isize[2];
    h_[0] = M_PI * 2 / n[0];
    h_[1] = M_PI * 2 / n[1];
    h_[2] = M_PI * 2 / n[2];
    lebesgue_measure_ = h_[0] * h_[1] * h_[2];

    // gather all isizes to isize_gathered_ : needed for nifti I/O which is serial: only zero
    // contains the values.
    int rank, nprocs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    if (rank == 0) {
      isize_gathered_ = new int[3*nprocs];
      isize_offset_ = new int[nprocs];
      isize_send_ = new int[nprocs];
      istart_gathered_ = new int[3*nprocs];
    }

    MPI_Gather(&isize_[0], 3, MPI_INT, isize_gathered_, 3, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Gather(&istart_[0], 3, MPI_INT, istart_gathered_, 3, MPI_INT, 0, PETSC_COMM_WORLD);
    int offset = 0;
    if (rank == 0) {
        for (int i = 0; i < nprocs; i++) {
        isize_send_[i] = 1;
        for (int j = 0; j < 3; j++) {
          isize_send_[i] *= isize_gathered_[3*i + j];
        }
        isize_offset_[i] = offset;
        offset += isize_send_[i];
      }
    }
  }

  ~Grid() {
    MPI_Comm_free(&c_comm_);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      delete[] isize_send_;
      delete[] isize_gathered_;
      delete[] istart_gathered_;
      delete[] isize_offset_;
    }
  }

  int n_[3];
  int isize_[3];
  int osize_[3];
  int istart_[3];
  int ostart_[3];
  int c_dims_[3];
  ScalarType h_[3];
  int nl_;
  int64_t ng_;
  int64_t accfft_alloc_max_;
  fft_plan *plan_;
  MPI_Comm c_comm_;
  ScalarType lebesgue_measure_;

  int *isize_gathered_;  // needed for nifti I/O
  int *istart_gathered_;
  int *isize_send_;
  int *isize_offset_;
};


/* #### ------------------------------------------------------------------- #### */
/* #### ========             Tumor Solver Parameters               ======== #### */
/* #### ------------------------------------------------------------------- #### */

class Parameters {
  public:
    Parameters() :
      opt_(),
      optf_(),
      tu_(),
      grid_() {

      opt_ = std::make_shared<OptimizerSettings>();
      optf_ = std::make_shared<OptimizerFeedback>();
      tu_ = std::make_shared<TumorParameters>();
      // grid is initialized once n is known
    }

    // initialize distributed grid
    void createGrid(int *n, int *isize, int *osize, int *istart, int *ostart,
          fft_plan *plan, MPI_Comm c_comm, int *c_dims) {
      grid_ = std::make_shared<Grid>(n, isize, osize, istart, ostart, plan, c_comm, c_dims);
    }
    // inline int get_nk() {return opt_->diffusivity_inversion_ ? tu_->nk_ : 0;}
    inline int get_nk() {return (opt_->diffusivity_inversion_ || opt_->reaction_inversion_) ? tu_->nk_ : 0;} // TODO is this correct always? I think yes bc we never invert for rho and not for k
    inline int get_nr() {return opt_->reaction_inversion_ ? tu_->nr_ : 0;}

    virtual ~Parameters() {}

    // parameter structs
    std::shared_ptr<OptimizerSettings> opt_;  // optimizer settings
    std::shared_ptr<OptimizerFeedback> optf_; // optimizer feedback
    std::shared_ptr<TumorParameters> tu_;     // tumor parameters
    std::shared_ptr<Grid> grid_;              // computationl grid
};



/* #### ------------------------------------------------------------------- #### */
/* #### ========         Application Code Parameters               ======== #### */
/* #### ------------------------------------------------------------------- #### */

struct FilePaths {
  public:
    FilePaths() :
      seg_(), wm_(), gm_(), vt_(), csf_(),
      p_seg_(), p_wm_(), p_gm_(), p_vt_(), p_csf_(),
      data_t1_(), data_t0_(), data_support_(), data_support_data_(),
      data_comps_(), data_comps_data_(), obs_filter_(), mri_(),
      velocity_x1_(), velocity_x2_(), velocity_x3_(),
      pvec_(), phi_()
    {}

    // material properties atlas
    std::string seg_;
    std::string wm_;
    std::string gm_;
    std::string vt_;
    std::string csf_;
    // material properties patient
    std::string p_seg_;
    std::string p_wm_;
    std::string p_gm_;
    std::string p_vt_;
    std::string p_csf_;
    // data
    std::string data_t1_;
    std::string data_t0_;
    std::string data_support_;
    std::string data_support_data_;
    std::string data_comps_;
    std::string data_comps_data_;
    std::string obs_filter_;
    std::string mri_;
    // velocity
    std::string velocity_x1_;
    std::string velocity_x2_;
    std::string velocity_x3_;
    // warmstart solution
    std::string pvec_;
    std::string phi_;
};


struct SyntheticData {
public:
  SyntheticData() :
    enabled_(false),
    rho_(10),
    k_(1E-2),
    forcing_factor_(1E5),
    ox_hypoxia_(0.65),
    death_rate_(0.9),
    alpha_0_(0.15),
    ox_consumption_(8),
    ox_source_(55),
    beta_0_(0.02),
    dt_(0.01),
    nt_(100),
    testcase_(0),
    pre_adv_time_(-1),
    user_cms_()
  {}

  // attributes
  bool enabled_;
  ScalarType rho_;
  ScalarType k_;
  ScalarType forcing_factor_;
  ScalarType ox_hypoxia_;
  ScalarType death_rate_;
  ScalarType alpha_0_;
  ScalarType ox_consumption_;
  ScalarType ox_source_;
  ScalarType beta_0_;
  ScalarType dt_;
  int nt_;
  int testcase_;
  ScalarType pre_adv_time_;
  std::vector< std::array<ScalarType, 4> > user_cms_;
};

struct Prediction {
public:
  Prediction() :
    enabled_(false),
    dt_(0.01),
    t_pred_(),
    true_data_path_(),
    wm_path_(),
    gm_path_(),
    vt_path_(),
    csf_path_()
  {}

  // attributes
  bool enabled_;
  ScalarType dt_;
  std::vector< ScalarType > t_pred_;
  std::vector< std::string > true_data_path_;
  std::vector< std::string > wm_path_;
  std::vector< std::string > gm_path_;
  std::vector< std::string > vt_path_;
  std::vector< std::string > csf_path_;
};

/// @brief only used in Solver for setup and tests
class ApplicationSettings {
  public:
    ApplicationSettings()
    :
      inject_solution_(false),
      gaussian_selection_mode_(1),
      path_(),
      syn_(),
      pred_(),
      atlas_seg_(),
      patient_seg_() {

      path_ = std::make_shared<FilePaths>();
      syn_ = std::make_shared<SyntheticData>();
      pred_ = std::make_shared<Prediction>();
    }

    virtual ~ApplicationSettings() {}

    // attributes
    bool inject_solution_;
    int gaussian_selection_mode_;
    std::shared_ptr<FilePaths> path_;
    std::shared_ptr<SyntheticData> syn_;
    std::shared_ptr<Prediction> pred_;
    std::vector<int> atlas_seg_;
    std::vector<int> patient_seg_;
};

#endif
