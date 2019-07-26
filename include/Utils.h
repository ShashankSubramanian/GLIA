#ifndef _UTILS_H
#define _UTILS_H

#include <petsc.h>
#include <stdlib.h>
#include <iomanip>
#include "petsctao.h"
#include <accfft.h>
#include <accfft_operators.h>
#include <math.h>
#include <memory>
#include <complex>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <omp.h>
#include <complex>
#include <cmath>
#include <vector>
#include <queue>
#include <accfft_utils.h>
#include <assert.h>
#include <sys/stat.h>
#include "EventTimings.hpp"



class Phi;

enum {QDFS = 0, SLFS = 1};
enum {CONSTCOEF = 1, SINECOEF = 2, BRAIN = 0, BRAINNEARMF = 3, BRAINFARMF = 4};
enum {GAUSSNEWTON = 0, QUASINEWTON = 1};
enum {ARMIJO = 0, MT = 1};
enum {L1 = 0, L2 = 1, wL2 = 3, L2b = 4};

struct OptimizerSettings {
    double beta;                 /// @brief regularization parameter
    double opttolgrad;           /// @brief l2 gradient tolerance for optimization
    double ftol;                 /// @brief l1 function and solution tolerance
    double ls_minstep;           /// @brief minimum step length of linesearch
    double gtolbound;            /// @brief minimum reduction of gradient (even if maxiter hit earlier)
    double grtol;                /// @brief rtol TAO (relative tolerance for gradient, not used)
    double gatol;                /// @brief atol TAO (absolute tolerance for gradient)
    int    newton_maxit;         /// @brief maximum number of allowed newton iterations
    int    gist_maxit;           /// @brief maximum number of GIST iterations
    int    krylov_maxit;         /// @brief maximum number of allowed krylov iterations
    int    newton_minit;         /// @brief minimum number of newton steps
    int    iterbound;            /// @brief if GRADOBJ conv. crit is used, max number newton it
    int    fseqtype;             /// @brief type of forcing sequence (quadratic, superlinear)
    int    newtonsolver;         /// @brief type of newton slver (0=GN, 1=QN, 2=GN/QN)
    int    linesearch;           /// @brief type of line-search used (0=armijo, 1=mt)
    int    regularization_norm;  /// @brief defines the type of regularization (L1, L2, or weighted-L2)
    int    verbosity;            /// @brief controls verbosity of solver
    bool   lmvm_set_hessian;     /// @brief if true lmvm initial hessian ist set as matvec routine
    bool   reset_tao;            /// @brief if true TAO is destroyed and re-created for every new inversion solve, if not, old structures are kept.
    bool   diffusivity_inversion;/// @brief if true, we also invert for k_i scalings of material properties to construct isotropic part of diffusion coefficient

    OptimizerSettings ()
    :
    beta (0E-3),
    opttolgrad (1E-5),
    ftol (1E-5),
    ls_minstep (1E-9),
    gtolbound (0.8),
    grtol (1E-5),
    gatol (1E-8),
    newton_maxit (30),
    gist_maxit (5),
    krylov_maxit (30),
    newton_minit (1),
    iterbound (200),
    fseqtype (SLFS),
    newtonsolver (QUASINEWTON),
    linesearch (MT),
    regularization_norm (L2),
    reset_tao (false),
    lmvm_set_hessian (false),
    diffusivity_inversion(false),
    verbosity (1)
    {}
};


struct OptimizerFeedback {
    int nb_newton_it;            /// @brief stores the number of required Newton iterations for the last inverse tumor solve
    int nb_krylov_it;            /// @brief stores the number of required (accumulated) Krylov iterations for the last inverse tumor solve
    int nb_matvecs;              /// @brief stores the number of required (accumulated) matvecs per tumor solve
    int nb_objevals;             /// @brief stores the number of required (accumulated) objective function evaluations per tumor solve
    int nb_gradevals;            /// @brief stores the number of required (accumulated) gradient evaluations per tumor solve
    std::string solverstatus;    /// @brief gives information about the termination reason of inverse tumor TAO solver
    double gradnorm;             /// @brief final gradient norm
    double gradnorm0;            /// @brief norm of initial gradient (with p = intial guess)
    double j0;                   /// @brief initial objective : needed for L1 convergence tests
    double jval;                 /// @brief orbjective function value
    bool converged;              /// @brief true if solver converged within bounds

    OptimizerFeedback ()
    :
    nb_newton_it (-1),
    nb_krylov_it (-1),
    nb_matvecs(-1),
    nb_objevals(-1),
    nb_gradevals(-1),
    solverstatus (),
    gradnorm (0.),
    gradnorm0 (0.),
    j0 (0.),
    jval(0.),
    converged (false)
    {}
};

struct TumorSettings {
    int    tumor_model;
    double time_step_size;
    int    time_steps;
    double time_horizon;
    int    np;
    int    nk;
    double betap;
    bool   writeOutput;
    int    verbosity;
    double obs_threshold;
    double diff_coeff_scale;        /// @brief (scalar) diffusion rate
    double diff_coeff_scale_anisotropic; /// @brief (scalar) anisotropic diffusion rate
    double reaction_coeff_scale;    /// @brief (scalar) reaction rate
    double diffusion_ratio_gm_wm;   /// @brief ratio of diffusion coefficient between wm and gm
    double diffusion_ratio_glm_wm;  /// @brief ratio of diffusion coefficient between wm and gm
    double reaction_ratio_gm_wm;    /// @brief ratio of reaction coefficient between wm and gm
    double reaction_ratio_glm_wm;   /// @brief ratio of reaction coefficient between wm and gm
    int rho_linear;                 /// @brief used linearization
    std::array<double, 3> phi_center_of_mass; /// @brief center of mass of the tumor, center of the Gaussian mesh
    double phi_spacing_factor;      /// @brief defines spacing of Gaussian ansatz functions as multiple of sigma
    double phi_sigma;               /// @brief standard deviation of Gaussians
    double phi_sigma_data_driven;   /// @brief standard deviation for data driven selection of gaussians
    double gaussian_volume_fraction;/// @brief defines the volume frqction of tumor cells within sigma such that gaussian is enabled, when selection mode is adaptive datadriven
    double target_sparsity;         /// @brief defines the target sparsity of a solution causing the L1 solve to terminate
    int phi_selection_mode_bbox;    /// @brief flag for phi selectin mode. If set, initialize bounding box
    bool diffusivity_inversion;     /// @brief if true, we also invert for k_i scalings of material properties to construct isotropic part of diffusion coefficient

    TumorSettings () :
    tumor_model(1),
    time_step_size(0.01),
    time_steps(16),
    time_horizon(0.16),
    np(27),
    nk(2),
    betap(1E-3),
    writeOutput(false),
    verbosity(3),
    obs_threshold(0.0),
    diff_coeff_scale(1E-2),
    diff_coeff_scale_anisotropic(0.0),
    reaction_coeff_scale(15),
    diffusion_ratio_gm_wm(1.0 / 10.0),
    diffusion_ratio_glm_wm(0.0),
    reaction_ratio_gm_wm(1.0 / 5.0),
    reaction_ratio_glm_wm(0.0),
    rho_linear(0),
    phi_center_of_mass{ {0.5f*2 * PETSC_PI, 0.5*2 * PETSC_PI, 0.5*2 * PETSC_PI} },
    phi_spacing_factor (1.5),
    phi_sigma (PETSC_PI/10),
    phi_sigma_data_driven(2*PETSC_PI/256),
    gaussian_volume_fraction(0),
    target_sparsity(0.99),
    phi_selection_mode_bbox(1),
    diffusivity_inversion(false)
    {}
};

struct TumorStatistics {
  int nb_state_solves;            /// @brief number of state equation solves
  int nb_adjoint_solves;          /// @brief number of adjoint equation solves
  int nb_grad_evals;              /// @brief number of gradient evaluations
  int nb_obj_evals;               /// @brief number of objective evaluations
  int nb_hessian_evals;           /// @brief number of hessian evaluations

  int nb_state_solves_acc;        /// @brief number of state equation solves
  int nb_adjoint_solves_acc;      /// @brief number of adjoint equation solves
  int nb_grad_evals_acc;          /// @brief number of gradient evaluations
  int nb_obj_evals_acc;           /// @brief number of objective evaluations
  int nb_hessian_evals_acc;       /// @brief number of hessian evaluations

public:
  TumorStatistics() :
  nb_state_solves(0),
  nb_adjoint_solves(0),
  nb_grad_evals(0),
  nb_obj_evals(0),
  nb_hessian_evals(0),
  nb_state_solves_acc(0),
  nb_adjoint_solves_acc(0),
  nb_grad_evals_acc(0),
  nb_obj_evals_acc(0),
  nb_hessian_evals_acc(0)
  {}

  void reset() {
    nb_state_solves_acc     += nb_state_solves;
    nb_adjoint_solves_acc   += nb_adjoint_solves;
    nb_grad_evals_acc       += nb_grad_evals;
    nb_obj_evals_acc        += nb_obj_evals;
    nb_hessian_evals_acc    += nb_hessian_evals;
    nb_state_solves         = 0;
    nb_adjoint_solves       = 0;
    nb_grad_evals           = 0;
    nb_obj_evals            = 0;
    nb_hessian_evals        = 0;
  }

  void reset0() {
    nb_state_solves_acc     = 0;
    nb_adjoint_solves_acc   = 0;
    nb_grad_evals_acc       = 0;
    nb_obj_evals_acc        = 0;
    nb_hessian_evals_acc    = 0;
    nb_state_solves         = 0;
    nb_adjoint_solves       = 0;
    nb_grad_evals           = 0;
    nb_obj_evals            = 0;
    nb_hessian_evals        = 0;
  }

  PetscErrorCode print();
};


class NMisc {
    public:
        NMisc (int *n, int *isize, int *osize, int *istart, int *ostart, accfft_plan *plan, MPI_Comm c_comm, int *c_dims, int testcase = BRAIN)
        : model_ (1)                            //Reaction Diffusion --  1 , Positivity -- 2
                                                // Modified Obj -- 3
                                                // Mass effect -- 4
        , dt_ (0.5)                             // Time step
        , nt_(1)                                // Total number of time steps
        , np_ (1)                               // Number of gaussians for bounding box
        , nk_ (1)                               // Number of k_i that we like to invert for (1-3)
        , nr_ (1)                               // number of rho_i that we like to invert for (1-3)
        , k_ (0E-1)                             // Isotropic diffusion coefficient
        , kf_(0.0)                              // Anisotropic diffusion coefficient
        , rho_ (10)                             // Reaction coefficient
        , p_scale_ (0.0)                        // Scaling factor for initial guess
        , p_scale_true_ (1.0)                   // Scaling factor for synthetic data generation
        , noise_scale_(0.0)                     // Noise scale
        , low_freq_noise_scale_ (0.25)          // Low freq noise scale
        , beta_ (0e-3)                          // Regularization parameter
        , lambda_ (1e5)                         // Regularization parameter for L1
        , lambda_continuation_ (true)           // bool for parameter continuation
        , target_sparsity_ (0.99)               // target sparsity for L1 continuation
        , writeOutput_ (1)                      // Print flag for paraview visualization
        , verbosity_ (1)                        // Print flag for optimization routines
        , k_gm_wm_ratio_ (0.0 / 1.0)            // gm to wm diffusion coeff ratio
        , k_glm_wm_ratio_ (0.0)                 // glm to wm diffusion coeff ratio
        , r_gm_wm_ratio_ (0.0 / 5.0)            // gm to wm reaction coeff ratio
        , r_glm_wm_ratio_ (0.0)                 // glm to wm diffusion coeff ratio
        , phi_sigma_ (2 * M_PI / 64)            // Gaussian standard deviation for bounding box
        , phi_sigma_data_driven_ (2 * M_PI / 256) // Sigma for data-driven gaussians
        , phi_spacing_factor_ (1.5)             // Gaussian spacing for bounding box
        , obs_threshold_ (0.0)                  // Observation threshold
        , statistics_()                         //
        , exp_shift_ (10.0)                     // Parameter for positivity shift
        , penalty_ (1E-4)                       // Parameter for positivity objective function
        , data_threshold_ (0.05)                // Data threshold to set custom gaussians
        , gaussian_vol_frac_ (0.0)              // Volume fraction of gaussians to set custom basis functions
        , bounding_box_ (0)                     // Flag to set bounding box for gaussians
        , testcase_ (testcase)                  // Testcases
        , nk_fixed_ (true)                      // if true, nk cannot be changed anymore
        , regularization_norm_(L2b)             // defines the tumor regularization norm, L1, L2, or weighted L2
        , diffusivity_inversion_ (false)        // if true, we also invert for k_i scalings of material properties to construct isotropic part of diffusion coefficient
        , reaction_inversion_ (false)           // Automatically managed inside the code: We can only invert for reaction given some constraints on the solution
        , flag_reaction_inv_ (false)            // This switch is turned on automatically when reaction iversion is used for the separate final tao solver
        , beta_changed_ (false)                 // if true, we overwrite beta with user provided beta: only for tumor inversion standalone
        , write_p_checkpoint_(true)             // if true, p vector and corresponding Gaussian centers are written to file at certain checkpoints
        , newton_solver_ (QUASINEWTON)          // Newton solver type
        , linesearch_ (MT)                      // Line-search type
        , newton_maxit_ (30)                    // Newton max itr
        , gist_maxit_ (50)                      // GIST max itr
        , krylov_maxit_ (30)                    // Krylov max itr
        , opttolgrad_ (1E-5)                    // Relative gradient tolerance of L2 solves
        , sparsity_level_ (5)                   // Level of sparsity for L1 solves
        , smoothing_factor_ (1)                 // Smoothing factor
        , max_p_location_ (0)                   // Location of maximum gaussian scale concentration - this is used to set bounds for reaction inversion
        , ic_max_ (0)                           // Maximum value of reconstructed initial condition with wrong reaction coefficient - this is used to rescale the ic to 1
        , predict_flag_ (0)                     // Flag to perform future tumor growth prediction after inversion
        , order_ (2)                            // Order of accuracy for PDE solves
        , nu_healthy_ (0.4)                     // Poisson's ratio of wm and gm
        , nu_tumor_ (0.45)                      // Poisson's ratio of tumor
        , nu_bg_ (0.48)                         // Poisson's ratio of background
        , nu_csf_ (0.1)                         // Poisson's ratio of CSF
        , E_healthy_ (2100)                     // Young's modulus of wm and gm
        , E_bg_ (15000)                         // Young's modulus of background
        , E_tumor_ (10000)                      // Young's modulus of tumor
        , E_csf_ (10)                           // Young's modulus of CSF
        , screen_low_ (0)                       // low screening coefficient
        , screen_high_ (1E4)                    // high screening
        , forcing_factor_ (2.0E5)               // mass effect forcing factor
        , prune_components_ (1)                 // prunes L2 solution based on components
        , multilevel_ (0)                       // scales INT_Omega phi(x) dx = const across levels
        , phi_store_ (false)                    // Flag to store phis 
        , adjoint_store_ (true)                 // Flag to store half-step concentrations for adjoint solve to speed up time to solution
        , k_lb_ (1E-3)                           // Lower bound on kappa - depends on mesh; 1E-3 for 128^3 1E-4 for 256^3
        , k_ub_ (1)                              // Upper bound on kappa
                                {


            time_horizon_ = nt_ * dt_;
            if (testcase_ == BRAIN || testcase_ == BRAINFARMF || testcase_ == BRAINNEARMF) {
                // user_cm_[0] = 4.0;
                // user_cm_[1] = 2.53;
                // user_cm_[2] = 2.57;

                // tumor is near the top edge -- more wm tracts visible
                // user_cm_[0] = 2 * M_PI / 128 * 68;//82  //Z
                // user_cm_[1] = 2 * M_PI / 128 * 88;//64  //Y
                // user_cm_[2] = 2 * M_PI / 128 * 72;//52  //X

                // tumor is top middle
                // user_cm_[0] = 2 * M_PI / 128 * 80;//82  //Z
                // user_cm_[1] = 2 * M_PI / 128 * 64;//64  //Y
                // user_cm_[2] = 2 * M_PI / 128 * 76;//52  //X

                // tc1 and 2
                user_cm_[0] = 2 * M_PI / 128 * 56;//82  //Z
                user_cm_[1] = 2 * M_PI / 128 * 68;//64  //Y
                user_cm_[2] = 2 * M_PI / 128 * 72;//52  //X

                // casebrats for mass effect
                // user_cm_[0] = 2 * M_PI / 128 * 72;//82  //Z
                // user_cm_[1] = 2 * M_PI / 128 * 92;//64  //Y
                // user_cm_[2] = 2 * M_PI / 128 * 72;//52  //X

                // tc2
                // user_cm_[0] = 2 * M_PI / 128 * 72;//82  //Z
                // user_cm_[1] = 2 * M_PI / 128 * 84;//64  //Y
                // user_cm_[2] = 2 * M_PI / 128 * 80;//52  //X

                // user_cm_[0] = 2 * M_PI / 64 * 32;//82  //Z
                // user_cm_[1] = 2 * M_PI / 64 * 24;//64  //Y
                // user_cm_[2] = 2 * M_PI / 64 * 40;//52  //X

                user_cms_.push_back (user_cm_[0]);
                user_cms_.push_back (user_cm_[1]);
                user_cms_.push_back (user_cm_[2]);
                user_cms_.push_back (1.); // this is the default scaling

            }
            else {
                user_cm_[0] = M_PI;
                user_cm_[1] = M_PI;
                user_cm_[2] = M_PI;

                user_cms_.push_back (user_cm_[0]);
                user_cms_.push_back (user_cm_[1]);
                user_cms_.push_back (user_cm_[2]);
                user_cms_.push_back (1.); // this is the default scaling
            }

            memcpy (n_, n, 3 * sizeof(int));
            memcpy (c_dims_, c_dims, 2 * sizeof(int));
            memcpy (isize_, isize, 3 * sizeof(int));
            memcpy (osize_, osize, 3 * sizeof(int));
            memcpy (istart_, istart, 3 * sizeof(int));
            memcpy (ostart_, ostart, 3 * sizeof(int));

            plan_ = plan;
            c_comm_ = c_comm;
            accfft_alloc_max_ = plan->alloc_max;

            n_global_ = n[0] * n[1] * n[2];
            n_local_ = isize[0] * isize[1] * isize[2];
            h_[0] = M_PI * 2 / n[0];
            h_[1] = M_PI * 2 / n[1];
            h_[2] = M_PI * 2 / n[2];

            lebesgue_measure_ = h_[0] * h_[1] * h_[2];

            for(int i=0; i < 7; ++i)
                timers_[i] = 0;

            //Read and write paths
            readpath_ << "./brain_data/" << n_[0] <<"/";
            writepath_ << "./results/";
        }

        double k_lb_;
        double k_ub_;

        double ic_max_;
        int predict_flag_;
        int prune_components_;


        bool phi_store_;
        bool adjoint_store_;

        int testcase_;
        int n_[3];
        int isize_[3];
        int osize_[3];
        int istart_[3];
        int ostart_[3];
        int c_dims_[2];
        double h_[3];
        double lebesgue_measure_;

        int np_;
        int nk_;
        int nr_;
        double time_horizon_;
        double dt_;
        int nt_;

        int model_;
        int regularization_norm_;
        int bounding_box_;
        int writeOutput_;
        int verbosity_;

        double exp_shift_;
        double penalty_;

        double k_;
        double kf_;
        double k_gm_wm_ratio_;
        double k_glm_wm_ratio_;
        double r_gm_wm_ratio_;
        double r_glm_wm_ratio_;
        double rho_;
        std::array<double, 3> user_cm_;
        double p_scale_;
        double p_scale_true_;
        double noise_scale_;
        double low_freq_noise_scale_;
        double beta_;
        double lambda_;

        bool beta_changed_;
        bool write_p_checkpoint_;

        double phi_sigma_;
        double phi_sigma_data_driven_;
        double phi_spacing_factor_;
        double data_threshold_;
        double gaussian_vol_frac_;

        double obs_threshold_;

        bool nk_fixed_;
        bool diffusivity_inversion_;
        bool lambda_continuation_;
        bool reaction_inversion_;
        bool flag_reaction_inv_;
        bool multilevel_;

        double target_sparsity_;

        int max_p_location_;

        int sparsity_level_;

        int order_;

        TumorStatistics statistics_;
        std::array<double, 7> timers_;

        int64_t accfft_alloc_max_;
        int64_t n_local_;
        int64_t n_global_;

        accfft_plan *plan_;
        MPI_Comm c_comm_;

        std::stringstream readpath_;
        std::stringstream writepath_;

        int newton_solver_, linesearch_, newton_maxit_, gist_maxit_, krylov_maxit_;
        double opttolgrad_;

        std::vector<int> support_;      // support of cs guess

        std::vector<double> user_cms_;  // stores the cms for synthetic user data

        double smoothing_factor_;

        double E_csf_, E_healthy_, E_tumor_, E_bg_;
        double nu_csf_, nu_healthy_, nu_tumor_, nu_bg_;

        double screen_low_, screen_high_;

        double forcing_factor_;
};

class VecField {
    public:
        VecField (int nl, int ng);
        Vec x_;
        Vec y_;
        Vec z_;
        Vec magnitude_;
        ~VecField () {
            PetscErrorCode ierr = 0;
            ierr = VecDestroy (&x_);
            ierr = VecDestroy (&y_);
            ierr = VecDestroy (&z_);
            ierr = VecDestroy (&magnitude_);
        }

        PetscErrorCode computeMagnitude ();
        PetscErrorCode copy (std::shared_ptr<VecField> field);
        PetscErrorCode getComponentArrays (double *&x_ptr, double *&y_ptr, double *&z_ptr);
        PetscErrorCode restoreComponentArrays (double *&x_ptr, double *&y_ptr, double *&z_ptr);
        PetscErrorCode setIndividualComponents (Vec in);  // uses indivdual components from in and sets it to x,y,z
        PetscErrorCode getIndividualComponents (Vec in);  // uses x,y,z to populate in
};

/**
    Context structure for user-defined linesearch routines needed
    for L1 minimization problems
**/
struct LSCtx {
    Vec x_work_1;   //Temporary vector for storing steepest descent guess
    Vec x_work_2; //Work vector
    Vec x_sol;
    double sigma; //Sufficient decrease parameter
    double lambda; //Regularization parameter for L1: Linesearch needs
                   //this application specific info
    PetscReal J_old;
};

int weierstrassSmoother (double *Wc, double *c, std::shared_ptr<NMisc> n_misc, double sigma); //TODO: Clean up .cpp file
PetscErrorCode enforcePositivity (Vec c, std::shared_ptr<NMisc> n_misc);
PetscErrorCode checkClipping (Vec c, std::shared_ptr<NMisc> n_misc);

/// @brief computes geometric tumor coupling m1 = m0(1-c(1))
PetscErrorCode geometricCoupling(
  Vec m1_wm, Vec m1_gm, Vec m1_csf, Vec m1_glm, Vec m1_bg,
  Vec m0_wm, Vec m0_gm, Vec m0_csf, Vec m0_glm, Vec m0_bg,
  Vec c1, std::shared_ptr<NMisc> nmisc);

/** @brief computes difference xi = m_data - m_geo
 *  - function assumes that on input, xi = m_geo * (1-c(1))
 */
PetscErrorCode geometricCouplingAdjoint(PetscScalar *sqrdl2norm,
	Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg,
	Vec m_geo_wm, Vec m_geo_gm, Vec m_geo_csf, Vec m_geo_glm, Vec m_geo_bg,
	Vec m_data_wm, Vec m_data_gm, Vec m_data_csf, Vec m_data_glm, Vec m_data_bg);

/// @brief computes difference diff = x - y
PetscErrorCode computeDifference(PetscScalar *sqrdl2norm,
	Vec diff_wm, Vec diff_gm, Vec diff_csf, Vec diff_glm, Vec diff_bg,
	Vec x_wm, Vec x_gm, Vec x_csf, Vec x_glm, Vec x_bg,
	Vec y_wm, Vec y_gm, Vec y_csf, Vec y_glm, Vec y_bg);

//Read/Write function prototypes
void dataIn (double *A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataIn (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataOut (double *A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataOut (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname);
/// @reads in binary vector, serial
PetscErrorCode readBIN(Vec* x, int size2, std::string f);
/// @brief writes out vector im binary format, serial
PetscErrorCode writeBIN(Vec x, std::string f);
/// @reads in p_vec vector from txt file
PetscErrorCode readPVec(Vec* x, int size, int np, std::string f);
/// @brief reads in Gaussian centers from file
PetscErrorCode readPhiMesh(std::vector<double> &centers, std::shared_ptr<NMisc> n_misc, std::string f, bool read_comp_data = false, std::vector<int> *comps = nullptr);
/// @brief reads connected component data from file
PetscErrorCode readConCompDat(std::vector<double> &weights, std::vector<double> &centers, std::string f);
/// @brief write checkpoint for p-vector and Gaussian centers
PetscErrorCode writeCheckpoint(Vec p, std::shared_ptr<Phi> phi, std::string path, std::string suffix);
/// @brief returns only filename
PetscErrorCode getFileName(std::string& filename, std::string file);
/// @brief returns filename, extension and path
PetscErrorCode getFileName(std::string& path, std::string& filename, std::string& extension, std::string file);
/// @brief checks if file exists
bool fileExists(const std::string& filename);

/* helper methods for print out to console */
PetscErrorCode tuMSG(std::string msg, int size = 98);
PetscErrorCode tuMSGstd(std::string msg, int size = 98);
PetscErrorCode tuMSGwarn(std::string msg, int size = 98);
PetscErrorCode _tuMSG(std::string msg, std::string color, int size);

/* accfft differential operators */
void accfft_grad (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, accfft_plan *plan, std::bitset<3> *pXYZ, double *timers);
void accfft_divergence (Vec div, Vec dx, Vec dy, Vec dz, accfft_plan *plan, double *timers);

PetscErrorCode vecSign (Vec x); //signum of petsc vector
PetscErrorCode vecSparsity (Vec x, double &sparsity); //Hoyer measure for sparsity of vector

/* definition of tumor assert */
#ifndef NDEBUG
#   define TU_assert(Expr, Msg) \
    __TU_assert(#Expr, Expr, __FILE__, __LINE__, Msg);
#else
#   define TU_assert(Expr, Msg);
#endif
void __TU_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg);

PetscErrorCode hardThreshold (Vec x, int sparsity_level, int sz, std::vector<int> &support, int &nnz);
PetscErrorCode hardThreshold (Vec x, int sparsity_level, int sz, std::vector<int> &support, std::vector<int> labels, std::vector<double> weights, int &nnz, int num_components);

double myDistance (double *c1, double *c2);

PetscErrorCode computeCenterOfMass (Vec x, int *isize, int *istart, double *h, double *cm);

#endif // end _UTILS_H
