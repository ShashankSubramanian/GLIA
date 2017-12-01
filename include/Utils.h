#ifndef _UTILS_H
#define _UTILS_H

//#define POSITIVITY
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
#include <accfft_utils.h>
#include "EventTimings.hpp"



enum {QDFS = 0, SLFS = 1};
enum {CONSTCOEF = 1, SINECOEF = 2, BRAIN = 0};
enum {GAUSSNEWTON = 0, QUASINEWTON = 1};

struct OptimizerSettings {
    double beta;                 /// @brief regularization parameter
    double opttolgrad;           /// @brief l2 gradient tolerance for optimization
    double gtolbound;            /// @brief minimum reduction of gradient (even if maxiter hit earlier)
    double grtol;                /// @brief rtol TAO (relative tolerance for gradient, not used)
    double gatol;                /// @brief atol TAO (absolute tolerance for gradient)
    int    newton_maxit;         /// @brief maximum number of allowed newton iterations
    int    krylov_maxit;         /// @brief maximum number of allowed krylov iterations
    int    newton_minit;         /// @brief minimum number of newton steps
    int    iterbound;            /// @brief if GRADOBJ conv. crit is used, max number newton it
    int    fseqtype;             /// @brief type of forcing sequence (quadratic, superlinear)
    int    newtonsolver;         /// @brief type of newton slver (0=GN, 1=QN, 2=GN/QN)
    int    verbosity;            /// @brief controls verbosity of solver
    bool   lmvm_set_hessian;     /// @brief if true lmvm initial hessian ist set as matvec routine
    bool   reset_tao;            /// @brief if true TAO is destroyed and re-created for every new inversion solve, if not, old structures are kept.

    OptimizerSettings ()
    :
    beta (1E-3),
    opttolgrad (1E-3),
    gtolbound (0.8),
    grtol (1E-12),
    gatol (1E-6),
    newton_maxit (500),
    krylov_maxit (30),
    newton_minit (1),
    iterbound (200),
    fseqtype (SLFS),
    newtonsolver(GAUSSNEWTON),
    reset_tao(false),
    lmvm_set_hessian(false),
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
    converged (false)
    {}
};

struct TumorSettings {
    int    tumor_model;
    double time_step_size;
    int    time_steps;
    double time_horizon;
    int    np;
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

    TumorSettings () :
    tumor_model(1),
    time_step_size(0.01),
    time_steps(16),
    time_horizon(0.16),
    np(27),
    betap(1E-3),
    writeOutput(false),
    verbosity(1),
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
    phi_sigma (PETSC_PI/10)
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
        NMisc (int *n, int *isize, int *osize, int *istart, int *ostart, accfft_plan *plan, MPI_Comm c_comm, int testcase = BRAIN)
        : model_ (1)   //Reaction Diffusion --  1 , Positivity -- 2
                       // Modified Obj -- 3
        , dt_ (0.01)
        , nt_(16)
        , np_ (27)
        , k_ (1.0E-2)
        , kf_(0.0)
        , rho_ (10)
        , p_scale_ (0.0)
        , p_scale_true_ (1.0)
        , noise_scale_(0.0)
        , beta_ (1e-3)
        , writeOutput_ (0)
        , verbosity_ (1)
        , k_gm_wm_ratio_ (1.0 / 10.0)
        , k_glm_wm_ratio_ (0.0)
        , r_gm_wm_ratio_ (1.0 / 5.0)
        , r_glm_wm_ratio_ (1.0)
        , phi_sigma_ (PETSC_PI / 10)
        , phi_spacing_factor_ (1.5)
        , obs_threshold_ (-1.0)
        , statistics_()
        , exp_shift_ (10.0)
        , penalty_ (1E-4)
        , testcase_ (testcase) {

            time_horizon_ = nt_ * dt_;
            if (testcase_ == BRAIN) {
                user_cm_[0] = 4.0;
                user_cm_[1] = 2.03;
                user_cm_[2] = 2.07;
            }
            else {
                user_cm_[0] = M_PI;
                user_cm_[1] = M_PI;
                user_cm_[2] = M_PI;
            }

            memcpy (n_, n, 3 * sizeof(int));
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

            for(int i=0; i < 7; ++i)
                timers_[i] = 0;
        }

        int testcase_;
        int n_[3];
        int isize_[3];
        int osize_[3];
        int istart_[3];
        int ostart_[3];
        double h_[3];

        int np_;
        double time_horizon_;
        double dt_;
        int nt_;

        int model_;
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
        double beta_;

        double phi_sigma_;
        double phi_spacing_factor_;

        double obs_threshold_;

        TumorStatistics statistics_;
        std::array<double, 7> timers_;

        int64_t accfft_alloc_max_;
        int64_t n_local_;
        int64_t n_global_;

        accfft_plan *plan_;
        MPI_Comm c_comm_;

};

int weierstrassSmoother (double *Wc, double *c, std::shared_ptr<NMisc> n_misc, double sigma); //TODO: Clean up .cpp file
PetscErrorCode enforcePositivity (Vec c, std::shared_ptr<NMisc> n_misc);
PetscErrorCode geometricCoupling(Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg, Vec mR_wm, Vec mR_gm, Vec mR_csf, Vec mR_glm, Vec mR_bg, Vec c1, std::shared_ptr<NMisc> nmisc);
PetscErrorCode geometricCouplingAdjoint(PetscScalar *sqrdl2norm,
	Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg,
	Vec mR_wm, Vec mR_gm, Vec mR_csf, Vec mR_glm, Vec mR_bg,
	Vec mT_wm, Vec mT_gm, Vec mT_csf, Vec mT_glm, Vec mT_bg);

//Read/Write function prototypes
void dataIn (double *A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataIn (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataOut (double *A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataOut (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname);

/* helper methods for print out to console */
PetscErrorCode tuMSG(std::string msg, int size = 98);
PetscErrorCode tuMSGstd(std::string msg, int size = 98);
PetscErrorCode tuMSGwarn(std::string msg, int size = 98);
PetscErrorCode _tuMSG(std::string msg, std::string color, int size);

/* accfft differential operators */
void accfft_grad (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, accfft_plan *plan, std::bitset<3> *pXYZ, double *timers);
void accfft_divergence (Vec div, Vec dx, Vec dy, Vec dz, accfft_plan *plan, double *timers);

/* definition of tumor assert */
#ifndef NDEBUG
#   define TU_assert(Expr, Msg) \
    __TU_assert(#Expr, Expr, __FILE__, __LINE__, Msg);
#else
#   define TU_assert(Expr, Msg);
#endif
void __TU_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg);


#endif // end _UTILS_H
