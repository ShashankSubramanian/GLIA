/**
 *  SIBIA (Scalable Biophysics-Based Image Analysis)
 *
 *  Copyright (C) 2017-2020, The University of Texas at Austin
 *  This file is part of the SIBIA library.
 *
 *  Authors: Klaudius Scheufele, Shashank Subramanian
 *
 *  SIBIA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SIBIA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <stdlib.h>

struct OptimizerSettings {
public:
  OptimizerSettings ()
  :
    beta (0E-3),
    opttolgrad (1E-3),
    ftol (1E-3),
    ls_minstep (1E-9),
    gtolbound (0.8),
    grtol (1E-5),
    gatol (1E-4),
    newton_maxit (30),
    newton_minit (1),
    gist_maxit (5),
    krylov_maxit (30),
    iterbound (200),
    fseqtype (SLFS),
    newtonsolver (QUASINEWTON),
    linesearch (MT),
    regularization_norm (L2),
    reset_tao (false),
    cosamp_stage (0),
    lmvm_set_hessian (false),
    diffusivity_inversion(true),
    reaction_inversion(false),
    verbosity (3),
    k_lb (1E-3),
    k_ub (1
  {}

  ScalarType beta;             /// @brief regularization parameter
  ScalarType opttolgrad;       /// @brief l2 gradient tolerance for optimization
  ScalarType ftol;             /// @brief l1 function and solution tolerance
  ScalarType ls_minstep;       /// @brief minimum step length of linesearch
  ScalarType gtolbound;        /// @brief minimum reduction of gradient (even if maxiter hit earlier)
  ScalarType grtol;            /// @brief rtol TAO (relative tolerance for gradient, not used)
  ScalarType gatol;            /// @brief atol TAO (absolute tolerance for gradient)
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
  int    cosamp_stage;         /// @brief indicates stage of cosamp solver for warmstart
  bool   lmvm_set_hessian;     /// @brief if true lmvm initial hessian ist set as matvec routine
  bool   reset_tao;            /// @brief if true TAO is destroyed and re-created for every new inversion solve, if not, old structures are kept.
  bool   diffusivity_inversion;/// @brief if true, we also invert for k_i scalings of material properties to construct isotropic part of diffusion coefficient
  bool   reaction_inversion;   /// @brief if true, we also invert for rho
  ScalarType k_lb;             /// @brief lower bound on kappa - depends on mesh; 1E-3 for 128^3 1E-4 for 256^3
  ScalarType k_ub;             /// @brief upper bound on kappa
};

struct OptimizerFeedback {
public:
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

  int nb_newton_it;            /// @brief stores the number of required Newton iterations for the last inverse tumor solve
  int nb_krylov_it;            /// @brief stores the number of required (accumulated) Krylov iterations for the last inverse tumor solve
  int nb_matvecs;              /// @brief stores the number of required (accumulated) matvecs per tumor solve
  int nb_objevals;             /// @brief stores the number of required (accumulated) objective function evaluations per tumor solve
  int nb_gradevals;            /// @brief stores the number of required (accumulated) gradient evaluations per tumor solve
  std::string solverstatus;    /// @brief gives information about the termination reason of inverse tumor TAO solver
  ScalarType gradnorm;         /// @brief final gradient norm
  ScalarType gradnorm0;        /// @brief norm of initial gradient (with p = intial guess)
  ScalarType j0;               /// @brief initial objective : needed for L1 convergence tests
  ScalarType jval;             /// @brief orbjective function value
  bool converged;              /// @brief true if solver converged within bounds
};

struct TumorParameters {
  // TODO
};

struct AuxiliaryParameters {
public:
  AuxiliaryParameters() :
    transport_mri_(false)
  {}

  bool transport_mri_;
};



struct Grid {
public:
  Grid (int *n, int *isize, int *osize, int *istart, int *ostart,
        fft_plan *plan, MPI_Comm c_comm, int *c_dims)
  {
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
  }

  int[3] n_;
  int[3] h_;
  int[3] isize_;
  int[3] osize_;
  int[3] istart_;
  int[3] ostart_;
  int[3] c_dims_;
  int nl_;
  int64_t ng_;
  int64_t accfft_alloc_max_;
  ScalarType lebesgue_measure_;
};


struct FilePaths {
  public:
    FilePaths() :
      wm_(), gm_(), csf_(), ve_(), glm_(), data_t1_(), data_t0_(),
      data_support_(), data_support_data_(), data_comps(), data_comps_data_(),
      obs_filter_(), mri_(), velocity_x_1(), velocity_x_2(), velocity_x_3(),
      pvec_(), phi_(),
      writepath_(), readpath_(), ext_(".nc")
    {}

    // material properties
    std::string wm_;
    std::string gm_;
    std::string csf_;
    std::string ve_;
    std::string glm_;
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
    // io paths
    std::string writepath_;
    std::string readpath_;
    // .nc or nifty output
    std::string ext_; // extension ".nc" or ".nii.gz"
};


struct SyntheticData {
public:
  SyntheticData() :
    rho_(10),
    k_(1E-2),
    dt_(0.01),
    nt_(100),
    pre_adv_time_(-1),
    user_cms_()
  {}

  ScalarType rho_;
  ScalarType k_;
  ScalarType dt_;
  int nt_;
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
    csf_path_()
  {}

  bool enabled_;
  ScalarType dt_;
  std::vector< ScalarType > t_pred_;
  std::vector< std::string > true_data_path_;
  std::vector< std::string > wm_path_;
  std::vector< std::string > gm_path_;
  std::vector< std::string > csf_path_;
};

class Parameters {
  public:
    Parameters() :
      obs_threshold_0_(-1),
      obs_threshold_1_(-1),
      smooth_fac_data_(1.5),
      opt_(),
      optf_(),
      tu_(),
      path_(),
      grid_(),
      syn(),
      pred_(),
      aux_() {

      opt_ = std::make_shared<OptimizerSettings>();
      optf_ = std::make_shared<OptimizerFeedback>();
      tu_ = std::make_shared<TumorParameters>();
      path_ = std::make_shared<FilePaths>();
      syn_ = std::make_shared<SyntheticData>();
      pred_ = std::make_shared<Prediction>();
      aux_ = std::make_shared<AuxiliaryParameters>();

      path_->readpath_ = "./brain_data/" << grid_->n_[0] <<"/";
      path_->writepath_ = "./results/";
    }

    // initialize distributed grid
    createGrid(int *n, int *isize, int *osize, int *istart, int *ostart,
          fft_plan *plan, MPI_Comm c_comm, int *c_dims) {
      grid_ = std::make_shared<Grid>(n, isize, osize, istart, ostart, plan, c_comm, c_dims);
    }
    inline int get_nk() {return tu_->diffusivity_inversion_ ? params_->tu_->nk_ : 0;}
    inline int get_nr() {return tu_->reaction_inversion_ ? params_->tu_->nr_ : 0;}

    virtual ~Parameters() {}

  private:
    ScalarType obs_threshold_0_;
    ScalarType obs_threshold_1_;
    ScalarType smooth_fac_data_;

    bool relative_obs_threshold_;
    bool inject_coarse_sol_;
    bool two_time_points_;
    bool time_history_off_;
    bool use_c0_; // use c(0) directly, never use phi*p
    bool write_output_;

    int sparsity_level_;
    int model_;

    std::shared_ptr<OptimizerSettings> opt_;
    std::shared_ptr<OptimizerFeedback> optf_;
    std::shared_ptr<TumorParameters> tu_;
    std::shared_ptr<FilePaths> path_;
    std::shared_ptr<Grid> grid_;
    std::shared_ptr<SyntheticData> syn_;
    std::shared_ptr<Prediction> pred_;
    std::shared_ptr<AuxiliaryParameters> aux_;
};

#endif
