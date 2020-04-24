#ifndef PDEOPERATORS_H_
#define PDEOPERATORS_H_

#include "SpectralOperators.h"
#include "AdvectionSolver.h"
#include "DiffusionSolver.h"
#include "ElasticitySolver.h"
#include "Tumor.h"

class PdeOperators {
 public:
  PdeOperators(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : tumor_(tumor), params_(params), spec_ops_(spec_ops) {
    diff_solver_ = std::make_shared<DiffusionSolver>(params, spec_ops, tumor->k_);
    nt_ = params->tu_->nt_;
    diff_ksp_itr_state_ = 0;
    diff_ksp_itr_adj_ = 0;
  }

  std::shared_ptr<Tumor> tumor_;
  std::shared_ptr<DiffusionSolver> diff_solver_;
  std::shared_ptr<Parameters> params_;
  std::shared_ptr<SpectralOperators> spec_ops_;

  // @brief time history of state variable
  std::vector<Vec> c_;
  // @brief time history of adjoint variable
  std::vector<Vec> p_;
  // half-time history of state variables
  std::vector<Vec> c_half_;

  // Accumulated number of KSP solves for diff solver in one forward and adj solve
  int diff_ksp_itr_state_, diff_ksp_itr_adj_;

  virtual PetscErrorCode solveState(int linearized) = 0;
  virtual PetscErrorCode solveAdjoint(int linearized) = 0;
  virtual PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4) = 0;
  virtual PetscErrorCode resizeTimeHistory(std::shared_ptr<Parameters> params) = 0;
  virtual PetscErrorCode reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor = {}) = 0;
  virtual PetscErrorCode getModelSpecificVector(Vec *x) { PetscFunctionReturn(0); }

  virtual PetscErrorCode preAdvection (Vec &wm, Vec &gm, Vec &csf, Vec &mri, ScalarType adv_time) {PetscFunctionReturn(0);};

  virtual ~PdeOperators() {}

 protected:
  /// @brief local copy of nt, bc if parameters change, pdeOperators needs to
  /// be re-constructed. However, the destructor has to use the nt value that
  /// was used upon construction of that object, not the changed value in nmisc
  int nt_;
};

class PdeOperatorsRD : public PdeOperators {
 public:
  PdeOperatorsRD(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops);

  virtual PetscErrorCode solveState(int linearized);
  virtual PetscErrorCode reaction(int linearized, int i);
  virtual PetscErrorCode reactionAdjoint(int linearized, int i);
  virtual PetscErrorCode solveAdjoint(int linearized);
  virtual PetscErrorCode resizeTimeHistory(std::shared_ptr<Parameters> params);
  virtual PetscErrorCode reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor = {});

  // This solves the diffusivity update part of the incremental forward equation
  PetscErrorCode solveIncremental(Vec c_tilde, std::vector<Vec> c_history, ScalarType dt, int iter, int mode);

  /** @brief computes effect of varying/moving material properties, i.e.,
   *  computes q = int_T dK / dm * (grad c)^T grad * \alpha + dRho / dm c(1-c) * \alpha dt
   */
  virtual PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4);
  virtual PetscErrorCode preAdvection (Vec &wm, Vec &gm, Vec &csf, Vec &mri, ScalarType adv_time);
  virtual ~PdeOperatorsRD();

  // for explicit advection of material properties; only allocated if used
  std::shared_ptr<AdvectionSolver> adv_solver_;
};

class PdeOperatorsMassEffect : public PdeOperatorsRD {
 public:
  PdeOperatorsMassEffect(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : PdeOperatorsRD(tumor, params, spec_ops) {
    PetscErrorCode ierr = 0;
    adv_solver_ = std::make_shared<SemiLagrangianSolver>(params, tumor, spec_ops);
    // adv_solver_ = std::make_shared<TrapezoidalSolver> (params, tumor, spec_ops);
    elasticity_solver_ = std::make_shared<VariableLinearElasticitySolver>(params, tumor, spec_ops);
    displacement_old_ = std::make_shared<VecField>(params_->grid_->nl_, params_->grid_->ng_);
    ierr = VecDuplicate(tumor->work_[0], &magnitude_);
    temp_ = new Vec[3];
    for (int i = 0; i < 3; i++) {
      temp_[i] = tumor->work_[11 - i];
    }
  }

  std::shared_ptr<AdvectionSolver> adv_solver_;
  std::shared_ptr<ElasticitySolver> elasticity_solver_;
  std::shared_ptr<VecField> displacement_old_;

  Vec *temp_;
  Vec magnitude_;

  virtual PetscErrorCode solveState(int linearized);
  PetscErrorCode conserveHealthyTissues();
  PetscErrorCode updateReacAndDiffCoefficients(Vec c, std::shared_ptr<Tumor> tumor);
  virtual PetscErrorCode reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor = {});
  virtual PetscErrorCode getModelSpecificVector(Vec *x) {
    *x = magnitude_;
    PetscFunctionReturn(0);
  }

  virtual ~PdeOperatorsMassEffect() {
    PetscErrorCode ierr = 0;
    ierr = VecDestroy(&magnitude_);
    delete[] temp_;
  }
};

class PdeOperatorsMultiSpecies : public PdeOperatorsRD {
 public:
  PdeOperatorsMultiSpecies(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : PdeOperatorsRD(tumor, params, spec_ops) {
    PetscErrorCode ierr = 0;
    adv_solver_ = std::make_shared<SemiLagrangianSolver>(params, tumor, spec_ops);
    // adv_solver_ = std::make_shared<TrapezoidalSolver> (params, tumor);
    elasticity_solver_ = std::make_shared<VariableLinearElasticitySolver>(params, tumor, spec_ops);
    displacement_old_ = std::make_shared<VecField>(params_->grid_->nl_, params_->grid_->ng_);
    ierr = VecDuplicate(tumor->work_[0], &magnitude_);
  }

  std::shared_ptr<AdvectionSolver> adv_solver_;
  std::shared_ptr<ElasticitySolver> elasticity_solver_;
  std::shared_ptr<VecField> displacement_old_;

  Vec magnitude_;

  virtual PetscErrorCode solveState(int linearized);
  PetscErrorCode computeReactionRate(Vec m);
  PetscErrorCode computeTransition(Vec alpha, Vec beta);
  PetscErrorCode computeThesholder(Vec h);
  PetscErrorCode computeSources(Vec p, Vec i, Vec n, Vec O, ScalarType dt);
  PetscErrorCode updateReacAndDiffCoefficients(Vec c, std::shared_ptr<Tumor> tumor);
  virtual PetscErrorCode reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor = {});
  virtual PetscErrorCode getModelSpecificVector(Vec *x) {
    *x = magnitude_;
    PetscFunctionReturn(0);
  }
  virtual ~PdeOperatorsMultiSpecies() {
    PetscErrorCode ierr = 0;
    ierr = VecDestroy(&magnitude_);
  }
};

// Cuda helpers
void conserveHealthyTissuesCuda(ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *sum_ptr, ScalarType *scale_gm_ptr, ScalarType *scale_wm_ptr, ScalarType dt, int64_t sz);
void computeReactionRateCuda(ScalarType *m_ptr, ScalarType *ox_ptr, ScalarType *rho_ptr, ScalarType ox_hypoxia, int64_t sz);
void computeTransitionCuda(ScalarType *alpha_ptr, ScalarType *beta_ptr, ScalarType *ox_ptr, ScalarType *p_ptr, ScalarType *i_ptr, ScalarType alpha_0, ScalarType beta_0, ScalarType ox_inv,
                           ScalarType thres, int64_t sz);
void computeThesholderCuda(ScalarType *h_ptr, ScalarType *ox_ptr, ScalarType ox_hypoxia, int64_t sz);
void computeSourcesCuda(ScalarType *p_ptr, ScalarType *i_ptr, ScalarType *n_ptr, ScalarType *m_ptr, ScalarType *al_ptr, ScalarType *bet_ptr, ScalarType *h_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr,
                        ScalarType *ox_ptr, ScalarType *di_ptr, ScalarType dt, ScalarType death_rate, ScalarType ox_source, ScalarType ox_consumption, int64_t sz);
void logisticReactionCuda(ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt, int64_t sz, int linearized);
void updateReacAndDiffCoefficientsCuda(ScalarType *rho_ptr, ScalarType *k_ptr, ScalarType *bg_ptr, ScalarType *gm_ptr, ScalarType *vt_ptr, ScalarType *csf_ptr, ScalarType rho, ScalarType k,
                                       int64_t sz);
#endif
