#ifndef DERIVATIVEOPERATORS_H_
#define DERIVATIVEOPERATORS_H_

#include "PdeOperators.h"

class DerivativeOperators {
 public:
  DerivativeOperators(std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) : pde_operators_(pde_operators), params_(params), tumor_(tumor) {
    VecDuplicate(tumor_->c_0_, &temp_);
    VecDuplicate(tumor_->p_, &ptemp_);
    VecDuplicate(tumor_->p_, &p_current_);

    disable_verbose_ = true;
  }

  std::shared_ptr<PdeOperators> pde_operators_;
  std::shared_ptr<Tumor> tumor_;
  std::shared_ptr<Parameters> params_;

  bool disable_verbose_;  // temp flag for log file vis purposes

  Vec temp_;
  Vec ptemp_;
  Vec p_current_;  // Current solution vector in newton iteration

  virtual PetscErrorCode evaluateObjective(PetscReal *J, Vec x, Vec data) = 0;
  virtual PetscErrorCode evaluateGradient(Vec dJ, Vec x, Vec data) = 0;
  virtual PetscErrorCode evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, Vec data) { PetscFunctionReturn(0); };
  virtual PetscErrorCode evaluateHessian(Vec y, Vec x) = 0;
  virtual PetscErrorCode evaluateConstantHessianApproximation(Vec y, Vec x) { PetscFunctionReturn(0); };

  virtual PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) { PetscFunctionReturn(0); }
  virtual PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) { PetscFunctionReturn(0); }
  virtual PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) { PetscFunctionReturn(0); }
  virtual PetscErrorCode setMaterialProperties(Vec gm, Vec wm, Vec csf, Vec glm) { PetscFunctionReturn(0); }
  virtual PetscErrorCode checkGradient(Vec p, Vec data);
  virtual PetscErrorCode checkHessian(Vec p, Vec data);
  // reset vector sizes
  virtual PetscErrorCode reset(Vec p, std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor);

  virtual ~DerivativeOperators() {
    if (temp_ != nullptr) {
      VecDestroy(&temp_);
      temp_ = nullptr;
    }
    if (ptemp_ != nullptr) {
      VecDestroy(&ptemp_);
      ptemp_ = nullptr;
    }
    if (p_current_ != nullptr) {
      VecDestroy(&p_current_);
      p_current_ = nullptr;
    }
  }
};

class DerivativeOperatorsRD : public DerivativeOperators {
 public:
  DerivativeOperatorsRD(std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) : DerivativeOperators(pde_operators, params, tumor) {
    // tuMSGstd (" ----- Setting reaction-diffusion derivative operators --------");
  }

  PetscErrorCode evaluateObjective(PetscReal *J, Vec x, Vec data);
  PetscErrorCode evaluateGradient(Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateHessian(Vec y, Vec x);
  virtual PetscErrorCode evaluateConstantHessianApproximation(Vec y, Vec x);
  // virtual PetscErrorCode reset (Vec p, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor);
  ~DerivativeOperatorsRD() {}

  // Vec work_np_;  // vector of size np to compute objective and part of gradient related to p
};

class DerivativeOperatorsKL : public DerivativeOperators {
 public:
  DerivativeOperatorsKL(std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) : DerivativeOperators(pde_operators, params, tumor) {
    // tuMSGstd (" ----- Setting reaction-diffusion derivative operators --------");
    eps_ = 1E-2;
  }

  ScalarType eps_;
  PetscErrorCode evaluateObjective(PetscReal *J, Vec x, Vec data);
  PetscErrorCode evaluateGradient(Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateHessian(Vec y, Vec x);
  ~DerivativeOperatorsKL() {}

  // Vec work_np_;  // vector of size np to compute objective and part of gradient related to p
};


class DerivativeOperatorsMassEffect : public DerivativeOperators {
 public:
  DerivativeOperatorsMassEffect(std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) : DerivativeOperators(pde_operators, params, tumor) {
    tuMSGstd(" ----- Setting RD derivative operators with mass-effect objective --------");
    VecCreateSeq(PETSC_COMM_SELF, 3, &delta_);
    setupVec(delta_, SEQ);
    VecSet(delta_, 0.);
    disable_verbose_ = false;
  }

  Vec delta_;

  PetscErrorCode evaluateObjective(PetscReal *J, Vec x, Vec data);
  PetscErrorCode evaluateGradient(Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateHessian(Vec y, Vec x);

  PetscErrorCode computeMisfitBrain(PetscReal *J);
  PetscErrorCode setMaterialProperties(Vec gm, Vec wm, Vec vt, Vec csf) {
    gm_ = gm;
    wm_ = wm;
    vt_ = vt;
    csf_ = csf;
    PetscFunctionReturn(0);
  }

  PetscErrorCode checkGradient(Vec p, Vec data);
  virtual PetscErrorCode reset(Vec p, std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor);

  ~DerivativeOperatorsMassEffect() { VecDestroy(&delta_); }

 private:
  Vec gm_, wm_, vt_, csf_;
};

class DerivativeOperatorsRDObj : public DerivativeOperators {
 public:
  DerivativeOperatorsRDObj(std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) : DerivativeOperators(pde_operators, params, tumor) {
    tuMSGstd(" ----- Setting RD derivative operators with modified objective --------");
  }

  PetscErrorCode evaluateObjective(PetscReal *J, Vec x, Vec data);
  PetscErrorCode evaluateGradient(Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, Vec data);
  PetscErrorCode evaluateHessian(Vec y, Vec x);

  /** @brief: Sets the image vectors for the simulation geometry material properties
   *  - MOVING PATIENT: mA(0) (= initial helathy atlas)
   *  - MOVING ATLAS:   mA(1) (= initial helathy patient)
   */
  PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
    m_geo_wm_ = wm;
    m_geo_gm_ = gm;
    m_geo_csf_ = csf;
    m_geo_glm_ = glm;
    m_geo_bg_ = bg;
    nc_ = (wm != nullptr) + (gm != nullptr) + (csf != nullptr) + (glm != nullptr);
    PetscFunctionReturn(0);
  }

  /** @brief: Sets the image vectors for the target (patient) geometry material properties
   *  - MOVING PATIENT: mP(1) (= advected patient)
   *  - MOVING ATLAS:   mR    (= patient data)
   */
  PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
    m_data_wm_ = wm;
    m_data_gm_ = gm;
    m_data_csf_ = csf;
    m_data_glm_ = glm;
    m_data_bg_ = bg;
    PetscFunctionReturn(0);
  }

  /** @brief: Sets the image vectors for the distance measure difference
   *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2
   *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2
   */
  PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
    xi_wm_ = wm;
    xi_gm_ = gm;
    xi_csf_ = csf;
    xi_glm_ = glm;
    xi_bg_ = bg;
    PetscFunctionReturn(0);
  }

  // virtual PetscErrorCode reset (Vec p, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor);
  ~DerivativeOperatorsRDObj() {}

 private:
  /** @brief: Image vectors for the simulation geometry material properties (memory from outsie;)
   *  - MOVING PATIENT: mA(0) (= initial helathy atlas)      (reference image)
   *  - MOVING ATLAS:   mA(1) (= initial helathy patient)    (template image)
   */
  Vec m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_, m_geo_bg_;
  /** @brief: Image vectors for the target (patient) geometry material properties (memory from outsie;)
   *  - MOVING PATIENT: mP(1) (= advected patient)           (template image)
   *  - MOVING ATLAS:   mR    (= patient data)               (reference image)
   */
  Vec m_data_wm_, m_data_gm_, m_data_csf_, m_data_glm_, m_data_bg_;

  /** @brief: Image vectors for the distance measure difference
   *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2  (negative of the inner term)
   *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2  (negative of the inner term)
   */
  Vec xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_;
  // / number of components in objective function
  int nc_;
};

// Derivative obj helpers
/// @brief computes geometric tumor coupling m1 = m0(1-c(1))
PetscErrorCode geometricCoupling(Vec m1_wm, Vec m1_gm, Vec m1_csf, Vec m1_glm, Vec m1_bg, Vec m0_wm, Vec m0_gm, Vec m0_csf, Vec m0_glm, Vec m0_bg, Vec c1, std::shared_ptr<Parameters> nmisc);
// @brief computes difference xi = m_data - m_geo - function assumes that on input, xi = m_geo * (1-c(1))
PetscErrorCode geometricCouplingAdjoint(ScalarType *sqrdl2norm, Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg, Vec m_geo_wm, Vec m_geo_gm, Vec m_geo_csf, Vec m_geo_glm, Vec m_geo_bg, Vec m_data_wm, Vec m_data_gm, Vec m_data_csf, Vec m_data_glm, Vec m_data_bg);
/// @brief computes difference diff = x - y
PetscErrorCode computeDifference(ScalarType *sqrdl2norm, Vec diff_wm, Vec diff_gm, Vec diff_csf, Vec diff_glm, Vec diff_bg, Vec x_wm, Vec x_gm, Vec x_csf, Vec x_glm, Vec x_bg, Vec y_wm, Vec y_gm, Vec y_csf, Vec y_glm, Vec y_bg);


// Cuda helpers
void computeCrossEntropyCuda(ScalarType *ce_ptr, ScalarType *d_ptr, ScalarType *c_ptr, ScalarType eps, int64_t sz);
void computeCrossEntropyAdjointICCuda(ScalarType *a_ptr, ScalarType *d_ptr, ScalarType *c_ptr, ScalarType eps, int64_t sz);
#endif
