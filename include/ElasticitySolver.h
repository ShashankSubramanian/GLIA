#ifndef ELASTICITYSOLVER_H_
#define ELASTICITYSOLVER_H_

#include "SpectralOperators.h"
#include "Tumor.h"

struct CtxElasticity {
  CtxElasticity(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : params_(params), tumor_(tumor), spec_ops_(spec_ops) {
    PetscErrorCode ierr = 0;
    ierr = VecDuplicate(tumor_->mat_prop_->gm_, &mu_);
    ierr = VecDuplicate(mu_, &lam_);
    ierr = VecDuplicate(mu_, &screen_);

    displacement_ = tumor->displacement_;
    force_ = tumor->work_field_;

    ierr = VecSet(mu_, 0.);
    ierr = VecSet(lam_, 0.);
    ierr = VecSet(screen_, 0.);
  }
  std::shared_ptr<Parameters> params_;
  std::shared_ptr<Tumor> tumor_;
  std::shared_ptr<SpectralOperators> spec_ops_;

  // vector fields to store KSP vectors while solving
  std::shared_ptr<VecField> displacement_;
  std::shared_ptr<VecField> force_;

  ScalarType mu_avg_, lam_avg_, screen_avg_;

  ComplexType *fx_hat_, *fy_hat_, *fz_hat_;
  ComplexType *ux_hat_, *uy_hat_, *uz_hat_;

  Vec mu_;
  Vec lam_;
  Vec screen_;
  Vec disp_;

  ScalarType computeMu(ScalarType E, ScalarType nu) { return (E / (2 * (1 + nu))); }

  ScalarType computeLam(ScalarType E, ScalarType nu) { return (nu * E / ((1 + nu) * (1 - 2 * nu))); }

  ~CtxElasticity() {
    PetscErrorCode ierr = 0;
    ierr = VecDestroy(&mu_);
    ierr = VecDestroy(&lam_);
    ierr = VecDestroy(&screen_);
    ierr = VecDestroy(&disp_);

    fft_free(ux_hat_);
    fft_free(uy_hat_);
    fft_free(uz_hat_);
    fft_free(fx_hat_);
    fft_free(fy_hat_);
    fft_free(fz_hat_);
  }
};

class ElasticitySolver {
 public:
  ElasticitySolver(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops);  // tumor is needed for its work vectors

  KSP ksp_;
  Mat A_;
  Vec rhs_;

  PC pc_;

  std::shared_ptr<CtxElasticity> ctx_;

  virtual PetscErrorCode solve(std::shared_ptr<VecField> displacement, std::shared_ptr<VecField> rhs) = 0;

  virtual ~ElasticitySolver();
};

class VariableLinearElasticitySolver : public ElasticitySolver {
 public:
  VariableLinearElasticitySolver(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : ElasticitySolver(params, tumor, spec_ops) {}
  PetscErrorCode computeMaterialProperties();
  PetscErrorCode smoothMaterialProperties();
  virtual PetscErrorCode solve(std::shared_ptr<VecField> displacement, std::shared_ptr<VecField> rhs);
  virtual ~VariableLinearElasticitySolver() {}
};

// Helper functions
PetscErrorCode operatorConstantCoefficients(PC pc, Vec x, Vec y);  // this is used as a preconditioner
PetscErrorCode operatorVariableCoefficients(Mat A, Vec x, Vec y);
PetscErrorCode operatorCreateVecsElas(Mat A, Vec *left, Vec *right);
PetscErrorCode elasticitySolverKSPMonitor(KSP ksp, PetscInt its, PetscReal rnorm, void *ptr);

// Cuda helpers
void initElasticityCudaConstants(int *n, int *ostart);
void precFactorElasticityCuda(CudaComplexType *ux_hat, CudaComplexType *uy_hat, CudaComplexType *uz_hat, CudaComplexType *fx_hat, CudaComplexType *fy_hat, CudaComplexType *fz_hat, ScalarType lam_avg, ScalarType mu_avg, ScalarType screen_avg, int *sz);
void computeScreeningCuda(ScalarType *screen_ptr, ScalarType *c_ptr, ScalarType *bg_ptr, ScalarType screen_low, ScalarType screen_high, int64_t sz);
void computeTumorLameCuda(ScalarType *mu_ptr, ScalarType *lam_ptr, ScalarType *c_ptr, ScalarType mu_tumor, ScalarType lam_tumor, int64_t sz);
#endif
