#ifndef DIFFSOLVER_H_
#define DIFFSOLVER_H_

#include "DiffCoef.h"
#include "SpectralOperators.h"

struct Ctx {
  std::shared_ptr<DiffCoef> k_;
  std::shared_ptr<Parameters> params_;
  std::shared_ptr<SpectralOperators> spec_ops_;
  fft_plan *plan_;
  ScalarType dt_;
  Vec temp_;
  ScalarType *precfactor_;
  ScalarType *work_cuda_;

  ~Ctx() {}
};

class DiffusionSolver {
 public:
  DiffusionSolver(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<DiffCoef> k);

  KSP ksp_;
  Mat A_;
  PC pc_;
  Vec rhs_;
  int ksp_itr_;
  std::shared_ptr<Ctx> ctx_;

  PetscErrorCode solve(Vec c, ScalarType dt);
  PetscErrorCode precFactor();

  virtual ~DiffusionSolver();
};

// Helper functions for KSP solve
PetscErrorCode operatorA(Mat A, Vec x, Vec y);
PetscErrorCode operatorCreateVecs(Mat A, Vec *left, Vec *right);
PetscErrorCode precFactor(ScalarType *precfactor, std::shared_ptr<Ctx> ctx);
PetscErrorCode applyPC(PC pc, Vec x, Vec y);
PetscErrorCode diffSolverKSPMonitor(KSP ksp, PetscInt its, PetscReal rnorm, void *ptr);

#endif
