#ifndef SOLVER_H_
#define SOLVER_H_

#include "IO.h"
#include "Parameters.h"
#include "SolverInterface.h"
#include "Utils.h"

class ForwardSolver : public SolverInterface {
 public:
  ForwardSolver() : SolverInterface() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~ForwardSolver() {}
};

class InverseL2Solver : public SolverInterface {
 public:
  InverseL2Solver() : SolverInterface() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~InverseL2Solver() {}
};

class InverseL1Solver : public SolverInterface {
 public:
  InverseL1Solver() : SolverInterface() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~InverseL1Solver() {}
};

class InverseReactionDiffusionSolver : public SolverInterface {
 public:
  InverseReactionDiffusionSolver() : SolverInterface() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~InverseReactionDiffusionSolver() {}
};

class InverseMassEffectSolver : public SolverInterface {
 public:
  InverseMassEffectSolver() : SolverInterface() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  PetscErrorCode readPatient();

  virtual ~InverseMassEffectSolver() {
    if (p_wm_ != nullptr) VecDestroy(&p_wm_);
    if (p_gm_ != nullptr) VecDestroy(&p_gm_);
    if (p_vt_ != nullptr) VecDestroy(&p_vt_);
    if (p_csf_ != nullptr) VecDestroy(&p_csf_);
  }

 private:
  ScalarType gamma_;
  Vec p_wm_;
  Vec p_gm_;
  Vec p_vt_;
  Vec p_csf_;
};

class MultiSpeciesSolver : public SolverInterface {
  public:
  MultiSpeciesSolver() : SolverInterface() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~MultiSpeciesSolver() {}
};
#endif
