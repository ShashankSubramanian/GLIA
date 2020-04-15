/**
 *  NAME_OF_THE_CODE
 *
 *  Copyright (C) 2017-2020, The University of Texas at Austin
 *  This file is part of the NAME_OF_THE_CODE framework.
 *
 *  Main Contributers:    Shashank Subramanian
 *                        Klaudius Scheufele
 *
 *  Further Contributers: Naveen Himthani
 *                        George Biros
 *
 *  NAME_OF_THE_CODE is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  NAME_OF_THE_CODE is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/

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
