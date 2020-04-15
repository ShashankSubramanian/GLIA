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
#include "TumorSolverInterface.h"
#include "Utils.h"

class Solver {
 public:
  Solver();

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~Solver() {
    if(wm_ != nullptr) VecDestroy(&wm_);
    if(gm_ != nullptr) VecDestroy(&gm_);
    if(vt_ != nullptr) VecDestroy(&vt_);
    if(csf_ != nullptr) VecDestroy(&csf_);
    if(mri_ != nullptr) VecDestroy(&mri_);
    if(tmp_ != nullptr) VecDestroy(&tmp_);
    if(data_t1_ != nullptr) VecDestroy(&data_t1_);
    if(data_t0_ != nullptr) VecDestroy(&data_t0_);
    if(!app_settings_->syn_->enabled_ && !app_settings_->path_->data_support_.empty()) {
      if(data_support_ != nullptr) VecDestroy(&data_support_);
    }
    if(data_comps_ != nullptr) VecDestroy(&data_comps_);
    if(obs_filter_ != nullptr) VecDestroy(&obs_filter_);
    if(p_rec_ != nullptr) VecDestroy(&p_rec_);
    if(velocity_ != nullptr) VecDestroy(&velocity_);
  }

 protected:
  virtual PetscErrorCode initializeOperators();
  virtual PetscErrorCode resetOperators(Vec p, bool ninv_changed=true, bool nt_changed=false);
  virtual PetscErrorCode readAtlas();
  virtual PetscErrorCode readData();
  virtual PetscErrorCode readDiffusionFiberTensor();  // TODO(K) implement.
  virtual PetscErrorCode readVelocity();
  virtual PetscErrorCode createSynthetic();
  virtual PetscErrorCode initializeGaussians();
  virtual PetscErrorCode predict();

  std::shared_ptr<Parameters> params_;
  std::shared_ptr<ApplicationSettings> app_settings_;
  // std::shared_ptr<TumorSolverInterface> solver_interface_;
  std::shared_ptr<DerivativeOperators> derivative_operators_;
  std::shared_ptr<PdeOperators> pde_operators_;
  std::shared_ptr<SpectralOperators> spec_ops_;
  std::shared_ptr<Tumor> tumor_;
  std::shared_ptr<Optimizer> optimizer_;

  bool custom_obs_;
  bool warmstart_p_;
  bool has_dt0_;

  Vec wm_;
  Vec gm_;
  Vec vt_;
  Vec csf_;
  Vec mri_;
  Vec tmp_;
  Vec data_t1_;
  Vec data_t0_;
  Vec data_support_;
  Vec data_comps_;
  Vec obs_filter_;
  Vec p_rec_;
  Vec velocity_;
  std::shared_ptr<Data> data_;
};

class ForwardSolver : public Solver {
 public:
  ForwardSolver() : Solver() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~ForwardSolver() {}
};

class InverseL2Solver : public Solver {
 public:
  InverseL2Solver() : Solver() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~InverseL2Solver() {}
};

class InverseL1Solver : public Solver {
 public:
  InverseL1Solver() : Solver() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~InverseL1Solver() {}
};

class InverseReactionDiffusionSolver : public Solver {
 public:
  InverseReactionDiffusionSolver() : Solver() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~InverseReactionDiffusionSolver() {}
};

class InverseMassEffectSolver : public Solver {
 public:
  InverseMassEffectSolver() : Solver() {}

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

class MultiSpeciesSolver : public Solver {
  public:
  MultiSpeciesSolver() : Solver() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~MultiSpeciesSolver() {}
};
#endif
