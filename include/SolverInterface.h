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

#ifndef SOLVER_INTERFACE_H_
#define SOLVER_INTERFACE_H_

#include "IO.h"
#include "Parameters.h"
#include "Utils.h"

class SolverInterface {
 public:
  SolverInterface();

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~SolverInterface();

 protected:
  virtual PetscErrorCode initializeOperators();
  virtual PetscErrorCode resetOperators(Vec p, bool ninv_changed=true, bool nt_changed=false);
  virtual PetscErrorCode createSynthetic();
  virtual PetscErrorCode initializeGaussians();

  PetscErrorCode updateTumorCoefficients(Vec wm, Vec gm, Vec csf, Vec vt, Vec bg);
  PetscErrorCode readAtlas();
  PetscErrorCode readData();
  PetscErrorCode readVelocity();
  PetscErrorCode readDiffusionFiberTensor();  // TODO(K) implement.
  PetscErrorCode predict();

  std::shared_ptr<Parameters> params_;
  std::shared_ptr<ApplicationSettings> app_settings_;
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

#endif
