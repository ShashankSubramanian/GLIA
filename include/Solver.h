/**
 *  SIBIA (Scalable Biophysics-Based Image Analysis)
 *
 *  Copyright (C) 2017-2020, The University of Texas at Austin
 *  This file is part of the SIBIA library.
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

#ifndef SOLVER_H_
#define SOLVER_H_

#include "IO.h"
#include "Utils.h"
#include "Parameters.h"
#include "TumorSolverInterface.h"

class Solver {
  public:
    Solver(std::shared_ptr<SpectralOperators> spec_ops);

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run() = 0;

    virtual ~Solver() {}

  protected:
    virtual PetscErrorCode readAtlas();
    virtual PetscErrorCode readData();
    virtual PetscErrorCode readDiffusionFiberTensor(); // TODO(K) implement.
    virtual PetscErrorCode readVelocity();
    virtual PetscErrorCode readUserCMs(); // TODO(K) implement.
    virtual PetscErrorCode createSynthetic(); // TODO(K) implement.
    virtual PetscErrorCode initializeGaussians();
    virtual PetscErrorCode predict(); // TODO(K) implement.

    std::shared_ptr<Parameters> params_;
    std::shared_ptr<TumorSolverInterface> solver_interface_;
    std::shared_ptr<SpectralOperators> spec_ops_;
    std::shared_ptr<Tumor> tumor_;

    bool custom_obs_;
    bool warmstart_p_;
    bool synthetic_;
    bool has_dt0_;

    ScalarType smooth_fac_data_; // TODO(K)

    std::vector<int> user_cms_; // TODO(K)

    Vec wm_;
    Vec gm_;
    Vec csf_;
    Vec ve_;
    Vec glm_;
    Vec mri_;
    Vec tmp_;
    Vec data_t1_;
    Vec data_t0_;
    Vec data_support_;
    Vec data_comps_;
    Vec obs_filter_;

    Vec p_rec_;
    VecField velocity_;
};

class ForwardSolver : public Solver {
  public:
    ForwardSolver(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run();

    virtual ~ForwardSolver() {
      if(wm_ != nullptr) VecDestroy(&wm_);
      if(gm_ != nullptr) VecDestroy(&gm_);
      if(csf_ != nullptr) VecDestroy(&csf_);
      if(glm_ != nullptr) VecDestroy(&glm_);
      if(ve_ != nullptr) VecDestroy(&ve_);
      if(mri_ != nullptr) VecDestroy(&mri_);
      if(tmp_ != nullptr) VecDestroy(&tmp_);
      if(data_t1_ != nullptr) VecDestroy(&data_t1_);
      if(data_t0_ != nullptr) VecDestroy(&data_t0_);
      if(data_support_ != nullptr) VecDestroy(&data_support_);
      if(data_comps_ != nullptr) VecDestroy(&data_comps_);
      if(obs_filter_ != nullptr) VecDestroy(&obs_filter_);
      if(p_rec_ != nullptr) VecDestroy(&p_rec_);
    }
};

class InverseL2Solver : public Solver {
  public:
    InverseL2Solver(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run();

    virtual ~InverseL2Solver();
};

class InverseL1Solver : public Solver {
  public:
    InverseL1Solver(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run();

    virtual ~InverseL1Solver();
};

class InverseReactionDiffusionSolver : public Solver {
  public:
    InverseReactionDiffusionSolver(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run();

    virtual ~InverseReactionDiffusionSolver();
};

class InverseMassEffectSolver : public Solver {
  public:
    InverseMassEffectSolver(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize ();
    virtual PetscErrorCode initialize (std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run ();

    virtual ~InverseMassEffectSolver () {}

  private:
    ScalarType gamma_;
};

class InverseMultiSpeciesSolver : public Solver {
  public:
    InverseMassEffectSolver(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run();

    virtual ~InverseMultiSpeciesSolver() {}
};


class TestSuite : public Solver {
  public:
    TestSuite(std::shared_ptr<SpectralOperators> spec_ops)
    : Solver(spec_ops)
    {}

    virtual PetscErrorCode finalize();
    virtual PetscErrorCode initialize(std::shared_ptr<Parameters> params = {});
    virtual PetscErrorCode run();

    virtual ~TestSuite() {}
};

#endif
