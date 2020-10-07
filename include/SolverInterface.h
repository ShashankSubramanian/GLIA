#ifndef SOLVER_INTERFACE_H_
#define SOLVER_INTERFACE_H_

#include "IO.h"
#include "Parameters.h"
#include "SpectralOperators.h"
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Optimizer.h"
#include "Tumor.h"
#include "Utils.h"

class SolverInterface {
 public:
  SolverInterface();

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~SolverInterface();


  // timers
  PetscErrorCode initializeEvent();
  PetscErrorCode finalizeEvent();
  // getter
  std::shared_ptr<Tumor> getTumor() {return tumor_;} // returns tumor so that test-cases can access concentration maps
  // setter
  PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec bg);
  PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec bg);
  PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec bg);
  PetscErrorCode updateTumorCoefficients(Vec wm, Vec gm, Vec csf, Vec vt, Vec bg);
  // helpers for cython
  PetscErrorCode readNetCDF(Vec A, std::string filename);
  PetscErrorCode writeNetCDF(Vec A, std::string filename);
  PetscErrorCode smooth(Vec x, ScalarType num_voxels);
  // helpers for sibia
  PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q4);

 protected:
  virtual PetscErrorCode initializeOperators();
  virtual PetscErrorCode resetOperators(Vec p, bool ninv_changed=true, bool nt_changed=false);
  virtual PetscErrorCode createSynthetic();

  PetscErrorCode setupData();
  PetscErrorCode readAtlas();
  PetscErrorCode readData();
  PetscErrorCode readVelocity();
  PetscErrorCode readDiffusionFiberTensor();  // TODO(K) implement.
  PetscErrorCode predict();
  PetscErrorCode initializeGaussians();

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
  bool data_t1_from_seg_;

  int n_inv_;

  Vec wm_;
  Vec gm_;
  Vec vt_;
  Vec csf_;
  Vec ed_;
  Vec mri_;
  Vec tmp_;
  Vec data_t1_;
  Vec data_t0_;
  Vec data_support_;
  Vec data_comps_;
  Vec obs_filter_;
  Vec p_rec_;
  Vec velocity_;
  Vec tc_seg_; // in case data_t1 is changed for the inversion; store the tc seg so that we can compute the dice
  std::shared_ptr<Data> data_;
};

#endif
