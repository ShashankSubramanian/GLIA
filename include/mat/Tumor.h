/*
Tumor class
*/
#ifndef TUMOR_H_
#define TUMOR_H_

#include "MatProp.h"
#include "DiffCoef.h"
#include "ReacCoef.h"
#include "Obs.h"
#include "Phi.h"

class Tumor {
 public:
  Tumor(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops);

  std::shared_ptr<DiffCoef> k_;
  std::shared_ptr<ReacCoef> rho_;
  std::shared_ptr<Phi> phi_;
  std::shared_ptr<Obs> obs_;

  std::shared_ptr<MatProp> mat_prop_;

  std::shared_ptr<Parameters> params_;
  std::shared_ptr<SpectralOperators> spec_ops_;

  // parametrization
  Vec p_;
  // state variables
  Vec c_t_;
  Vec c_0_;
  // adjoint Variables
  Vec p_t_;
  Vec p_0_;
  // work vectors
  Vec *work_;

  // segmentation based on max voxel-wise prop
  Vec seg_;

  // For multiple species
  std::map<std::string, Vec> species_;
  std::map<std::string, Vec> data_species_;

  // mass effect parameters
  // velocity
  std::shared_ptr<VecField> velocity_;
  std::shared_ptr<VecField> displacement_;
  std::shared_ptr<VecField> force_;
  std::shared_ptr<VecField> work_field_;

  PetscErrorCode initialize(Vec p, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Phi> phi = {}, std::shared_ptr<MatProp> mat_prop = {});
  PetscErrorCode setParams(Vec p, std::shared_ptr<Parameters> params, bool npchanged = false);
  PetscErrorCode setSinusoidalCoefficients(std::shared_ptr<Parameters> params);
  PetscErrorCode computeEdema();
  PetscErrorCode computeSegmentation();
  PetscErrorCode computeSpeciesNorms();
  PetscErrorCode clipTumor();
  PetscErrorCode getTCRecon(Vec x);
  PetscErrorCode getHealthyBrain(Vec x); 
  // mass effect functions
  PetscErrorCode computeForce(Vec c);

  ~Tumor();
};

// Cuda helpers
void nonlinearForceScalingCuda(ScalarType *c_ptr, ScalarType *fx_ptr, ScalarType *fy_ptr, ScalarType *fz_ptr, int64_t sz);
void computeTumorSegmentationCuda(ScalarType *bg_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *csf_ptr, ScalarType *glm_ptr, ScalarType *c_ptr, ScalarType *seg_ptr, int64_t sz);
void getTCReconCuda(ScalarType *seg_ptr, ScalarType *x_ptr, int64_t sz);
void getHealthyBrainCuda(ScalarType *seg_ptr, ScalarType *x_ptr, int64_t sz);
#endif
