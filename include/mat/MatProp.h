#ifndef MATPROP_H_
#define MATPROP_H_

#include "Parameters.h"
#include "IO.h"
#include "SpectralOperators.h"

class MatProp {
 public:
  MatProp(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops);

  Vec gm_;
  Vec wm_;
  Vec vt_;
  Vec csf_;
  Vec bg_;
  Vec filter_;

  Vec kfxx_;
  Vec kfxy_;
  Vec kfxz_;
  Vec kfyy_;
  Vec kfyz_;
  Vec kfzz_;
  
  ScalarType force_factor_;
  ScalarType edema_threshold_;

  std::shared_ptr<Parameters> params_;
  std::shared_ptr<SpectralOperators> spec_ops_;

  // undeformed -- this is never changed; so use as pointers
  Vec gm_0_;
  Vec wm_0_;
  Vec vt_0_;
  Vec csf_0_;
  Vec kfxx_0_;
  Vec kfxy_0_;
  Vec kfxz_0_;
  Vec kfyy_0_;
  Vec kfyz_0_; 
  Vec kfzz_0_;

  // mri
  Vec mri_;

  PetscErrorCode setValues(std::shared_ptr<Parameters> params);
  PetscErrorCode setValuesCustom(Vec gm, Vec wm, Vec csf, Vec vt, Vec bg, std::shared_ptr<Parameters> params);
  PetscErrorCode setDiffusionFiber(Vec kfxx, Vec kfxy, Vec kfxz, Vec kfyy, Vec kfyz, Vec kfzz, std::shared_ptr<Parameters> params);
  PetscErrorCode setValuesSinusoidal(std::shared_ptr<Parameters> params);
  PetscErrorCode clipHealthyTissues();
  PetscErrorCode filterBackgroundAndSmooth(Vec in);
  PetscErrorCode filterTumor(Vec c);
  PetscErrorCode setAtlas(Vec gm, Vec wm, Vec csf, Vec vt, Vec bg);
  PetscErrorCode resetValues();

  ~MatProp();
};

#endif
