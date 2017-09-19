#include "DiffCoef.h"

DiffCoef::DiffCoef (std::shared_ptr<NMisc> n_misc) {
    PetscErrorCode ierr;
    ierr = VecCreate (PETSC_COMM_WORLD, &kxx_);
    ierr = VecSetSizes (kxx_, n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (kxx_);

    ierr = VecDuplicate (kxx_, &kxy_);
    ierr = VecDuplicate (kxx_, &kxz_);
    ierr = VecDuplicate (kxx_, &kyy_);
    ierr = VecDuplicate (kxx_, &kyz_);
    ierr = VecDuplicate (kxx_, &kzz_);

    temp_ = new Vec[7];
    for (int i = 0; i < 7; i++) {
        ierr = VecDuplicate (kxx_, &temp_[i]);
        ierr = VecSet (temp_[i] , 0);
    }

    ierr = VecSet (kxx_ , 0);
    ierr = VecSet (kxy_ , 0);
    ierr = VecSet (kxz_ , 0);
    ierr = VecSet (kyy_ , 0);
    ierr = VecSet (kyz_ , 0);
    ierr = VecSet (kzz_ , 0);

    smooth_flag_ = 0;
}

PetscErrorCode DiffCoef::setValues (double k_scale, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
  PetscFunctionBegin;
    PetscErrorCode ierr;
  k_scale_ = k_scale;
    double dk_dm_gm =  k_scale / 5.0;        //GM
    double dk_dm_wm = k_scale;               //WM
    double dk_dm_glm = 3.0 / 5.0 * k_scale;  //GLM

    ierr = VecAXPY (kxx_, dk_dm_gm, mat_prop->gm_);                     CHKERRQ (ierr);
    ierr = VecAXPY (kxx_, dk_dm_wm, mat_prop->wm_);                     CHKERRQ (ierr);
    ierr = VecAXPY (kxx_, dk_dm_glm, mat_prop->glm_);                   CHKERRQ (ierr);

    ierr = VecCopy (kxx_, kyy_);                                        CHKERRQ (ierr);
    ierr = VecCopy (kxx_, kzz_);                                        CHKERRQ (ierr);

    if (smooth_flag_) {
        ierr = this->smooth (n_misc); CHKERRQ (ierr);
  }

    PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::smooth (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    double sigma = 2.0 * M_PI / n_misc->n_[0];
    double *kxx_ptr, *kxy_ptr, *kxz_ptr, *kyy_ptr, *kyz_ptr, *kzz_ptr;

  ierr = VecGetArray (kxx_, &kxx_ptr);                              CHKERRQ (ierr);
  ierr = VecGetArray (kxy_, &kxy_ptr);                              CHKERRQ (ierr);
  ierr = VecGetArray (kxz_, &kxz_ptr);                              CHKERRQ (ierr);
  ierr = VecGetArray (kyy_, &kyy_ptr);                              CHKERRQ (ierr);
  ierr = VecGetArray (kyz_, &kyz_ptr);                              CHKERRQ (ierr);
  ierr = VecGetArray (kzz_, &kzz_ptr);                              CHKERRQ (ierr);

    ierr = weierstrassSmoother (kxx_ptr, kxx_ptr, n_misc, sigma);
  ierr = weierstrassSmoother (kxy_ptr, kxy_ptr, n_misc, sigma);
  ierr = weierstrassSmoother (kxz_ptr, kxz_ptr, n_misc, sigma);
  ierr = weierstrassSmoother (kyy_ptr, kyy_ptr, n_misc, sigma);
  ierr = weierstrassSmoother (kyz_ptr, kyz_ptr, n_misc, sigma);
  ierr = weierstrassSmoother (kzz_ptr, kzz_ptr, n_misc, sigma);

  ierr = VecRestoreArray (kxx_, &kxx_ptr);                          CHKERRQ (ierr);
  ierr = VecRestoreArray (kxy_, &kxy_ptr);                          CHKERRQ (ierr);
  ierr = VecRestoreArray (kxz_, &kxz_ptr);                          CHKERRQ (ierr);
  ierr = VecRestoreArray (kyy_, &kyy_ptr);                          CHKERRQ (ierr);
  ierr = VecRestoreArray (kyz_, &kyz_ptr);                          CHKERRQ (ierr);
  ierr = VecRestoreArray (kzz_, &kzz_ptr);                          CHKERRQ (ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::applyK (Vec x, Vec y, Vec z) {
    PetscErrorCode ierr = 0;
    for (int i = 1; i < 4; i++) {
        ierr = VecSet (temp_[i] , 0);                                   CHKERRQ (ierr);
    }
    //X
    ierr = VecPointwiseMult (temp_[0], kxx_, x);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[1], 1.0, temp_[0]);                           CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kxy_, y);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[1], 1.0, temp_[0]);                           CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kxz_, z);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[1], 1.0, temp_[0]);                           CHKERRQ (ierr);

    //Y
    ierr = VecPointwiseMult (temp_[0], kxy_, x);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[2], 1.0, temp_[0]);                           CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kyy_, y);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[2], 1.0, temp_[0]);                           CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kyz_, z);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[2], 1.0, temp_[0]);                           CHKERRQ (ierr);

    //Z
    ierr = VecPointwiseMult (temp_[0], kxz_, x);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[3], 1.0, temp_[0]);                           CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kyz_, y);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[3], 1.0, temp_[0]);                           CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kzz_, z);                        CHKERRQ (ierr);
    ierr = VecAXPY (temp_[3], 1.0, temp_[0]);                           CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::applyD (Vec dc, Vec c, accfft_plan *plan) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    std::bitset<3> XYZ;
    XYZ[0] = 1;
    XYZ[1] = 1;
    XYZ[2] = 1;

    double *timer = NULL;   //Used for calling accfft routines

    /*For debug*/
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    /*End Debug context*/

    accfft_grad (temp_[4], temp_[5], temp_[6], c, plan, &XYZ, timer);
    ierr = applyK (temp_[4], temp_[5], temp_[6]); CHKERRQ (ierr);
    accfft_divergence (dc, temp_[1], temp_[2], temp_[3], plan, timer);

    PetscFunctionReturn(0);
}


DiffCoef::~DiffCoef () {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    ierr = VecDestroy (&kxx_);                          CHKERRQ (ierr);
    ierr = VecDestroy (&kxy_);                          CHKERRQ (ierr);
    ierr = VecDestroy (&kxz_);                          CHKERRQ (ierr);
    ierr = VecDestroy (&kyy_);                          CHKERRQ (ierr);
    ierr = VecDestroy (&kyz_);                          CHKERRQ (ierr);
    ierr = VecDestroy (&kzz_);                          CHKERRQ (ierr);
}
