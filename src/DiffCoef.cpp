#include "DiffCoef.h"

DiffCoef::DiffCoef (std::shared_ptr<NMisc> n_misc) :
  k_scale_(1E-2)
, k_gm_wm_ratio_(1.0 / 5.0)
, k_glm_wm_ratio_(3.0 / 5.0)
, smooth_flag_(0)
, filter_avg_ (0.0) {
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

    temp_accfft_ = (double * ) accfft_alloc (n_misc->accfft_alloc_max_);

    ierr = VecSet (kxx_ , 0);
    ierr = VecSet (kxy_ , 0);
    ierr = VecSet (kxz_ , 0);
    ierr = VecSet (kyy_ , 0);
    ierr = VecSet (kyz_ , 0);
    ierr = VecSet (kzz_ , 0);

    kxx_avg_ = kxy_avg_ = kxz_avg_ = kyy_avg_ = kyz_avg_ = kzz_avg_ = 0.0;
}

PetscErrorCode DiffCoef::setValues (double k_scale, double k_gm_wm_ratio, double k_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    k_scale_ = k_scale;
    k_gm_wm_ratio_  = k_gm_wm_ratio;
    k_glm_wm_ratio_ = k_glm_wm_ratio;

    double dk_dm_gm  = k_scale * k_gm_wm_ratio_;        //GM
    double dk_dm_wm  = k_scale;                         //WM
    double dk_dm_glm = k_scale * k_glm_wm_ratio_;       //GLM
    // if ratios <= 0, only diffuse in white matter
    dk_dm_gm   = (dk_dm_gm <= 0)  ? 0.0 : dk_dm_gm;
    dk_dm_glm  = (dk_dm_glm <= 0) ? 0.0 : dk_dm_glm;

    if (n_misc->testcase_ != BRAIN) {
        double *kxx_ptr, *kyy_ptr, *kzz_ptr;
        ierr = VecGetArray (kxx_, &kxx_ptr);                                CHKERRQ (ierr);
        ierr = VecGetArray (kyy_, &kyy_ptr);                                CHKERRQ (ierr);
        ierr = VecGetArray (kzz_, &kzz_ptr);                                CHKERRQ (ierr);
        int64_t X, Y, Z, index;
        double amp;
        if (n_misc->testcase_ == CONSTCOEF)
            amp = 0.0;
        else if (n_misc->testcase_ == SINECOEF)
            amp = std::min (1.0, k_scale_);

        double freq = 4.0;
        for (int x = 0; x < n_misc->isize_[0]; x++) {
            for (int y = 0; y < n_misc->isize_[1]; y++) {
                for (int z = 0; z < n_misc->isize_[2]; z++) {
                    X = n_misc->istart_[0] + x;
                    Y = n_misc->istart_[1] + y;
                    Z = n_misc->istart_[2] + z;

                    index = x * n_misc->isize_[1] * n_misc->isize_[2] + y * n_misc->isize_[2] + z;

                    kxx_ptr[index] = k_scale + amp * sin (freq * 2.0 * M_PI / n_misc->n_[0] * X)
                                                   * sin (freq * 2.0 * M_PI / n_misc->n_[1] * Y)
                                                   * sin (freq * 2.0 * M_PI / n_misc->n_[2] * Z);

                    // kxx_ptr[index] = 1E-2 + 0.5 * sin (2.0 * M_PI / n_misc->n_[0] * X) * cos (2.0 * M_PI / n_misc->n_[1] * Y)
                    //                             + 0.5 + 1E-3 * (1 - 0.5 * sin (2.0 * M_PI / n_misc->n_[0] * X) * cos (2.0 * M_PI / n_misc->n_[1] * Y)
                    //                                     + 0.5);

                    kyy_ptr[index] = kxx_ptr[index];
                    kzz_ptr[index] = kxx_ptr[index];
                }
            }
        }
        ierr = VecRestoreArray (kxx_, &kxx_ptr);                            CHKERRQ (ierr);
        ierr = VecRestoreArray (kyy_, &kyy_ptr);                            CHKERRQ (ierr);
        ierr = VecRestoreArray (kzz_, &kzz_ptr);                            CHKERRQ (ierr);
    }
    else {
        ierr = VecSet  (kxx_, 0.0);                                         CHKERRQ (ierr);
        ierr = VecAXPY (kxx_, dk_dm_gm, mat_prop->gm_);                     CHKERRQ (ierr);
        ierr = VecAXPY (kxx_, dk_dm_wm, mat_prop->wm_);                     CHKERRQ (ierr);
        ierr = VecAXPY (kxx_, dk_dm_glm, mat_prop->glm_);                   CHKERRQ (ierr);

        ierr = VecCopy (kxx_, kyy_);                                        CHKERRQ (ierr);
        ierr = VecCopy (kxx_, kzz_);                                        CHKERRQ (ierr);
    }

    //Average diff coeff values for preconditioner for diffusion solve
    ierr = VecSum (kxx_, &kxx_avg_);                                    CHKERRQ (ierr);
    ierr = VecSum (kxy_, &kxy_avg_);                                    CHKERRQ (ierr);
    ierr = VecSum (kxz_, &kxz_avg_);                                    CHKERRQ (ierr);
    ierr = VecSum (kyy_, &kyy_avg_);                                    CHKERRQ (ierr);
    ierr = VecSum (kyz_, &kyz_avg_);                                    CHKERRQ (ierr);
    ierr = VecSum (kzz_, &kzz_avg_);                                    CHKERRQ (ierr);
    ierr = VecSum (mat_prop->filter_, &filter_avg_);                    CHKERRQ (ierr);

    kxx_avg_ *= 1.0 / filter_avg_;
    kxy_avg_ *= 1.0 / filter_avg_;
    kxz_avg_ *= 1.0 / filter_avg_;
    kyy_avg_ *= 1.0 / filter_avg_;
    kyz_avg_ *= 1.0 / filter_avg_;
    kzz_avg_ *= 1.0 / filter_avg_;

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

    accfft_grad (temp_[4], temp_[5], temp_[6], c, plan, &XYZ, timer);
    ierr = applyK (temp_[4], temp_[5], temp_[6]);
    accfft_divergence (dc, temp_[1], temp_[2], temp_[3], plan, timer);

    PetscFunctionReturn(0);
}


DiffCoef::~DiffCoef () {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    ierr = VecDestroy (&kxx_);
    ierr = VecDestroy (&kxy_);
    ierr = VecDestroy (&kxz_);
    ierr = VecDestroy (&kyy_);
    ierr = VecDestroy (&kyz_);
    ierr = VecDestroy (&kzz_);
    for (int i = 0; i < 7; i++) {
        ierr = VecDestroy (&temp_[i]);
    }
    delete[] temp_;
    accfft_free (temp_accfft_);
}
