#include "ReacCoef.h"

ReacCoef::ReacCoef (std::shared_ptr<NMisc> n_misc) {
    PetscErrorCode ierr;
    ierr = VecCreate (PETSC_COMM_WORLD, &rho_vec_);
    ierr = VecSetSizes (rho_vec_, n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (rho_vec_);
    ierr = VecSet(rho_vec_, 0);

    smooth_flag_ = 0;
}

PetscErrorCode ReacCoef::setValues (double rho_scale, double r_gm_wm_ratio, double r_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    rho_scale_      = rho_scale;
    r_gm_wm_ratio_  = r_gm_wm_ratio;
    r_glm_wm_ratio_ = r_glm_wm_ratio;
    double dr_dm_gm = rho_scale * r_gm_wm_ratio;                //GM
    double dr_dm_wm = rho_scale;                                //WM
    double dr_dm_glm = rho_scale * r_glm_wm_ratio;              //GLM

    n_misc->r_gm_wm_ratio_  = r_gm_wm_ratio_;    // update values in n_misc
    n_misc->r_glm_wm_ratio_ = r_glm_wm_ratio_;
    n_misc->rho_              = rho_scale_;

    dr_dm_gm   = (dr_dm_gm <= 0)  ? 0.0 : dr_dm_gm;
    dr_dm_glm  = (dr_dm_glm <= 0) ? 0.0 : dr_dm_glm;

    if (n_misc->testcase_ != BRAIN && n_misc->testcase_ != BRAINNEARMF && n_misc->testcase_ != BRAINFARMF) {
        double *rho_vec_ptr;
        ierr = VecGetArray (rho_vec_, &rho_vec_ptr);             CHKERRQ (ierr);
        int64_t X, Y, Z, index;
        double amp;
        if (n_misc->testcase_ == CONSTCOEF)
            amp = 0.0;
        else if (n_misc->testcase_ == SINECOEF)
            amp = std::min (1.0, rho_scale_);

        double freq = 4.0;
        for (int x = 0; x < n_misc->isize_[0]; x++) {
            for (int y = 0; y < n_misc->isize_[1]; y++) {
                for (int z = 0; z < n_misc->isize_[2]; z++) {
                    X = n_misc->istart_[0] + x;
                    Y = n_misc->istart_[1] + y;
                    Z = n_misc->istart_[2] + z;

                    index = x * n_misc->isize_[1] * n_misc->isize_[2] + y * n_misc->isize_[2] + z;

                    rho_vec_ptr[index] = rho_scale + amp * sin (freq * 2.0 * M_PI / n_misc->n_[0] * X)
                                                   * sin (freq * 2.0 * M_PI / n_misc->n_[1] * Y)
                                                   * sin (freq * 2.0 * M_PI / n_misc->n_[2] * Z);
                }
            }
        }
        ierr = VecGetArray (rho_vec_, &rho_vec_ptr);             CHKERRQ (ierr);
    }
    else {
        ierr = VecSet (rho_vec_, 0.0);                           CHKERRQ (ierr);
        ierr = VecAXPY (rho_vec_, dr_dm_gm, mat_prop->gm_);      CHKERRQ (ierr);
        ierr = VecAXPY (rho_vec_, dr_dm_wm, mat_prop->wm_);      CHKERRQ (ierr);
        ierr = VecAXPY (rho_vec_, dr_dm_glm, mat_prop->glm_);    CHKERRQ (ierr);
    }

    if (smooth_flag_)
        this->smooth (n_misc);

    if (n_misc->writeOutput_) {
        dataOut (rho_vec_, n_misc, "rho.nc");
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ReacCoef::updateIsotropicCoefficients (double rho_1, double rho_2, double rho_3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // compute new ratios
  rho_scale_        = rho_1;
  r_gm_wm_ratio_    = (n_misc->nr_ == 1) ? n_misc->r_gm_wm_ratio_ : rho_2 / rho_1;
  r_glm_wm_ratio_   = (n_misc->nr_ == 1) ? n_misc->r_glm_wm_ratio_ : rho_3 / rho_1;
  // and set the values
  setValues (rho_scale_, r_gm_wm_ratio_, r_glm_wm_ratio_, mat_prop, n_misc);
  PetscFunctionReturn (0);
}

PetscErrorCode ReacCoef::smooth (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    double sigma = 2.0 * M_PI / n_misc->n_[0];

    double *rho_vec_ptr;
    ierr = VecGetArray (rho_vec_, &rho_vec_ptr);                 CHKERRQ (ierr);
    ierr = weierstrassSmoother (rho_vec_ptr, rho_vec_ptr, n_misc, sigma);
    ierr = VecRestoreArray (rho_vec_, &rho_vec_ptr);             CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ReacCoef::applydRdm(Vec x1, Vec x2, Vec x3, Vec x4, Vec input) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // Event e ("tumor-reac-coeff-apply-dRdm");
  // std::array<double, 7> t = {0};
  // double self_exec_time = -MPI_Wtime ();

  PetscScalar dr_dm_gm  = rho_scale_ * r_gm_wm_ratio_;        // GM
  PetscScalar dr_dm_wm  = rho_scale_;                         // WM
  PetscScalar dr_dm_glm = rho_scale_ * r_glm_wm_ratio_;       // GLM
  // if ratios <= 0, only diffuse in white matter
  dr_dm_gm   = (dr_dm_gm <= 0)  ? 0.0 : dr_dm_gm;
  dr_dm_glm  = (dr_dm_glm <= 0) ? 0.0 : dr_dm_glm;
  // compute dKdm
  // compute dK/dm * (grad c)^T grad \alpha ..
  // assumes that geometry map has ordered components (WM, GM ,CSF, GLM)^T
  if(x1 != nullptr) {ierr = VecAXPY (x1, dr_dm_wm, input);      CHKERRQ (ierr);} // WM
  if(x2 != nullptr) {ierr = VecAXPY (x2, dr_dm_gm, input);      CHKERRQ (ierr);} // GM
  //if(x3 != nullptr) {ierr = VecAXPY (x3, 0, input);             CHKERRQ (ierr);} // CSF
  if(x4 != nullptr) {ierr = VecAXPY (x4, dr_dm_glm, input);     CHKERRQ (ierr);} // GLM

  // self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time); e.addTimings (t); e.stop ();
  PetscFunctionReturn(0);
}

ReacCoef::~ReacCoef () {
    PetscErrorCode ierr;
    ierr = VecDestroy (&rho_vec_);
}
