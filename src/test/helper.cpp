#include "helper.h"

/* helper routines for unit tests */

PetscErrorCode createPVec(Vec &x, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int n_inv = 1 + params->get_nk() + params->get_nr();
  ierr = VecCreateSeq(PETSC_COMM_SELF, n_inv, &x); CHKERRQ(ierr);
  ierr = setupVec(x, SEQ); CHKERRQ(ierr);
  ierr = VecSet(x, 0); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode createTestFunction(Vec x, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr;
  ScalarType R = std::sqrt(2.) * (2 * M_PI) / 64;
  ScalarType dx, dy, dz, ratio, r;
  int64_t ptr, X, Y, Z;
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
  for (int x = 0; x < params->grid_->isize_[0]; x++)
    for (int y = 0; y < params->grid_->isize_[1]; y++)
      for (int z = 0; z < params->grid_->isize_[2]; z++) {
        X = params->grid_->istart_[0] + x;
        Y = params->grid_->istart_[1] + y;
        Z = params->grid_->istart_[2] + z;
        ptr = x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z;

        dx = params->grid_->h_[0] * X - M_PI;
        dy = params->grid_->h_[1] * Y - M_PI;
        dz = params->grid_->h_[2] * Z - M_PI;

        r = sqrt(dx * dx + dy * dy + dz * dz);
        ratio = r / R;
        x_ptr[ptr] = std::exp(-ratio * ratio);
      }
  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode createTestField(std::shared_ptr<VecField> v, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr, *y_ptr, *z_ptr;
  ierr = VecGetArray(v->x_, &x_ptr); CHKERRQ (ierr);
  ierr = VecGetArray(v->y_, &y_ptr); CHKERRQ (ierr);
  ierr = VecGetArray(v->z_, &z_ptr); CHKERRQ (ierr);
  ScalarType freq = 4;
  int64_t X, Y, Z, index;
  for (int x = 0; x < params->grid_->isize_[0]; x++) {
    for (int y = 0; y < params->grid_->isize_[1]; y++) {
      for (int z = 0; z < params->grid_->isize_[2]; z++) {
        X = params->grid_->istart_[0] + x;
        Y = params->grid_->istart_[1] + y;
        Z = params->grid_->istart_[2] + z;

        index = x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z;

        x_ptr[index] = 1 + 0.25 * sin (freq * 2.0 * M_PI / params->grid_->n_[0] * X)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[1] * Y)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[2] * Z);
        y_ptr[index] = 1.5 + 0.1 * sin (freq * 2.0 * M_PI / params->grid_->n_[0] * X)
                                  * cos (freq * 2.0 * M_PI / params->grid_->n_[1] * Y)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[2] * Z);
        z_ptr[index] = 1 + cos (freq * 2.0 * M_PI / params->grid_->n_[0] * X)
                                  * cos (freq * 2.0 * M_PI / params->grid_->n_[1] * Y)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[2] * Z);
      }
    }
  }
  ierr = VecRestoreArray(v->x_, &x_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray(v->y_, &y_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray(v->z_, &z_ptr); CHKERRQ (ierr);

  PetscFunctionReturn(ierr);
}
