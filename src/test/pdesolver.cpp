#include "Solver.h"
#include "IO.h"
#include "catch.hpp"

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


/* run a simple diffusion equation \partial c + Dc = 0 with variable coefficients and test the result */
TEST_CASE( "Running diffusion solver", "[pdesolver]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(ACCFFT);
#endif

  DISABLE_VERBOSE = true;

  // initialize a grid
  initializeGrid(64, params, spec_ops);
  // create a test diffusion coefficient
  std::shared_ptr<DiffCoef> k = std::make_shared<DiffCoef>(params, spec_ops);
  // create work vectors
  Vec *work = new Vec[8];
  for (int i = 0; i < 8; i++) {
    VecDuplicate(k->kxx_, &work[i]);
    VecSet(work[i], 0);
  }
  // setting work vecs for diffusion coefficient (first 8)
  k->setWorkVecs(work);
  // set sinusoidal diffusion coefficient
  k->setValuesSinusoidal(params, 1E-2);
  // create a diffusion solver
  std::shared_ptr<DiffusionSolver> diff_solver = std::make_shared<DiffusionSolver>(params, spec_ops, k);
  // create a valid function
  Vec c;
  VecDuplicate(work[0], &c);
  VecSet(c, 0);
  createTestFunction(c, params);
  // solve a diffusion equation for a few time steps
  diff_solver->precFactor();
  for (int i = 0; i < 10; i++)
    diff_solver->solve(c, 0.02);

  // test the norm of c
  ScalarType nrm;
  VecNorm(c, NORM_2, &nrm);
  REQUIRE(nrm == Approx(2.0487));
  // test number of ksp iter for last solve
  REQUIRE(diff_solver->ksp_itr_ == 5);

  // deallocate memory
  for (int i = 0; i < 8; i++) {
    VecDestroy(&work[i]);
  }
  delete[] work;
  VecDestroy(&c);
}

/* run a simple advection equation \partial c + v.\gradc = 0 with non-uniform velocity and advect back with negative velocity */
TEST_CASE( "Running advection solver", "[pdesolver]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(ACCFFT);
#endif

  DISABLE_VERBOSE = true;
  ScalarType norm_c, norm;
  Vec c, c_copy;

  // initialize a grid ~ use good resolution to minimize discretization errors
  initializeGrid(256, params, spec_ops);
  // create tumor class
  params->tu_->model_ = 4; // creates velocities, displacements
  std::shared_ptr<Tumor> tumor = std::make_shared<Tumor>(params, spec_ops);
  // create a test velocity
  createTestField(tumor->velocity_, params);
  // create a test function
  VecDuplicate(tumor->c_t_, &c);
  VecDuplicate(tumor->c_t_, &c_copy);
  VecSet(c, 0);
  createTestFunction(c, params);
  VecCopy(c, c_copy);
  // get norm of c
  VecNorm(c, NORM_2, &norm_c);
  // make an advection solver
  std::shared_ptr<AdvectionSolver> adv_solver = std::make_shared<SemiLagrangianSolver>(params, tumor, spec_ops);
  adv_solver->advection_mode_ = 2; // pure advection
  // run adv solver forward
  for (int i = 0; i < 4; i++) {
    adv_solver->solve(c, tumor->velocity_, 0.02);
  }
  // run inverse advection solver
  tumor->velocity_->scale(-1); // reverse the velocity
  adv_solver->trajectoryIsComputed_ = false; // recompute the trajectories
  for (int i = 0; i < 4; i++) {
    adv_solver->solve(c, tumor->velocity_, 0.02);
  }
  // get new c norm
  VecAXPY(c, -1, c_copy);
  VecNorm(c, NORM_2, &norm);

  // less than 1% error
  REQUIRE(norm / norm_c < 1E-2);

  VecDestroy(&c);
  VecDestroy(&c_copy);
}


