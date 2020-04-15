#include "SolverInterface.h"
#include "TestSuite.h"
/* #### ------------------------------------------------------------------- #### */
/* #### ========                      TestSuite                    ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
TestSuite::TestSuite(Test test) : SolverInterface() {
  testcase_ = test;
}

PetscErrorCode TestSuite::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  ierr = tuMSGwarn(" Initializing for TestSuite."); CHKERRQ(ierr);
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode TestSuite::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;

  // simulator test
  switch(testcase_) {
    case DEFAULTTEST: // run any default tests: TODO
      break;
    case FORWARDTEST:
      ierr = tuMSGwarn(" Forward simulator test."); CHKERRQ(ierr);
      ierr = forwardTest(); CHKERRQ(ierr);
      break;
    case INVERSETEST:
      ierr = tuMSGwarn(" Inverse simulator test."); CHKERRQ(ierr);
      ierr = inverseTest(); CHKERRQ(ierr);
      break;
    default:
      ierr = tuMSGwarn(" Invalid test. Exiting..."); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode TestSuite::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  ierr = tuMSGwarn(" Test suite complete."); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                SIMULATOR LEVEL TESTS              ======== #### */
/* #### ------------------------------------------------------------------- #### */

PetscErrorCode TestSuite::forwardTest() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  /* run forward solver */
  ierr = createSynthetic(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode TestSuite::inverseTest() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  /* create some data; no observation op */
  ierr = createSynthetic(); CHKERRQ(ierr);
  data_support_ = data_t1_;
  data_->set(data_t1_, nullptr);
  /* set Gaussians */
  ierr = initializeGaussians(); CHKERRQ(ierr);
  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  // ierr = solver_interface_->setParams(p_rec_, nullptr);
  ierr = resetOperators(p_rec_); CHKERRQ(ierr);
  // ierr = solver_interface_->setInitialGuess(0.); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  /* run sparse til solver */
  ierr = tuMSGwarn(" Beginning Test Inversion for Sparse TIL, and Diffusion/Reaction."); CHKERRQ(ierr);
  SolverInterface::run(); CHKERRQ(ierr);

  // set the reg norm as L2
  params_->opt_->regularization_norm_ = L2;
  // inv_solver_->getInverseSolverContext()->cosamp_->inexact_nits = params_->opt_->newton_maxit_; // TODO(K) restart version
  optimizer_->ctx_->cosamp_->maxit_newton = params_->opt_->newton_maxit_;
  ierr = optimizer_->setData(data_); CHKERRQ(ierr);
  ierr = optimizer_->setInitialGuess(p_rec_); CHKERRQ(ierr);         // TODO: requires length of np + nk + nr
  ierr = VecCopy(optimizer_->getSolution(), p_rec_); CHKERRQ(ierr);
  // ierr = solver_interface_->solveInverseCoSaMp(p_rec_, data_t1_, nullptr); CHKERRQ(ierr);

  // TODO: compare results

  /* finalize with error metrics */
  ierr = tuMSGwarn(" Finalizing Test Inversion for Sparse TIL, and Reaction/Diffusion."); CHKERRQ(ierr);
  ierr = SolverInterface::finalize(); CHKERRQ(ierr);


  PetscFunctionReturn(ierr);
}




























// TODO (S,K) implement test-suite

// diffcoef sinusoidal test
// if (n_misc->testcase_ != BRAIN && n_misc->testcase_ != BRAINNEARMF && n_misc->testcase_ != BRAINFARMF) {
//         ScalarType *kxx_ptr, *kyy_ptr, *kzz_ptr;
//         ierr = VecGetArray (kxx_, &kxx_ptr);                     CHKERRQ (ierr);
//         ierr = VecGetArray (kyy_, &kyy_ptr);                     CHKERRQ (ierr);
//         ierr = VecGetArray (kzz_, &kzz_ptr);                     CHKERRQ (ierr);
//         int64_t X, Y, Z, index;
//         ScalarType amp;
//         if (n_misc->testcase_ == CONSTCOEF)
//             amp = 0.0;
//         else if (n_misc->testcase_ == SINECOEF)
//             amp = std::min ((ScalarType)1.0, k_scale_);

//         ScalarType freq = 4.0;
//         for (int x = 0; x < n_misc->isize_[0]; x++) {
//             for (int y = 0; y < n_misc->isize_[1]; y++) {
//                 for (int z = 0; z < n_misc->isize_[2]; z++) {
//                     X = n_misc->istart_[0] + x;
//                     Y = n_misc->istart_[1] + y;
//                     Z = n_misc->istart_[2] + z;

//                     index = x * n_misc->isize_[1] * n_misc->isize_[2] + y * n_misc->isize_[2] + z;

//                     kxx_ptr[index] = k_scale + amp * sin (freq * 2.0 * M_PI / params->grid_->n_[0] * X)
//                                                    * sin (freq * 2.0 * M_PI / params->grid_->n_[1] * Y)
//                                                    * sin (freq * 2.0 * M_PI / params->grid_->n_[2] * Z);

//                     // kxx_ptr[index] = 1E-2 + 0.5 * sin (2.0 * M_PI / params->grid_->n_[0] * X) * cos (2.0 * M_PI / params->grid_->n_[1] * Y)
//                     //                             + 0.5 + 1E-3 * (1 - 0.5 * sin (2.0 * M_PI / params->grid_->n_[0] * X) * cos (2.0 * M_PI / params->grid_->n_[1] * Y)
//                     //                                     + 0.5);

//                     kyy_ptr[index] = kxx_ptr[index];
//                     kzz_ptr[index] = kxx_ptr[index];
//                 }
//             }
//         }
//         ierr = VecRestoreArray (kxx_, &kxx_ptr);                 CHKERRQ (ierr);
//         ierr = VecRestoreArray (kyy_, &kyy_ptr);                 CHKERRQ (ierr);
//         ierr = VecRestoreArray (kzz_, &kzz_ptr);                 CHKERRQ (ierr);
//     }

// reaccoeff sinusoidal testcase
// if (n_misc->testcase_ != BRAIN && n_misc->testcase_ != BRAINNEARMF && n_misc->testcase_ != BRAINFARMF) {
//         ScalarType *rho_vec_ptr;
//         ierr = VecGetArray (rho_vec_, &rho_vec_ptr);             CHKERRQ (ierr);
//         int64_t X, Y, Z, index;
//         ScalarType amp;
//         if (n_misc->testcase_ == CONSTCOEF)
//             amp = 0.0;
//         else if (n_misc->testcase_ == SINECOEF)
//             amp = std::min ((ScalarType)1.0, rho_scale_);

//         ScalarType freq = 4.0;
//         for (int x = 0; x < n_misc->isize_[0]; x++) {
//             for (int y = 0; y < n_misc->isize_[1]; y++) {
//                 for (int z = 0; z < n_misc->isize_[2]; z++) {
//                     X = n_misc->istart_[0] + x;
//                     Y = n_misc->istart_[1] + y;
//                     Z = n_misc->istart_[2] + z;

//                     index = x * n_misc->isize_[1] * n_misc->isize_[2] + y * n_misc->isize_[2] + z;

//                     rho_vec_ptr[index] = rho_scale + amp * sin (freq * 2.0 * M_PI / n_misc->n_[0] * X)
//                                                    * sin (freq * 2.0 * M_PI / n_misc->n_[1] * Y)
//                                                    * sin (freq * 2.0 * M_PI / n_misc->n_[2] * Z);
//                 }
//             }
//         }
//         ierr = VecRestoreArray (rho_vec_, &rho_vec_ptr);             CHKERRQ (ierr);
//     }
