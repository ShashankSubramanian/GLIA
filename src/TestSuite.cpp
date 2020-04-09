
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
//         ierr = VecGetArray (rho_vec_, &rho_vec_ptr);             CHKERRQ (ierr);
//     }