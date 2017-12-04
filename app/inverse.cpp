#include "Utils.h"
#include "TumorSolverInterface.h"

static char help[] = "Inverse Driver \
\n Testcase 1 - Constant reaction and diffusion coefficient \
\n Testcase 2 - Sinusoidal reaction and diffusion coefficient";

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode generateSinusoidalData (Vec &d, std::shared_ptr<NMisc> n_misc);
PetscErrorCode computeError (double &error_norm, Vec p_rec, Vec data, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode readData (Vec &data, std::shared_ptr<NMisc> n_misc);
PetscErrorCode createMFData (std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);

int main (int argc, char** argv) {
 /* ACCFFT, PETSC setup begin */
    PetscErrorCode ierr;
    PetscInitialize (&argc, &argv, (char*) 0, help);
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    int n[3];
    n[0] = 64;
    n[1] = 64;
    n[2] = 64;
    int testcase = 0;

    PetscOptionsBegin (PETSC_COMM_WORLD, NULL, "Tumor Inversion Options", "");
    PetscOptionsInt ("-nx", "NX", "", n[0], &n[0], NULL);
    PetscOptionsInt ("-ny", "NY", "", n[1], &n[1], NULL);
    PetscOptionsInt ("-nz", "NZ", "", n[2], &n[2], NULL);
    PetscOptionsInt ("-testcase", "Test Cases", "", testcase, &testcase, NULL);
    PetscOptionsEnd ();

    PCOUT << " ----- Grid Size: " << n[0] << "x" << n[1] << "x" << n[2] << " ---- " << std::endl;
    switch (testcase) {
        case CONSTCOEF: {
            PCOUT << " ----- Test Case 1: No brain, Constant reaction and diffusion ---- " << std::endl;
            break;
        }
        case SINECOEF: {
            PCOUT << " ----- Test Case 2: No brain, Sinusoidal reaction and diffusion ---- " << std::endl;
            break;
        }
        case BRAIN: {
            PCOUT << " ----- Full brain test ---- " << std::endl;
            break;
        }
        default: break;
    }

    accfft_init();
    MPI_Comm c_comm;
    int c_dims[2] = { 0 };
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
    int isize[3], osize[3], istart[3], ostart[3];
    int64_t alloc_max = accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    double *c_0 = (double*) accfft_alloc (alloc_max);
    Complex *c_hat = (Complex*) accfft_alloc (alloc_max);
    accfft_plan *plan = accfft_plan_dft_3d_r2c (n, c_0, (double*) c_hat, c_comm, ACCFFT_MEASURE);
    accfft_free (c_0);
    accfft_free (c_hat);
/* ACCFFT, PETSC setup end */
/* --------------------------------------------------------------------------------------------------------------*/

{
    EventRegistry::initialize ();
    Event e1 ("solve-tumor-inverse-tao");
    std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, osize, istart, ostart, plan, c_comm, c_dims, testcase);   //This class contains all required parameters
    std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc, nullptr, nullptr);

    n_misc->phi_sigma_ = 0.2;
    n_misc->phi_spacing_factor_ = 1.0;
    createMFData (solver_interface, n_misc);
    n_misc->phi_sigma_ = PETSC_PI / 10;
    n_misc->phi_spacing_factor_ = 1.5;
    exit (1);

    Vec c_0, data, p_rec;
    PetscErrorCode ierr = 0;
    PCOUT << "Generating Synthetic Data --->" << std::endl;
    ierr = generateSyntheticData (c_0, data, p_rec, solver_interface, n_misc);
    PCOUT << "Data Generated: Inverse solve begin --->" << std::endl;

    //Solve interpolation
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = solver_interface->solveInterpolation (data, p_rec, tumor->phi_, n_misc);

    exit (1);

    double self_exec_time = -MPI_Wtime ();
    std::array<double, 7> timers = {0};

    ierr = solver_interface->setInitialGuess(0.);
    ierr = solver_interface->solveInverse (p_rec, data, nullptr);

    self_exec_time += MPI_Wtime ();

    double prec_norm;
    ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
    PCOUT << "\nReconstructed P Norm: " << prec_norm << std::endl;

    double l2_rel_error = 0.0;
    ierr = computeError (l2_rel_error, p_rec, data, solver_interface, n_misc);
    PCOUT << "\nL2 Error in Reconstruction: " << l2_rel_error << std::endl;

    accumulateTimers (n_misc->timers_, timers, self_exec_time);
    e1.addTimings (timers);
    e1.stop ();
    EventRegistry::finalize ();
    if (procid == 0) {
        EventRegistry r;
        r.print ();
        r.print ("EventsTimings.log", true);
    }
}

/* --------------------------------------------------------------------------------------------------------------*/
    accfft_destroy_plan (plan);
    PetscFinalize ();
}

PetscErrorCode createMFData (std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec p;
    ierr = VecCreate (PETSC_COMM_WORLD, &p);                            CHKERRQ (ierr);
    ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
    ierr = VecSetFromOptions (p);                                       CHKERRQ (ierr);
    ierr = VecSet (p, 0.5);                                            CHKERRQ (ierr);

    std::array<double, 3> cm;
    cm[0] = 4.0; cm[1] = 2.53; cm[2] = 2.57;
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->phi_->setGaussians (cm, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = tumor->phi_->setValues (tumor->mat_prop_);
    Vec c, cf;
    ierr = VecCreate (PETSC_COMM_WORLD, &c);                            CHKERRQ (ierr);
    ierr = VecSetSizes (c, n_misc->n_local_, n_misc->n_global_);        CHKERRQ (ierr);
    ierr = VecSetFromOptions (c);                                       CHKERRQ (ierr);
    ierr = VecDuplicate (c, &cf);                                       CHKERRQ (ierr);
    ierr = VecSet (cf, 0);                                              CHKERRQ (ierr);

    ierr = tumor->phi_->apply (c, p);                                   CHKERRQ (ierr);
    // ierr = solver_interface->solveForward (c, c);
    // ierr = tumor->obs_->apply (c, c);
    ierr = VecAXPY (cf, 1.0, c);                                        CHKERRQ (ierr);

    //---------------------------------------------------------------------------------
    cm[0] = 4.0; cm[1] = 2.03; cm[2] = 4.07;
    ierr = tumor->phi_->setGaussians (cm, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = tumor->phi_->setValues (tumor->mat_prop_);
    ierr = tumor->phi_->apply (c, p);                                   CHKERRQ (ierr);
    // ierr = solver_interface->solveForward (c, c);
    // ierr = tumor->obs_->apply (c, c);
    ierr = VecAXPY (cf, 1.0, c);                                        CHKERRQ (ierr);

    //---------------------------------------------------------------------------------
    cm[0] = 4.0; cm[1] = 4.03; cm[2] = 2.07;
    ierr = tumor->phi_->setGaussians (cm, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = tumor->phi_->setValues (tumor->mat_prop_);
    ierr = tumor->phi_->apply (c, p);                                   CHKERRQ (ierr);
    // ierr = solver_interface->solveForward (c, c);
    // ierr = tumor->obs_->apply (c, c);
    ierr = VecAXPY (cf, 1.0, c);                                        CHKERRQ (ierr);

    //---------------------------------------------------------------------------------
    cm[0] = 4.0; cm[1] = 4.03; cm[2] = 4.07;
    ierr = tumor->phi_->setGaussians (cm, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = tumor->phi_->setValues (tumor->mat_prop_);
    ierr = tumor->phi_->apply (c, p);                                   CHKERRQ (ierr);
    // ierr = solver_interface->solveForward (c, c);
    // ierr = tumor->obs_->apply (c, c);
    ierr = VecAXPY (cf, 1.0, c);                                        CHKERRQ (ierr);

    double *ptr;
    ierr = VecGetArray (cf, &ptr);                                      CHKERRQ (ierr);
    ierr = weierstrassSmoother (ptr, ptr, n_misc, 0.0003);              CHKERRQ(ierr);
    // for (int i = 0; i < n_misc->n_local_; i++) {
    //     ptr[i] = 1 / (1 + exp(-ptr[i] + 10));
    // }
    ierr = VecRestoreArray (cf, &ptr);                                  CHKERRQ (ierr);

    n_misc->writepath_ << "./brain_data/" << n_misc->n_[0] <<"/";
    dataOut (cf, n_misc, "multifocal.nc");
    n_misc->writepath_ << "./results/";

    PetscFunctionReturn (0);
}

PetscErrorCode readData (Vec &data, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &data);                             CHKERRQ (ierr);
    ierr = VecSetSizes (data, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = VecSetFromOptions (data);                                        CHKERRQ (ierr);

    dataIn (data, n_misc, "multifocal.nc");

    PetscFunctionReturn (0);
}

PetscErrorCode computeError (double &error_norm, Vec p_rec, Vec data, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Vec c_rec_0, c_rec;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    double data_norm;
    ierr = VecDuplicate (data, &c_rec_0);                                   CHKERRQ (ierr);
    ierr = VecDuplicate (data, &c_rec);                                     CHKERRQ (ierr);
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->phi_->apply (c_rec_0, p_rec);

    double *c0_ptr;

    if (n_misc->model_ == 2) {
        ierr = VecGetArray (c_rec_0, &c0_ptr);                              CHKERRQ (ierr);
        for (int i = 0; i < n_misc->n_local_; i++) {
            c0_ptr[i] = 1 / (1 + exp(-c0_ptr[i] + n_misc->exp_shift_));
        }
        ierr = VecRestoreArray (c_rec_0, &c0_ptr);                          CHKERRQ (ierr);
    }

    ierr = solver_interface->solveForward (c_rec, c_rec_0);

    double max, min;
    ierr = VecMax (c_rec, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_rec, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Reconstructed Max and Min (Before Observation) : " << max << " " << min << std::endl;

    ierr = tumor->obs_->apply (c_rec, c_rec);   //Apply observation to reconstructed C to compare
                                                //values to data

    if (n_misc->writeOutput_)
        dataOut (c_rec, n_misc, "CRecon.nc");

    ierr = VecAXPY (c_rec, -1.0, data);                                     CHKERRQ (ierr);
    ierr = VecNorm (data, NORM_2, &data_norm);                              CHKERRQ (ierr);
    ierr = VecNorm (c_rec, NORM_2, &error_norm);                            CHKERRQ (ierr);

    error_norm /= data_norm;
    PetscFunctionReturn (0);
}

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    //Create p_rec
    ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
    ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
    ierr = VecSetFromOptions (p_rec);                                       CHKERRQ (ierr);

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = VecSetFromOptions (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->setTrueP (n_misc);
    ierr = tumor->phi_->apply (c_0, tumor->p_true_);

    double *c0_ptr;

    if (n_misc->model_ == 2) {
        ierr = VecGetArray (c_0, &c0_ptr);                                  CHKERRQ (ierr);
        for (int i = 0; i < n_misc->n_local_; i++) {
            c0_ptr[i] = 1 / (1 + exp(-c0_ptr[i] + n_misc->exp_shift_));
        }
        ierr = VecRestoreArray (c_0, &c0_ptr);                              CHKERRQ (ierr);
    }

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    if (n_misc->writeOutput_)
        dataOut (c_0, n_misc, "c0.nc");

    double max, min;
    ierr = VecMax (c_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Data IC Max and Min : " << max << " " << min << std::endl;

    ierr = solver_interface->solveForward (c_t, c_0);   //Observation operator is applied in InvSolve ()

    ierr = tumor->obs_->apply (c_t, c_t);

    if (n_misc->writeOutput_) {
        dataOut (c_t, n_misc, "data.nc");
    }

    PetscFunctionReturn (0);
}
