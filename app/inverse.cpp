#include "Utils.h"
#include "TumorSolverInterface.h"

static char help[] = "Inverse Driver \
\n Testcase 1 - Constant reaction and diffusion coefficient \
\n Testcase 2 - Sinusoidal reaction and diffusion coefficient";

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);
PetscErrorCode generateSinusoidalData (Vec &d, std::shared_ptr<NMisc> n_misc);
PetscErrorCode computeError (double &error_norm, Vec p_rec, Vec data, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);

int main (int argc, char** argv) {
 /* ACCFFT, PETSC setup begin */
    google::InitGoogleLogging (argv[0]);
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
    std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, osize, istart, ostart, plan, c_comm, testcase);   //This class contains all required parameters
    std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc);

    Vec c_0, data, p_rec;
    PetscErrorCode ierr = 0;
    PCOUT << "Generating Synthetic Data --->" << std::endl;
    ierr = generateSyntheticData (c_0, data, p_rec, solver_interface, n_misc);
    PCOUT << "Data Generated: Inverse solve begin --->" << std::endl;

    ierr = solver_interface->solveInverse (p_rec, data, nullptr);

    double prec_norm;
    ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
    PCOUT << "\nReconstructed P Norm: " << prec_norm << std::endl;

    double l2_rel_error = 0.0;
    ierr = computeError (l2_rel_error, p_rec, data, solver_interface, n_misc);
    PCOUT << "\nL2 Error in Reconstruction: " << l2_rel_error << std::endl;
}

/* --------------------------------------------------------------------------------------------------------------*/
    accfft_destroy_plan (plan);
    PetscFinalize ();
}

PetscErrorCode computeError (double &error_norm, Vec p_rec, Vec data, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Vec c_rec_0, c_rec; 

    double data_norm;
    ierr = VecDuplicate (data, &c_rec_0);                                   CHKERRQ (ierr);
    ierr = VecDuplicate (data, &c_rec);                                     CHKERRQ (ierr);
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->phi_->apply (c_rec_0, p_rec); 

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_rec_0, n_misc);
    #endif

    ierr = solver_interface->solveForward (c_rec, c_rec_0);

    double max, min;
    ierr = VecMax (c_rec, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_rec, NULL, &min);                                      CHKERRQ (ierr);
    std::cout << "\nC Reconstructed Max and Min (Before Observation) : " << max << " " << min << std::endl;

    ierr = tumor->obs_->apply (c_rec, c_rec);   //Apply observation to reconstructed C to compare
                                                //values to data

    if (n_misc->writeOutput_) 
        dataOut (c_rec, n_misc, "results/CRecon.nc");

    ierr = VecAXPY (c_rec, -1.0, data);                                     CHKERRQ (ierr); 
    ierr = VecNorm (data, NORM_2, &data_norm);                              CHKERRQ (ierr);         
    ierr = VecNorm (c_rec, NORM_2, &error_norm);                            CHKERRQ (ierr); 

    error_norm /= data_norm;
    PetscFunctionReturn (0);
}

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
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
    ierr = tumor->setTrueP (n_misc->p_scale_true_);
    ierr = tumor->phi_->apply (c_0, tumor->p_true_);

    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    if (n_misc->writeOutput_) 
        dataOut (c_0, n_misc, "results/c0.nc");

    ierr = solver_interface->solveForward (c_t, c_0);   //Observation operator is applied in InvSolve ()

    if (n_misc->writeOutput_)
        dataOut (c_t, n_misc, "results/data.nc");

    PetscFunctionReturn (0);
}

