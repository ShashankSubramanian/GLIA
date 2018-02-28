#include "Utils.h"
#include "TumorSolverInterface.h"

static char help[] = "Inverse Driver \
\n Testcase 1 - Constant reaction and diffusion coefficient \
\n Testcase 2 - Sinusoidal reaction and diffusion coefficient";

PetscErrorCode readData (Vec &data, std::shared_ptr<NMisc> n_misc);
PetscErrorCode readDataAndAtlas (Vec &data, Vec &wm, Vec &gm, Vec &glm, Vec &csf, std::shared_ptr<NMisc> n_misc);
PetscErrorCode computeError (double &error_norm, Vec p_rec, Vec data, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);

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
    Vec data, p_rec, wm, gm, glm, csf;
    PetscErrorCode ierr = 0;

    wm = nullptr; gm = nullptr; glm = nullptr; csf = nullptr;

    // Data read only
    PCOUT << "Read raw Data --->" << std::endl;
    ierr = readData (data, n_misc);
    PCOUT << "Data Read: Inverse solve begin --->" << std::endl;
    
    //Data and altas read
    // PCOUT << " ------- Reading data and atlas: --------- " << std::endl;
    // ierr = readDataAndAtlas (data, gm, wm, glm, csf, n_misc);
    // PCOUT << " ------- Data and atlas read -------- " <<std::endl; 

    std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc);
    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    if (!n_misc->bounding_box_) {
        // ierr = tumor->mat_prop_->setValuesCustom (gm, wm, glm, csf, n_misc);    //Overwrite Matprop with custom atlas
        ierr = tumor->phi_->setGaussians (data);                                   //Overwrites bounding box phis with custom phis
        ierr = tumor->phi_->setValues (tumor->mat_prop_);
    }

     //Create p_rec
    int np = n_misc->np_;
    int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec);                 CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);                            CHKERRQ (ierr);
        ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);                  CHKERRQ (ierr);
        ierr = VecSetFromOptions (p_rec);                                       CHKERRQ (ierr);
    #endif

    ierr = solver_interface->setParams (p_rec, nullptr);
    //Solve interpolation
    // ierr = solver_interface->solveInterpolation (data, p_rec, tumor->phi_, n_misc);
    
    ierr = VecSet (p_rec, 0);                                               CHKERRQ (ierr);
    //Solve tumor inversion
    ierr = solver_interface->solveInverse (p_rec, data, nullptr);

    double prec_norm;
    ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
    PCOUT << "\nReconstructed P Norm: " << prec_norm << std::endl;

    double l2_rel_error = 0.0;
    ierr = computeError (l2_rel_error, p_rec, data, solver_interface, n_misc);
    PCOUT << "\nL2 Error in Reconstruction: " << l2_rel_error << std::endl;

    e1.addTimings (n_misc->timers_);
    e1.stop ();
    EventRegistry::finalize ();
    if (procid == 0) {
        EventRegistry r;
        r.print ();
        r.print ("EventsTimings.log", true);
    }

    //Destroy vectors
    ierr = VecDestroy (&data);                                              CHKERRQ (ierr);
    if (gm != nullptr) {ierr = VecDestroy (&gm);                            CHKERRQ (ierr);}
    if (wm != nullptr) {ierr = VecDestroy (&wm);                            CHKERRQ (ierr);}
    if (glm != nullptr) {ierr = VecDestroy (&glm);                          CHKERRQ (ierr);}
    if (csf != nullptr) {ierr = VecDestroy (&csf);                          CHKERRQ (ierr);}
    ierr = VecDestroy (&p_rec);                                             CHKERRQ (ierr);
}

/* --------------------------------------------------------------------------------------------------------------*/
    accfft_destroy_plan (plan);
    PetscFinalize ();
}

PetscErrorCode readDataAndAtlas (Vec &data, Vec &wm, Vec &gm, Vec &glm, Vec &csf, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &data);                             CHKERRQ (ierr);
    ierr = VecSetSizes (data, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = VecSetFromOptions (data);                                        CHKERRQ (ierr);

    ierr = VecDuplicate (data, &wm);                                        CHKERRQ (ierr);
    ierr = VecDuplicate (data, &gm);                                        CHKERRQ (ierr);
    ierr = VecDuplicate (data, &glm);                                       CHKERRQ (ierr);
    ierr = VecDuplicate (data, &csf);                                       CHKERRQ (ierr);

    dataIn (data, n_misc, "tu.nc");
    dataIn (wm, n_misc, "jakob_wm.nc");
    dataIn (gm, n_misc, "jakob_gm.nc");
    dataIn (csf, n_misc, "jakob_csf.nc");

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

    if (n_misc->writeOutput_)
        dataOut (c_rec_0, n_misc, "CRecon0.nc");

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

PetscErrorCode readData (Vec &data, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCreate (PETSC_COMM_WORLD, &data);                             CHKERRQ (ierr);
    ierr = VecSetSizes (data, n_misc->n_local_, n_misc->n_global_);         CHKERRQ (ierr);
    ierr = VecSetFromOptions (data);                                        CHKERRQ (ierr);

    dataIn (data, n_misc, "tuAAAN.nc");

    PetscFunctionReturn (0);
}
