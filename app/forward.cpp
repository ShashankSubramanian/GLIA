#include "petsctao.h"
#include "petsc.h"
#include "accfft.h"

#include "Utils.h"
#include "Tumor.h"
#include "TumorSolverInterface.h"
#include "PdeOperators.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

static char help[] = "Forward Driver";

int main (int argc, char** argv) {
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
    double *c_0;
    Complex *c_hat;
    #ifdef CUDA
        int64_t alloc_max = accfft_local_size_dft_r2c_gpu (n, isize, istart, osize, ostart, c_comm);
        cudaMalloc ((void**) &c_0, alloc_max);
        cudaMalloc ((void**) &c_hat, alloc_max);
        fft_plan *plan = accfft_plan_dft_3d_r2c_gpu (n, c_0, (double*) c_hat, c_comm, ACCFFT_MEASURE);

        // define constants for the gpu
        initCudaConstants (isize, osize, istart, ostart, n);
    #else
        int64_t alloc_max = accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
        c_0= (double*) accfft_alloc (alloc_max);
        c_hat = (Complex*) accfft_alloc (alloc_max);
        fft_plan *plan = accfft_plan_dft_3d_r2c (n, c_0, (double*) c_hat, c_comm, ACCFFT_MEASURE);        
    #endif
    fft_free (c_0);
    fft_free (c_hat);

/* --------------------------------------------------------------------------------------------------------------*/

{

    EventRegistry::initialize ();
    Event e1 ("solve-tumor-forward");
	std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, osize, istart, ostart, plan, c_comm, c_dims, testcase);   //This class contains all required parameters
	std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc);

    double self_exec_time = -MPI_Wtime ();
    std::array<double, 7> timers = {0};
	//Create IC
	Vec c_0, c_t;
    #ifdef SERIAL
        Vec p;
        ierr = VecCreateSeq (PETSC_COMM_SELF, n_misc->np_, &p);                            CHKERRQ (ierr);
        PetscScalar val[2] = {.9, .2};
        PetscInt center = (int) std::floor(n_misc->np_ / 2.);
        PetscInt idx[2] = {center-1, center};
        ierr = VecSetValues(p, 2, idx, val, INSERT_VALUES );        CHKERRQ(ierr);
        ierr = VecAssemblyBegin(p);                                 CHKERRQ(ierr);
        ierr = VecAssemblyEnd(p);                                   CHKERRQ(ierr);
    #endif

	ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = setupVec (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();


    #ifdef SERIAL
        ierr = tumor->setTrueP (p);
    #else
        ierr = tumor->setTrueP (n_misc);
    #endif


    ierr = tumor->phi_->apply (c_0, tumor->p_true_);
    if (n_misc->writeOutput_)
        dataOut (c_0, n_misc, "forward_IC.nc");
    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    double max, min;
    ierr = VecMax (c_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Data IC Max and Min : " << max << " " << min << std::endl;
    PCOUT << "Forward solve begin" << std::endl;

	ierr = solver_interface->solveForward (c_t , c_0); 

	if (n_misc->writeOutput_)
        dataOut (c_t, n_misc, "forward_data.nc");
    PCOUT << "Forward solve done" << std::endl;
    ierr = VecMax (c_t, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_t, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC fwd solve IC Max and Min : " << max << " " << min << std::endl;
     self_exec_time += MPI_Wtime ();
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
