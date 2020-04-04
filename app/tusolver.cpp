
// system includes

#include "Utils.h"
#include "Parameters.h"
#include "TumorSolverInterface.h"


enum RunMode = {FORWARD, INVERSE_L2, INVERSE_L1, INVERSE_RD, INVERSE_ME, INVERSE_MS, TEST};


void openFiles(std::shared_ptr<Parameters> params) {
  std::stringstream ss;
  ss.str(std::string()); ss.clear();
  if (params->verbosity_ >= 2) {
      if (procid == 0) {
          ss << params->writepath_.str().c_str() << "x_it.dat";
          params->outfile_sol_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          ss << params->writepath_.str().c_str() << "g_it.dat";
          params->outfile_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          ss << params->writepath_.str().c_str() << "glob_g_it.dat";
          params->outfile_glob_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          params->outfile_sol_ << std::setprecision(16)<<std::scientific;
          params->outfile_grad_ << std::setprecision(16)<<std::scientific;
          params->outfile_glob_grad_ << std::setprecision(16)<<std::scientific;
      }
  }
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc, &argv, reinterpret_cast<char*>(NULL), reinterpret_cast<char*>(NULL)); CHKERRQ(ierr);

  accfft_init();
  MPI_Comm c_comm;
  int c_dims[2] = { 0 };
  accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
  int isize[3], osize[3], istart[3], ostart[3];

  std::shared_ptr<SpectralOperators> spec_ops;
  #if defined(CUDA) && !defined(MPICUDA)
      spec_ops = std::make_shared<SpectralOperators> (CUFFT);
  #else
      spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
  #endif
  spec_ops->setup (n, isize, istart, osize, ostart, c_comm);
  int64_t alloc_max = spec_ops->alloc_max_;
  fft_plan *plan = spec_ops->plan_;

  RunMode run_mode = FORWARD;
  std::stringstream ss;
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

  // TODO(K) parse config file and populate into parameters; also set run_mode
  // TODO(K) params->opt_settings_ have to be populated in arg parse

  std::unique_ptr<Solver> solver;
  switch(run_mode) {
    case FORWARD:
      solve = std::make_unique<ForwardSolver>(spec_ops);
      break;
    case INVERSE_L2:
      solve = std::make_unique<InverseL2Solver>(spec_ops);
      break;
    case INVERSE_L1:
      solve = std::make_unique<InverseL1Solver>(spec_ops);
      break;
    case INVERSE_RD:
      solve = std::make_unique<InverseReactionDiffusionSolver>(spec_ops);
      break;
    case INVERSE_ME:
      solve = std::unique_ptr<InverseMassEffectSolver>(spec_ops);
      break;
    case INVERSE_MS:
      solve = std::make_unique<InverseMultiSpeciesSolver>(spec_ops);
      break;
    case TEST:
      solve = std::make_unique<TestSuite>(spec_ops);
      break;
    default:
      ierr = cplMSGwarn("Configuration invalid: solver mode not recognized. Exiting."); CHKERRQ(ierr);
      PetscFunctionReturn(ierr);
  }

  ierr = openFiles(); CHKERRQ(ierr); // opens all txt output files on one core only
  // ensures data for specific solver is read and code is set up for running mode
  ierr = solver->initialize(); CHKERRQ(ierr);
  ierr = solver->run(); CHKERRQ(ierr);
  // compute errors, segmentations, other measures of interest, and shutdown solver
  ierr = solver->finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
