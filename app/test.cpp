
// system includes
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>

#include "IO.h"
#include "Utils.h"
#include "Solver.h"
#include "Parameters.h"
#include "SpectralOperators.h"
#include "SolverInterface.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

// Not needed: directly modified catch.hpp to allow for printing from rank 0. TODO(S): delete.
// void ConsoleReporter::testRunEnded(TestRunStats const& _testRunStats) {
//   int rank id = -1;
//   MPI Comm rank(MPI COMM WORLD,&rank id);
//   if(rank id != 0 && testRunStats.totals.testCases.allPassed())
//       return;
//   printTotalsDivider(_testRunStats.totals);
//   printTotals(_testRunStats.totals);
//   stream << std::endl;
//   StreamingReporterBase::testRunEnded(_testRunStats);
// }

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, reinterpret_cast<char*>(NULL), reinterpret_cast<char*>(NULL)); CHKERRQ(ierr);
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  // verbose
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###"); CHKERRQ (ierr);
  ierr = tuMSG("###                                         TUMOR INVERSION: UNIT TESTS                                   ###"); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###"); CHKERRQ (ierr);

  EventRegistry::initialize();
  ierr = tuMSGwarn("Running tests with Catch2..."); CHKERRQ(ierr);
  int result = Catch::Session().run();

  EventRegistry::finalize ();
  if (procid == 0) {
      EventRegistry r;
      r.print();
      r.print("EventsTimings.log", true);
  }

  ierr = PetscFinalize();
  return(result);
}
