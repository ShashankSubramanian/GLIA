#include <mpi.h>
#include "Utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <iomanip> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <accfft.h>
#include <accfft_operators.h>

static bool isLittleEndian () {
	uint16_t number = 0x1;
	uint8_t *numPtr = (uint8_t*) &number;
	return (numPtr[0] == 1);
}

void dataIn (double *A, NMisc* n_misc, const char *fname) {
	MPI_Comm c_comm = n_misc->c_comm_;
	int nprocs, procid;
	MPI_Comm_rank (c_comm, &procid);
	MPI_Comm_size (c_comm, &nprocs);

	if (A == NULL) {
		PCOUT << "Error in DataOut ---> Input data is null" << std::endl;
		return;
	}
	int *istart = n_misc->istart_;
	int *isize = n_misc->isize_;

	MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
	MPI_Offset isize_mpi[3] = { isize[0], isize[1], isize[2] };

	std::stringstream str;
	const char *prefix = "./brain_data/";
	str << prefix << "/" << n_misc->n_[0] << "/" << fname;	
	read_pnetcdf(str.str().c_str(), istart_mpi, isize_mpi, c_comm, n_misc->n_, A);
	return;
}

void dataIn (Vec A, NMisc *n_misc, const char *fname) {
  double *a_ptr;
  PetscErrorCode ierr;
  ierr = VecGetArray (A, &a_ptr);							
  dataIn(a_ptr, n_misc, fname);
  ierr = VecRestoreArray (A, &a_ptr);						
}
