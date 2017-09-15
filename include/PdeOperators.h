#ifndef PDEOPERATORS_H_
#define PDEOPERATORS_H_

#include "Utils.h"
#include "Tumor.h"
#include <mpi.h>
#include <omp.h>


PetscErrorCode reaction (Vec c_t, NMisc *n_misc, Tumor *tumor, double dt);

#endif