#ifndef _HELPER_H
#define _HELPER_H

#include "Parameters.h"

PetscErrorCode createTestFunction(Vec x, std::shared_ptr<Parameters> params);
PetscErrorCode createTestField(std::shared_ptr<VecField> v, std::shared_ptr<Parameters> params);
PetscErrorCode createPVec(Vec &x, std::shared_ptr<Parameters> params);

#endif