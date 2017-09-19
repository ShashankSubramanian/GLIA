#include "DerivativeOperators.h"

PetscErrorCode DerivativeOperatorsRD::evaluateObjective (PetscReal *J, Vec x) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    

    PetscFunctionReturn(0);
}
PetscErrorCode DerivativeOperatorsRD::evaluateGradient (Vec dJ, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscFunctionReturn(0);
}
PetscErrorCode DerivativeOperatorsRD::evaluateHessian (Vec x, Vec y){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscFunctionReturn(0);
}
