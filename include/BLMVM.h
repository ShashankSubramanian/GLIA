/*
Environment for creating new Tao solver for PDE-constrained L1 minimization
*/
#ifndef _BLMVM_SOLVER_H
#define _BLMVM_SOLVER_H

#include "petsc/private/taoimpl.h"
#include "petsc/private/taolinesearchimpl.h"
#undef __FUNCT__
#define __FUNCT__ __func__

/*
 Context for limited memory variable metric method for bound constrained
 optimization.
*/
typedef struct {

  Mat M;

  Vec unprojected_gradient;
  Vec Xold;
  Vec Gold;

  PetscInt n_free;
  PetscInt n_bind;

  PetscInt grad;
  PetscInt reset;
  Mat      H0;

  PetscReal last_ls_step;
} TAO_BLMVM_M;


PetscErrorCode TaoCreate_BLMVM_M (Tao tao);

PETSC_EXTERN PetscErrorCode MatLMVMReset(Mat);
PETSC_EXTERN PetscErrorCode MatLMVMUpdate(Mat,Vec, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMSetDelta(Mat,PetscReal);
PETSC_EXTERN PetscErrorCode MatLMVMSetScale(Mat,Vec);
PETSC_EXTERN PetscErrorCode MatLMVMGetRejects(Mat,PetscInt*);
PETSC_EXTERN PetscErrorCode MatLMVMSetH0(Mat,Mat);
PETSC_EXTERN PetscErrorCode MatLMVMGetH0(Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatLMVMGetH0KSP(Mat,KSP*);
PETSC_EXTERN PetscErrorCode MatLMVMSetPrev(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatLMVMGetX0(Mat,Vec);
PETSC_EXTERN PetscErrorCode MatLMVMRefine(Mat, Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMAllocateVectors(Mat m, Vec v);
PETSC_EXTERN PetscErrorCode MatLMVMSolve(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatCreateLMVM(MPI_Comm,PetscInt,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatView_LMVM(Mat,PetscViewer);
PETSC_EXTERN PetscErrorCode MatDestroy_LMVM(Mat);


#endif
