#ifndef _UTILS_H
#define _UTILS_H

/* General Utilies */
#include <accfft_utils.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <petsc.h>
#include <petscmat.h>
#include <pnetcdf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <vector>
#include "EventTimings.hpp"
#include "petsctao.h"
#include "TypeDefs.h"
#ifdef NIFTIIO
#include "nifti1_io.h"
#endif
#ifdef CUDA
#include "UtilsCuda.h"
#endif

#define throwError(str) throwErrorMsg(str, __LINE__, __FILE__)

struct TumorStatistics {
  int nb_state_solves;    /// @brief number of state equation solves
  int nb_adjoint_solves;  /// @brief number of adjoint equation solves
  int nb_grad_evals;      /// @brief number of gradient evaluations
  int nb_obj_evals;       /// @brief number of objective evaluations
  int nb_hessian_evals;   /// @brief number of hessian evaluations

  int nb_state_solves_acc;    /// @brief number of state equation solves
  int nb_adjoint_solves_acc;  /// @brief number of adjoint equation solves
  int nb_grad_evals_acc;      /// @brief number of gradient evaluations
  int nb_obj_evals_acc;       /// @brief number of objective evaluations
  int nb_hessian_evals_acc;   /// @brief number of hessian evaluations

 public:
  TumorStatistics()
      : nb_state_solves(0),
        nb_adjoint_solves(0),
        nb_grad_evals(0),
        nb_obj_evals(0),
        nb_hessian_evals(0),
        nb_state_solves_acc(0),
        nb_adjoint_solves_acc(0),
        nb_grad_evals_acc(0),
        nb_obj_evals_acc(0),
        nb_hessian_evals_acc(0) {}

  void reset() {
    nb_state_solves_acc += nb_state_solves;
    nb_adjoint_solves_acc += nb_adjoint_solves;
    nb_grad_evals_acc += nb_grad_evals;
    nb_obj_evals_acc += nb_obj_evals;
    nb_hessian_evals_acc += nb_hessian_evals;
    nb_state_solves = 0;
    nb_adjoint_solves = 0;
    nb_grad_evals = 0;
    nb_obj_evals = 0;
    nb_hessian_evals = 0;
  }

  void reset0() {
    nb_state_solves_acc = 0;
    nb_adjoint_solves_acc = 0;
    nb_grad_evals_acc = 0;
    nb_obj_evals_acc = 0;
    nb_hessian_evals_acc = 0;
    nb_state_solves = 0;
    nb_adjoint_solves = 0;
    nb_grad_evals = 0;
    nb_obj_evals = 0;
    nb_hessian_evals = 0;
  }

  PetscErrorCode print();
};


class Data {
 public:
  Data() :
  t1(nullptr),
  t0(nullptr),
  two_snapshot(false)
  {}
  void set(Vec dt1, Vec dt0){
      t1 = dt1;
      if (dt0 != nullptr) {
          t0 = dt0; two_snapshot = true;
      }
  }
  void setT1(Vec dt1) {
      t1 = dt1;
  }
  void setT0(Vec dt0) {
      t0 = dt0;
      two_snapshot = true;
  }
  Vec dt1() {return t1;}
  Vec dt0() {return (two_snapshot) ? t0 : nullptr;}
  Vec t1;
  Vec t0;
  bool two_snapshot;
};

/* Encapsulates vector fields with 3 components */
class VecField {
 public:
  VecField(int nl, int ng);
  Vec x_;
  Vec y_;
  Vec z_;
  ~VecField() {
    PetscErrorCode ierr = 0;
    ierr = VecDestroy(&x_);
    ierr = VecDestroy(&y_);
    ierr = VecDestroy(&z_);
  }

  PetscErrorCode computeMagnitude(Vec);
  PetscErrorCode copy(std::shared_ptr<VecField> field);
  PetscErrorCode set(ScalarType scalar);
  PetscErrorCode scale(ScalarType scalar);
  PetscErrorCode getComponentArrays(ScalarType *&x_ptr, ScalarType *&y_ptr, ScalarType *&z_ptr);
  PetscErrorCode restoreComponentArrays(ScalarType *&x_ptr, ScalarType *&y_ptr, ScalarType *&z_ptr);
  PetscErrorCode setIndividualComponents(Vec in);  // uses indivdual components from in and sets it to x,y,z
  PetscErrorCode getIndividualComponents(Vec in);  // uses x,y,z to populate in
};


/* custom getters/restorers for cuda and cpu pointers */
PetscErrorCode vecGetArray(Vec x, ScalarType **x_ptr);
PetscErrorCode vecRestoreArray(Vec x, ScalarType **x_ptr);

/* custom vec routines */
PetscErrorCode vecMax(Vec x, PetscInt *p, PetscReal *val);
PetscErrorCode vecSum(Vec x, PetscScalar *val);
PetscErrorCode vecSign(Vec x);                            // signum of petsc vector
PetscErrorCode vecSparsity(Vec x, ScalarType &sparsity);  // Hoyer measure for sparsity of vector
PetscErrorCode setupVec(Vec x, int type = MPIVEC);
PetscErrorCode printVecBounds(Vec c, std::string str = "c");

/* helper methods for print out to console */
PetscErrorCode tuMSG(std::string msg, int size = 111);
PetscErrorCode tuMSGstd(std::string msg, int size = 111);
PetscErrorCode tuMSGwarn(std::string msg, int size = 111);
PetscErrorCode _tuMSG(std::string msg, std::string color, int size);
/* definition of tumor assert */
#ifndef NDEBUG
#define TU_assert(Expr, Msg) __TU_assert(#Expr, Expr, __FILE__, __LINE__, Msg);
#else
#define TU_assert(Expr, Msg) ;
#endif
void __TU_assert(const char *expr_str, bool expr, const char *file, int line, const char *msg);

/* custom routines for l0 helpers */
PetscErrorCode hardThreshold(Vec x, int sparsity_level, int sz, std::vector<int> &support, int &nnz);
ScalarType myDistance(ScalarType *c1, ScalarType *c2);
PetscErrorCode hardThreshold(Vec x, int sparsity_level, int sz, std::vector<int> &support, std::vector<int> labels, std::vector<ScalarType> weights, int &nnz, int num_components, double thresh_component_weight, bool double_mode=false);
PetscErrorCode computeCenterOfMass(Vec x, int *isize, int *istart, ScalarType *h, ScalarType *cm);

PetscErrorCode splitSegmentation(Vec seg, Vec wm, Vec gm, Vec vt, Vec csf, Vec tu, Vec ed, int nl, std::vector<int> &labels);
PetscErrorCode createEdemaBasedObservationMask(Vec mask, Vec tc, Vec ed, double lambda, int nl, std::vector<int> &labels);

PetscErrorCode computeDice(Vec in, Vec truth, ScalarType &dice);
PetscErrorCode computeVolume(Vec x, Vec roi, Vec temp, ScalarType measure, ScalarType *vol_roi, ScalarType *sum_roi=nullptr);
PetscErrorCode computeVolume(Vec x, ScalarType measure, ScalarType *vol, ScalarType *sum=nullptr);
PetscErrorCode computeStd(Vec x, Vec temp, Vec roi, PetscInt n_roi, ScalarType mean_roi, ScalarType *std_roi); 
PetscErrorCode computeStd(Vec x, Vec temp, PetscInt n, ScalarType mean, ScalarType *std);
PetscErrorCode computeQuantile(Vec x, Vec roi, Vec temp, Vec roi_temp, ScalarType *val, PetscInt roi_size, ScalarType quantile = 0.5, bool sort = true);
PetscErrorCode computeQuantile(Vec x, Vec temp, ScalarType *val, ScalarType quantile = 0.5);
PetscErrorCode vecScatter(Vec x, Vec x_scatter, Vec seq); 
PetscErrorCode setSequence(Vec x);
ScalarType computeDeterminant(std::array<ScalarType, 9> matrix);
std::array<ScalarType, 9> computeStressTensor(std::array<ScalarType, 9> E, ScalarType mu, ScalarType lam);
std::array<ScalarType, 9> computeStrainTensor(std::array<ScalarType, 9> F);
PetscErrorCode setupKSPEigenvalues(KSP *ksp, Mat *A, Vec *rhs, Vec *sol);
PetscErrorCode computeIndicatorFunction(Vec i, Vec x, ScalarType x_star);

#endif  // end _UTILS_H
