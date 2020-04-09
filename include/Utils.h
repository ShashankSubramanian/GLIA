#ifndef _UTILS_H
#define _UTILS_H

/* General Utilies */

#include <petsc.h>
#include <stdlib.h>
#include <iomanip>
#include "petsctao.h"
#include <limits>
#include <cfloat>
#include <math.h>
#include <memory>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <omp.h>
#include <complex>
#include <cmath>
#include <vector>
#include <queue>
#include <accfft_utils.h>
#include <assert.h>
#include <sys/stat.h>
#include <map>
#include "EventTimings.hpp"
#include <pnetcdf.h>
#ifdef NIFTIIO
  #include "nifti1_io.h"
#endif
#include "TypeDefs.h"

#ifdef CUDA
    #include "UtilsCuda.h"
#endif

enum {ACCFFT = 0, CUFFT = 1};
class Phi;

#define throwError(str) throwErrorMsg(str, __LINE__, __FILE__)

struct TumorStatistics {
  int nb_state_solves;            /// @brief number of state equation solves
  int nb_adjoint_solves;          /// @brief number of adjoint equation solves
  int nb_grad_evals;              /// @brief number of gradient evaluations
  int nb_obj_evals;               /// @brief number of objective evaluations
  int nb_hessian_evals;           /// @brief number of hessian evaluations

  int nb_state_solves_acc;        /// @brief number of state equation solves
  int nb_adjoint_solves_acc;      /// @brief number of adjoint equation solves
  int nb_grad_evals_acc;          /// @brief number of gradient evaluations
  int nb_obj_evals_acc;           /// @brief number of objective evaluations
  int nb_hessian_evals_acc;       /// @brief number of hessian evaluations

public:
  TumorStatistics() :
  nb_state_solves(0),
  nb_adjoint_solves(0),
  nb_grad_evals(0),
  nb_obj_evals(0),
  nb_hessian_evals(0),
  nb_state_solves_acc(0),
  nb_adjoint_solves_acc(0),
  nb_grad_evals_acc(0),
  nb_obj_evals_acc(0),
  nb_hessian_evals_acc(0)
  {}

  void reset() {
    nb_state_solves_acc     += nb_state_solves;
    nb_adjoint_solves_acc   += nb_adjoint_solves;
    nb_grad_evals_acc       += nb_grad_evals;
    nb_obj_evals_acc        += nb_obj_evals;
    nb_hessian_evals_acc    += nb_hessian_evals;
    nb_state_solves         = 0;
    nb_adjoint_solves       = 0;
    nb_grad_evals           = 0;
    nb_obj_evals            = 0;
    nb_hessian_evals        = 0;
  }

  void reset0() {
    nb_state_solves_acc     = 0;
    nb_adjoint_solves_acc   = 0;
    nb_grad_evals_acc       = 0;
    nb_obj_evals_acc        = 0;
    nb_hessian_evals_acc    = 0;
    nb_state_solves         = 0;
    nb_adjoint_solves       = 0;
    nb_grad_evals           = 0;
    nb_obj_evals            = 0;
    nb_hessian_evals        = 0;
  }

  PetscErrorCode print();
};

PetscErrorCode vecGetArray (Vec x, ScalarType **x_ptr);
PetscErrorCode vecRestoreArray (Vec x, ScalarType **x_ptr);

class VecField {
    public:
        VecField (int nl, int ng);
        Vec x_;
        Vec y_;
        Vec z_;
        ~VecField () {
            PetscErrorCode ierr = 0;
            ierr = VecDestroy (&x_);
            ierr = VecDestroy (&y_);
            ierr = VecDestroy (&z_);
        }

        PetscErrorCode computeMagnitude (Vec);
        PetscErrorCode copy (std::shared_ptr<VecField> field);
        PetscErrorCode set (ScalarType scalar);
        PetscErrorCode scale (ScalarType scalar);
        PetscErrorCode getComponentArrays (ScalarType *&x_ptr, ScalarType *&y_ptr, ScalarType *&z_ptr);
        PetscErrorCode restoreComponentArrays (ScalarType *&x_ptr, ScalarType *&y_ptr, ScalarType *&z_ptr);
        PetscErrorCode setIndividualComponents (Vec in);  // uses indivdual components from in and sets it to x,y,z
        PetscErrorCode getIndividualComponents (Vec in);  // uses x,y,z to populate in
};

/**
    Context structure for user-defined linesearch routines needed
    for L1 minimization problems
**/
struct LSCtx {
    Vec x_work_1;   //Temporary vector for storing steepest descent guess
    Vec x_work_2; //Work vector
    Vec x_sol;
    ScalarType sigma; //Sufficient decrease parameter
    ScalarType lambda; //Regularization parameter for L1: Linesearch needs
                   //this application specific info
    PetscReal J_old;
};


//PetscErrorCode enforcePositivity (Vec c, std::shared_ptr<NMisc> n_misc);
//PetscErrorCode checkClipping (Vec c, std::shared_ptr<NMisc> n_misc);

/// @brief computes geometric tumor coupling m1 = m0(1-c(1))
/*
PetscErrorCode geometricCoupling(
  Vec m1_wm, Vec m1_gm, Vec m1_csf, Vec m1_glm, Vec m1_bg,
  Vec m0_wm, Vec m0_gm, Vec m0_csf, Vec m0_glm, Vec m0_bg,
  Vec c1, std::shared_ptr<NMisc> nmisc);

// @brief computes difference xi = m_data - m_geo
//  - function assumes that on input, xi = m_geo * (1-c(1))
///
PetscErrorCode geometricCouplingAdjoint(ScalarType *sqrdl2norm,
	Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg,
	Vec m_geo_wm, Vec m_geo_gm, Vec m_geo_csf, Vec m_geo_glm, Vec m_geo_bg,
	Vec m_data_wm, Vec m_data_gm, Vec m_data_csf, Vec m_data_glm, Vec m_data_bg);

/// @brief computes difference diff = x - y
PetscErrorCode computeDifference(ScalarType *sqrdl2norm,
	Vec diff_wm, Vec diff_gm, Vec diff_csf, Vec diff_glm, Vec diff_bg,
	Vec x_wm, Vec x_gm, Vec x_csf, Vec x_glm, Vec x_bg,
	Vec y_wm, Vec y_gm, Vec y_csf, Vec y_glm, Vec y_bg);
*/

/* helper methods for print out to console */
PetscErrorCode tuMSG(std::string msg, int size = 111);
PetscErrorCode tuMSGstd(std::string msg, int size = 111);
PetscErrorCode tuMSGwarn(std::string msg, int size = 111);
PetscErrorCode _tuMSG(std::string msg, std::string color, int size);

PetscErrorCode vecMax (Vec x, PetscInt *p, PetscReal *val);
PetscErrorCode vecSign (Vec x); //signum of petsc vector
PetscErrorCode vecSparsity (Vec x, ScalarType &sparsity); //Hoyer measure for sparsity of vector

/* definition of tumor assert */
#ifndef NDEBUG
#   define TU_assert(Expr, Msg) \
    __TU_assert(#Expr, Expr, __FILE__, __LINE__, Msg);
#else
#   define TU_assert(Expr, Msg);
#endif
void __TU_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg);

PetscErrorCode hardThreshold (Vec x, int sparsity_level, int sz, std::vector<int> &support, int &nnz);
ScalarType myDistance (ScalarType *c1, ScalarType *c2);
PetscErrorCode hardThreshold (Vec x, int sparsity_level, int sz, std::vector<int> &support, std::vector<int> labels, std::vector<ScalarType> weights, int &nnz, int num_components);


PetscErrorCode computeCenterOfMass (Vec x, int *isize, int *istart, ScalarType *h, ScalarType *cm);
PetscErrorCode setupVec (Vec x, int type = MPI);

#endif // end _UTILS_H
