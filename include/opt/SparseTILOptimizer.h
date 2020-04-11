#ifndef SPARSE_TIL_OPTIMIZER_H_
#define SPARSE_TIL_OPTIMIZER_H_


#include "Optimizer.h"
#include "Parameters.h"

/* #### ------------------------------------------------------------------- #### */
/* #### ========          CoSaMp (SparseTIL) Context               ======== #### */
/* #### ------------------------------------------------------------------- #### */

struct CtxCoSaMp {
    int cosamp_stage;               // indicates solver state of CoSaMp function when using warmstart
    int its_l1;                     // cosamp iterations
    int np_full;                    // size of unrestricted subspace
    int maxit_newton;               // global maxit for L2 Newton solver
    int nits;                       // global iterations performed for L2 Newton solver
    int inexact_nits;               // Newton its per inexact solve
    bool compute_reference_values;  // if true, compute and store reference objective and gradient
    bool converged_l1;              // indicates if L1 solver converged
    bool converged_l2;              // indicates if L2 solver converged
    bool converged_error_l2;        // indicates if L2 solver diverged/failed
    bool initialized;               // indicates if vectors are allocated or destroyed
    PetscReal J;                    // objective function value
    PetscReal J_prev;               // previous objective function value
    PetscReal J_ref;                // reference objective function value
    PetscReal g_norm;               // norm of reference gradient
    PetscReal f_tol;                // CoSaMp iteration tolerance
    Vec g;                          // gradient
    Vec x_full;                     // solution vector full space
    Vec x_full_prev;                // solution vector full space
    Vec x_sub;                      // solution vector subspace
    Vec work;

    CtxCoSaMp ()
    :
      cosamp_stage(INIT)
    , its_l1(0)
    , np_full(0)
    , maxit_newton(50)
    , inexact_nits(4)
    , nits(0)
    , compute_reference_values(true)
    , converged_l1(false)
    , converged_l2(false)
    , converged_error_l2(false)
    , initialized(false)
    , J(0)
    , J_prev(0)
    , J_ref(0)
    , g_norm(0)
    , f_tol(1E-5)
    , g(nullptr)
    , x_full(nullptr)
    , x_full_prev(nullptr)
    , x_sub(nullptr)
    {}

    PetscErrorCode initialize(Vec p) {
        PetscFunctionBegin;
        PetscErrorCode ierr = 0;
        ierr = VecDuplicate (p, &g);            CHKERRQ (ierr);
        ierr = VecDuplicate (p, &x_full);       CHKERRQ (ierr);
        ierr = VecDuplicate (p, &x_full_prev);  CHKERRQ (ierr);
        ierr = VecDuplicate (p, &work);         CHKERRQ (ierr);
        ierr = VecSet       (g, 0.0);           CHKERRQ (ierr);
        ierr = VecSet       (x_full_prev, 0.0); CHKERRQ (ierr);
        ierr = VecSet       (work, 0.0);        CHKERRQ (ierr);
        ierr = VecCopy      (p, x_full);        CHKERRQ (ierr);
        initialized = true;
        PetscFunctionReturn(ierr);
    }

    PetscErrorCode cleanup() {
        PetscFunctionBegin;
        PetscErrorCode ierr = 0;
        if(initialized) {
            if (g != nullptr)           { VecDestroy (&g);           g           = nullptr;}
            if (x_full != nullptr)      { VecDestroy (&x_full);      x_full      = nullptr;}
            if (x_full_prev != nullptr) { VecDestroy (&x_full_prev); x_full_prev = nullptr;}
            if (work != nullptr)        { VecDestroy (&work);        work        = nullptr;}
        }
        initialized = false;
        PetscFunctionReturn(ierr);
    }

    ~CtxCoSaMp () {
        if (initialized) {cleanup();}
    }
};

class SparseTILOptimizer : public Optimizer {
public :
  SparseTILOptimizer()
  : Optimizer() {
    cosamp_ = std::make_shared<CtxCoSaMp>();
    til_opt_ = std::make_shared<TILOptimizer>();
    rd_opt_ = std::make_shared<RDOptimizer>();
  }

  virtual PetscErrorCode initialize (
            std::shared_ptr <DerivativeOperators> derivative_operators,
            std::shared_ptr <PdeOperators> pde_operators,
            std::shared_ptr <Parameters> params,
            std::shared_ptr <Tumor> tumor);

  virtual PetscErrorCode allocateTaoObjects();
  virtual PetscErrorCode setTaoOptions (Tao tao, CtxInv* ctx);
  virtual PetscErrorCode reset(Vec p);
  virtual PetscErrorCode solve();

  virtual ~SparseTILOptimizerstd(); // TODO(K) implement destructor

private:
  // local methods
  PetscErrorCode restrictSubspace (Vec *x_restricted, Vec x_full, std::shared_ptr<CtxInv> itctx, bool create_rho_dofs);
  PetscErrorCode prolongateSubspace (Vec x_full, Vec *x_restricted, std::shared_ptr<CtxInv> itctx, int np_full, bool reset_operators);
  PetscErrorCode cosampMonitor(int its, PetscReal J, PetscReal J_rel, PetscReal g_norm, PetscReal p_rel_norm, Vec x_L1);

  std::shared_ptr<CtxCoSaMp> cosamp_;   // cosamp soler context
  std::shared_ptr<Optimizer> til_opt_;  // TIL optimizer
  std::shared_ptr<Optimizer> rd_opt_;   // RD optimizer
};
#endif
