
#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <memory>
#include "DerivativeOperators.h"
#include "Parameters.h"



/* #### ------------------------------------------------------------------- #### */
/* #### ========                Optimizer Context                  ======== #### */
/* #### ------------------------------------------------------------------- #### */
struct CtxInv {
public:
    ScalarType jvalold;                 // old value of objective function (previous newton iteration)
    ScalarType last_ls_step;            // remeber line-search step of previous solve
    ScalarType step_init;                // init step length for line search
    ScalarType ksp_gradnorm0;            // reference gradient for hessian PCG
    bool update_reference_gradient_hessian_ksp;
    bool update_reference_gradient;
    bool update_reference_objective;

    Vec tmp;
    Vec c0_old;    // previous initial condition \Phi p^k-1
    Vec x_old;     // previous solution
    Vec data;      // data for tumor inversion

    std::vector<std::string> convergence_message; // convergence message

    /// @brief evalJ evalDJ, eval D2J
    std::shared_ptr<DerivativeOperators> derivative_operators_;
    /// @brief required to reset derivative_operators_
    std::shared_ptr<PdeOperators> pde_operators_;
    /// @brief common settings/ parameters
    std::shared_ptr<Parameters> params_;
    /// @brief accumulates all tumor related fields and methods
    std::shared_ptr<Tumor> tumor_;
    /// @brief context for CoSaMp L1 solver
    void* optctx_;
    // std::shared_ptr<CtxCoSaMp> cosamp_; // TODO(K) make void ptr data, only used for sparseTILOptimizer

    CtxInv ()
    :
      jvalold(0)
    , last_ls_step(1.0)
    , step_init(1.0)
    , ksp_gradnorm0(1)
    , update_reference_gradient_hessian_ksp(true)
    , update_reference_gradient(true)
    , update_reference_objective(true)
    , tmp(nullptr)
    , c0_old(nullptr)
    , x_old(nullptr)
    , data(nullptr)
    , derivative_operators_ ()
    , params_ ()
    , tumor_ ()
    , data (nullptr)
    , convergence_message ()
    , optctx_(nullptr)
    {}

    ~CtxInv () {
        if (x_old != nullptr) { VecDestroy (&x_old); x_old = nullptr;}
        if (c0_old != nullptr) { VecDestroy (&c0_old); c0_old = nullptr;}
        if (tmp != nullptr) { VecDestroy (&tmp); tmp = nullptr;}
    }
};




/* #### ------------------------------------------------------------------- #### */
/* #### ========                Optimizer Class                    ======== #### */
/* #### ------------------------------------------------------------------- #### */
class Optimizer {
public :

  Optimizer()
  :
    initialized_(false),
  , tao_reset_(true)
  , data_(nullptr)
  , ctx_(nullptr)
  , tao_(nullptr)
  , xrec_(nullptr)
  , H_(nullptr) {}

  virtual PetscErrorCode initialize (
            std::shared_ptr <DerivativeOperators> derivative_operators,
            std::shared_ptr <PdeOperators> pde_operators,
            std::shared_ptr <Parameters> params,
            std::shared_ptr <Tumor> tumor);

  //TODO(K) implement, should contain allocateTaoObjectsMassEffect
  virtual PetscErrorCode allocateTaoObjects();

  // TODO(K) implement; should contain setTaoOptionsMassEffect
  virtual PetscErrorCode setTaoOptions (Tao tao, CtxInv* ctx);
  virtual PetscErrorCode reset(Vec p);

  // TODO(K) implement
  virtual PetscErrorCode solve();
  // PetscErrorCode resetTao(std::shared_ptr<Parameters> params); //TODO(K) do we need reset TAO?
  // PetscErrorCode resetOperators (Vec p); // TODO(K) do we need this? I think cosamp needs it
  // PetscErrorCode setParams (             // TODO(K) do we need this, I think no!
            // std::shared_ptr<DerivativeOperators> derivative_operators,
            // std::shared_ptr <PdeOperators> pde_operators,
            // std::shared_ptr<Parameters> params,
            // std::shared_ptr<Tumor> tumor, bool npchanged = false);

  // setter
  void setData (Vec d) {data_ = d;}
  initialized() {return initialized_;}

  // TODO(K) implement destructor
  virtual ~Optimizer();

  // inline PetscErrorCode evaluateGradient (Vec x, Vec dJ) {
  //     PetscFunctionBegin; PetscErrorCode ierr = 0;
  //     ierr = ctx_->derivative_operators_->evaluateGradient (dJ, x, data_);
  //     PetscFunctionReturn(0);
  // }
  // inline PetscErrorCode evaluateObjective (Vec x, PetscReal *J) {
  //     PetscFunctionBegin; PetscErrorCode ierr = 0;
  //     ierr = ctx_->derivative_operators_->evaluateObjective (J, x, data_);
  //     PetscFunctionReturn(0);
  // }
  // inline PetscErrorCode evaluateObjectiveAndGradient (Vec x, PetscReal *J, Vec dJ) {
  //     PetscFunctionBegin; PetscErrorCode ierr = 0;
  //     ierr = ctx_->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, data_);
  //     PetscFunctionReturn(0);
  // }

protected:
  std::shared_ptr<CtxInv> ctx_;
  bool initialized_;
  bool tao_reset_;   // TODO(K) at the end: check if needed

  int n_inv;
  Vec data_;            // TODO(K) at the end: check if needed
  Vec xrec_;
  // Vec xrec_rd_;      // TODO(K) we only need one
  Tao tao_;
  Mat H_;               // TODO(K) at the end: check if needed
};

#endif
