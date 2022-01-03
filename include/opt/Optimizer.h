
#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <memory>
#include "DerivativeOperators.h"
#include "Parameters.h"
#include "TaoInterface.h"


struct CtxCoSaMp;

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
    std::shared_ptr<Data> data; // data for tumor inversion
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
    std::shared_ptr<CtxCoSaMp> cosamp_;
		
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
    , derivative_operators_()
    , params_()
    , tumor_()
    , convergence_message()
    , cosamp_(nullptr)
    {}

    ~CtxInv () {
        if (x_old  != nullptr) {VecDestroy(&x_old); x_old = nullptr;}
        if (c0_old != nullptr) {VecDestroy(&c0_old); c0_old = nullptr;}
        if (tmp    != nullptr) {VecDestroy(&tmp); tmp = nullptr;}
    }
};

struct CtxInvCMA {

public:
    ScalarType jvalold;                  // old value of the objective function
    ScalarType cma_sigma;                // variance for the cma optimizer

    Vec tmp;
    Vec c0_old;                          // previous solution
    Vec x_old;                           // previous test solution
    std::shared_ptr<Data> data;           // data for tumor inversion
    /// @brief evalJ evalDJ, eval D2J
    std::shared_ptr<DerivativeOperators> derivative_operators_;
    /// @brief required to reset derivative_operators_
    std::shared_ptr<PdeOperators> pde_operators_;
    /// @brief common settings/ parameters
    std::shared_ptr<Parameters> params_;
    /// @brief accumulates all tumor related fields and methods
    std::shared_ptr<Tumor> tumor_;
    CtxInvCMA()
    :
      jvalold(0)
    , cma_sigma(1.0)
    , tmp(nullptr)
    , c0_old(nullptr)
    , x_old(nullptr)
    , data(nullptr)
    , derivative_operators_()
    , params_()
    , tumor_()
    {}

    ~CtxInvCMA () {
        if (x_old != nullptr) {VecDestroy(&x_old); x_old = nullptr;}
        if (c0_old != nullptr) {VecDestroy(&c0_old); c0_old = nullptr;}
        if (tmp != nullptr) {VecDestroy(&tmp); tmp = nullptr;}
    }
};


/* #### ------------------------------------------------------------------- #### */
/* #### ========              CMA Optimizer Class                  ======== #### */
/* #### ------------------------------------------------------------------- #### */
class CMAOptimizer {
public :

  CMAOptimizer()
  :
    initialized_(false)
  , n_inv_(0)
  , data_(nullptr)
  , cma_ctx_(nullptr)
  , xrec_(nullptr)
  , xin_(nullptr)
  , xout_(nullptr)
 {}

  virtual PetscErrorCode initialize (
            std::shared_ptr <DerivativeOperators> derivative_operators,
            std::shared_ptr <PdeOperators> pde_operators,
            std::shared_ptr <Parameters> params,
            std::shared_ptr <Tumor> tumor);

  virtual PetscErrorCode allocateObjects();
  virtual PetscErrorCode solve() = 0;
  //virtual PetscErrorCode resetOperators(Vec p);
  virtual PetscErrorCode setInitialGuess(Vec x_init) = 0;
  //virtual PetscErrorCode setVariableBounds() = 0;
  
  void setData(std::shared_ptr<Data> d) {data_ = d;}
  void setData(Vec d1, Vec d0={}) {data_->set(d1, d0);}
  void setDataT1(Vec d1) {data_->setT1(d1);}
  void setDataT0(Vec d0) {data_->setT1(d0);}
  
  bool initialized() {return initialized_;}
  Vec getSolution() {return xout_;}

  virtual ~CMAOptimizer();
  inline PetscErrorCode evalObjective (Vec x, PetscReal *J) {
      PetscFunctionBegin; PetscErrorCode ierr = 0;
      ierr = cma_ctx_->derivative_operators_->evaluateObjective (J, x, data_);
      PetscFunctionReturn(0);
  }


  std::shared_ptr<CtxInvCMA> cma_ctx_;
protected:
  bool initialized_;
  
  int n_inv_;
  std::shared_ptr<Data> data_;
  Vec xrec_;
  Vec xin_;
  Vec xout_;
};



/* #### ------------------------------------------------------------------- #### */
/* #### ========                Optimizer Class                    ======== #### */
/* #### ------------------------------------------------------------------- #### */
class Optimizer {
public :

  Optimizer()
  :
    initialized_(false)
  , tao_reset_(true)
  , n_inv_(0)
  , data_(nullptr)
  , ctx_(nullptr)
  , tao_(nullptr)
  , xrec_(nullptr)
  , xin_(nullptr)
  , xout_(nullptr)
  , H_(nullptr) 
 {}

  virtual PetscErrorCode initialize (
            std::shared_ptr <DerivativeOperators> derivative_operators,
            std::shared_ptr <PdeOperators> pde_operators,
            std::shared_ptr <Parameters> params,
            std::shared_ptr <Tumor> tumor);

  virtual PetscErrorCode solve() = 0;

  virtual PetscErrorCode allocateTaoObjects();
  virtual PetscErrorCode setTaoOptions();
  virtual PetscErrorCode resetOperators(Vec p);
  virtual PetscErrorCode resetTao();

  virtual PetscErrorCode setInitialGuess(Vec x_init) = 0;
  virtual PetscErrorCode setVariableBounds() = 0;

  void setData(std::shared_ptr<Data> d) {data_ = d;}
  void setData(Vec d1, Vec d0={}) {data_->set(d1, d0);}
  void setDataT1(Vec d1) {data_->setT1(d1);}
  void setDataT0(Vec d0) {data_->setT1(d0);}

  void updateReferenceGradient(bool b) {ctx_->update_reference_gradient = b;}
  void updateReferenceObjective(bool b) {ctx_->update_reference_objective = b;}
  bool initialized() {return initialized_;}
  Vec getSolution() {return xout_;}

  // TODO(K) implement destructor, need to destroy vecs
  virtual ~Optimizer();

  inline PetscErrorCode evalGradient (Vec x, Vec dJ) {
      PetscFunctionBegin; PetscErrorCode ierr = 0;
      ierr = ctx_->derivative_operators_->evaluateGradient (dJ, x, data_);
      PetscFunctionReturn(0);
  }
  inline PetscErrorCode evalObjective (Vec x, PetscReal *J) {
      PetscFunctionBegin; PetscErrorCode ierr = 0;
      ierr = ctx_->derivative_operators_->evaluateObjective (J, x, data_);
      PetscFunctionReturn(0);
  }
  inline PetscErrorCode evalObjectiveAndGradient (Vec x, PetscReal *J, Vec dJ) {
      PetscFunctionBegin; PetscErrorCode ierr = 0;
      ierr = ctx_->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, data_);
      PetscFunctionReturn(0);
  }

  std::shared_ptr<CtxInv> ctx_;

protected:
  bool initialized_;
  bool tao_reset_;   // TODO(K) at the end: check if needed

  int n_inv_;
  std::shared_ptr<Data> data_;
  Vec xrec_;
  Vec xin_;
  Vec xout_;

  Tao tao_;
  Mat H_;
};

// // === non-class methods
// PetscErrorCode operatorCreateVecs(Mat A, Vec *left, Vec *right);

#endif
