
#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Parameters.h"
#include "Tumor.h"
#include "MatProp.h"
#include "PdeOperators.h"
#include "DerivativeOperators.h"
#include "SpectralOperators.h"
#include "InvSolver.h"



struct DataDistributionParameters {
    int64_t alloc_max;
    int nlocal;
    int nglobal;
    int n[3];
    int cdims[2];
    int istart[3];
    int isize[3];
    int ostart[3];
    int osize[3];
    fft_plan* plan;
    MPI_Comm comm;
    int fft_mode;

    DataDistributionParameters ()
    :
      alloc_max(0)
    , nlocal(0)
    , nglobal(0)
    , cdims{0,0}
    , n{256,256,256}
    , istart{0,0,0}
    , isize{0,0,0}
    , ostart{0,0,0}
    , osize{0,0,0}
    , plan(nullptr)
    , comm (MPI_COMM_WORLD)
    , fft_mode(ACCFFT)
    {}
};


class TumorSolverInterface {

public :

  /** @brief: Creates the TumorSolverInterface. If given, initializes nmisc, phi, matprob.
  *
  *  @param NMisc[in] nmisc     - struct with all tumor parameters
  *  @param Phi[in] phi         - set of basis functions
  *  @param MatProb[in] matprob - material properties, i.e., WM, GM, CSF
  */
  TumorSolverInterface (
    std::shared_ptr<Parameters> params = {},
    std::shared_ptr<SpectralOperators> spec_ops = {},
    std::shared_ptr<Phi> phi = {},
    std::shared_ptr<MatProp> mat_prop = {});


  ~TumorSolverInterface () {}

  /** @brief: Destroys accfft related objects.
  */
  PetscErrorCode finalize(DataDistributionParameters& ivars);

  /** @brief: Initializes accFFT, communication plan, data distribution, cartesian communicator, etc.
  *
  *  @param[out] DataDistributionParameters ivars - struct with all relevant parameters defining data distribution and communication (will be set in this function)
  */
  PetscErrorCode initializeFFT(DataDistributionParameters& ivars);

  /** @brief: Initializes the TumorSolverInterface. Creates nmisc, phi, pde_operators, mat_prob, tumor.
  *
  *  @param[in] DataDistributionParameters ivars - struct with all relevant parameters defining data distribution and communication
  */
  PetscErrorCode initialize (
      DataDistributionParameters& ivars,
      std::shared_ptr<TumorParameters> tumor_params = {});

  /** @brief: Initializes the TumorSolverInterface. Creates nmisc, phi, pde_operators, mat_prob, tumor.
  *
  *  @param NMisc[in] nmisc     - struct with all tumor parameters
  *  @param Phi[in] phi         - set of basis functions
  *  @param MatProb[in] matprob - material properties, i.e., WM, GM, CSF
  */
  PetscErrorCode initialize (
      std::shared_ptr<Parameters> params,
      std::shared_ptr<SpectralOperators> spec_ops = {},
      std::shared_ptr<Phi> phi = {},
      std::shared_ptr<MatProp> mat_prop = {});

  /** @brief: Sets tumor parameters; possibly re-creates operators, vectors and matrices if np, nt or model changed
  *
  *  @param Vec p                   - vector to initialize tumor
  *  @param TumorParameters tuparams  - struct with all relevant tumor parameters
  */
  PetscErrorCode setParams (
      Vec p,
      std::shared_ptr<TumorParameters> tumor_params);

  PetscErrorCode setParams (
      std::shared_ptr<TumorParameters> tumor_params);

  /** @brief: Solves the forward tumor problem, given initial concentration
  *         and tumor parameters
  *  @param Vec c0  - initial tumor concentration
  *  @param Vec cT  - target tumor concentration after simulation
  */
  PetscErrorCode solveForward (Vec c1, Vec c0);
  PetscErrorCode solveForward (
      Vec c1, Vec c0,
      std::map<std::string, Vec> *species);

  PetscErrorCode solveInverseMassEffect (ScalarType *xrec, Vec data, Vec data_gradeval = {});

  /** @brief: Solves the inverse tumor problem using Tao, given target concentration
   *
   *  @param Vec d1     - tumor inverse target data
   *  @param Vec p_rec, - reconstructed parameters for initial condition  c_rec = \Phi p_rec
   */
  PetscErrorCode solveInverse (
      Vec prec,
      Vec d1, Vec d1g = {});

  /** @brief: Solves only for rho and k, given a (scaled betwenn [0,1]) c(0) initial condition
   *
   *  @param Vec p_rec, - reconstructed parameters for initial condition  c_rec = \Phi p_rec
   *  @param Vec d1     - tumor inverse target data
   */
  PetscErrorCode solveInverseReacDiff(
      Vec prec,
      Vec d1, Vec d1g = {});

  /** @brief: Solves the L1 optimization problem using compressive sampling methods
   *
   *  @param Vec p_rec, - reconstructed parameters for initial condition  c_rec = \Phi p_rec
   *  @param Vec d1     - tumor inverse target data
   */
  PetscErrorCode solveInverseCoSaMp (
      Vec prec,
      Vec d1, Vec d1g = {});


  PetscErrorCode resetTaoSolver();

  /** @brief: updates the reaction and diffusion coefficients depending on
   *         the probability maps for healthy tissue
   */
  PetscErrorCode updateTumorCoefficients (Vec wm, Vec gm, Vec glm, Vec csf, Vec bg);

  /// @brief: evaluates gradient for given control variable p and data
  PetscErrorCode computeGradient(Vec dJ, Vec p, Vec data_gradeval);

  /** @brief: computes effect of varying/moving material properties, i.e.,
   *  computes q = int_T dK / dm * (grad c)^T grad * \alpha + dRho / dm c(1-c) * \alpha dt
   */
  PetscErrorCode setMassEffectData(Vec gm, Vec wm, Vec csf, Vec glm);

  PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3);

  //  ---------  getter functions -------------
  bool isInitialized()     {return initialized_;}
  int getNumberGaussians() {return n_misc_->np_;}
  std::shared_ptr<Tumor> getTumor()  {return tumor_;}
  std::shared_ptr<CtxInv> getITctx() {return inv_solver_->getInverseSolverContext();}     // TODO(K): remove?
  std::shared_ptr<OptimizerFeedback> getOptFeedback() {return inv_solver_->optfeedback_;} // TODO(K): remove?
  std::shared_ptr<OptimizerSettings> getOptSettings() {return inv_solver_->optsettings_;} // TODO(K): remove?
  std::shared_ptr<InvSolver> getInvSolver() {return inv_solver_;}
  std::shared_ptr<PdeOperators> getPdeOperators() {return pde_operators_;}
  std::shared_ptr<DerivativeOperators> getDerivativeOperators() {return derivative_operators_;}
  // std::vector<ScalarType> getSolverOutParams()  {return out_params_ = inv_solver_->getInvOutParams (); }

  // setter
  PetscErrorCode setOptimizerFeedback (std::shared_ptr<OptimizerFeedback> optfeed);
  PetscErrorCode setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset);
  PetscErrorCode setInitialGuess(Vec p);
  PetscErrorCode setInitialGuess(ScalarType d);
  PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec bg);
  PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec bg);
  PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec bg);
  PetscErrorCode setGaussians (Vec data);
  PetscErrorCode setGaussians (ScalarType* cm, ScalarType sigma, ScalarType spacing, int np);
  PetscErrorCode setGaussians (std::array<ScalarType, 3> cm, ScalarType sigma, ScalarType spacing, int np);
  PetscErrorCode applyPhi (Vec phi_p, Vec p);

  // timers
  PetscErrorCode initializeEvent();
  PetscErrorCode finalizeEvent();

  // helpers for cython
  PetscErrorCode readNetCDF (Vec A, std::string filename);
  PetscErrorCode writeNetCDF (Vec A, std::string filename);
  PetscErrorCode smooth (Vec x, ScalarType num_voxels);


private :
  bool initializedFFT_;
  bool initialized_;
  bool optimizer_settings_changed_;
  bool regularization_norm_changed_;
  bool newton_solver_type_changed_;
  std::shared_ptr<Parameters> params_;
  std::shared_ptr<SpectralOperators> spec_ops_;
  std::shared_ptr<DerivativeOperators> derivative_operators_;
  std::shared_ptr<Tumor> tumor_;
  std::shared_ptr<InvSolver> inv_solver_;
  std::shared_ptr<PdeOperators> pde_operators_;
  // std::vector<ScalarType> out_params_;
};

//} // namespace pglistr

#endif
