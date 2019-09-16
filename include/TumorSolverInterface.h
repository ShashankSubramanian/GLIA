/**
 *  SIBIA (Scalable Biophysics-Based Image Analysis)
 *
 *  Copyright (C) 2017-2020, The University of Texas at Austin
 *  This file is part of the SIBIA library.
 *
 *  SIBIA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SIBIA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/

#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "MatProp.h"
#include "PdeOperators.h"
#include "DerivativeOperators.h"
#include "SpectralOperators.h"
#include "InvSolver.h"

//namespace pglistr {

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
        int testcase;
        accfft_plan* plan;
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
        , testcase(0)
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
        std::shared_ptr<NMisc> n_misc = {},
        std::shared_ptr<SpectralOperators> spec_ops = {},
        std::shared_ptr<Phi> phi = {},
        std::shared_ptr<MatProp> mat_prop = {});

    /** @brief: Destructor
    */
    ~TumorSolverInterface () {}

    /** @brief: Destroys accfft related objects.
    */
    PetscErrorCode finalize (
        DataDistributionParameters& ivars);

    /** @brief: Initializes accFFT, communication plan, data distribution, cartesian communicator, etc.
    *
    *  @param[out] DataDistributionParameters ivars - struct with all relevant parameters defining data distribution and communication (will be set in this function)
    */
    PetscErrorCode initializeFFT (
        DataDistributionParameters& ivars);

    /** @brief: Initializes the TumorSolverInterface. Creates nmisc, phi, pde_operators, mat_prob, tumor.
    *
    *  @param[in] DataDistributionParameters ivars - struct with all relevant parameters defining data distribution and communication
    */
    PetscErrorCode initialize (
        DataDistributionParameters& ivars,
        std::shared_ptr<TumorSettings> tumor_params);

    /** @brief: Initializes the TumorSolverInterface. Creates nmisc, phi, pde_operators, mat_prob, tumor.
    *
    *  @param NMisc[in] nmisc     - struct with all tumor parameters
    *  @param Phi[in] phi         - set of basis functions
    *  @param MatProb[in] matprob - material properties, i.e., WM, GM, CSF
    */
    PetscErrorCode initialize (
        std::shared_ptr<NMisc> n_misc,
        std::shared_ptr<SpectralOperators> spec_ops = {},
        std::shared_ptr<Phi> phi = {},
        std::shared_ptr<MatProp> mat_prop = {});

    /** @brief: Sets tumor parameters; possibly re-creates operators, vectors and matrices if np, nt or model changed
    *
    *  @param Vec p                   - vector to initialize tumor
    *  @param TumorSettings tuparams  - struct with all relevant tumor parameters
    */
    PetscErrorCode setParams (
        Vec p,
        std::shared_ptr<TumorSettings> tumor_params);
    PetscErrorCode _setParams (
        std::shared_ptr<TumorSettings> tumor_params);

    /** @brief: Solves the forward tumor problem, given initial concentration
    *         and tumor parameters
    *  @param Vec c0  - initial tumor concentration
    *  @param Vec cT  - target tumor concentration after simulation
    */
    PetscErrorCode solveForward (Vec c1, Vec c0);
    PetscErrorCode solveForward (
        Vec c1, Vec c0,
        std::map<std::string, Vec> *species);

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

    /** @brief: Solves the projection problem \Phi p = d for p.
     *
     *  @param Vec data   - target data
     *  @param Vec p_rec, - reconstructed parameters p for projection  \Phi p = d
     *  @param phi        - set of basis functions
     *  @params n_misc    - tumor parameters
     */
    PetscErrorCode solveInterpolation (
        Vec data,
        Vec p_rec,
        std::shared_ptr<Phi> phi,
        std::shared_ptr<NMisc> n_misc);

    PetscErrorCode resetTaoSolver();

    /** @brief: updates the reaction and diffusion coefficients depending on
     *         the probability maps for GRAY MATTER, WHITE MATTER and CSF.
     *         A additional filter, that filters the admissable area for tumor
     *         growth has to be passed (updates the \Phi filter)
     */
    PetscErrorCode updateTumorCoefficients (
        Vec wm, Vec gm, Vec csf, Vec bg,
        Vec filter,
        std::shared_ptr<TumorSettings> tumor_params,
        bool use_nmisc = false);

    /// @brief: evaluates gradient for given control variable p and data
    PetscErrorCode computeGradient(Vec dJ, Vec p, Vec data_gradeval);

    // defines whether or not we have to update the reference gradeient for the inverse solve
    void updateReferenceGradient (bool b) {if (inv_solver_ != nullptr) inv_solver_->updateReferenceGradient(b);}

    /** @brief: computes effect of varying/moving material properties, i.e.,
     *  computes q = int_T dK / dm * (grad c)^T grad * \alpha + dRho / dm c(1-c) * \alpha dt
     */
    PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3);

    //  ---------  getter functions -------------
    bool isInitialized()     {return initialized_;}
    int getNumberGaussians() {return n_misc_->np_;}
    std::shared_ptr<Tumor> getTumor()  {return tumor_;}
    std::shared_ptr<CtxInv> getITctx() {return inv_solver_->getInverseSolverContext();}
    std::shared_ptr<OptimizerFeedback> getOptFeedback() {return inv_solver_->optfeedback_;}
    std::shared_ptr<OptimizerSettings> getOptSettings() {return inv_solver_->optsettings_;}
    std::shared_ptr<InvSolver> getInvSolver() {return inv_solver_;}
    std::shared_ptr<PdeOperators> getPdeOperators() {return pde_operators_;}
    std::vector<double> getSolverOutParams()  {return out_params_ = inv_solver_->getInvOutParams (); }

    // ---------- setter functions --------------
    PetscErrorCode setOptimizerFeedback (std::shared_ptr<OptimizerFeedback> optfeed);
    PetscErrorCode setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset);
    PetscErrorCode setInitialGuess(Vec p);
    PetscErrorCode setInitialGuess(ScalarType d);
    PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec bg);
    PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec bg);
    PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec bg);
    /// @brief: sets Gaussians adaptively based on data
    PetscErrorCode setGaussians (Vec data);
    /// @brief: sets Gaussians as bbox around center of mass
    PetscErrorCode setGaussians (double* cm, double sigma, double spacing, int np);
    PetscErrorCode setGaussians (std::array<double, 3> cm, double sigma, double spacing, int np);
    PetscErrorCode setTumorRegularizationNorm (int type);
    PetscErrorCode setTumorSolverType (int type);

    private :
    bool initializedFFT_;
    bool initialized_;
    bool optimizer_settings_changed_;
    bool regularization_norm_changed_;
    bool newton_solver_type_changed_;
    std::shared_ptr<NMisc> n_misc_;
    std::shared_ptr<SpectralOperators> spec_ops_;
    std::shared_ptr<Tumor> tumor_;
    std::shared_ptr<PdeOperators> pde_operators_;
    std::shared_ptr<DerivativeOperators> derivative_operators_;
    std::shared_ptr<InvSolver> inv_solver_;

    std::vector<double> out_params_;
};

//} // namespace pglistr

#endif
