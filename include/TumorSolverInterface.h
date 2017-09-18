#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"

class TumorSolverInterface {
	public :

	  // TODO: (pointer to) NMisc should also be a member of the class, can be passed
	  // from outside to constructor and/or initialize function, then stored inside the
	  // TumorSolverInterface class such that it doesn't need to be passed again when
	  // calling mehtods like solveForward()
		TumorSolverInterface (std::shared_ptr<NMisc> n_misc = {})
		:
		n_misc_(nmisc),
		tumor_(),
		pde_operators_
		{}

    // TODO: do all the initialization of TumorSolverInterface here, only absolutely
		// crutial initialization in the constructor
		// e.g.: call tumor_->initialize()
		//       call pde_operators_->initialize()
		//       etc.
		//       see SIBIA: src/coupling/TumorAdapter.cpp
    PetscErrorCode initialize (std::shared_ptr<NMisc> n_misc);

		// TODO: call forward tumor solver to produce c1 = T^fwd(c0)
		//       see SIBIA: src/coupling/TumorAdapter.cpp
		PetscErrorCode solveForward (std::shared_ptr<Image> c0, std::shared_ptr<Image> c1);
		// TODO: call tao inverse solver to get p_rec = T^inv(d1)
		//       see SIBIA: src/coupling/TumorAdapter.cpp
		PetscErrorCode solveInverse (std::shared_ptr<Image> d1, Vec& p_rec);

    // TODO: re-set values for diffusion coefficient, reaction coefficient, gaussian-filter,
		//       tumor parameters, see SIBIA: src/coupling/TumorAdapter.cpp
		PetscErrorCode updateTumorCoefficients(std::shared_ptr<MatProb> matprob, std::shared_ptr<TumorParameter> g = {});

    // TODO: set initial guess for tao tumor inversion
		//       see SIBIA: src/coupling/TumorAdapter.cpp
	  PetscErrorCode setInitialGuess(Vec p);

		~TumorSolverInterface ();
	private :

		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<PdeOperators> pde_operators_;
};

#endif
