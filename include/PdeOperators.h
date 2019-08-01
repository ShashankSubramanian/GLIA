#ifndef PDEOPERATORS_H_
#define PDEOPERATORS_H_

#include "Utils.h"
#include "Tumor.h"
#include "DiffSolver.h"
#include "AdvectionSolver.h"
#include "ElasticitySolver.h"
#include "SpectralOperators.h"

class PdeOperators {
	public:
		PdeOperators (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) : tumor_(tumor), n_misc_(n_misc), spec_ops_(spec_ops) {
			diff_solver_ = std::make_shared<DiffSolver> (n_misc, spec_ops, tumor->k_);
			nt_ = n_misc->nt_;
			diff_ksp_itr_state_ = 0;
			diff_ksp_itr_adj_ = 0;
		}

		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<DiffSolver> diff_solver_;
		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<SpectralOperators> spec_ops_;

		// @brief time history of state variable
		std::vector<Vec> c_;
		// @brief time history of adjoint variable
		std::vector<Vec> p_;

		// Accumulated number of KSP solves for diff solver in one forward and adj solve
		int diff_ksp_itr_state_, diff_ksp_itr_adj_;


		virtual PetscErrorCode solveState (int linearized) = 0;
		virtual PetscErrorCode solveAdjoint (int linearized) = 0;
		virtual PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4) = 0;
		virtual PetscErrorCode resizeTimeHistory (std::shared_ptr<NMisc> n_misc) = 0;

		virtual ~PdeOperators () {}


	protected:
			/// @brief local copy of nt, bc if parameters change, pdeOperators needs to
			/// be re-constructed. However, the destructor has to use the nt value that
			/// was used upon construction of that object, not the changed value in nmisc
			int nt_;
};

class PdeOperatorsRD : public PdeOperators {
	public:
		PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops);

		virtual PetscErrorCode solveState (int linearized);
		virtual PetscErrorCode reaction (int linearized, int i);
		virtual PetscErrorCode reactionAdjoint (int linearized, int i);
		virtual PetscErrorCode solveAdjoint (int linearized);
		virtual PetscErrorCode resizeTimeHistory (std::shared_ptr<NMisc> n_misc);

		/** @brief computes effect of varying/moving material properties, i.e.,
		 *  computes q = int_T dK / dm * (grad c)^T grad * \alpha + dRho / dm c(1-c) * \alpha dt
		 */
		virtual PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4);
		virtual ~PdeOperatorsRD ();
};

class PdeOperatorsMassEffect : public PdeOperatorsRD {
	public:
		PdeOperatorsMassEffect (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) : PdeOperatorsRD (tumor, n_misc, spec_ops) {
			PetscErrorCode ierr = 0;
			adv_solver_ = std::make_shared<SemiLagrangianSolver> (n_misc, tumor, spec_ops);
			// adv_solver_ = std::make_shared<TrapezoidalSolver> (n_misc, tumor);
			elasticity_solver_ = std::make_shared<VariableLinearElasticitySolver> (n_misc, tumor, spec_ops);

			temp_ = new Vec[3];
			for (int i = 0; i <3; i++) {
				ierr = VecDuplicate (tumor->work_[0], &temp_[i]);
				ierr = VecSet (temp_[i], 0.);
			}
		}

		std::shared_ptr<AdvectionSolver> adv_solver_;
		std::shared_ptr<ElasticitySolver> elasticity_solver_;

		Vec *temp_;

		virtual PetscErrorCode solveState (int linearized);
		PetscErrorCode conserveHealthyTissues ();

		virtual ~PdeOperatorsMassEffect () {
			PetscErrorCode ierr = 0;
			for (int i = 0; i < 3; i++)
				ierr = VecDestroy (&temp_[i]);
			delete [] temp_;
		}
};

#endif
