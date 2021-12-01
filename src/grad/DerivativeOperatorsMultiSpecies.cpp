
#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>




PetscErrorCode DerivativeOperatorMultiSpecies::reset(Vec p, std::shared_ptr<PdeOperators>, pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor)
 {

  PetscFunctionBegin; 
  PetscErrorCode ierr = 0;
  
  // delete and re-create p vectors
  if (ptemp_ != nullptr) {
    ierr = VecDestroy(&ptemp_); CHKERRQ(ierr);
    ptemp_ = nullptr;
  }
  ierr = VecDuplicate(delta_, &ptemp_); CHKERRQ(ierr);
  if (temp_ != nullptr) {
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
  }
  
  pde_operators_ = pde_operators;
  tumor_ = tumor;
  params_ = params;
  PetscFunctionReturn(ierr);
  
}


PetscErrorCode DerivativeOperatorsMultiSpecies::computeMisfitBrain(PetscRreal *J) { 

  PetscFunctionBegign;
  PetscErrorCode ierr = 0;
  
  PetscReal ms = 0;
  *J = 0;

  ierr = VecCopyy(tumor_->mat_prop_->vt_, temp_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, vt_); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &ms); CHKERR(ierr);
  *J += ms;

  PetscFunctionReturn(ierr);

}


PetscErrorCode DerivativeOperatorsMultiSpecies::evaluateObjective(PetscReal *J, ScalarType *x_ptr, std::shared_ptr<Data> data_inv) {

  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  const ScalarType *x_ptr;
  
  Vec data = data_inv->dt1;
   
 
  std::stringstream s;
  ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
  if (params_->opt_->invert_mass_effect_){
  
    // x[0] : forcing factor for the masseffect
    // x[1] : rho 
    // x[2] : kappa
    // x[3] : ox_hypoxia
    // x[4] : death_rate (deathrate)
    // x[5] : alpha_0 (trans from p to i
    // x[6] : ox_consumption 
    // x[7] : ox_source
    // x[8] : beta_0 (trans from i to p)
  
    params_->tu_->forcing_factor_ = x_ptr[0];
    params_->tu_->rho_ = x_ptr[1];
    params_->tu_->k_ = x_ptr[2];
    params_->tu_->ox_hypoxia_ = x_ptr[3];
    params_->tu_->death_rate_ = x_ptr[4];
    params_->tu_->alpha_0_ = x_ptr[5];
    params_->tu_->ox_consumption_ = x_ptr[6];
    params_->tu_->ox_source_ = x_ptr[7];
    params_->tu_->beta_0_ = x_ptr[8];
  } else {
    params_->tu_->rho_ = x_ptr[0];
    params_->tu_->k_ = x_ptr[1];
    params_->tu_->ox_hypoxia_ = x_ptr[2];
    params_->tu_->death_rate_ = x_ptr[3];
    params_->tu_->alpha_0_ = x_ptr[4];
    params_->tu_->ox_consumption_ = x_ptr[5];
    params_->tu_->ox_source_ = x_ptr[6];
    params_->tu_->beta_0_ = x_ptr[7];
  }

  ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);


  if (!disable_verbose_){
    s << " Forcing factor at current guess = " << params_->tu_->forcing_factor_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Reaction at current guess       = " << params_->tu_->rho_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Diffusivity at current guess    = " << params_->tu_->k_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Hypoxia threshold at current guess    = " << params_->tu_->ox_hypoxia_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Deathrate at current guess    = " << params_->tu_->death_rate_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Transition rate from p to i at current guess    = " << params_->tu_->alpha_0_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Oxygen consumption at current guess    = " << params_->tu_->ox_consumption_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Oxygen source at current guess    = " << params_->tu_->ox_source_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Transition rate from i to p at current guess    = " << params_->tu_->beta_0_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  // Rest mat-props and parameters, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->velocity_->set(0);
  ierr = tumor_->displacement_->set(0);
  
  ierr = pde_operators_->solveState(0);
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, data); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, J); CHKERRQ(ierr);
  (*J) *= 0.5 * params_->grid_->lebesgue_measure_;
  ierr = computeMisfitBrain (&misfit_brain);
  misfit_brain *= 0.5 * params_->grid_->lebesgue_measure_;
  if (!disable_verbose_) {
    s << "J = misfit_tu + misfit_brain = " << std::setprecision(12) << *J << " + " << misfit_brain << " = " << (*J) + misfit_brain;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }
  
  (*J) += misfit_brain;

  PetscFunctionReturn(ierr);

}

















}
