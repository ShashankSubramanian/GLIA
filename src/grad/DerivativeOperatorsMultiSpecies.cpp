
#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>


/*
PetscErrorCode DerivativeOperatorsMultiSpecies::reset(Vec p, std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor)
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
*/


PetscErrorCode DerivativeOperatorsMultiSpecies::computeMisfitBrain(PetscReal *J) { 

  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  
  PetscReal ms = 0;
  *J = 0;

  ierr = VecCopy(tumor_->mat_prop_->vt_, temp_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, vt_); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &ms); CHKERRQ(ierr);
  *J += ms;

  PetscFunctionReturn(ierr);

}


PetscErrorCode DerivativeOperatorsMultiSpecies::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data_inv) {

  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  const ScalarType *x_ptr;
  
  Vec data = data_inv->dt1();
   
 
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
  PetscReal misfit_brain = 0.;
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


PetscErrorCode DerivativeOperatorsMultiSpecies::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_grad_evals++;

  Vec data = data_inv->dt1();
  disable_verbose_ = true;
  // Finite difference gradient -- forward for now
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f, J_b;

  ierr = evaluateObjective(&J_b, x, data_inv); CHKERRQ(ierr);
  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ScalarType const *x_ptr;
  ierr = VecGetSize(x, &sz); CHKERRQ(ierr);
  ierr = VecGetArray(dJ, &dj_ptr); CHKERRQ(ierr);
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  ScalarType small = 1E-5;//PETSC_SQRT_MACHINE_EPSILON;
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy(x, delta_); CHKERRQ(ierr);
    ierr = VecGetArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = evaluateObjective(&J_f, delta_, data_inv); CHKERRQ(ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(dJ, &dj_ptr); CHKERRQ(ierr);

  disable_verbose_ = false;

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsMultiSpecies::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;

  Vec data = data_inv->dt1();
  ierr = evaluateObjective(J, x, data_inv); CHKERRQ(ierr);
  // Finite difference gradient -- forward for now
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f;

  disable_verbose_ = true;
  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ScalarType const *x_ptr;
  ierr = VecGetSize(x, &sz); CHKERRQ(ierr);
  ierr = VecGetArray(dJ, &dj_ptr); CHKERRQ(ierr);

  ScalarType scale = 1;
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};


  ScalarType J_b = (*J);
  ScalarType small = 1E-5;
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy(x, delta_); CHKERRQ(ierr);
    ierr = VecGetArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = evaluateObjective(&J_f, delta_, data_inv); CHKERRQ(ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(dJ, &dj_ptr); CHKERRQ(ierr);

  disable_verbose_ = false;

  PetscFunctionReturn(ierr);
}



PetscErrorCode DerivativeOperatorsMultiSpecies::evaluateHessian(Vec y, Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_hessian_evals++;

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  Event e("tumor-eval-hessian");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ierr = VecCopy(x, y); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}





  



