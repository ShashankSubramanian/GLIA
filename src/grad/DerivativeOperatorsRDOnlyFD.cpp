#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>


/* #### --------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Finite Diff. {rho,kappa} for RD Model ======== #### */
/* #### --------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateObjective (PetscReal *J, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;

  PetscReal m1 = 0, m0 = 0, reg = 0;
  std::stringstream s;


  // Reset mat-props and diffusion and reaction operators, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues (); CHKERRQ (ierr);
  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

  // compute mismatch ||Oc(1) - d1||
  ierr = pde_operators_->solveState(0); CHKERRQ (ierr);               // c(1)
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ (ierr); // Oc(1)
  ierr = VecAXPY(temp_, -1.0, data->dt1()); CHKERRQ (ierr);           // Oc(1) - d1
  ierr = VecDot(temp_, temp_, &m1); CHKERRQ (ierr);                   // ||.||^2

  // compute mismatch ||Oc(0) - d0||
  if (params_->tu_->two_time_points_) {
      ierr = tumor_->obs_->apply (temp_, tumor_->c_0_, 0); CHKERRQ (ierr); // Oc(0)
      ierr = VecAXPY (temp_, -1.0, data->dt0()); CHKERRQ (ierr);           // Oc(0) - d0
      ierr = VecDot (temp_, temp_, &m0); CHKERRQ (ierr);                   // ||.||^2
      // compute regularization (modified observation operator)
      //ierr = tumor_->obs_->apply (temp_, tumor_->c_t_, 1, true);  CHKERRQ (ierr); // I-Oc(1)
      //ierr = VecDot (temp_, temp_, &reg);                          CHKERRQ (ierr); // ||.||^2
      //reg *= 0.5 * params_->opt_->beta_;
  }

  // objective function value
  (*J) = params_->grid_->lebesgue_measure_ * 0.5 *(m1 + m0) + reg;

  if (params_->tu_->two_time_points_) {
    s << "  obj: J(p) = D(c1) + D(c0) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<<  params_->grid_->lebesgue_measure_ *0.5*m1 <<" + " << std::setprecision(12)<<  params_->grid_->lebesgue_measure_ * 0.5 * m0 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  } else {
    s << "  obj: J(p) = D(c1) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<<  params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateGradient (Vec dJ, Vec x, std::shared_ptr<Data> data){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  const ScalarType *x_ptr;
  params_->tu_->statistics_.nb_grad_evals++;

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  //ScalarType scale_rho = 1;
  //ScalarType scale_kap = 1E-2;

  // Finite difference gradient (forward differences)
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f, J_b;
  Vec delta_;

  // TODO: remove creation and deletion of delta every solve - move delta to class.
  ierr = VecDuplicate(x, &delta_); CHKERRQ (ierr);
  ierr = evaluateObjective (&J_b, x, data); CHKERRQ (ierr);
  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ierr = VecGetSize (x, &sz); CHKERRQ (ierr);
  ierr = VecGetArray (dJ, &dj_ptr); CHKERRQ (ierr);
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  std::stringstream s;
  //std::array<ScalarType, 3> characteristic_scale = {scale_kap * 1, scale_rho * 1, 1};
  #ifdef SINGLE
  ScalarType small = 3.45266983e-04F;
  #else
  ScalarType small = 3.45266983e-04;
  #endif
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy (x, delta_); CHKERRQ (ierr);
    ierr = VecGetArray (delta_, &delta_ptr);  CHKERRQ (ierr);
    ierr = VecGetArrayRead (x, &x_ptr); CHKERRQ (ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray (delta_, &delta_ptr); CHKERRQ (ierr);
    ierr = evaluateObjective (&J_f, delta_, data); CHKERRQ (ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead (x, &x_ptr); CHKERRQ (ierr);
  }
  ierr = VecRestoreArray (dJ, &dj_ptr); CHKERRQ (ierr);


  if(procid == 0) { ierr = VecView(dJ, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
  if(delta_ != nullptr) {ierr = VecDestroy(&delta_); CHKERRQ(ierr);}

  PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateObjectiveAndGradient (PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  //ScalarType scale_rho = 1;
  //ScalarType scale_kap = 1E-02

  ierr = evaluateObjective (J, x, data); CHKERRQ(ierr);

  // Finite difference gradient (forward differences)
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f;
  Vec delta_;
  ierr = VecDuplicate(x, &delta_); CHKERRQ (ierr);

  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ScalarType const *x_ptr;
  ierr = VecGetSize (x, &sz); CHKERRQ (ierr);
  ierr = VecGetArray (dJ, &dj_ptr); CHKERRQ (ierr);

  ScalarType scale = 1;
  // std::array<ScalarType, 3> characteristic_scale = {params_->opt_->k_scale_, params_->opt_->kf_scale_, params_->opt_->rho_scale_};
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  //std::array<ScalarType, 3> characteristic_scale = {scale_kap * 1, scale_rho * 1, 1};
  #ifdef SINGLE
  ScalarType small = 3.45266983e-04F;
  #else
  ScalarType small = 3.45266983e-04;
  #endif
  ScalarType J_b = (*J);

  for (int i = 0; i < sz; i++) {
    ierr = VecCopy (x, delta_); CHKERRQ (ierr);
    ierr = VecGetArray (delta_, &delta_ptr); CHKERRQ (ierr);
    ierr = VecGetArrayRead (x, &x_ptr); CHKERRQ (ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray (delta_, &delta_ptr); CHKERRQ (ierr);
    ierr = evaluateObjective (&J_f, delta_, data); CHKERRQ (ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead (x, &x_ptr); CHKERRQ (ierr);
  }

  ierr = VecRestoreArray (dJ, &dj_ptr); CHKERRQ (ierr);

  if(procid == 0) { ierr = VecView(dJ, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
  if(delta_ != nullptr) {ierr = VecDestroy(&delta_); CHKERRQ(ierr);}

  PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    params_->tu_->statistics_.nb_hessian_evals++;

    std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
    Event e ("tumor-eval-hessian");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    // no-op, i.e., gradient descent
    ierr = VecCopy(x, y); CHKERRQ (ierr);

    self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}
