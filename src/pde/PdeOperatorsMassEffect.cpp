#include "PdeOperators.h"

/* #### --------------------------------------------------------------------------- #### */
/* #### ========                 PDE Ops. Mass effect model.               ======== #### */
/* #### --------------------------------------------------------------------------- #### */
PetscErrorCode PdeOperatorsMassEffect::reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  // no-op so far

  params_ = params;
  if (tumor != nullptr) tumor_ = tumor;

  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMassEffect::conserveHealthyTissues() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-conserve-healthy");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  // gm, wm is conserved with rhs g/(g + w) * (Dc + Rc) : treated explicity
  ScalarType *c_ptr, *rho_ptr, *gm_ptr, *wm_ptr, *scale_gm_ptr, *scale_wm_ptr, *sum_ptr;
  ScalarType dt = params_->tu_->dt_;
  ierr = VecCopy(tumor_->c_t_, temp_[1]); CHKERRQ(ierr);
  ierr = tumor_->k_->applyD(temp_[0], temp_[1]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[1], tumor_->c_t_, tumor_->c_t_); CHKERRQ(ierr);
  ierr = VecAYPX(temp_[1], -1.0, tumor_->c_t_); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[1], temp_[1], tumor_->rho_->rho_vec_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[0], 1.0, temp_[1]); CHKERRQ(ierr);

  ierr = vecGetArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(temp_[0], &sum_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(temp_[1], &scale_gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(temp_[2], &scale_wm_ptr); CHKERRQ(ierr);

#ifdef CUDA
  conserveHealthyTissuesCuda(gm_ptr, wm_ptr, sum_ptr, scale_gm_ptr, scale_wm_ptr, dt, params_->grid_->nl_);
#else
  ScalarType threshold = 1E-3;
  for (int i = 0; i < params_->grid_->nl_; i++) {
    scale_gm_ptr[i] = 0;
    scale_wm_ptr[i] = 0;

    if ((gm_ptr[i] > threshold || wm_ptr[i] > threshold) && (wm_ptr[i] + gm_ptr[i] > threshold)) {
      scale_gm_ptr[i] = -dt * gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
      scale_wm_ptr[i] = -dt * wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
    }
    gm_ptr[i] += scale_gm_ptr[i] * sum_ptr[i];
    wm_ptr[i] += scale_wm_ptr[i] * sum_ptr[i];
  }
#endif

  ierr = vecRestoreArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(temp_[0], &sum_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(temp_[1], &scale_gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(temp_[2], &scale_wm_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMassEffect::updateReacAndDiffCoefficients(Vec seg, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *bg_ptr, *gm_ptr, *vt_ptr, *csf_ptr, *rho_ptr, *k_ptr;
  ierr = vecGetArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->k_->kxx_, &k_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->csf_, &csf_ptr); CHKERRQ(ierr);

  ScalarType temp = 1.;
  ScalarType gm_k_scale = 1 - params_->tu_->k_gm_wm_ratio_;
  ScalarType gm_r_scale = 1 - params_->tu_->r_gm_wm_ratio_;
#ifdef CUDA
  updateReacAndDiffCoefficientsCuda(rho_ptr, k_ptr, bg_ptr, gm_ptr, vt_ptr, csf_ptr, params_->tu_->rho_, params_->tu_->k_, gm_r_scale, gm_k_scale, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    temp = (1 - (bg_ptr[i] + gm_r_scale * gm_ptr[i] + vt_ptr[i] + csf_ptr[i]));
    temp = (temp < 0) ? 0 : temp;
    rho_ptr[i] = temp * params_->tu_->rho_;
    temp = (1 - (bg_ptr[i] + gm_k_scale * gm_ptr[i] + vt_ptr[i] + csf_ptr[i]));
    temp = (temp < 0) ? 0 : temp;
    k_ptr[i] = temp * params_->tu_->k_;
  }
#endif

  ierr = vecRestoreArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->k_->kxx_, &k_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->csf_, &csf_ptr); CHKERRQ(ierr);

  // copy kxx to other directions
  ierr = VecCopy(tumor_->k_->kxx_, tumor_->k_->kyy_); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->k_->kxx_, tumor_->k_->kzz_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMassEffect::computeTumorQuants(std::shared_ptr<Tumor> tumor, Vec dcdt, Vec gradc, ScalarType *c_star) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  std::array<double, 7> t = {0};
  // gradc
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  // gradc mag 
  Vec c = tumor->c_t_;
  std::shared_ptr<VecField> work_field = tumor->work_field_;
  ierr = spec_ops_->computeGradient(work_field->x_, work_field->y_, work_field->z_, c, &XYZ, t.data());
  ierr = work_field->computeMagnitude(gradc); CHKERRQ(ierr);

  // dcdt = Rc + Dc - Ac
  ierr = VecCopy(c, dcdt); CHKERRQ(ierr);
  // dcdt = Dc
  ierr = tumor_->k_->applyD(dcdt, dcdt); CHKERRQ(ierr); // because applyD uses temp vectors always use pdeop's work for temp here
  ierr = VecPointwiseMult(work_[2], c, c); CHKERRQ(ierr);
  ierr = VecAYPX(work_[2], -1.0, c); CHKERRQ(ierr);
  ierr = VecPointwiseMult(work_[2], work_[2], tumor_->rho_->rho_vec_); CHKERRQ(ierr);
  // work[2] is Rc
  ierr = VecAXPY(dcdt, 1.0, work_[2]); CHKERRQ(ierr);
  // dcdt = Rc + Dc here; add -Ac = -div (cv)
  // compute cv
  ierr = VecPointwiseMult(work_field->x_, c, tumor->velocity_->x_); CHKERRQ(ierr);
  ierr = VecPointwiseMult(work_field->y_, c, tumor->velocity_->y_); CHKERRQ(ierr);
  ierr = VecPointwiseMult(work_field->z_, c, tumor->velocity_->z_); CHKERRQ(ierr);
  // compute div(cv)
  ierr = spec_ops_->computeDivergence(work_[2], work_field->x_, work_field->y_, work_field->z_, t.data());
  ierr = VecAXPY(dcdt, -1.0, work_[2]); CHKERRQ(ierr);

  // compute location of max gradient
  ierr = VecCopy(gradc, work_[2]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(work_[2], gradc, tc_atlas_); CHKERRQ(ierr);
  ierr = VecAYPX(work_[2], -1.0, gradc); CHKERRQ(ierr); //work is now gradc(1-tc)
  PetscInt max_loc;
  ScalarType max_gradc;
  ierr = VecMax(work_[2], &max_loc, &max_gradc); CHKERRQ(ierr);
  ScalarType *c_ptr;
  ierr = vecGetArray(c, &c_ptr); CHKERRQ(ierr);
#ifdef CUDA
  cudaMemcpy(c_star, &c_ptr[max_loc], sizeof(ScalarType), cudaMemcpyDeviceToHost);
#else
  (*c_star) = c_ptr[max_loc];
#endif
  ierr = vecRestoreArray(c, &c_ptr); CHKERRQ(ierr);
 
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMassEffect::writeStats(Vec x, std::stringstream &feature_stream) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType vol, sum, mean, std, quart;
  // compute \integral (x)
  ierr = computeVolume(x, params_->grid_->lebesgue_measure_, &vol, &sum); CHKERRQ(ierr);
  feature_stream << vol << ",";
  // compute std in x
  mean = sum / params_->grid_->ng_;
  ierr = VecPointwiseMult(work_[2], x, x); CHKERRQ(ierr);
  ierr = vecSum(work_[2], &sum); CHKERRQ(ierr);
  sum /= params_->grid_->ng_;
  std = std::sqrt(sum - mean * mean); // sqrt(E(x^2) - E(x)^2)
  feature_stream << std << ",";
  // compute 75% quantile
  ierr = computeQuantile(x, work_[2], &quart, 0.75); CHKERRQ(ierr);
  feature_stream << quart << ",";

  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMassEffect::computeStress(Vec *gradu, Vec jacobian, Vec trace_stress, Vec max_shear) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-solve-stress-compute");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  ScalarType **gradu_ptr = new ScalarType*[9];
  ScalarType *jac_ptr, *trace_ptr, *max_shear_ptr, *mu_ptr, *lam_ptr;

  CtxElasticity *ctx;
  ierr = MatShellGetContext(elasticity_solver_->A_, &ctx); CHKERRQ(ierr);
  Vec mu = ctx->mu_;
  Vec lam = ctx->lam_;

  for (int i = 0; i < 9; i++) {
    ierr = VecGetArray(gradu[i], &gradu_ptr[i]); CHKERRQ(ierr);
  }
  ierr = VecGetArray(jacobian, &jac_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(trace_stress, &trace_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(max_shear, &max_shear_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mu, &mu_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(lam, &lam_ptr); CHKERRQ(ierr);

  std::array<ScalarType, 9> F; //deformation gradient
  std::array<ScalarType, 9> E; //strain tensor
  std::array<ScalarType, 9> S; //stress tensor
  
  /*
  // use petsc to compute eigenvalues because Eigen has issues with xlc compilers
  // characteristic polynomial method for CUDA is TODO; need to check numerical stability
  // setup a ksp object for eigenvalues
  KSP ksp = nullptr;
  Mat A = nullptr;
  Vec rhs = nullptr;
  Vec sol = nullptr;
  ierr = setupKSPEigenvalues(&ksp, &A, &rhs, &sol); CHKERRQ(ierr);

  PetscScalar *a = new PetscScalar[9];
  PetscReal r[3], c[3];
  const int id[3] = {0,1,2};
  int size = 3;
  std::vector<PetscScalar> eigenvalues(3);
  */

  for (int i = 0; i < params_->grid_->nl_; i++) {
    for (int j = 0; j < F.size(); j++) F[j] = gradu_ptr[j][i];
    F[0] += 1; 
    F[4] += 1;
    F[8] += 1; // F = I + \gradu

    jac_ptr[i] = computeDeterminant(F);
    E = computeStrainTensor(F);
    S = computeStressTensor(E, mu_ptr[i], lam_ptr[i]); 
    trace_ptr[i] = S[0] + S[4] + S[8]; // trace of stress tensor
    /*
    // compute eigenvalues
    for (int i = 0; i < 9; i++) a[i] = S[i];
    // set stress tensor into petsc mat object
    ierr = MatSetValues(A, size, id, size, id, a, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    // eigenvalues into real and complex arrays
    ierr = KSPComputeEigenvaluesExplicitly(ksp, 3, r, c); CHKERRQ(ierr);
    for (int i = 0; i < 3; i++) eigenvalues[i] = r[i];
    std::sort(eigenvalues.begin(), eigenvalues.end());
    max_shear_ptr[i] = 0.5 * (eigenvalues[2] - eigenvalues[0]);
    TU_assert(std::accumulate(eigenvalues.begin(), eigenvalues.end(), decltype(eigenvalues)::value_type(0)) == trace_ptr[i], "Sum of eigenvalues not equal to trace");
    */
  }

  for (int i = 0; i < 9; i++) {
    ierr = VecRestoreArray(gradu[i], &gradu_ptr[i]); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(jacobian, &jac_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(trace_stress, &trace_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(max_shear, &max_shear_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mu, &mu_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(lam, &lam_ptr); CHKERRQ(ierr);


  delete[] gradu_ptr;
  /*
  delete[] a;

  if (ksp != nullptr) {
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  }
  if (A != nullptr) {
    ierr = MatDestroy(&A); CHKERRQ(ierr);
  }
  if (rhs != nullptr) {
    ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  }
  if (sol != nullptr) {
    ierr = VecDestroy(&sol); CHKERRQ(ierr);
  }
  */
  
  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}
  

PetscErrorCode PdeOperatorsMassEffect::computeBiophysicalFeatures(std::stringstream &feature_stream, int time_step) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-solve-feature-compute");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  std::vector<std::string> feature_list{"c","dcdt","gradc","u","trT","tau","jac","vel"};

  if (time_step == 0) {
    feature_stream << "timestep,";
    for (auto feature : feature_list) {
      feature_stream << "vol_" << feature << "," << "sa_" << feature << "," << "std_" << feature << "," << "quart_" << feature;
    }
  }

  // compute tc 
  tc_atlas_ = work_[5];
  ierr = tumor_->getTCRecon(tc_atlas_); CHKERRQ(ierr);

  feature_stream << "\n" << time_step << ",";
  // concentration-based features
  Vec c = tumor_->c_t_;
  Vec dcdt = work_[0];
  Vec gradc = work_[1];
  ScalarType c_star = 0; // value of c at max gradient 
  ierr = computeTumorQuants(tumor_, dcdt, gradc, &c_star); CHKERRQ(ierr);
  std::stringstream s;
  s << "c(x) at location of max gradient outside TC = " << c_star;
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  Vec indicator = work_[4];
  // compute |c - c_star| < epsilon --> used for surface integrals computed thro' vol integrals
  ierr = computeIndicatorFunction(indicator, c, c_star); CHKERRQ(ierr);

  ierr = writeStats(c, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(dcdt, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(gradc, feature_stream); CHKERRQ(ierr);
  
  // displacement-based features
  // work is done; can be resued
  ierr = tumor_->displacement_->computeMagnitude(work_[2]); CHKERRQ(ierr); //work is mag of disp
  ierr = writeStats(work_[2], feature_stream); CHKERRQ(ierr);
  // stresses
  // compute gradu 
  Vec *tumor_work = tumor_->work_; // reuse tumor work here; 12 work vectors
  ierr = spec_ops_->computeGradient(tumor_work[0], tumor_work[1], tumor_work[2], tumor_->displacement_->x_, &XYZ, t.data());
  ierr = spec_ops_->computeGradient(tumor_work[3], tumor_work[4], tumor_work[5], tumor_->displacement_->y_, &XYZ, t.data());
  ierr = spec_ops_->computeGradient(tumor_work[6], tumor_work[7], tumor_work[8], tumor_->displacement_->z_, &XYZ, t.data());
 
  Vec jacobian = work_[0];
  Vec trace_stress = work_[1];
  Vec max_shear = work_[2];

  ierr = computeStress(tumor_work, jacobian, trace_stress, max_shear); CHKERRQ(ierr);
  ierr = writeStats(jacobian, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(trace_stress, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(max_shear, feature_stream); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMassEffect::solveState(int linearized) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-solve-state");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;
  params_->tu_->statistics_.nb_state_solves++;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  ierr = VecCopy(tumor_->c_0_, tumor_->c_t_); CHKERRQ(ierr);
  ierr = displacement_old_->set(0); CHKERRQ(ierr);

  ScalarType k1, k2, k3, r1, r2, r3;
  k1 = params_->tu_->k_;
  k2 = params_->tu_->k_gm_wm_ratio_ * params_->tu_->k_;
  k3 = 0;
  r1 = params_->tu_->rho_;
  r2 = params_->tu_->r_gm_wm_ratio_ * params_->tu_->rho_;
  r3 = 0;

  // filter matprop
  ierr = tumor_->mat_prop_->filterTumor(tumor_->c_t_); CHKERRQ(ierr);
  // force compute
  ierr = tumor_->computeForce(tumor_->c_t_);
  // displacement compute through elasticity solve
  ierr = elasticity_solver_->solve(displacement_old_, tumor_->force_);

  std::stringstream ss;
  ScalarType vel_max;
  ScalarType cfl;
  std::stringstream s;
  ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / params_->grid_->n_[0];
  bool flag_smooth_velocity = true;
  bool write_output_and_break = false;

  ScalarType max_cfl = 14;

  std::ofstream feature_file;
  std::stringstream feature_stream;
      
  for (int i = 0; i < nt + 1; i++) {
    if (params_->tu_->verbosity_ > 1) {
      s << "Time step = " << i;
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
    }
    
    // compute CFL
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ierr = tumor_->velocity_->computeMagnitude(magnitude_);
    ierr = VecMax(magnitude_, NULL, &vel_max); CHKERRQ(ierr);
    cfl = dt * vel_max / params_->grid_->h_[0];
    if (params_->tu_->verbosity_ > 1) {
      s << "CFL = " << cfl;
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
    }
    // Adaptively time step if CFL is too large
    if (cfl >= max_cfl) {
      s << "CFL is too large (>=" << max_cfl << "); consider using smaller forcing factor; exiting solver...";
      ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
      write_output_and_break = true;
    }

    if (((!params_->tu_->feature_compute_) && (params_->tu_->write_output_ && params_->tu_->verbosity_ > 1 && i % 5 == 0)) || write_output_and_break) {
      ss << "velocity_t[" << i << "].nc";
      dataOut(magnitude_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ierr = displacement_old_->computeMagnitude(magnitude_);
      ss << "displacement_t[" << i << "].nc";
      dataOut(magnitude_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "seg_t[" << i << "].nc";
      dataOut(tumor_->seg_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "c_t[" << i << "].nc";
      dataOut(tumor_->c_t_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      if (params_->tu_->verbosity_ > 2) {
        ss << "rho_t[" << i << "].nc";
        dataOut(tumor_->rho_->rho_vec_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "kxx_t[" << i << "].nc";
        dataOut(tumor_->k_->kxx_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "lam_t[" << i << "].nc";
        dataOut(elasticity_solver_->ctx_->lam_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "mu_t[" << i << "].nc";
        dataOut(elasticity_solver_->ctx_->mu_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "scr_t[" << i << "].nc";
        dataOut(elasticity_solver_->ctx_->screen_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ierr = tumor_->force_->computeMagnitude(magnitude_);
        ss << "force_t[" << i << "].nc";
        dataOut(magnitude_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "vt_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->vt_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "csf_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->csf_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "wm_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->wm_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "gm_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->gm_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "mri_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->mri_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
      }
    }

    if (write_output_and_break) break;

    // Update diffusivity and reaction coefficient
    ierr = updateReacAndDiffCoefficients(tumor_->seg_, tumor_); CHKERRQ(ierr);
    CHKERRQ(ierr); CHKERRQ(ierr);

    // need to update prefactors for diffusion KSP preconditioner, as k changed
    ierr = diff_solver_->precFactor(); CHKERRQ(ierr);

    if (params_->tu_->feature_compute_) {
      // compute biophysical temporal features
      s << "computing biophysical features for time step " << i;
      ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
      ierr = computeBiophysicalFeatures(feature_stream, i);
    }
    // Advection of tumor and healthy tissue
    // first compute trajectories for semi-Lagrangian solve as velocity is changing every itr
    adv_solver_->trajectoryIsComputed_ = false;
    ierr = adv_solver_->solve(tumor_->mat_prop_->gm_, tumor_->velocity_, dt); CHKERRQ(ierr);
    ierr = adv_solver_->solve(tumor_->mat_prop_->wm_, tumor_->velocity_, dt); CHKERRQ(ierr);
    adv_solver_->advection_mode_ = 2;  // pure advection for vt
    ierr = adv_solver_->solve(tumor_->mat_prop_->vt_, tumor_->velocity_, dt); CHKERRQ(ierr);
    ierr = adv_solver_->solve(tumor_->mat_prop_->csf_, tumor_->velocity_, dt); CHKERRQ(ierr);
    adv_solver_->advection_mode_ = 1;  // reset to mass conservation
    ierr = adv_solver_->solve(tumor_->c_t_, tumor_->velocity_, dt); CHKERRQ(ierr);

    if (tumor_->mat_prop_->mri_ != nullptr) {
      // transport mri
      adv_solver_->advection_mode_ = 2;
      ierr = adv_solver_->solve(tumor_->mat_prop_->mri_, tumor_->velocity_, dt); CHKERRQ(ierr);
      adv_solver_->advection_mode_ = 1;
    }

    // All solves complete except elasticity: clip values to ensure positivity
    // clip healthy tissues ~ this is non-differentiable. careful.
    // ierr = tumor_->mat_prop_->clipHealthyTissues ();                          CHKERRQ (ierr);

    // Diffusion of tumor
    ierr = diff_solver_->solve(tumor_->c_t_, dt);

    // Reaction of tumor
    ierr = reaction(linearized, i); CHKERRQ(ierr);

    // force compute
    ierr = tumor_->computeForce(tumor_->c_t_); CHKERRQ(ierr);
    // displacement compute through elasticity solve: Linv(force_) = displacement_
    ierr = elasticity_solver_->solve(tumor_->displacement_, tumor_->force_); CHKERRQ(ierr);
    // compute velocity
    ierr = VecWAXPY(tumor_->velocity_->x_, -1.0, displacement_old_->x_, tumor_->displacement_->x_); CHKERRQ(ierr);
    ierr = VecWAXPY(tumor_->velocity_->y_, -1.0, displacement_old_->y_, tumor_->displacement_->y_); CHKERRQ(ierr);
    ierr = VecWAXPY(tumor_->velocity_->z_, -1.0, displacement_old_->z_, tumor_->displacement_->z_); CHKERRQ(ierr);
    ierr = VecScale(tumor_->velocity_->x_, (1.0 / dt)); CHKERRQ(ierr);
    ierr = VecScale(tumor_->velocity_->y_, (1.0 / dt)); CHKERRQ(ierr);
    ierr = VecScale(tumor_->velocity_->z_, (1.0 / dt)); CHKERRQ(ierr);

    // smooth the velocity
    if (flag_smooth_velocity) {
      ierr = spec_ops_->weierstrassSmoother(tumor_->velocity_->x_, tumor_->velocity_->x_, params_, sigma_smooth); CHKERRQ(ierr);
      ierr = spec_ops_->weierstrassSmoother(tumor_->velocity_->y_, tumor_->velocity_->y_, params_, sigma_smooth); CHKERRQ(ierr);
      ierr = spec_ops_->weierstrassSmoother(tumor_->velocity_->z_, tumor_->velocity_->z_, params_, sigma_smooth); CHKERRQ(ierr);
    }

    ScalarType vel_x_norm, vel_y_norm, vel_z_norm;
    ierr = VecNorm(tumor_->velocity_->x_, NORM_2, &vel_x_norm); CHKERRQ(ierr);
    ierr = VecNorm(tumor_->velocity_->y_, NORM_2, &vel_y_norm); CHKERRQ(ierr);
    ierr = VecNorm(tumor_->velocity_->z_, NORM_2, &vel_z_norm); CHKERRQ(ierr);
    if (params_->tu_->verbosity_ > 1) {
      s << "Norm of velocity (x,y,z) = (" << vel_x_norm << ", " << vel_y_norm << ", " << vel_z_norm << ")";
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
    }

    // copy displacement to old vector
    ierr = displacement_old_->copy(tumor_->displacement_);
  }

  if (params_->tu_->verbosity_ >= 3) {
    s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

#ifdef CUDA
  if (params_->tu_->verbosity_ > 1) cudaPrintDeviceMemory();
#endif

  if ((params_->tu_->write_output_ && params_->tu_->verbosity_ > 1 && !write_output_and_break)) {
    // for mass-effect inversion, write the very last one too. TODO: change loc of print statements instead.
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "seg_final.nc";
    dataOut(tumor_->seg_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ss << "c_final.nc";
    dataOut(tumor_->c_t_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ss << "vt_final.nc";
    dataOut(tumor_->mat_prop_->vt_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ss << "csf_final.nc";
    dataOut(tumor_->mat_prop_->csf_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ss << "wm_final.nc";
    dataOut(tumor_->mat_prop_->wm_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ss << "gm_final.nc";
    dataOut(tumor_->mat_prop_->gm_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ierr = tumor_->displacement_->computeMagnitude(magnitude_);
    ss << "displacement_final.nc";
    dataOut(magnitude_, params_, ss.str().c_str());
    ss.str(std::string());
    ss.clear();
    ScalarType mag_norm, mm;
    ierr = VecNorm(magnitude_, NORM_2, &mag_norm); CHKERRQ(ierr);
    ierr = VecMax(magnitude_, NULL, &mm); CHKERRQ(ierr);
    ss << "norm of displacement: " << mag_norm << "; max of displacement: " << mm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();

    if (tumor_->mat_prop_->mri_ != nullptr) {
      ss << "mri_final.nc";
      dataOut(tumor_->mat_prop_->mri_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
    }
  }


  if (params_->tu_->feature_compute_) {
    if (procid == 0) {
      feature_file.open(params_->tu_->writepath_ + "biophysical_features.csv", std::ios_base::out);
      feature_file << feature_stream.str() << std::endl;
      feature_file.close();
    }
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}
