#include "PdeOperators.h"

/* #### --------------------------------------------------------------------------- #### */
/* #### ========                 PDE Ops. Mass effect model.               ======== #### */
/* #### --------------------------------------------------------------------------- #### */

PdeOperatorsMassEffect::PdeOperatorsMassEffect(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : PdeOperatorsRD(tumor, params, spec_ops) {
  adv_solver_ = std::make_shared<SemiLagrangianSolver>(params, tumor, spec_ops);
  // adv_solver_ = std::make_shared<TrapezoidalSolver> (params, tumor, spec_ops);
  elasticity_solver_ = std::make_shared<VariableLinearElasticitySolver>(params, tumor, spec_ops);
  displacement_old_ = std::make_shared<VecField>(params_->grid_->nl_, params_->grid_->ng_);
  VecDuplicate(tumor->work_[0], &magnitude_);
  temp_ = new Vec[3];
  for (int i = 0; i < 3; i++) {
    temp_[i] = tumor->work_[11 - i];
  }
  work_ = new Vec[num_work_vecs_];
  for (int i = 0; i < num_work_vecs_; i++) work_[i] = nullptr;
  if (params->tu_->feature_compute_) {
    for (int i = 0; i < num_work_vecs_; i++) {
      VecDuplicate(tumor->c_t_, &work_[i]);
      VecSet(work_[i], 0);
    }
  }
  tc_ = nullptr;
  tc_atlas_ = nullptr;
  indicator_ = nullptr;
  num_tc_voxels_ = 0;
}

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
  
  std::array<ScalarType, 3> eigenvalues;
#ifdef CUDA
  computeStressQuantsCuda(gradu_ptr, jac_ptr, trace_ptr, max_shear_ptr, mu_ptr, lam_ptr, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    for (int j = 0; j < F.size(); j++) F[j] = gradu_ptr[j][i];
    F[0] += 1; 
    F[4] += 1;
    F[8] += 1; // F = I + \gradu

    jac_ptr[i] = computeDeterminant(F);
    E = computeStrainTensor(F);
    S = computeStressTensor(E, mu_ptr[i], lam_ptr[i]); 
    trace_ptr[i] = S[0] + S[4] + S[8]; // trace of stress tensor

    computeEigenValues(S.data(), eigenvalues.data());
    std::sort(eigenvalues.begin(), eigenvalues.end());
//    eigenvalues[0] = eigenvalues[0] > 0 ? eigenvalues[0] : 0;
    max_shear_ptr[i] = 0.5 * (eigenvalues[2] - eigenvalues[0]);
//    max_shear_ptr[i] = (std::isnan(max_shear_ptr[i]) || std::isinf(max_shear_ptr[i])) ? 0 : max_shear_ptr[i];
  }
#endif

  for (int i = 0; i < 9; i++) {
    ierr = VecRestoreArray(gradu[i], &gradu_ptr[i]); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(jacobian, &jac_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(trace_stress, &trace_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(max_shear, &max_shear_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mu, &mu_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(lam, &lam_ptr); CHKERRQ(ierr);

  delete[] gradu_ptr;
  
  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
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

  ScalarType vol, sa, sum, mean, std, quart;
  ScalarType vol_roi, sum_roi, mean_roi, std_roi, quart_roi;
  ScalarType vol_roi_c, sum_roi_c, mean_roi_c, std_roi_c, quart_roi_c;
  //
  // set roi as tc and its complement as the healthy brain
  Vec roi = tc_atlas_;
  PetscInt n_roi = num_tc_voxels_;
  Vec roi_c = healthy_brain_;
  PetscInt n_roi_c = num_healthy_voxels_;

  // compute \integral (x)
  ScalarType vol_measure = params_->grid_->lebesgue_measure_;
  ierr = computeVolume(x, vol_measure, &vol, &sum); CHKERRQ(ierr);
  ierr = computeVolume(x, roi, work_[2], vol_measure, &vol_roi, &sum_roi); CHKERRQ(ierr);
  ierr = computeVolume(x, roi_c, work_[2], vol_measure, &vol_roi_c, &sum_roi_c); CHKERRQ(ierr);
  feature_stream << vol << "," << vol_roi << "," << vol_roi_c << ",";
  
  // compute SA (x) = (N/2pi) * \integral I \hadamard x
  ierr = VecPointwiseMult(work_[2], indicator_, x); CHKERRQ(ierr);
  ScalarType sa_measure = vol_measure *  params_->grid_->n_[0] / (2.0 * M_PI);
  ierr = computeVolume(work_[2], sa_measure, &sa, nullptr); CHKERRQ(ierr); 
  feature_stream << sa << ",";
 
  // compute std in x
  mean = sum / params_->grid_->ng_;
  mean_roi = sum_roi / n_roi;
  mean_roi_c = sum_roi_c / n_roi_c;
  ierr = computeStd(x, work_[2], params_->grid_->ng_, mean, &std); CHKERRQ(ierr);
  ierr = computeStd(x, work_[2], roi, n_roi, mean_roi, &std_roi); CHKERRQ(ierr);
  ierr = computeStd(x, work_[2], roi_c, n_roi_c, mean_roi_c, &std_roi_c); CHKERRQ(ierr);
  feature_stream << std << "," << std_roi << "," << std_roi_c << ",";
 
  // compute quantile
  ierr = computeQuantile(x, work_[2], &quart, 0.875); CHKERRQ(ierr);
  ierr = computeQuantile(x, roi, work_[2], work_[7], &quart_roi, n_roi, 0.875, false); CHKERRQ(ierr);
  ierr = computeQuantile(x, roi_c, work_[2], work_[7], &quart_roi_c, n_roi_c, 0.875, false); CHKERRQ(ierr);
  feature_stream << quart << "," << quart_roi << "," << quart_roi_c << ",";
  // compute quantile
  ierr = computeQuantile(x, work_[2], &quart, 0.95); CHKERRQ(ierr);
  ierr = computeQuantile(x, roi, work_[2], work_[7], &quart_roi, n_roi, 0.95, false); CHKERRQ(ierr);
  ierr = computeQuantile(x, roi_c, work_[2], work_[7], &quart_roi_c, n_roi_c, 0.95, false); CHKERRQ(ierr);
  feature_stream << quart << "," << quart_roi << "," << quart_roi_c << ",";
  // compute quantile
  ierr = computeQuantile(x, work_[2], &quart, 0.99); CHKERRQ(ierr);
  ierr = computeQuantile(x, roi, work_[2], work_[7], &quart_roi, n_roi, 0.99, false); CHKERRQ(ierr);
  ierr = computeQuantile(x, roi_c, work_[2], work_[7], &quart_roi_c, n_roi_c, 0.99, false); CHKERRQ(ierr);
  feature_stream << quart << "," << quart_roi << "," << quart_roi_c << ",";

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

  std::vector<std::string> feature_list{"c","dcdt","gradc","u","jac","trT","tau","vel"};
  std::vector<std::string> stats{"vol","sa","std","quartmed","95","99"};
  if (time_step == 0) {
    feature_stream << "t,volc/volb,";
    for (auto feature : feature_list) {
      for (auto stat: stats) {
        if (stat == "sa") feature_stream << stat << "_" << feature << ",";
        else feature_stream << stat << "_" << feature << "," << stat << "_tc_" << feature << "," << stat << "_b_" << feature << ",";
      }
    }
  }

  // compute tc 
  tc_atlas_ = work_[5];
  healthy_brain_ = work_[6];
  ierr = tumor_->getTCRecon(tc_atlas_); CHKERRQ(ierr);
  ierr = vecSum(tc_atlas_, &num_tc_voxels_); CHKERRQ(ierr);
  // healthy brain is simply 1 - bg - tc_atlas
  ierr = tumor_->getHealthyBrain(healthy_brain_); CHKERRQ(ierr);
  ierr = vecSum(healthy_brain_, &num_healthy_voxels_); CHKERRQ(ierr);

  feature_stream << "\n" << time_step * params_->tu_->dt_  << ",";
  feature_stream << num_tc_voxels_/ (num_healthy_voxels_ + num_tc_voxels_) << ",";
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
  indicator_ = work_[4];
  // compute |c - c_star| < epsilon --> used for surface integrals computed thro' vol integrals
  ierr = computeIndicatorFunction(indicator_, c, c_star); CHKERRQ(ierr);
//    std::stringstream ss;
//    if (time_step%5==0) {
//      ss << "indicator_t[" << time_step << "].nc";
//      dataOut(indicator_, params_, ss.str().c_str());
//      ss.str(std::string());
//      ss.clear();
//    }

  ierr = writeStats(c, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(dcdt, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(gradc, feature_stream); CHKERRQ(ierr);
  
  // displacement-based features
  // work is done; can be resued
  ierr = tumor_->displacement_->computeMagnitude(work_[3]); CHKERRQ(ierr); //work is mag of disp
  ierr = writeStats(work_[3], feature_stream); CHKERRQ(ierr);
  // stresses
  // compute gradu 
  Vec *tumor_work = tumor_->work_; // reuse tumor work here; 12 work vectors
  ierr = spec_ops_->computeGradient(tumor_work[0], tumor_work[1], tumor_work[2], tumor_->displacement_->x_, &XYZ, t.data());
  ierr = spec_ops_->computeGradient(tumor_work[3], tumor_work[4], tumor_work[5], tumor_->displacement_->y_, &XYZ, t.data());
  ierr = spec_ops_->computeGradient(tumor_work[6], tumor_work[7], tumor_work[8], tumor_->displacement_->z_, &XYZ, t.data());
 
  Vec jacobian = work_[0];
  Vec trace_stress = work_[1];
  Vec max_shear = work_[3];

  ierr = computeStress(tumor_work, jacobian, trace_stress, max_shear); CHKERRQ(ierr);
  ierr = writeStats(jacobian, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(trace_stress, feature_stream); CHKERRQ(ierr);
  ierr = writeStats(max_shear, feature_stream); CHKERRQ(ierr);
//    std::stringstream ss;
//    if (time_step%5==0) {
//      ss << "max_shear_t[" << time_step << "].nc";
//      dataOut(max_shear, params_, ss.str().c_str());
//      ss.str(std::string());
//      ss.clear();
//    }
  // velocity
  ierr = tumor_->velocity_->computeMagnitude(work_[3]); CHKERRQ(ierr); //work is mag of disp
  ierr = writeStats(work_[3], feature_stream); CHKERRQ(ierr);

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
  ierr = tumor_->computeForce(tumor_->c_t_); CHKERRQ(ierr);
  // displacement compute through elasticity solve
  ierr = elasticity_solver_->solve(displacement_old_, tumor_->force_); CHKERRQ(ierr);
  //ierr = tumor_->displacement_->copy(displacement_old_); CHKERRQ(ierr);

  std::stringstream ss;
  ScalarType vel_max;
  ScalarType tu_ratio;
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
      // kill solver if volume of tumor is larger than 10% of the brain
      tu_ratio = num_tc_voxels_ / (num_healthy_voxels_ + num_tc_voxels_);
      s << "tumor ratio with brain = " << tu_ratio;
      ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
      if (std::abs(tu_ratio - 0.1) < 5E-3 || tu_ratio > 0.1) { 
        s << "tumor is large, ratio = " << tu_ratio << ", with diff from 10% = " << std::abs(tu_ratio-0.1) << "; exiting solver...";
        ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
        s.str("");
        s.clear();
        break;
      } 
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
