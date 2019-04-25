#include "Tumor.h"

Tumor::Tumor (std::shared_ptr<NMisc> n_misc) : n_misc_ (n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    k_ = std::make_shared<DiffCoef> (n_misc);
    rho_ = std::make_shared<ReacCoef> (n_misc);
    obs_ = std::make_shared<Obs> (n_misc);

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t_);
    ierr = VecSetSizes (c_t_, n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (c_t_);
    ierr = VecDuplicate (c_t_, &c_0_);
    ierr = VecDuplicate (c_t_, &p_0_);
    ierr = VecDuplicate (c_0_, &p_t_);

    // allocating memory for work vectors
    work_ = new Vec[12];
    for (int i = 0; i < 12; i++) {
        ierr = VecDuplicate (c_t_, &work_[i]);
        ierr = VecSet (work_[i] , 0);
    }
    // setting work vecs for diffusion coefficient (first 8)
    k_->setWorkVecs (work_);

    ierr = VecSet (c_t_, 0);
    ierr = VecSet (c_0_, 0);
    ierr = VecSet (p_0_, 0);
    ierr = VecSet (p_t_, 0);

    ierr = VecDuplicate (c_t_, &seg_);                                        
    ierr = VecSet (seg_, 0);                                            


    if (n_misc->model_ == 4) { // mass effect model -- allocate space for more variables
        velocity_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
        force_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
        displacement_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
    }
}


PetscErrorCode Tumor::initialize (Vec p, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (mat_prop == nullptr) {
        mat_prop_ = std::make_shared<MatProp> (n_misc);
        ierr = mat_prop_->setValues (n_misc);
    }
    else
        mat_prop_ = mat_prop;
    ierr = k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = VecDuplicate (p, &p_);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_true_);                            CHKERRQ (ierr);
    ierr = VecDuplicate (p, &weights_);                           CHKERRQ (ierr);
    ierr = VecCopy (p, p_);                                       CHKERRQ (ierr);
    ierr = setTrueP (n_misc);                                     CHKERRQ (ierr);

    if (phi == nullptr) {
        phi_ = std::make_shared<Phi> (n_misc);
        ierr = phi_->setGaussians (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
        ierr = phi_->setValues (mat_prop_);
    }
    else
        phi_ = phi;
    ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn(0);
}

PetscErrorCode Tumor::setParams (Vec p, std::shared_ptr<NMisc> n_misc, bool npchanged) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    if (npchanged) {
      // re-create p vectors
      if (p_ != nullptr) {ierr = VecDestroy (&p_);                  CHKERRQ (ierr);}
      if (p_true_ != nullptr) {ierr = VecDestroy (&p_true_);        CHKERRQ (ierr);}
      if (weights_ != nullptr) {ierr = VecDestroy (&weights_);      CHKERRQ (ierr);}
      ierr = VecDuplicate (p, &p_);                                 CHKERRQ (ierr);
      ierr = VecDuplicate (p, &p_true_);                            CHKERRQ (ierr);
      ierr = VecDuplicate (p, &weights_);                           CHKERRQ (ierr);
      // re-create phi mesh (deletes old instance and creates new instance)
      // phi_ = std::make_shared<Phi> (n_misc);
    }
    ierr = VecCopy (p, p_);                                         CHKERRQ (ierr);
    ierr = VecCopy (p, p_true_);                                    CHKERRQ (ierr);

    // set new values
    ierr = k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, mat_prop_, n_misc);
    // ierr = phi_->setGaussians (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = phi_->setValues (mat_prop_);
    ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn(0);
}

PetscErrorCode Tumor::setTrueP (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscScalar val;

    // if (n_misc->smoothing_factor_ == 1) {
    //     val = 1.38;
    // } else if (n_misc->smoothing_factor_ == 1.5) {
    //     val = 1.95;
    // } else if (n_misc->smoothing_factor_ == 2) {
    //     val = 2.8;
    // } else {
    //     val = 1.;
    // }

    val = 1.;
    double *p_ptr;
    PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    ierr = VecSet (p_true_, 0);                                     CHKERRQ (ierr);
    ierr = VecGetArray (p_true_, &p_ptr);                           CHKERRQ (ierr);
    if (n_misc->np_ == 1) {
        p_ptr[0] = val;
    } else {
        p_ptr[center] = val;
    }

    ierr = VecRestoreArray (p_true_, &p_ptr);                       CHKERRQ (ierr);


    // if (n_misc->np_ == 1) {
    //     ierr = VecSet (p_true_, val);                                 CHKERRQ (ierr);
    //     PetscFunctionReturn (0);
    // }
    // // PetscScalar val[2] = {.9, .2}; 
    // // PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    // // PetscInt idx[2] = {center-1, center};
    // // ierr = VecSetValues(p_true_, 2, idx, val, INSERT_VALUES );        CHKERRQ(ierr);
    // // ierr = VecAssemblyBegin(p_true_);                                 CHKERRQ(ierr);
    // // ierr = VecAssemblyEnd(p_true_);                                   CHKERRQ(ierr);
    // // PetscFunctionReturn (0);
    
    // PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    // PetscInt idx = center;
    // ierr = VecSetValues(p_true_, 1, &idx, &val, INSERT_VALUES);         CHKERRQ(ierr);
    // ierr = VecAssemblyBegin(p_true_);                                 CHKERRQ(ierr);
    // ierr = VecAssemblyEnd(p_true_);                                   CHKERRQ(ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode Tumor::setTrueP (std::shared_ptr<NMisc> n_misc, PetscScalar val) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    double *p_ptr;
    PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    ierr = VecSet (p_true_, 0);                                     CHKERRQ (ierr);
    ierr = VecGetArray (p_true_, &p_ptr);                           CHKERRQ (ierr);
    if (n_misc->np_ == 1) {
        p_ptr[0] = val;
    } else {
        p_ptr[center] = val;
    }

    ierr = VecRestoreArray (p_true_, &p_ptr);                       CHKERRQ (ierr);

    PetscFunctionReturn (0);
}


PetscErrorCode Tumor::setTrueP (Vec p) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ierr = VecCopy (p, p_true_);                                      CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode Tumor::computeForce (Vec c1) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-compute-force");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    std::bitset<3> XYZ;
    XYZ[0] = 1;
    XYZ[1] = 1;
    XYZ[2] = 1;

    double *c_ptr, *fx_ptr, *fy_ptr, *fz_ptr;

    // snafu: smooth
    Vec c;
    ierr = VecDuplicate (c1, &c);      CHKERRQ (ierr);
    ierr = VecCopy (c1, c);            CHKERRQ (ierr);

    ierr = VecGetArray (c, &c_ptr);                                  CHKERRQ (ierr);
    double sigma_smooth = 1.0 * 2.0 * M_PI / n_misc_->n_[0];
    ierr = weierstrassSmoother (c_ptr, c_ptr, n_misc_, sigma_smooth);
    ierr = VecRestoreArray (c, &c_ptr);                              CHKERRQ (ierr);


    accfft_grad (force_->x_, force_->y_, force_->z_, c, n_misc_->plan_, &XYZ, t.data());

    ierr = force_->getComponentArrays (fx_ptr, fy_ptr, fz_ptr);
    ierr = VecGetArray (c, &c_ptr);                                  CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        fx_ptr[i] *= n_misc_->forcing_factor_ * tanh (c_ptr[i]);
        fy_ptr[i] *= n_misc_->forcing_factor_ * tanh (c_ptr[i]);
        fz_ptr[i] *= n_misc_->forcing_factor_ * tanh (c_ptr[i]);
    }
    ierr = VecRestoreArray (c, &c_ptr);                              CHKERRQ (ierr);
    ierr = force_->restoreComponentArrays (fx_ptr, fy_ptr, fz_ptr); 

    ierr = VecDestroy (&c);             CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();

    PetscFunctionReturn (0);
}

PetscErrorCode Tumor::computeSegmentation () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecSet (seg_, 0);                                                  CHKERRQ(ierr);

    // compute seg_ of gm, wm, csf, bg, tumor
    std::vector<double> v;
    std::vector<double>::iterator seg_component;
    double *bg_ptr, *gm_ptr, *wm_ptr, *csf_ptr, *c_ptr, *seg_ptr;
    ierr = VecGetArray (mat_prop_->bg_, &bg_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->gm_, &gm_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->wm_, &wm_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->csf_, &csf_ptr);                   CHKERRQ(ierr);
    ierr = VecGetArray (c_t_, &c_ptr);                                CHKERRQ(ierr);
    ierr = VecGetArray (seg_, &seg_ptr);                               CHKERRQ(ierr);

    // segmentation for c0
    for (int i = 0; i < n_misc_->n_local_; i++) {    
        v.push_back (bg_ptr[i]); 
        v.push_back (c_ptr[i]);
        v.push_back (gm_ptr[i]);
        v.push_back (wm_ptr[i]);
        v.push_back (csf_ptr[i]);
        
        seg_component = std::max_element (v.begin(), v.end());
        seg_ptr[i] = std::distance (v.begin(), seg_component);

        v.clear();
    }   
    
    double sigma_smooth = 1.0 * M_PI / n_misc_->n_[0];
    ierr = weierstrassSmoother (seg_ptr, seg_ptr, n_misc_, sigma_smooth);
    
    ierr = VecRestoreArray (mat_prop_->bg_, &bg_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->gm_, &gm_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->wm_, &wm_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->csf_, &csf_ptr);                   CHKERRQ(ierr);
    ierr = VecRestoreArray (c_t_, &c_ptr);                                CHKERRQ(ierr);
    ierr = VecRestoreArray (seg_, &seg_ptr);                               CHKERRQ(ierr); 

    PetscFunctionReturn(0);
}


Tumor::~Tumor () {
    PetscErrorCode ierr;
    ierr = VecDestroy (&c_t_);
    ierr = VecDestroy (&c_0_);
    ierr = VecDestroy (&p_t_);
    ierr = VecDestroy (&p_0_);
    ierr = VecDestroy (&p_);
    ierr = VecDestroy (&p_true_);

    for (int i = 0; i < 12; i++) {
        ierr = VecDestroy (&work_[i]);
    }
    delete[] work_;
    ierr = VecDestroy (&weights_);

    ierr = VecDestroy (&seg_);
}
