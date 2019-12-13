#include "Tumor.h"

Tumor::Tumor (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) : n_misc_ (n_misc), spec_ops_ (spec_ops) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    k_ = std::make_shared<DiffCoef> (n_misc, spec_ops);
    rho_ = std::make_shared<ReacCoef> (n_misc, spec_ops);
    obs_ = std::make_shared<Obs> (n_misc);

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t_);
    ierr = VecSetSizes (c_t_, n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (c_t_);
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


    if (n_misc->model_ == 4 || n_misc_->model_ == 5) { // mass effect model -- allocate space for more variables
        velocity_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
        force_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
        displacement_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
        work_field_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
    }

    if (n_misc_->model_ == 5) {
        std::vector<Vec> c (n_misc->num_species_);
        for (int i = 0; i < c.size(); i++) {
            ierr = VecDuplicate (c_t_, &c[i]); 
            ierr = VecSet (c_t_, 0.);
        }
        // Insert the different species
        species_.insert (std::pair<std::string, Vec> ("proliferative", c[0]));
        species_.insert (std::pair<std::string, Vec> ("infiltrative", c[1]));
        species_.insert (std::pair<std::string, Vec> ("necrotic", c[2]));
        species_.insert (std::pair<std::string, Vec> ("oxygen", c[3]));
        species_.insert (std::pair<std::string, Vec> ("edema", c[4]));
    }
}


PetscErrorCode Tumor::initialize (Vec p, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (mat_prop == nullptr) {
        mat_prop_ = std::make_shared<MatProp> (n_misc, spec_ops);
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
        phi_ = std::make_shared<Phi> (n_misc, spec_ops);
        ierr = phi_->setGaussians (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
        ierr = phi_->setValues (mat_prop_);
    }
    else
        phi_ = phi;
    // TODO: for SIBIA this might be needed
    // ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn (ierr);
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

    // TODO: for sibia this might be needed
    // ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn (ierr);
}

PetscErrorCode Tumor::setTrueP (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType val;

    val = 1.;
    ScalarType *p_ptr;
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
    //     PetscFunctionReturn (ierr);
    // }
    // // ScalarType val[2] = {.9, .2}; 

    // // PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    // // PetscInt idx[2] = {center-1, center};
    // // ierr = VecSetValues(p_true_, 2, idx, val, INSERT_VALUES );        CHKERRQ(ierr);
    // // ierr = VecAssemblyBegin(p_true_);                                 CHKERRQ(ierr);
    // // ierr = VecAssemblyEnd(p_true_);                                   CHKERRQ(ierr);
    // // PetscFunctionReturn (ierr);

    // PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    // PetscInt idx = center;
    // ierr = VecSetValues(p_true_, 1, &idx, &val, INSERT_VALUES);         CHKERRQ(ierr);
    // ierr = VecAssemblyBegin(p_true_);                                 CHKERRQ(ierr);
    // ierr = VecAssemblyEnd(p_true_);                                   CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode Tumor::setTrueP (std::shared_ptr<NMisc> n_misc, ScalarType val) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType *p_ptr;
    PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    ierr = VecSet (p_true_, 0);                                     CHKERRQ (ierr);
    ierr = VecGetArray (p_true_, &p_ptr);                           CHKERRQ (ierr);
    if (n_misc->np_ == 1) {
        p_ptr[0] = val;
    } else {
        p_ptr[center] = val;
    }

    ierr = VecRestoreArray (p_true_, &p_ptr);                       CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}


PetscErrorCode Tumor::setTrueP (Vec p) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ierr = VecCopy (p, p_true_);                                      CHKERRQ (ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode Tumor::computeForce (Vec c1) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-compute-force");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    std::bitset<3> XYZ;
    XYZ[0] = 1;
    XYZ[1] = 1;
    XYZ[2] = 1;

    ScalarType *c_ptr, *fx_ptr, *fy_ptr, *fz_ptr;
    ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / n_misc_->n_[0]; // smooth because c might be too sharp at csf boundaries

    ierr = VecCopy (c1, work_[0]);            CHKERRQ (ierr);
    ierr = spec_ops_->weierstrassSmoother (work_[0], work_[0], n_misc_, sigma_smooth);
    spec_ops_->computeGradient (force_->x_, force_->y_, force_->z_, work_[0], &XYZ, t.data());

    // scale force by constant
    ierr = force_->scale (n_misc_->forcing_factor_);                                     CHKERRQ (ierr);

    if (n_misc_->use_tanh_scaling_) {
        ierr = force_->getComponentArrays (fx_ptr, fy_ptr, fz_ptr);
        ierr = vecGetArray (work_[0], &c_ptr);                                          CHKERRQ (ierr);
        #ifdef CUDA
            nonlinearForceScalingCuda (c_ptr, fx_ptr, fy_ptr, fz_ptr, n_misc_->n_local_);
        #else
            for (int i = 0; i < n_misc_->n_local_; i++) {
                fx_ptr[i] *= tanh (c_ptr[i]);
                fy_ptr[i] *= tanh (c_ptr[i]);
                fz_ptr[i] *= tanh (c_ptr[i]);
            }
        #endif
        ierr = vecRestoreArray (work_[0], &c_ptr);                                  CHKERRQ (ierr);
        ierr = force_->restoreComponentArrays (fx_ptr, fy_ptr, fz_ptr); 
    }

    self_exec_time += MPI_Wtime();
    accumulateTimers (n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();

    PetscFunctionReturn (ierr);
}

PetscErrorCode Tumor::computeEdema () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType *i_ptr, *ed_ptr;
    ierr = VecGetArray (species_["infiltrative"], &i_ptr);                  CHKERRQ (ierr);
    ierr = VecGetArray (species_["edema"], &ed_ptr);                        CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) 
        ed_ptr[i] = (i_ptr[i] > n_misc_->invasive_threshold_) ? 1.0 : 0.0;

    ierr = VecRestoreArray (species_["infiltrative"], &i_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (species_["edema"], &ed_ptr);                        CHKERRQ (ierr);

    // smooth
    ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / n_misc_->n_[0];
    ierr = spec_ops_->weierstrassSmoother (species_["edema"], species_["edema"], n_misc_, sigma_smooth);  CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}

PetscErrorCode Tumor::computeSegmentation () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecSet (seg_, 0);                                                  CHKERRQ(ierr);
    // compute seg_ of gm, wm, csf, bg, tumor
    std::vector<ScalarType> v;
    std::vector<ScalarType>::iterator seg_component;
    ScalarType *bg_ptr, *gm_ptr, *wm_ptr, *csf_ptr, *c_ptr, *glm_ptr, *seg_ptr, *p_ptr, *n_ptr, *ed_ptr;
    ierr = VecGetArray (mat_prop_->bg_, &bg_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->gm_, &gm_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->wm_, &wm_ptr);                     CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->csf_, &csf_ptr);                   CHKERRQ(ierr);
    ierr = VecGetArray (mat_prop_->glm_, &glm_ptr);                   CHKERRQ(ierr);
    ierr = VecGetArray (seg_, &seg_ptr);                              CHKERRQ(ierr);

    // if (n_misc_->model_ == 5) {
    //     ierr = VecGetArray (species_["proliferative"], &p_ptr);       CHKERRQ(ierr);
    //     ierr = VecGetArray (species_["necrotic"], &n_ptr);            CHKERRQ(ierr);
    //     ierr = VecGetArray (species_["edema"], &ed_ptr);              CHKERRQ(ierr);

    //     for (int i = 0; i < n_misc_->n_local_; i++) {
    //         v.push_back (bg_ptr[i]);
    //         v.push_back (p_ptr[i]);
    //         v.push_back (n_ptr[i]);
    //         v.push_back (ed_ptr[i]);
    //         v.push_back (wm_ptr[i]);
    //         v.push_back (gm_ptr[i]);
    //         v.push_back (csf_ptr[i]);
    //         v.push_back (glm_ptr[i]);

    //         seg_component = std::max_element (v.begin(), v.end());
    //         seg_ptr[i] = std::distance (v.begin(), seg_component);

    //         v.clear();
    //     }   
    // } else {
        ierr = VecGetArray (c_t_, &c_ptr);                                CHKERRQ(ierr);
        for (int i = 0; i < n_misc_->n_local_; i++) {
            v.push_back (bg_ptr[i]);
            v.push_back (c_ptr[i]);
            v.push_back (wm_ptr[i]);
            v.push_back (gm_ptr[i]);
            v.push_back (csf_ptr[i]);
            v.push_back (glm_ptr[i]);

            seg_component = std::max_element (v.begin(), v.end());
            seg_ptr[i] = std::distance (v.begin(), seg_component);

            v.clear();
        }   
    // }
    
    ierr = VecRestoreArray (mat_prop_->bg_, &bg_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->gm_, &gm_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->wm_, &wm_ptr);                     CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->csf_, &csf_ptr);                   CHKERRQ(ierr);
    ierr = VecRestoreArray (mat_prop_->glm_, &glm_ptr);                   CHKERRQ(ierr);
    // if (n_misc_->model_ == 5) {
    //     ierr = VecRestoreArray (species_["proliferative"], &p_ptr);       CHKERRQ(ierr);
    //     ierr = VecRestoreArray (species_["necrotic"], &n_ptr);            CHKERRQ(ierr);
    //     ierr = VecRestoreArray (species_["edema"], &ed_ptr);              CHKERRQ(ierr);
    // } else {
        ierr = VecRestoreArray (c_t_, &c_ptr);                                CHKERRQ(ierr);
    // }
    ierr = VecRestoreArray (seg_, &seg_ptr);                               CHKERRQ(ierr);

    PetscFunctionReturn (ierr);
}

PetscErrorCode Tumor::computeSpeciesNorms () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType nrm;
    std::stringstream s;

    if (n_misc_->model_ == 5) {
        ierr = VecNorm (species_["proliferative"], NORM_2, &nrm);              CHKERRQ (ierr);
        s << "Proliferative cell concentration norm = " << nrm;
        ierr = tuMSGstd (s.str());                                             CHKERRQ(ierr);
        s.str (""); s.clear ();
        ierr = VecNorm (species_["infiltrative"], NORM_2, &nrm);              CHKERRQ (ierr);
        s << "Infiltrative cell concentration norm = " << nrm;
        ierr = tuMSGstd (s.str());                                             CHKERRQ(ierr);
        s.str (""); s.clear ();
        ierr = VecNorm (species_["necrotic"], NORM_2, &nrm);              CHKERRQ (ierr);
        s << "Necrotic cell concentration norm = " << nrm;
        ierr = tuMSGstd (s.str());                                             CHKERRQ(ierr);
        s.str (""); s.clear ();
    }

    PetscFunctionReturn (ierr);
}


PetscErrorCode Tumor::clipTumor () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType *c_ptr;
    ierr = vecGetArray (c_t_, &c_ptr);                          CHKERRQ (ierr);

    if (n_misc_->model_ == 5) {
        ScalarType *p_ptr, *i_ptr, *n_ptr;
        ierr = vecGetArray (species_["proliferative"], &p_ptr); CHKERRQ (ierr);
        ierr = vecGetArray (species_["infiltrative"], &i_ptr);  CHKERRQ (ierr);
        ierr = vecGetArray (species_["necrotic"], &n_ptr);      CHKERRQ (ierr);

        #ifdef CUDA
            clipVectorCuda (p_ptr, n_misc_->n_local_);
            clipVectorCuda (i_ptr, n_misc_->n_local_);
            clipVectorCuda (n_ptr, n_misc_->n_local_);
        #else
            for (int i = 0; i < n_misc_->n_local_; i++) {
                p_ptr[i] = (p_ptr[i] <= 0.) ? 0. : p_ptr[i];
                i_ptr[i] = (i_ptr[i] <= 0.) ? 0. : i_ptr[i];
                n_ptr[i] = (n_ptr[i] <= 0.) ? 0. : n_ptr[i];
            }
        #endif

        ierr = vecRestoreArray (species_["proliferative"], &p_ptr); CHKERRQ (ierr);
        ierr = vecRestoreArray (species_["infiltrative"], &i_ptr);  CHKERRQ (ierr);
        ierr = vecRestoreArray (species_["necrotic"], &n_ptr);      CHKERRQ (ierr);

    } else {
        #ifdef CUDA
            clipVectorCuda (c_ptr, n_misc_->n_local_);
            clipVectorAboveCuda (c_ptr, n_misc_->n_local_);
        #else
            for (int i = 0; i < n_misc_->n_local_; i++) {
                c_ptr[i] = (c_ptr[i] <= 0.) ? 0. : c_ptr[i];
                c_ptr[i] = (c_ptr[i] > 1.) ? 1. : c_ptr[i];
            }
        #endif
    }

    ierr = vecRestoreArray (c_t_, &c_ptr);                      CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
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

    for (std::map<std::string, Vec>:: iterator it = species_.begin(); it != species_.end(); it++) {
        ierr = VecDestroy (&it->second);
    }
}
