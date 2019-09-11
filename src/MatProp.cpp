#include "MatProp.h"

MatProp::MatProp (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) : spec_ops_ (spec_ops) {
	PetscErrorCode ierr;
	ierr = VecCreate (PETSC_COMM_WORLD, &gm_);
	ierr = VecSetSizes (gm_, n_misc->n_local_, n_misc->n_global_);
	ierr = setupVec (gm_);

	ierr = VecDuplicate (gm_, &wm_);
	ierr = VecDuplicate (gm_, &csf_);
	ierr = VecDuplicate (gm_, &glm_);
	ierr = VecDuplicate (gm_, &bg_);
	ierr = VecDuplicate (gm_, &filter_);

	ierr = VecSet (gm_ , 0);
	ierr = VecSet (wm_ , 0);
	ierr = VecSet (csf_ , 0);
	ierr = VecSet (glm_ , 0);
	ierr = VecSet (bg_, 0);
	ierr = VecSet (filter_ , 0);

	n_misc_ = n_misc;
}

PetscErrorCode MatProp::setValues (std::shared_ptr<NMisc> n_misc) {
	PetscFunctionBegin;
	PetscErrorCode ierr;
	ScalarType *gm_ptr, *wm_ptr, *csf_ptr, *glm_ptr, *filter_ptr;
	int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

	if (n_misc->testcase_ != BRAIN && n_misc->testcase_ != BRAINNEARMF && n_misc->testcase_ != BRAINFARMF) {
		ierr = VecSet (filter_, 1.0);						  CHKERRQ (ierr);
		ierr = VecSet (wm_, 1.0); 							  CHKERRQ (ierr);
		n_misc->nk_ = 1;
	}
	else {
		std::stringstream str;
        std::stringstream ss;
		const char *prefix = "./brain_data/";
		str << prefix << "/" << n_misc->n_[0] << "/gray_matter.nc";
		std::ifstream data_file (str.str().c_str());
		if (!data_file.is_open()) {
			//Do nothing
			ss << " ---- brain default data not read: expecting user data ---- "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
		}
		else {
			str.str(std::string());
			str << prefix << "/" << n_misc->n_[0] << "/gray_matter.nc";
			dataIn (gm_, n_misc, str.str().c_str());
			str.str(std::string());
			str << prefix << "/" << n_misc->n_[0] << "/white_matter.nc";
			dataIn (wm_, n_misc, str.str().c_str());
			str.str(std::string());
			str << prefix << "/" << n_misc->n_[0] << "/csf.nc";
			dataIn (csf_, n_misc, str.str().c_str());
			str.str(std::string());
			str << prefix << "/" << n_misc->n_[0] << "/glial_matter.nc";
			dataIn (glm_, n_misc, str.str().c_str());
			
			ScalarType sigma_smooth = n_misc->smoothing_factor_ * 2 * M_PI / n_misc->n_[0];

			ierr = spec_ops_->weierstrassSmoother (gm_, gm_, n_misc, sigma_smooth);
			ierr = spec_ops_->weierstrassSmoother (wm_, wm_, n_misc, sigma_smooth);
			ierr = spec_ops_->weierstrassSmoother (glm_, glm_, n_misc, sigma_smooth);
			ierr = spec_ops_->weierstrassSmoother (csf_, csf_, n_misc, sigma_smooth);

			ierr = VecGetArray (gm_, &gm_ptr);                    CHKERRQ (ierr);
			ierr = VecGetArray (wm_, &wm_ptr);                    CHKERRQ (ierr);
			ierr = VecGetArray (csf_, &csf_ptr);                  CHKERRQ (ierr);
			ierr = VecGetArray (glm_, &glm_ptr);                  CHKERRQ (ierr);
			ierr = VecGetArray (filter_, &filter_ptr);            CHKERRQ (ierr);


			for (int i = 0; i < n_misc->n_local_; i++) {
				if ((wm_ptr[i] > 0.1 || gm_ptr[i] > 0.1) && csf_ptr[i] < 0.8)
					filter_ptr[i] = 1.0;
				else
					filter_ptr[i] = 0.0;
			}

			if(n_misc->writeOutput_) {
				dataOut (gm_ptr, n_misc, "gray_matter.nc");
				dataOut (wm_ptr, n_misc, "white_matter.nc");
				dataOut (csf_ptr, n_misc, "csf.nc");
				dataOut (glm_ptr, n_misc, "glial_matter.nc");
				dataOut (filter_ptr, n_misc, "filter_zero.nc");
			}

			ierr = VecRestoreArray (gm_, &gm_ptr);                    CHKERRQ (ierr);
			ierr = VecRestoreArray (wm_, &wm_ptr);                    CHKERRQ (ierr);
			ierr = VecRestoreArray (csf_, &csf_ptr);                  CHKERRQ (ierr);
			ierr = VecRestoreArray (glm_, &glm_ptr);                  CHKERRQ (ierr);
			ierr = VecRestoreArray (filter_, &filter_ptr);            CHKERRQ (ierr);
		}
	}

	PetscFunctionReturn (ierr);
}

PetscErrorCode MatProp::setValuesCustom (Vec gm, Vec wm, Vec glm, Vec csf, Vec bg, std::shared_ptr<NMisc> n_misc) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	int nk = 0;
	if(wm != nullptr)      { ierr = VecCopy (wm, wm_); nk++;  CHKERRQ(ierr); }
	else                   { ierr = VecSet (wm_, 0.0);        CHKERRQ(ierr); }
	if(gm != nullptr)      { ierr = VecCopy (gm, gm_); nk++;  CHKERRQ(ierr); }
	else                   { ierr = VecSet (gm_, 0.0);        CHKERRQ(ierr); }
	if(csf != nullptr)     { ierr = VecCopy (csf, csf_);      CHKERRQ(ierr); }
	else                   { ierr = VecSet (csf_, 0.0);       CHKERRQ(ierr); }
	if(glm != nullptr)     { ierr = VecCopy (glm, glm_); nk++; CHKERRQ(ierr); }
	else                   { ierr = VecSet (glm_, 0.0);       CHKERRQ(ierr); }
	if(bg != nullptr)      { ierr = VecCopy (bg, bg_);		  CHKERRQ(ierr); }
	else                   { ierr = VecSet (bg_, 0.0);       CHKERRQ(ierr); }
	if (!n_misc->nk_fixed_) n_misc->nk_ = nk;

	ScalarType *gm_ptr, *wm_ptr, *csf_ptr, *glm_ptr, *filter_ptr, *bg_ptr;
	ierr = VecGetArray (gm_, &gm_ptr);                    CHKERRQ (ierr);
	ierr = VecGetArray (wm_, &wm_ptr);                    CHKERRQ (ierr);
	ierr = VecGetArray (csf_, &csf_ptr);                  CHKERRQ (ierr);
	ierr = VecGetArray (glm_, &glm_ptr);                  CHKERRQ (ierr);
	ierr = VecGetArray (filter_, &filter_ptr);            CHKERRQ (ierr);
	ierr = VecGetArray (bg_, &bg_ptr);					  CHKERRQ (ierr);

	for (int i = 0; i < n_misc->n_local_; i++) {
		if ((wm_ptr[i] > 0.1 || gm_ptr[i] > 0.1) && csf_ptr[i] < 0.8)
			filter_ptr[i] = 1.0;
		else
			filter_ptr[i] = 0.0;
	}

	if(n_misc->writeOutput_) {
		dataOut (gm_ptr, n_misc, "gray_matter.nc");
		dataOut (wm_ptr, n_misc, "white_matter.nc");
		dataOut (csf_ptr, n_misc, "csf.nc");
		dataOut (glm_ptr, n_misc, "glial_matter.nc");
		dataOut (filter_ptr, n_misc, "filter_zero.nc");
		dataOut (bg_ptr, n_misc, "bg.nc");
	}

	ierr = VecRestoreArray (gm_, &gm_ptr);                    CHKERRQ (ierr);
	ierr = VecRestoreArray (wm_, &wm_ptr);                    CHKERRQ (ierr);
	ierr = VecRestoreArray (csf_, &csf_ptr);                  CHKERRQ (ierr);
	ierr = VecRestoreArray (glm_, &glm_ptr);                  CHKERRQ (ierr);
	ierr = VecRestoreArray (filter_, &filter_ptr);            CHKERRQ (ierr);
	ierr = VecRestoreArray (bg_, &bg_ptr);					  CHKERRQ (ierr);
	PetscFunctionReturn (ierr);
}

PetscErrorCode MatProp::filterTumor (Vec c) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ScalarType *c_ptr, *wm_ptr, *gm_ptr;
	ierr = VecGetArray (gm_, &gm_ptr);						  CHKERRQ (ierr);
	ierr = VecGetArray (wm_, &wm_ptr);						  CHKERRQ (ierr);
	ierr = VecGetArray (c, &c_ptr);							  CHKERRQ (ierr);

	for (int i = 0; i < n_misc_->n_local_; i++) {
		wm_ptr[i] *= (1. - c_ptr[i]);
		gm_ptr[i] *= (1. - c_ptr[i]);		
	}

	ierr = VecRestoreArray (gm_, &gm_ptr);						  CHKERRQ (ierr);
	ierr = VecRestoreArray (wm_, &wm_ptr);						  CHKERRQ (ierr);
	ierr = VecRestoreArray (c, &c_ptr);							  CHKERRQ (ierr);


	PetscFunctionReturn (ierr);
}

PetscErrorCode MatProp::filterBackgroundAndSmooth (Vec in) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ierr = VecShift (bg_, -1.0);						CHKERRQ (ierr); // bg - 1
	ierr = VecScale (bg_, -1.0);						CHKERRQ (ierr); // 1 - bg
	ierr = VecPointwiseMult (in, in, bg_); 				CHKERRQ (ierr); // in .* (1 - bg);
	ierr = VecScale (bg_, -1.0);						CHKERRQ (ierr); // bg - 1
	ierr = VecShift (bg_, 1.0);							CHKERRQ (ierr); // bg

	ScalarType sigma_smooth = 1. * n_misc_->smoothing_factor_ * 2 * M_PI / n_misc_->n_[0];
	ierr = spec_ops_->weierstrassSmoother (in, in, n_misc_, sigma_smooth);

	PetscFunctionReturn (ierr);
}

MatProp::~MatProp() {
	PetscErrorCode ierr;
	ierr = VecDestroy (&gm_);
	ierr = VecDestroy (&wm_);
	ierr = VecDestroy (&csf_);
	ierr = VecDestroy (&glm_);
	ierr = VecDestroy (&bg_);
	ierr = VecDestroy (&filter_);
}
