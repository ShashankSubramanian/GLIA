#include "Obs.h"

Obs::Obs (std::shared_ptr<Parameters> params) :
    params_ (params)
{
    PetscErrorCode ierr;
    two_snapshot_ = params->tu_->two_snapshot_;
    threshold_1_ = params->tu_->obs_threshold_1_;
    threshold_0_ = params->tu_->obs_threshold_0_;
    ierr = VecCreate (PETSC_COMM_WORLD, &filter_1_);
    ierr = VecSetSizes (filter_1_, params->grid_->nl_, params_->grid_->ng_);
    ierr = setupVec (filter_1_);
    ierr = VecSet (filter_1_, 1.0);

    // only allocate memory for T=0 if two snapshots
    if (two_snapshot_) {
        ierr = VecCreate (PETSC_COMM_WORLD, &filter_0_);
        ierr = VecSetSizes (filter_0_, params->grid_->nl_, params_->grid_->ng_);
        ierr = setupVec (filter_0_);
        ierr = VecSet (filter_0_, 1.0);
        ierr = VecDuplicate (filter_0_, &one);
        ierr = VecSet (one, 1.0);
    } else {
        filter_0_ = nullptr;
    }
}

PetscErrorCode Obs::setDefaultFilter (Vec data, int time_point, ScalarType thr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ScalarType *filter_ptr, *data_ptr;

    ScalarType th =0;
    if (time_point==0){
        if (not two_snapshot_) {ierr = tuMSGwarn("Error: Cannot apply Obs(T=0), not a two snapshot scenario."); CHKERRQ(ierr); }
        if (thr > 0) threshold_0_ = thr;
        ierr = VecGetArray (filter_0_, &filter_ptr);                                CHKERRQ (ierr);
        th = threshold_0_;
    } else {
        if (thr > 0) threshold_1_ = thr;
        ierr = VecGetArray (filter_1_, &filter_ptr);                                CHKERRQ (ierr);
        th = threshold_1_;
    }
    ierr = VecGetArray (data, &data_ptr);                                           CHKERRQ (ierr);
    for (int i = 0; i < params_->grid_->nl_; i++) {
        filter_ptr[i] = ScalarType (data_ptr[i] > th);
    }
    if (time_point==0){
        ierr = VecRestoreArray (filter_0_, &filter_ptr);                            CHKERRQ (ierr);
    } else {
        ierr = VecRestoreArray (filter_1_, &filter_ptr);                            CHKERRQ (ierr);
    }
    ierr = VecRestoreArray (data, &data_ptr);                                       CHKERRQ (ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode Obs::setCustomFilter (Vec custom_filter, int time_point) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    if (time_point==0){
        if (not two_snapshot_) {ierr = tuMSGwarn("Error: Cannot apply Obs(T=0), not a two snapshot scenario."); CHKERRQ(ierr); }
        ierr = VecCopy (custom_filter, filter_0_);                                  CHKERRQ (ierr);
    } else {
        ierr = VecCopy (custom_filter, filter_1_);                                  CHKERRQ (ierr);
    }
    PetscFunctionReturn (ierr);
}

PetscErrorCode Obs::apply(Vec y, Vec x, int time_point, bool complement) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    if (complement) {tuMSGstd(" applying complement obs op:  I-O ");}
    if (time_point==0){
        if (not two_snapshot_) {ierr = tuMSGwarn("Error: Cannot apply Obs(T=0), not a two snapshot scenario."); CHKERRQ(ierr); }
        if (complement) {ierr = VecScale(filter_0_, -1 ); CHKERRQ(ierr); VecAXPY (filter_0_, 1, one); CHKERRQ(ierr);}
        ierr = VecPointwiseMult (y, x, filter_0_);                                  CHKERRQ (ierr);
        if (complement) { VecAXPY (filter_0_, -1, one); CHKERRQ(ierr); ierr = VecScale(filter_0_, -1); CHKERRQ(ierr);}
    } else {
        if (complement) {ierr = VecScale(filter_1_, -1 ); CHKERRQ(ierr); VecAXPY (filter_1_, 1, one); CHKERRQ(ierr);}
        ierr = VecPointwiseMult (y, x, filter_1_);                                  CHKERRQ (ierr);
        if (complement) { VecAXPY (filter_1_, -1, one); CHKERRQ(ierr); ierr = VecScale(filter_1_, -1); CHKERRQ(ierr);}
    }
    PetscFunctionReturn (ierr);
}


PetscErrorCode Obs::applyT(Vec y, Vec x, int time_point, bool complement) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    if (time_point==0){
        if (not two_snapshot_) {ierr = tuMSGwarn("Error: Cannot apply Obs(T=0), not a two snapshot scenario."); CHKERRQ(ierr); }
        if (complement) {ierr = VecScale(filter_0_, -1 ); CHKERRQ(ierr); VecAXPY (filter_0_, 1, one); CHKERRQ(ierr);}
        ierr = VecPointwiseMult (y, x, filter_0_);                                  CHKERRQ (ierr);
        if (complement) { VecAXPY (filter_0_, -1, one); CHKERRQ(ierr); ierr = VecScale(filter_0_, -1); CHKERRQ(ierr);}
    } else {
        if (complement) {ierr = VecScale(filter_1_, -1 ); CHKERRQ(ierr); VecAXPY (filter_1_, 1, one); CHKERRQ(ierr);}
        ierr = VecPointwiseMult (y, x, filter_1_);                                  CHKERRQ (ierr);
        if (complement) { VecAXPY (filter_1_, -1, one); CHKERRQ(ierr); ierr = VecScale(filter_1_, -1); CHKERRQ(ierr);}
    }
    PetscFunctionReturn (ierr);
}

Obs::~Obs () {
    PetscErrorCode ierr;
    ierr = VecDestroy (&filter_1_);
    if (two_snapshot_) {VecDestroy (&filter_0_); VecDestroy(&one);}
}
