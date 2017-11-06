#include "Phi.h"

Phi::Phi (std::shared_ptr<NMisc> n_misc) : n_misc_ (n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;

    n_local_ = n_misc->n_local_;

    np_ = n_misc->np_;
    Vec v;
    for (int i = 0; i < np_; i++) {
        ierr = VecCreate (PETSC_COMM_WORLD, &v);
        ierr = VecSetSizes (v, n_misc->n_local_, n_misc->n_global_);
        ierr = VecSetFromOptions (v);
        ierr = VecSet (v, 0);
        phi_vec_.push_back (v);
    }
}

PetscErrorCode Phi::setValues (std::array<double, 3>& user_cm, double sigma, double spacing_factor, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PCOUT << " ----- Bounding box for Phi set with NP: " << np_ << " --------" << std::endl;
    memcpy (cm_, user_cm.data(), 3 * sizeof(double));
    double center[3 * np_];
    double *phi_ptr;
    double sigma_smooth = 2.0 * M_PI / n_misc->n_[0];

    sigma_ = sigma;
    spacing_factor_ = spacing_factor;

    ierr = phiMesh (center);

    for (int i = 0; i < np_; i++) {
        ierr = VecGetArray (phi_vec_[i], &phi_ptr);                             CHKERRQ (ierr);
        initialize (phi_ptr, n_misc, &center[3 * i]);
        ierr = VecRestoreArray (phi_vec_[i], &phi_ptr);                         CHKERRQ (ierr);

        ierr = VecPointwiseMult (phi_vec_[i], mat_prop->filter_, phi_vec_[i]);  CHKERRQ (ierr);

        if (n_misc->testcase_ == BRAIN) {  //BRAIN
            ierr = VecGetArray (phi_vec_[i], &phi_ptr);                             CHKERRQ (ierr);
            ierr = weierstrassSmoother (phi_ptr, phi_ptr, n_misc, sigma_smooth);
            ierr = VecRestoreArray (phi_vec_[i], &phi_ptr);                         CHKERRQ (ierr);
        }

        if(n_misc->writeOutput_) {
            dataOut (phi_vec_[i], n_misc, "results/phi.nc");
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode Phi::phiMesh (double *center) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int nprocs, procid;
	MPI_Comm_rank(PETSC_COMM_WORLD, &procid);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    int h = round (std::pow (np_, 1.0 / 3.0));
    double space[3];

    #ifdef VISUALIZE_PHI
     std::stringstream phivis;
     phivis <<" sigma = "<<sigma_<<", spacing = "<<spacing_factor_ * sigma_<<std::endl;
     phivis <<" centers = ["<<std::endl;
    #endif

    space[0] =  spacing_factor_ * sigma_;
    space[1] =  spacing_factor_ * sigma_;
    space[2] =  spacing_factor_ * sigma_;

    if (np_ == 1) {
        center[0] = cm_[0];
        center[1] = cm_[1];
        center[2] = cm_[2];
        PetscFunctionReturn(0);
    }
    if (np_ % 2 == 1) {
        int ptr = 0;
        for (int k = -(h - 1) / 2; k <= (h - 1) / 2; k++)
            for (int j = -(h - 1) / 2; j <= (h - 1) / 2; j++)
                for (int i = -(h - 1) / 2; i <= (h - 1) / 2; i++) {
                    center[ptr + 0] = i * space[0] + cm_[0];
                    center[ptr + 1] = j * space[1] + cm_[1];
                    center[ptr + 2] = k * space[2] + cm_[2];
                    #ifdef VISUALIZE_PHI
                     phivis << " " << center[ptr + 0] <<", " << center[ptr + 1] << ", "  << center[ptr + 2] << std::endl;
                    #endif
                    ptr += 3;
                }
    }
    else if (np_ % 2 == 0) {
        int ptr = 0;
        for (int k = -(h) / 2; k <= (h) / 2; k++)
            for (int j = -(h) / 2; j <= (h) / 2; j++)
                for (int i = -(h) / 2; i <= (h) / 2; i++) {
                    if ((i != 0) && (j != 0) && (k != 0)) {
                        center[ptr + 0] = i * space[0] + cm_[0];
                        center[ptr + 1] = j * space[1] + cm_[1];
                        center[ptr + 2] = k * space[2] + cm_[2];
                        #ifdef VISUALIZE_PHI
                         phivis << " " << center[ptr + 0] <<", " << center[ptr + 1] << ", "  << center[ptr + 2] <<std::endl;
                        #endif
                        ptr += 3;
                    }
                 }
    }
    #ifdef VISUALIZE_PHI
    phivis << "];"<<std::endl;
    std::fstream phifile;
    static int ct = 0;
    if(procid == 0) {
      std::stringstream ssct; ssct<<ct;
      phifile.open(std::string("phi-mesh-"+ssct.str()+".dat"), std::ios_base::out);
      phifile << phivis.str()<<std::endl;
      phifile.close();
      ct++;
    }
    #endif

    PetscFunctionReturn(0);
}

PetscErrorCode Phi::initialize (double *out, std::shared_ptr<NMisc> n_misc, double *center) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double twopi = 2.0 * M_PI;
    const double R = std::sqrt(2.) * sigma_; //0.05*twopi;
    int64_t X, Y, Z;
    double dummy, r, ratio;
    int64_t ptr;
    double xc = center[0], yc = center[1], zc = center[2];
    double hx = twopi / n_misc->n_[0], hy = twopi / n_misc->n_[1], hz = twopi / n_misc->n_[2];

    for (int x = 0; x < n_misc->isize_[0]; x++)
        for (int y = 0; y < n_misc->isize_[1]; y++)
            for (int z = 0; z < n_misc->isize_[2]; z++) {
                X = n_misc->istart_[0] + x;
                Y = n_misc->istart_[1] + y;
                Z = n_misc->istart_[2] + z;
                r = sqrt((hx * X - xc) * (hx * X - xc) + (hy * Y - yc) * (hy * Y - yc) + (hz * Z - zc) * (hz * Z - zc));
                ratio = r / R;
                ptr = x * n_misc->isize_[1] * n_misc->isize_[2] + y * n_misc->isize_[2] + z;
                out[ptr] = std::exp(-ratio * ratio);

            }
  PetscFunctionReturn(0);
}

PetscErrorCode Phi::apply (Vec out, Vec p) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double * pg_ptr;
    Vec pg;
    ierr = VecCreateSeq (PETSC_COMM_SELF, np_, &pg);                            CHKERRQ (ierr);
    ierr = VecSetFromOptions (pg);                                              CHKERRQ (ierr);

    {
        VecScatter scatter; /* scatter context */
        IS from, to; /* index sets that define the scatter */
        int idx_from[np_], idx_to[np_];
        for (int i = 0; i < np_; i++) {
            idx_from[i] = i;
            idx_to[i] = i;
        }
        ierr = ISCreateGeneral (PETSC_COMM_SELF, np_, idx_from, PETSC_COPY_VALUES,  &from);         CHKERRQ (ierr);
        ierr = ISCreateGeneral (PETSC_COMM_SELF, np_, idx_to, PETSC_COPY_VALUES, &to);              CHKERRQ (ierr);
        ierr = VecScatterCreate (p, from, pg, to, &scatter);                                        CHKERRQ (ierr);
        ierr = VecScatterBegin (scatter, p, pg, INSERT_VALUES, SCATTER_FORWARD);                    CHKERRQ (ierr);
        ierr = VecScatterEnd (scatter, p, pg, INSERT_VALUES, SCATTER_FORWARD);                      CHKERRQ (ierr);
        ierr = ISDestroy (&from);                                                                   CHKERRQ (ierr);
        ierr = ISDestroy (&to);                                                                     CHKERRQ (ierr);
        ierr = VecScatterDestroy (&scatter);                                                        CHKERRQ (ierr);
    }

    Vec dummy;
    ierr = VecDuplicate (phi_vec_[0], &dummy);                                                      CHKERRQ (ierr);
    ierr = VecSet (dummy, 0);                                                                       CHKERRQ (ierr);

    ierr = VecGetArray (pg, &pg_ptr);                                                               CHKERRQ (ierr);

    for (int i = 0; i < np_; i++) {
        ierr = VecAXPY (dummy, pg_ptr[i], phi_vec_[i]);                                             CHKERRQ (ierr);
    }

    ierr = VecRestoreArray (pg, &pg_ptr);                                                           CHKERRQ (ierr);

    ierr = VecCopy(dummy, out);                                                                     CHKERRQ (ierr);

    ierr = VecDestroy (&dummy);                                                                     CHKERRQ (ierr);
    ierr = VecDestroy (&pg);                                                                        CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode Phi::applyTranspose (Vec pout, Vec in) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscScalar values[np_];
    int low, high;
    double *pout_ptr;
    ierr = VecGetArray (pout, &pout_ptr);                                                           CHKERRQ (ierr);
    Vec *v = &phi_vec_[0];
    ierr = VecMTDot (in, np_, v, values);                                                           CHKERRQ (ierr);
    ierr = VecGetOwnershipRange (pout, &low, &high);                                                CHKERRQ (ierr);
    for (int i = low; i < high; i++) {
        pout_ptr[i - low] = values[i];
        // ierr = VecSetValues (pout, 1, &i, &values[i], INSERT_VALUES);                               CHKERRQ (ierr);
    }
    ierr = VecRestoreArray (pout, &pout_ptr);                                                       CHKERRQ (ierr);
    // ierr = VecAssemblyBegin (pout);                                                                 CHKERRQ (ierr);
    // ierr = VecAssemblyEnd (pout);                                                                   CHKERRQ (ierr);

    PetscFunctionReturn (0);
}

int checkTumorExistence (int64_t x, int64_t y, int64_t z, double radius, double *data, std::shared_ptr<NMisc> n_misc) {
    int flag, num_tumor, cube_interior;
    num_tumor = 0;
    cube_interior = 0;
    double distance;
    double threshold = 0.2;

    int64_t ptr;
    for (int i = x - radius; i <= x + radius; i++) 
        for (int j = y - radius; j <= y + radius; j++)
            for (int k = z - radius; k <= z + radius; k++) {
                if (i < 0 || j < 0 || k < 0) continue;
                if (i >= n_misc->isize_[0] || 
                    j >= n_misc->isize_[1] ||
                    k >= n_misc->isize_[2]) continue;
                    
                distance = sqrt ((i - x) * (i - x) + 
                                 (j - y) * (j - y) +
                                 (k - z) * (k - z));
                if (distance < radius) {
                    ptr = i * n_misc->isize_[1] * n_misc->isize_[2] + j * n_misc->isize_[2] + k;
                    if (data[ptr] > threshold) {
                        num_tumor++;
                    }
                }
                cube_interior++;
            }
    if (num_tumor > 0. * cube_interior)   
        flag = 1;
    else
        flag = 0;
    return flag;
}

PetscErrorCode Phi::setGaussians (Vec data, std::shared_ptr<MatProp> mat_prop) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PCOUT << "----- Bounding box not set: Phis set to match data -----" << std::endl;

    double twopi = 2.0 * M_PI;
    int64_t X, Y, Z;
    double hx = twopi / n_misc_->n_[0], hy = twopi / n_misc_->n_[1], hz = twopi / n_misc_->n_[2];
    sigma_ = 2.0 * hx;

    double sigma_smooth = 2.0 * M_PI / n_misc_->n_[0];
    spacing_factor_ = 2.0;
    n_misc_->phi_spacing_factor_ = spacing_factor_;
    double space = spacing_factor_ * sigma_ / hx;

    int flag = 0;
    np_ = 0;
    std::vector<double> center;

    double *data_ptr;
    ierr = VecGetArray (data, &data_ptr);                                      CHKERRQ (ierr);

    PCOUT << "checkTumorExistence" << std::endl;

    for (int x = 0; x < n_misc_->isize_[0]; x += space)
        for (int y = 0; y < n_misc_->isize_[1]; y += space)
            for (int z = 0; z < n_misc_->isize_[2]; z += space) {
                X = n_misc_->istart_[0] + x;
                Y = n_misc_->istart_[1] + y;
                Z = n_misc_->istart_[2] + z;

                flag = checkTumorExistence (x, y, z, sigma_ / hx, data_ptr, n_misc_);

                if (flag == 1) {
                    np_++;
                    center.push_back (X * hx);
                    center.push_back (Y * hy);
                    center.push_back (Z * hz);
                }
            }    

    ierr = VecRestoreArray (data, &data_ptr);                                   CHKERRQ (ierr);


    int np_global;
    MPI_Allreduce (&np_, &np_global, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD); 

    std::vector<int> center_size, displs, rcount;
    std::vector<double> center_global;
    center_size.resize (nprocs);
    displs.resize (nprocs);
    rcount.resize (nprocs);
    int size = center.size();
    center_global.resize (3 * np_global);
    MPI_Allgather (&size, 1, MPI_INT, &center_size[0], 1, MPI_INT, PETSC_COMM_WORLD);

    displs[0] = 0;
    rcount[0] = center_size[0];
    for (int i = 1; i < nprocs; i++) {
        displs[i] = displs[i - 1] + center_size[i - 1];
        rcount[i] = center_size[i];
    }
    MPI_Allgatherv (&center[0], center.size(), MPI_DOUBLE, &center_global[0], &rcount[0], &displs[0], MPI_DOUBLE, PETSC_COMM_WORLD);

    np_ = np_global;
    n_misc_->np_ = np_;
    PCOUT << " ----- NP: " << np_ << " ------" << std::endl;

    Vec v;
    phi_vec_.clear();   //Clear any previously set phi_vec_
    for (int i = 0; i < np_; i++) {
        ierr = VecCreate (PETSC_COMM_WORLD, &v);                                CHKERRQ (ierr);
        ierr = VecSetSizes (v, n_misc_->n_local_, n_misc_->n_global_);          CHKERRQ (ierr);
        ierr = VecSetFromOptions (v);                                           CHKERRQ (ierr);
        ierr = VecSet (v, 0);                                                   CHKERRQ (ierr);
        phi_vec_.push_back (v);
    }

    double *phi_ptr;
    Vec all_phis;
    ierr = VecDuplicate (phi_vec_[0], &all_phis);                               CHKERRQ (ierr);
    ierr = VecSet (all_phis, 0);                                                CHKERRQ (ierr);

    for (int i = 0; i < np_; i++) {
        ierr = VecGetArray (phi_vec_[i], &phi_ptr);                             CHKERRQ (ierr);
        initialize (phi_ptr, n_misc_, &center_global[3 * i]);
        ierr = VecRestoreArray (phi_vec_[i], &phi_ptr);                         CHKERRQ (ierr);

        ierr = VecPointwiseMult (phi_vec_[i], mat_prop->filter_, phi_vec_[i]);  CHKERRQ (ierr);

        if (n_misc_->testcase_ == BRAIN) {  //BRAIN
            ierr = VecGetArray (phi_vec_[i], &phi_ptr);                             CHKERRQ (ierr);
            ierr = weierstrassSmoother (phi_ptr, phi_ptr, n_misc_, sigma_smooth);
            ierr = VecRestoreArray (phi_vec_[i], &phi_ptr);                         CHKERRQ (ierr);
        }

        ierr = VecAXPY (all_phis, 1.0, phi_vec_[i]);                            CHKERRQ (ierr);
    }
    if(n_misc_->writeOutput_) {
        dataOut (all_phis, n_misc_, "results/phiNew.nc");
    }
    PetscFunctionReturn (0);
}

Phi::~Phi () {
    PetscErrorCode ierr = 0;
    for (int i = 0; i < np_; i++) {
        ierr = VecDestroy (&phi_vec_[i]);
    }
}
