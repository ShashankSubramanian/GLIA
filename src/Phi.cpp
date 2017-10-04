#include "Phi.h"

Phi::Phi (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;

    np_ = n_misc->np_;
    n_local_ = n_misc->n_local_;

    phi_vec_ = (Vec *) malloc (sizeof (Vec *) * np_);

    ierr = VecCreate (PETSC_COMM_WORLD, &phi_vec_[0]);
    ierr = VecSetSizes (phi_vec_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (phi_vec_[0]);

    for (int i = 1; i < np_; i++) {
        ierr = VecDuplicate (phi_vec_[0], &phi_vec_[i]);
    }

    for (int i = 0; i < np_; i++) {
        ierr = VecSet (phi_vec_[i], 0);
    }
}

PetscErrorCode Phi::setValues (std::array<double, 3>& user_cm, double sigma, double spacing_factor, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
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
    int h = round (std::pow (np_, 1.0 / 3.0));

    double space[3];

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
                        ptr += 3;
                    }
                 }
    }

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
    ierr = VecMTDot (in, np_, phi_vec_, values);                                                    CHKERRQ (ierr);
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

Phi::~Phi () {
    PetscErrorCode ierr = 0;
    for (int i = 0; i < np_; i++) {
        ierr = VecDestroy (&phi_vec_[i]);
    }
}
