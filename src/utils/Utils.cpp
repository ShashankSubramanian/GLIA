#include "Utils.h"

// global variable
bool DISABLE_VERBOSE = false;

VecField::VecField(int nl, int ng) {
  PetscErrorCode ierr = 0;
  ierr = VecCreate(PETSC_COMM_WORLD, &x_);
  ierr = VecSetSizes(x_, nl, ng);
  ierr = setupVec(x_);
  ierr = VecSet(x_, 0.);

  ierr = VecDuplicate(x_, &y_);
  ierr = VecDuplicate(x_, &z_);
  ierr = VecSet(y_, 0.);
  ierr = VecSet(z_, 0.);
}

PetscErrorCode VecField::copy(std::shared_ptr<VecField> field) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecCopy(field->x_, x_); CHKERRQ(ierr);
  ierr = VecCopy(field->y_, y_); CHKERRQ(ierr);
  ierr = VecCopy(field->z_, z_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::set(ScalarType scalar) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecSet(x_, scalar); CHKERRQ(ierr);
  ierr = VecSet(y_, scalar); CHKERRQ(ierr);
  ierr = VecSet(z_, scalar); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::scale(ScalarType scalar) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecScale(x_, scalar); CHKERRQ(ierr);
  ierr = VecScale(y_, scalar); CHKERRQ(ierr);
  ierr = VecScale(z_, scalar); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::getComponentArrays(ScalarType *&x_ptr, ScalarType *&y_ptr, ScalarType *&z_ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

#ifdef CUDA
  ierr = VecCUDAGetArrayReadWrite(x_, &x_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(y_, &y_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(z_, &z_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(x_, &x_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(y_, &y_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(z_, &z_ptr); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode vecGetArray(Vec x, ScalarType **x_ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

#ifdef CUDA
  ierr = VecCUDAGetArrayReadWrite(x, x_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(x, x_ptr); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode vecRestoreArray(Vec x, ScalarType **x_ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

#ifdef CUDA
  ierr = VecCUDARestoreArrayReadWrite(x, x_ptr); CHKERRQ(ierr);
#else
  ierr = VecRestoreArray(x, x_ptr); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::restoreComponentArrays(ScalarType *&x_ptr, ScalarType *&y_ptr, ScalarType *&z_ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

#ifdef CUDA
  ierr = VecCUDARestoreArrayReadWrite(x_, &x_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(y_, &y_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(z_, &z_ptr); CHKERRQ(ierr);
#else
  ierr = VecRestoreArray(x_, &x_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(y_, &y_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(z_, &z_ptr); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::computeMagnitude(Vec magnitude) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *mag_ptr, *x_ptr, *y_ptr, *z_ptr;
  int sz;
  ierr = getComponentArrays(x_ptr, y_ptr, z_ptr);
  ierr = VecGetLocalSize(x_, &sz); CHKERRQ(ierr);

#ifdef CUDA
  ierr = VecCUDAGetArrayReadWrite(magnitude, &mag_ptr); CHKERRQ(ierr);
  computeMagnitudeCuda(mag_ptr, x_ptr, y_ptr, z_ptr, sz);
  ierr = VecCUDARestoreArrayReadWrite(magnitude, &mag_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(magnitude, &mag_ptr); CHKERRQ(ierr);
  for (int i = 0; i < sz; i++) {
    mag_ptr[i] = std::sqrt(x_ptr[i] * x_ptr[i] + y_ptr[i] * y_ptr[i] + z_ptr[i] * z_ptr[i]);
  }
  ierr = VecRestoreArray(magnitude, &mag_ptr); CHKERRQ(ierr);
#endif

  ierr = restoreComponentArrays(x_ptr, y_ptr, z_ptr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::setIndividualComponents(Vec x_in) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr, *y_ptr, *z_ptr, *in_ptr;
  int local_size = 0;
  ierr = VecGetLocalSize(x_in, &local_size); CHKERRQ(ierr);
  ierr = getComponentArrays(x_ptr, y_ptr, z_ptr);

#ifdef CUDA
  ierr = VecCUDAGetArrayReadWrite(x_in, &in_ptr); CHKERRQ(ierr);
  cudaMemcpy(x_ptr, in_ptr, sizeof(ScalarType) * local_size / 3, cudaMemcpyDeviceToDevice);
  cudaMemcpy(y_ptr, &in_ptr[local_size / 3], sizeof(ScalarType) * local_size / 3, cudaMemcpyDeviceToDevice);
  cudaMemcpy(z_ptr, &in_ptr[2 * local_size / 3], sizeof(ScalarType) * local_size / 3, cudaMemcpyDeviceToDevice);
  ierr = VecCUDARestoreArrayReadWrite(x_in, &in_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(x_in, &in_ptr); CHKERRQ(ierr);
  for (int i = 0; i < local_size / 3; i++) {
    x_ptr[i] = in_ptr[i];
    y_ptr[i] = in_ptr[i + local_size / 3];
    z_ptr[i] = in_ptr[i + 2 * local_size / 3];
  }
  ierr = VecRestoreArray(x_in, &in_ptr); CHKERRQ(ierr);
#endif

  ierr = restoreComponentArrays(x_ptr, y_ptr, z_ptr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VecField::getIndividualComponents(Vec x_in) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr, *y_ptr, *z_ptr, *in_ptr;
  int local_size = 0;
  ierr = VecGetLocalSize(x_in, &local_size); CHKERRQ(ierr);
  ierr = getComponentArrays(x_ptr, y_ptr, z_ptr);

#ifdef CUDA
  ierr = VecCUDAGetArrayReadWrite(x_in, &in_ptr); CHKERRQ(ierr);
  cudaMemcpy(in_ptr, x_ptr, sizeof(ScalarType) * local_size / 3, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&in_ptr[local_size / 3], y_ptr, sizeof(ScalarType) * local_size / 3, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&in_ptr[2 * local_size / 3], z_ptr, sizeof(ScalarType) * local_size / 3, cudaMemcpyDeviceToDevice);
  ierr = VecCUDARestoreArrayReadWrite(x_in, &in_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(x_in, &in_ptr); CHKERRQ(ierr);
  for (int i = 0; i < local_size / 3; i++) {
    in_ptr[i] = x_ptr[i];
    in_ptr[i + local_size / 3] = y_ptr[i];
    in_ptr[i + 2 * local_size / 3] = z_ptr[i];
  }
  ierr = VecRestoreArray(x_in, &in_ptr); CHKERRQ(ierr);
#endif

  ierr = restoreComponentArrays(x_ptr, y_ptr, z_ptr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode tuMSG(std::string msg, int size) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  std::string color = "\x1b[1;34m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode tuMSGstd(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[37m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode tuMSGwarn(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[1;31m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode _tuMSG(std::string msg, std::string color, int size) {
  PetscErrorCode ierr = 0;
  std::stringstream ss;
  PetscFunctionBegin;

  if(!DISABLE_VERBOSE) {
    int procid, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    ss << std::left << std::setw(size) << msg;
    msg = color + "[ " + ss.str() + "]\x1b[0m\n";
    // msg = "\x1b[1;34;40m[ "  + ss.str() + "]\x1b[0m\n";

    // display message
    ierr = PetscPrintf(PETSC_COMM_WORLD, msg.c_str()); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode TumorStatistics::print() {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  std::stringstream s;
  ierr = tuMSG("---- statistics ----"); CHKERRQ(ierr);
  s << std::setw(8) << "     " << std::setw(8) << " #state " << std::setw(8) << " #adj " << std::setw(8) << " #obj " << std::setw(8) << " #grad " << std::setw(8) << " #hess ";
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str(std::string());
  s.clear();
  s << std::setw(8) << " curr:" << std::setw(8) << nb_state_solves << std::setw(8) << nb_adjoint_solves << std::setw(8) << nb_obj_evals << std::setw(8) << nb_grad_evals << std::setw(8)
    << nb_hessian_evals;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str(std::string());
  s.clear();
  s << std::setw(8) << " acc: " << std::setw(8) << nb_state_solves + nb_state_solves_acc << std::setw(8) << nb_adjoint_solves + nb_adjoint_solves_acc << std::setw(8) << nb_obj_evals + nb_obj_evals_acc
    << std::setw(8) << nb_grad_evals + nb_grad_evals_acc << std::setw(8) << nb_hessian_evals + nb_hessian_evals_acc;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str(std::string());
  s.clear();
  ierr = tuMSG("--------"); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

/* definition of tumor assert */
void __TU_assert(const char *expr_str, bool expr, const char *file, int line, const char *msg) {
  if (!expr) {
    std::cerr << "Assert failed:\t" << msg << "\n"
              << "Expected:\t" << expr_str << "\n"
              << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
  }
}

static bool isLittleEndian() {
  uint16_t number = 0x1;
  uint8_t *numPtr = (uint8_t *)&number;
  return (numPtr[0] == 1);
}

PetscErrorCode printVecBounds(Vec c, std::string str) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream s;
  ScalarType max, min;
  ierr = VecMax(c, NULL, &max); CHKERRQ(ierr);
  ierr = VecMin(c, NULL, &min); CHKERRQ(ierr);
  ScalarType tol = 0.;
  s << "  " << str << " bounds: max = " << max << ", min = " << min;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  PetscFunctionReturn(ierr);
}

// Hoyer measure for sparsity of a vector
PetscErrorCode vecSparsity(Vec x, ScalarType &sparsity) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int size;
  ierr = VecGetSize(x, &size); CHKERRQ(ierr);
  ScalarType norm_1, norm_inf;
  ierr = VecNorm(x, NORM_1, &norm_1); CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm_inf); CHKERRQ(ierr);

  if (norm_inf == 0) {
    sparsity = 1.0;
    PetscFunctionReturn(ierr);
  }

  sparsity = (size - (norm_1 / norm_inf)) / (size - 1);

  PetscFunctionReturn(ierr);
}


PetscErrorCode vecSign(Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr;
  int size;
  ierr = VecGetSize(x, &size); CHKERRQ(ierr);
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);

  for (int i = 0; i < size; i++) {
    if (x_ptr[i] > 0)
      x_ptr[i] = 1.0;
    else if (x_ptr[i] == 0)
      x_ptr[i] = 0.0;
    else
      x_ptr[i] = -1.0;
  }

  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode hardThreshold(Vec x, int sparsity_level, int sz, std::vector<int> &support, int &nnz) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  nnz = 0;

  std::priority_queue<std::pair<PetscReal, int>> q;
  ScalarType *x_ptr;
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
  for (int i = 0; i < sz; i++) {
    q.push(std::pair<PetscReal, int>(x_ptr[i], i));  // Push values and idxes into a priiority queue
  }

  ScalarType tol = 0.0;  // 1E-10; // tolerance for specifying if signal is present: We don't need to add signal components which
                         // are (almost)zero to the support
  for (int i = 0; i < sparsity_level; i++) {
    if (std::abs(q.top().first) > tol) {
      nnz++;  // keeps track of how many non-zero (important) components of the signal there are
      support.push_back(q.top().second);
    } else {  // if top of the queue is not greater than tol, we are done since none of the elements
              // below it will every be greater than tol
      break;
    }
    q.pop();
  }

  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode hardThreshold(Vec x, int sparsity_level, int sz, std::vector<int> &support, std::vector<int> labels, std::vector<ScalarType> weights, int &nnz, int num_components, double thresh_component_weight, bool double_mode) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_rank(PETSC_COMM_WORLD, &procid);
  MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

  std::stringstream ss;
  nnz = 0;
  std::priority_queue<std::pair<PetscReal, int>> q;
  ScalarType *x_ptr;
  ScalarType tol = 0.0;  // 1E-10; // tolerance for specifying if signal is present: We don't need to add signal components which
                         // are (almost)zero to the support
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);

  if (double_mode)
    sparsity_level /= 2;

  std::vector<int> component_sparsity;
  int fin_spars;
  int sparsity;
  int ncc = 0;
  int ncc_t = 0;
  for (auto w : weights)
    if (w >= thresh_component_weight) ncc++;
  double thres_reduce = thresh_component_weight / 10.0; 
  for (auto w : weights)
    if (w < thresh_component_weight && w >= thres_reduce) ncc_t++;

  for (int nc = 0; nc < num_components; nc++) {
    if (nc != num_components - 1) {
      // sparsity level in total is 5 * #nc (number components); if the sparsity per component is 5.
      // every component gets at 3 degrees of freedom, the remaining 2 * #nc degrees of freedom are distributed based on component weight
      // sparsity = (weights[nc] > thresh_component_weight) ? (3 + std::floor(weights[nc] * (sparsity_level - 3 * ncc - (num_components - ncc)))) : 1;
      if (weights[nc] >= thresh_component_weight) {
        sparsity = 3 + std::floor(weights[nc] * (sparsity_level - 3 * ncc - ncc_t));
      } else if (weights[nc] < thresh_component_weight && weights[nc] >= thres_reduce) {
        sparsity = 1;
      } else {
        sparsity = 0;
      }
      
      if (double_mode)
      	sparsity *= 2;
      component_sparsity.push_back(sparsity);
      ss << "sparsity of component " << nc << ": " << component_sparsity.at(nc);
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    } else {  // last component is the remaining support
      int used = 0;
      for (auto x : component_sparsity) {
        used += x;
      }
      if (double_mode)
      	sparsity_level *= 2;
      fin_spars = sparsity_level - used;
      component_sparsity.push_back(fin_spars);
      ss << "sparsity of component " << nc << ": " << fin_spars;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }

    for (int i = 0; i < sz; i++) {
      if (labels[i] == nc + 1)                           // push the current components into the priority queue
        q.push(std::pair<PetscReal, int>(x_ptr[i], i));  // Push values and idxes into a priiority queue
    }

    for (int i = 0; i < component_sparsity[nc]; i++) {
      if (q.size() > 0) {
        if (std::abs(q.top().first) > tol) {
          nnz++;  // keeps track of how many non-zero (important) components of the signal there are
          support.push_back(q.top().second);
        } else {  // if top of the queue is not greater than tol, we are done since none of the elements
                  // below it will ever be greater than tol
          ss << "  ... some DOF not used in comp " << nc << "; p_i = " << std::abs(q.top().first) << " < " << tol << " = tolerance";
          ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
          ss.str("");
          ss.clear();
          break;
        }
        q.pop();
      } else {
        ss << "  ... insufficient DOF selected in comp. " << nc << "; insufficient value in queue (component weight, w=" << weights[nc] << "). ";
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
        ss.str("");
        ss.clear();
        break;
      }
    }
    q = std::priority_queue<std::pair<PetscReal, int>>();  // reset the queue
  }

  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

ScalarType myDistance(ScalarType *c1, ScalarType *c2) { return std::sqrt((c1[0] - c2[0]) * (c1[0] - c2[0]) + (c1[1] - c2[1]) * (c1[1] - c2[1]) + (c1[2] - c2[2]) * (c1[2] - c2[2])); }

PetscErrorCode computeCenterOfMass(Vec x, int *isize, int *istart, ScalarType *h, ScalarType *cm) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int64_t ptr_idx;
  ScalarType X, Y, Z;
  ScalarType *data_ptr;
  ScalarType com[3], sum;
  for (int i = 0; i < 3; i++) com[i] = 0.;
  sum = 0;
  ierr = VecGetArray(x, &data_ptr); CHKERRQ(ierr);
  for (int x = 0; x < isize[0]; x++) {
    for (int y = 0; y < isize[1]; y++) {
      for (int z = 0; z < isize[2]; z++) {
        X = h[0] * (istart[0] + x);
        Y = h[1] * (istart[1] + y);
        Z = h[2] * (istart[2] + z);

        ptr_idx = x * isize[1] * isize[2] + y * isize[2] + z;
        com[0] += (data_ptr[ptr_idx] * X);
        com[1] += (data_ptr[ptr_idx] * Y);
        com[2] += (data_ptr[ptr_idx] * Z);

        sum += data_ptr[ptr_idx];
      }
    }
  }

  ScalarType sm;
  MPI_Allreduce(&com, cm, 3, MPIType, MPI_SUM, PETSC_COMM_WORLD);
  MPI_Allreduce(&sum, &sm, 1, MPIType, MPI_SUM, PETSC_COMM_WORLD);

  for (int i = 0; i < 3; i++) {
    cm[i] /= sm;
  }

  ierr = VecRestoreArray(x, &data_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode setupVec(Vec x, int type) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

#ifdef CUDA
  if (type == SEQ)
    ierr = VecSetType(x, VECSEQCUDA);
  else
    ierr = VecSetType(x, VECCUDA);
#else
  ierr = VecSetFromOptions(x);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode vecSum(Vec x, PetscScalar *val) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

#ifdef CUDA
    ScalarType *x_ptr;
    int sz;
    ierr = vecGetArray (x, &x_ptr); CHKERRQ (ierr);
    ierr = VecGetSize (x, &sz); CHKERRQ (ierr);
    vecSumCuda (x_ptr, val, sz);
    ierr = vecRestoreArray (x, &x_ptr); CHKERRQ (ierr);
#else
  ierr = VecSum(x, val); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode vecMax(Vec x, PetscInt *p, PetscReal *val) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  // TODO: thrust max fails in frontera rtx queue somtimes; not sure why. revist if it happens again
  // cause NaNs to appear after vecMax
#ifdef CUDA
    ScalarType *x_ptr;
    int sz;
    ierr = vecGetArray (x, &x_ptr); CHKERRQ (ierr);
    ierr = VecGetSize (x, &sz); CHKERRQ (ierr);
    vecMaxCuda (x_ptr, p, val, sz);
    ierr = vecRestoreArray (x, &x_ptr); CHKERRQ (ierr);
#else
  ierr = VecMax(x, p, val); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(ierr);
}


PetscErrorCode createEdemaBasedObservationMask(Vec mask, Vec tc, Vec ed, double lambda, int nl, std::vector<int> &labels) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if(mask == nullptr) {ierr = tuMSGwarn(" Error: Observation mask is nullptr."); CHKERRQ(ierr); PetscFunctionReturn(1);}
  if(tc == nullptr) {ierr = tuMSGwarn(" Error: Tumor core is nullptr."); CHKERRQ(ierr); PetscFunctionReturn(1);}
  if(ed == nullptr) {ierr = tuMSGwarn(" Error: Edema is nullptr."); CHKERRQ(ierr); PetscFunctionReturn(1);}
  
  ScalarType *tc_ptr, *ed_ptr, *mask_ptr; 
  int tc_label  = (labels[4] > 0) ? labels[4] : -1;
  int ed_label  = (labels[7] > 0) ? labels[7] : -1;
  
  ierr = VecSet(mask, 0.0); CHKERRQ(ierr);
  ierr = VecGetArray(mask, &mask_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tc, &tc_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(ed, &ed_ptr); CHKERRQ(ierr);
  for (int i = 0; i < nl; i++) {
    // set observation operator to OBS = 1[TC] + lambda*1[B/WT]
    mask_ptr[i] = (tc_ptr[i] > 0.99) ? 1 : (ed_ptr[i] > 0.99) ? 0 : lambda;
  }
  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tc, &tc_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(ed, &ed_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode splitSegmentation(Vec seg, Vec wm, Vec gm, Vec vt, Vec csf, Vec tu, Vec ed, int nl, std::vector<int> &labels) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if(seg == nullptr) {ierr = tuMSGwarn(" Error: Segmentation is null."); CHKERRQ(ierr); PetscFunctionReturn(1);}

  ScalarType *gm_ptr, *wm_ptr, *vt_ptr, *tu_ptr, *csf_ptr, *seg_ptr, *ed_ptr;
  int wm_label  = labels[0], gm_label = labels[1], vt_label = labels[2];
  int csf_label = (labels[3] > 0) ? labels[3] : -1;
  int tc_label  = (labels[4] > 0) ? labels[4] : -1;
  int nec_label = (labels[5] > 0) ? labels[5] : -1;
  int en_label  = (labels[6] > 0) ? labels[6] : -1;
  int ed_label  = (labels[7] > 0) ? labels[7] : -1;
  ierr = VecGetArray(seg, &seg_ptr); CHKERRQ(ierr);
  ierr = VecSet(wm, 0.0); CHKERRQ(ierr);
  ierr = VecSet(gm, 0.0); CHKERRQ(ierr);
  ierr = VecSet(vt, 0.0); CHKERRQ(ierr);
  ierr = VecGetArray(wm, &wm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(gm, &gm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(vt, &vt_ptr); CHKERRQ(ierr);
  if(csf != nullptr) {
    ierr = VecSet(csf, 0.0); CHKERRQ(ierr);
    ierr = VecGetArray(csf, &csf_ptr); CHKERRQ(ierr);
  }
  if(tu  != nullptr) {
    ierr = VecSet(tu, 0.0); CHKERRQ(ierr);
    ierr = VecGetArray(tu, &tu_ptr); CHKERRQ(ierr);
  }
  if(ed  != nullptr) {
    ierr = VecSet(ed, 0.0); CHKERRQ(ierr);
    ierr = VecGetArray(ed, &ed_ptr); CHKERRQ(ierr);
  }


  for (int i = 0; i < nl; i++) {
    if(wm_label > 0) {wm_ptr[i] = (seg_ptr[i] ==  wm_label) ? 1 : wm_ptr[i];}
    if(gm_label > 0) {gm_ptr[i] = (seg_ptr[i] ==  gm_label) ? 1 : gm_ptr[i];}
    if(vt_label > 0) {vt_ptr[i] = (seg_ptr[i] ==  vt_label) ? 1 : vt_ptr[i];}
    if(csf != nullptr) {
      if(csf_label > 0) {csf_ptr[i] = (seg_ptr[i] ==  csf_label) ? 1 : csf_ptr[i];}
    }
    if(tu != nullptr) {
      if(tc_label > 0) {
        wm_ptr[i] = (seg_ptr[i] == tc_label) ? 1 : wm_ptr[i];
        tu_ptr[i] = (seg_ptr[i] == tc_label) ? 1 : tu_ptr[i];
      } else if (nec_label > 0 && en_label > 0) {
        wm_ptr[i] = (seg_ptr[i] == nec_label || seg_ptr[i] == en_label) ? 1 : wm_ptr[i];
        tu_ptr[i] = (seg_ptr[i] == nec_label || seg_ptr[i] == en_label) ? 1 : tu_ptr[i];
      }
      if(ed_label > 0) {
        wm_ptr[i] = (seg_ptr[i] == ed_label) ? 1 : wm_ptr[i];
        if(ed  != nullptr) {
          ed_ptr[i] = (seg_ptr[i] == ed_label) ? 1 : ed_ptr[i];
	      }
      }
    }
  }
  ierr = VecRestoreArray(seg, &seg_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(wm, &wm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(gm, &gm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(vt, &vt_ptr); CHKERRQ(ierr);
  if(csf != nullptr) {ierr = VecRestoreArray(csf, &csf_ptr); CHKERRQ(ierr);}
  if(tu  != nullptr) {ierr = VecRestoreArray(tu, &tu_ptr); CHKERRQ(ierr);}
  if(ed  != nullptr) {ierr = VecRestoreArray(ed, &ed_ptr); CHKERRQ(ierr);}

  PetscFunctionReturn(ierr);
}


PetscErrorCode computeDice(Vec in, Vec truth, ScalarType &dice) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  if (truth == nullptr || in == nullptr) {
    dice = -1;
    PetscFunctionReturn(ierr);
  }

  PetscInt sz;
  ierr = VecGetSize(truth, &sz); CHKERRQ(ierr);

  ScalarType dot, sum_in, sum_truth;
  ierr = VecDot(in, truth, &dot); CHKERRQ(ierr);
#ifdef CUDA
  cublasStatus_t status;
  cublasHandle_t handle;
  // cublas for vec scale
  PetscCUBLASGetHandle(&handle);
  ScalarType *truth_ptr, *in_ptr;
  ierr = vecGetArray(truth, &truth_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(in, &in_ptr); CHKERRQ(ierr);
  status = cublasSum(handle, sz, truth_ptr, 1, &sum_truth);
  cublasCheckError(status);
  status = cublasSum(handle, sz, in_ptr, 1, &sum_in);
  cublasCheckError(status);
  ierr = vecRestoreArray(truth, &truth_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(in, &in_ptr); CHKERRQ(ierr);
#else
  ierr = VecSum(truth, &sum_truth); CHKERRQ(ierr);
  ierr = VecSum(in, &sum_in); CHKERRQ(ierr);
#endif
  ScalarType denom = sum_truth + sum_in;
  denom = (denom > 0) ? denom : 1E-5; // some small value to avoid division by zero
  dice = 2. * dot / denom;

  PetscFunctionReturn(ierr);
}

PetscErrorCode computeVolume(Vec x, ScalarType measure, ScalarType *vol, ScalarType *sum) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = vecSum(x, sum); CHKERRQ(ierr);
  (*vol) = (*sum) * measure;

  PetscFunctionReturn(ierr);
}

PetscErrorCode vecSort (Vec xin, Vec xout, int64_t sz) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecCopy(xin, xout); CHKERRQ(ierr);
  ScalarType *x_ptr, *y_ptr;
  ierr = vecGetArray(xout, &y_ptr); CHKERRQ(ierr);
#ifdef CUDA
  vecSortCuda(y_ptr, sz); 
#else
  // TODO implement parallel merge sort
  // return -1 for now
  ierr = VecSet(xout, -1); CHKERRQ(ierr);
#endif
  ierr = vecRestoreArray(xout, &y_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode computeQuantile(Vec x, Vec temp, ScalarType *val, ScalarType quantile) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  PetscInt sz;
  ierr = VecGetSize(x, &sz); CHKERRQ(ierr);
  ScalarType *x_ptr;
  ierr = vecSort(x, temp, sz); CHKERRQ(ierr);
  ierr = vecGetArray(temp, &x_ptr); CHKERRQ(ierr);
  int index = std::floor(quantile * sz);
#ifdef CUDA
  cudaMemcpy(val, &x_ptr[index], sizeof(ScalarType), cudaMemcpyDeviceToHost);
#else
  (*val) = x_ptr[index];
#endif
  ierr = vecRestoreArray(temp, &x_ptr); CHKERRQ(ierr); 

  PetscFunctionReturn(ierr);
}

ScalarType computeDeterminant(std::array<ScalarType, 9> matrix) {
  // computes determinant of a 3x3 matrix
  ScalarType *a = matrix.data();
  return a[0] * (a[4]*a[8] - a[7]*a[5]) - a[1] * (a[3]*a[8] - a[6]*a[5]) + a[2] * (a[3]*a[7] - a[6]*a[4]);
}

void matMult(ScalarType **c, ScalarType **a, ScalarType **b, int m, int n, int o) {
  // multiplies mxn and nxo
  for (int i = 0; i < m; i++)
    for (int j = 0; j < o; j++)
      for (int k = 0; k < n; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
}

std::array<ScalarType, 9> computeStrainTensor(std::array<ScalarType, 9> F) {
  // computes the strain tensor given a deformation gradient
  // E = 0.5* (FTF - I); or 0.5 * (FT + F) - I if high order terms are dropped
  std::array<ScalarType, 9> E, Ft;
  ScalarType *e = E.data();
  ScalarType *f = F.data();
  ScalarType *ft = Ft.data();

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      ft[i*3+j] = f[j*3+i];
    }

//  matMult(e, ft, f, 3, 3, 3);
  for (int i = 0; i < 9; i++) e[i] = 0.5 * (ft[i] + f[i]);
  e[0] -= 1;
  e[4] -= 1;
  e[8] -= 1;

//  for (auto& x: E) x *= 0.5;

  return E;
}

std::array<ScalarType, 9> computeStressTensor(std::array<ScalarType, 9> E, ScalarType mu, ScalarType lam) {
  // computes stress tensor
  // S = lam tr(E) I + 2 mu E
  std::array<ScalarType, 9> S;
  ScalarType *e = E.data();
  ScalarType *s = S.data();
  ScalarType trE = e[0] + e[4] + e[8]; 

  for (int i = 0; i < 9; i++) s[i] = 2 * mu * e[i];
  s[0] += lam * trE;
  s[4] += lam * trE;
  s[8] += lam * trE;

  return S;
}

 PetscErrorCode setupKSPEigenvalues(KSP *ksp, Mat *A, Vec *rhs, Vec *sol) {
  // setup a ksp object with some dummy values so that it can be used for eigenvalue computations later
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int size = 3;
  ierr = MatCreate(PETSC_COMM_SELF, A); CHKERRQ(ierr);
  ierr = MatSetSizes(*A,size,size,size,size); CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSE); CHKERRQ(ierr);
  ierr = MatSetUp(*A); CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_SELF, ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(*ksp, *A, *A); CHKERRQ(ierr);
  ierr = KSPSetTolerances(*ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetType(*ksp, KSPCG); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(*ksp); CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, 3, rhs); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*rhs); CHKERRQ(ierr);
  ierr = VecSet(*rhs, 0); CHKERRQ(ierr);
  ierr = VecDuplicate(*rhs, sol); CHKERRQ(ierr);
  ierr = VecSet(*sol, 0); CHKERRQ(ierr);
  
  PetscScalar a[9] = {1,0,0,0,1,0,0,0,1}; // gen id matrix
  const int id[3] = {0,1,2};
  ierr = MatSetValues(*A, size, id, size, id, a, INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    
  ierr = KSPSolve(*ksp, *rhs, *sol); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
 }


 PetscErrorCode computeIndicatorFunction(Vec i, Vec x, ScalarType x_star) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr, *i_ptr;
  ScalarType threshold = 1E-3;
  PetscInt size;
  ierr = VecGetSize(x, &size); CHKERRQ(ierr);
  ierr = vecGetArray(i, &i_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(x, &x_ptr); CHKERRQ(ierr);

#ifdef CUDA
  computeIndicatorFunctionCuda(i_ptr, x_ptr, x_star, threshold, size);
#else
  for (int i = 0; i < size; i++) {
    if (PetscAbsReal(x_ptr[i] - x_star) < threshold) {
      i_ptr[i] = 1;
    } else {
      i_ptr[i] = 0;
    }
  }
#endif
  ierr = vecRestoreArray(i, &i_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(x, &x_ptr); CHKERRQ(ierr);


  PetscFunctionReturn(ierr);
 }
