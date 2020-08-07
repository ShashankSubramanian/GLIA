#include "Utils.h"
#include "Phi.h"

VecField::VecField (int nl , int ng) {
	PetscErrorCode ierr = 0;
    ierr = VecCreate (PETSC_COMM_WORLD, &x_);
    ierr = VecSetSizes (x_, nl, ng);
    ierr = VecSetFromOptions (x_);
    ierr = VecSet (x_, 0.);

    ierr = VecDuplicate (x_, &y_);
    ierr = VecDuplicate (x_, &z_);
    ierr = VecDuplicate (x_, &magnitude_);
    ierr = VecSet (y_, 0.);
    ierr = VecSet (z_, 0.);
    ierr = VecSet (magnitude_, 0.);
}


PetscErrorCode VecField::copy (std::shared_ptr<VecField> field) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ierr = VecCopy (field->x_, x_);			CHKERRQ (ierr);
	ierr = VecCopy (field->y_, y_);			CHKERRQ (ierr);
	ierr = VecCopy (field->z_, z_);			CHKERRQ (ierr);

	PetscFunctionReturn (0);
}

PetscErrorCode VecField::getComponentArrays (double *&x_ptr, double *&y_ptr, double *&z_ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ierr = VecGetArray (x_, &x_ptr);		CHKERRQ (ierr);
	ierr = VecGetArray (y_, &y_ptr);		CHKERRQ (ierr);
	ierr = VecGetArray (z_, &z_ptr);		CHKERRQ (ierr);

	PetscFunctionReturn (0);
}


PetscErrorCode VecField::restoreComponentArrays (double *&x_ptr, double *&y_ptr, double *&z_ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ierr = VecRestoreArray (x_, &x_ptr);		CHKERRQ (ierr);
	ierr = VecRestoreArray (y_, &y_ptr);		CHKERRQ (ierr);
	ierr = VecRestoreArray (z_, &z_ptr);		CHKERRQ (ierr);

	PetscFunctionReturn (0);
}

PetscErrorCode VecField::computeMagnitude () {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	double *mag_ptr, *x_ptr, *y_ptr, *z_ptr;
	int sz;
	ierr = VecGetLocalSize (x_, &sz); 				CHKERRQ (ierr);
	ierr = VecGetArray (magnitude_, &mag_ptr);		CHKERRQ (ierr);
	ierr = getComponentArrays (x_ptr, y_ptr, z_ptr);

	for (int i = 0; i < sz; i++) {
		mag_ptr[i] = std::sqrt (x_ptr[i] * x_ptr[i] + y_ptr[i] * y_ptr[i] + z_ptr[i] * z_ptr[i]);
	}

	ierr = VecRestoreArray (magnitude_, &mag_ptr);	CHKERRQ (ierr);
	ierr = restoreComponentArrays (x_ptr, y_ptr, z_ptr);

	PetscFunctionReturn (0);
}

PetscErrorCode VecField::setIndividualComponents (Vec x_in) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

	double *x_ptr, *y_ptr, *z_ptr, *in_ptr;
	int local_size = 0;
	ierr = VecGetLocalSize (x_in, &local_size);		CHKERRQ (ierr);
	ierr = getComponentArrays (x_ptr, y_ptr, z_ptr);
	ierr = VecGetArray (x_in, &in_ptr);			    CHKERRQ (ierr);
	for (int i = 0; i < local_size / 3; i++) {
		x_ptr[i] = in_ptr[i];
		y_ptr[i] = in_ptr[i + local_size / 3];
		z_ptr[i] = in_ptr[i + 2 * local_size / 3];
	}
	ierr = VecRestoreArray (x_in, &in_ptr);			CHKERRQ (ierr);
	ierr = restoreComponentArrays (x_ptr, y_ptr, z_ptr);

	PetscFunctionReturn (0);
}

PetscErrorCode VecField::getIndividualComponents (Vec x_in) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	double *x_ptr, *y_ptr, *z_ptr, *in_ptr;
	int local_size = 0;
	ierr = VecGetLocalSize (x_in, &local_size);		CHKERRQ (ierr);
	ierr = getComponentArrays (x_ptr, y_ptr, z_ptr);
	ierr = VecGetArray (x_in, &in_ptr);			    CHKERRQ (ierr);
	for (int i = 0; i < local_size / 3; i++) {
		in_ptr[i] = x_ptr[i];
		in_ptr[i + local_size / 3] = y_ptr[i];
		in_ptr[i + 2 * local_size / 3] = z_ptr[i];
	}
	ierr = VecRestoreArray (x_in, &in_ptr);			CHKERRQ (ierr);
	ierr = restoreComponentArrays (x_ptr, y_ptr, z_ptr);

	PetscFunctionReturn (0);
}


PetscErrorCode tuMSG(std::string msg, int size) {
	PetscFunctionBegin;
  PetscErrorCode ierr;
  std::string color = "\x1b[1;34m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tuMSGstd(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[37m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tuMSGwarn(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[1;31m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _tuMSG(std::string msg, std::string color, int size) {
    PetscErrorCode ierr = 0;
    std::stringstream ss;
    PetscFunctionBegin;

    int procid, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    ss << std::left << std::setw(size) << msg;
    msg = color+"[ "  + ss.str() + "]\x1b[0m\n";
    //msg = "\x1b[1;34;40m[ "  + ss.str() + "]\x1b[0m\n";

    // display message
    ierr = PetscPrintf(PETSC_COMM_WORLD,msg.c_str()); CHKERRQ(ierr);


    PetscFunctionReturn(0);
}

PetscErrorCode TumorStatistics::print() {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	std::stringstream s;
	ierr = tuMSG ("---- statistics                                                                            ----"); CHKERRQ(ierr);
	s << std::setw(8) << "     " << std::setw(8) << " #state " << std::setw(8) << " #adj " << std::setw(8) << " #obj "  << std::setw(8) << " #grad " << std::setw(8) << " #hess ";
	ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(std::string()); s.clear();
	s << std::setw(8) << " curr:" << std::setw(8) << nb_state_solves     << std::setw(8) << nb_adjoint_solves     << std::setw(8) << nb_obj_evals      << std::setw(8) << nb_grad_evals     << std::setw(8) << nb_hessian_evals;
	ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(std::string()); s.clear();
	s << std::setw(8) << " acc: " << std::setw(8) << nb_state_solves + nb_state_solves_acc << std::setw(8) << nb_adjoint_solves + nb_adjoint_solves_acc << std::setw(8) << nb_obj_evals + nb_obj_evals_acc  << std::setw(8) << nb_grad_evals + nb_grad_evals_acc << std::setw(8) << nb_hessian_evals + nb_hessian_evals_acc;
	ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(std::string()); s.clear();
	ierr = tuMSG ("----                                                                                        ----"); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

void accfft_grad (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, accfft_plan *plan, std::bitset<3> *pXYZ, double *timers) {
	PetscErrorCode ierr = 0;
	double *grad_x_ptr, *grad_y_ptr, *grad_z_ptr, *x_ptr;
	ierr = VecGetArray (grad_x, &grad_x_ptr);
	ierr = VecGetArray (grad_y, &grad_y_ptr);
	ierr = VecGetArray (grad_z, &grad_z_ptr);
	ierr = VecGetArray (x, &x_ptr);

	accfft_grad (grad_x_ptr, grad_y_ptr, grad_z_ptr, x_ptr, plan, pXYZ, timers);

	ierr = VecRestoreArray (grad_x, &grad_x_ptr);
	ierr = VecRestoreArray (grad_y, &grad_y_ptr);
	ierr = VecRestoreArray (grad_z, &grad_z_ptr);
	ierr = VecRestoreArray (x, &x_ptr);
}

void accfft_divergence (Vec div, Vec dx, Vec dy, Vec dz, accfft_plan *plan, double *timers) {
	PetscErrorCode ierr = 0;
	double *div_ptr, *dx_ptr, *dy_ptr, *dz_ptr;
	ierr = VecGetArray (div, &div_ptr);
	ierr = VecGetArray (dx, &dx_ptr);
	ierr = VecGetArray (dy, &dy_ptr);
	ierr = VecGetArray (dz, &dz_ptr);

	accfft_divergence (div_ptr, dx_ptr, dy_ptr, dz_ptr, plan, timers);

	ierr = VecRestoreArray (div, &div_ptr);
	ierr = VecRestoreArray (dx, &dx_ptr);
	ierr = VecRestoreArray (dy, &dy_ptr);
	ierr = VecRestoreArray (dz, &dz_ptr);
}

/* definition of tumor assert */
void __TU_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

static bool isLittleEndian () {
	uint16_t number = 0x1;
	uint8_t *numPtr = (uint8_t*) &number;
	return (numPtr[0] == 1);
}

PetscErrorCode throwErrorMsg(std::string msg, int line, const char *file) {                                                                                                                                                                                                                
    PetscErrorCode ierr = 0;
    std::stringstream ss; 
    std::stringstream ss2;

    PetscFunctionBegin;

    ss2 << file << ":" << line;
    ss << std::setw(98-ss2.str().size()) << std::left << msg << std::right << ss2.str();
    std::string errmsg = "\x1b[31mERROR: " + ss.str() + "\x1b[0m";
    ierr = PetscError(PETSC_COMM_WORLD, __LINE__, PETSC_FUNCTION_NAME, __FILE__, 1, PETSC_ERROR_INITIAL, errmsg.c_str()); CHKERRQ(ierr);

    PetscFunctionReturn(ierr);
}

PetscErrorCode myAssert(bool condition, std::string msg) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
                                                                                                                                                                                                                                                                                           
    if (condition == false) {
        ierr = throwError(msg); CHKERRQ(ierr);
    }   

    PetscFunctionReturn(ierr);
}


PetscErrorCode NCERRQ (int cerr) {
    int rank;
    PetscErrorCode ierr = 0;
    std::stringstream ss;                                                                                                                                                                                                                                                                  
    PetscFunctionBegin;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    if (cerr != NC_NOERR) {
        ss << ncmpi_strerror(cerr);
        ierr = throwError(ss.str()); CHKERRQ(ierr);
    }   

    PetscFunctionReturn(ierr);
}


PetscErrorCode dataIn (double *p_x, std::shared_ptr<NMisc> n_misc, const char *fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0; 
	// get local sizes
  MPI_Offset istart[3], isize[3];
  istart[0] = static_cast<MPI_Offset>(n_misc->istart_[0]);
  istart[1] = static_cast<MPI_Offset>(n_misc->istart_[1]);
  istart[2] = static_cast<MPI_Offset>(n_misc->istart_[2]);

  isize[0] = static_cast<MPI_Offset>(n_misc->isize_[0]);
  isize[1] = static_cast<MPI_Offset>(n_misc->isize_[1]);
  isize[2] = static_cast<MPI_Offset>(n_misc->isize_[2]);
  int ncerr, fileid, ndims, nvars, ngatts, unlimited, varid[1];  
  // open file
  ncerr = ncmpi_open (PETSC_COMM_WORLD, fname, NC_NOWRITE, MPI_INFO_NULL, &fileid);
  ierr = NCERRQ (ncerr);                                              CHKERRQ (ierr);
  // query info about field named "data"
  ncerr = ncmpi_inq (fileid, &ndims, &nvars, &ngatts, &unlimited);
  ierr = NCERRQ (ncerr);                                              CHKERRQ (ierr);
  ncerr = ncmpi_inq_varid (fileid, "data", &varid[0]);
  ierr = NCERRQ(ncerr);                                               CHKERRQ(ierr);
  ncerr = ncmpi_get_vara_all (fileid, varid[0], istart, isize, p_x, n_misc->n_local_, MPI_DOUBLE);
  ierr = NCERRQ (ncerr);                                              CHKERRQ(ierr);                                                                                                                                                                                                                                                   
  ncerr=ncmpi_close(fileid);
  PetscFunctionReturn(0);
}

PetscErrorCode dataIn (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname) {
	double *a_ptr;
	PetscErrorCode ierr;
	ierr = VecGetArray (A, &a_ptr); CHKERRQ(ierr);  
	dataIn (a_ptr, n_misc, fname);
	ierr = VecRestoreArray (A, &a_ptr); CHKERRQ(ierr);  
  PetscFunctionReturn (0);
}

PetscErrorCode dataOut (double *p_x, std::shared_ptr<NMisc> n_misc, const char *fname) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    int ncerr, mode, dims[3], varid[1], nx[3], iscdf5, fileid;
    int nl;
    MPI_Offset istart[3], isize[3];
    MPI_Comm c_comm;
    bool usecdf5 = false;   // CDF-5 is mandatory for large files (>= 2x10^9 cells)

    std::stringstream ss;
    ss << n_misc->writepath_.str().c_str() << fname;
    ierr = myAssert(p_x != NULL, "null pointer"); CHKERRQ(ierr);

    // file creation mode
    mode=NC_CLOBBER;
    if (usecdf5) {
        mode = NC_CLOBBER | NC_64BIT_DATA;
    } else {
        mode = NC_CLOBBER | NC_64BIT_OFFSET;
    }    

    c_comm = n_misc->c_comm_;

    // create netcdf file
    ncerr = ncmpi_create (c_comm, ss.str().c_str(), mode, MPI_INFO_NULL, &fileid);
    ierr = NCERRQ (ncerr); CHKERRQ (ierr);

    nx[0] = n_misc->n_[0];
    nx[1] = n_misc->n_[1];
    nx[2] = n_misc->n_[2];
    nl = n_misc->n_local_;

    // set size
    ncerr = ncmpi_def_dim(fileid, "x", nx[0], &dims[0]);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);
    ncerr = ncmpi_def_dim(fileid, "y", nx[1], &dims[1]);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);
    ncerr = ncmpi_def_dim(fileid, "z", nx[2], &dims[2]);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);

    // define name for output field
    ncerr = ncmpi_def_var(fileid, "data", NC_DOUBLE, 3, dims, &varid[0]);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);

    iscdf5 = usecdf5 ? 1 : 0; 
    ncerr = ncmpi_put_att_int(fileid, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &iscdf5);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);
    ncerr = ncmpi_enddef(fileid);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);

    // get local sizes
    istart[0] = static_cast<MPI_Offset>(n_misc->istart_[0]);
    istart[1] = static_cast<MPI_Offset>(n_misc->istart_[1]);
    istart[2] = static_cast<MPI_Offset>(n_misc->istart_[2]);

    isize[0] = static_cast<MPI_Offset>(n_misc->isize_[0]);
    isize[1] = static_cast<MPI_Offset>(n_misc->isize_[1]);
    isize[2] = static_cast<MPI_Offset>(n_misc->isize_[2]);

    ierr = myAssert(nl == isize[0]*isize[1]*isize[2], "size error"); CHKERRQ(ierr);

    // write data to file
    ncerr = ncmpi_put_vara_all(fileid, varid[0], istart, isize, p_x, nl, MPI_DOUBLE);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);

    // close file
    ncerr = ncmpi_close(fileid);
    ierr = NCERRQ(ncerr); CHKERRQ(ierr);  
    PetscFunctionReturn(0);
}

PetscErrorCode dataOut (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname) {
	double *a_ptr;
	PetscErrorCode ierr;
	ierr = VecGetArray (A, &a_ptr); CHKERRQ(ierr);  
	dataOut (a_ptr, n_misc, fname);
	ierr = VecRestoreArray (A, &a_ptr); CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// writeCheckpoint /////////////////////////////////// ###
PetscErrorCode writeCheckpoint(Vec p, std::shared_ptr<Phi> phi, std::string path, std::string suffix) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  // write p vector to bin
  std::string fname_p = path + "p-rec-" + suffix + ".bin";
  std::string fname_p_txt = path + "p-rec-" + suffix + ".txt";
  writeBIN(p, fname_p);

  // write p vector to txt
  double *p_ptr;
  ierr = VecGetArray (p, &p_ptr);
  std::stringstream pvis;
  pvis <<" p = ["<<std::endl;
  for (int ptr = 0; ptr < phi->np_; ++ptr) {
      pvis << " " << p_ptr[ptr] << std::endl;
  }
  ierr = VecRestoreArray (p, &p_ptr);
  pvis << "];"<<std::endl;

  // write Gaussian centers
  std::string fname_phi = path + "phi-mesh-" + suffix + ".txt";
  std::stringstream phivis;
  phivis <<" sigma = "<<phi->sigma_<<", spacing = "<<phi->spacing_factor_ * phi->sigma_<<std::endl;
  phivis <<" centers = ["<<std::endl;
  for (int ptr = 0; ptr < 3 * phi->np_; ptr += 3) {
      phivis << " " << phi->centers_[ptr + 0] <<", " << phi->centers_[ptr + 1] << ", "  << phi->centers_[ptr + 2] << std::endl;
  }
  phivis << "];"<<std::endl;
  std::fstream phifile;
  std::fstream pfile;
  if(procid == 0) {
      phifile.open(fname_phi, std::ios_base::out);
      phifile << phivis.str()<<std::endl;
      phifile.close();
      pfile.open(fname_p_txt, std::ios_base::out);
      pfile << pvis.str()<<std::endl;
      pfile.close();
  }
  PetscFunctionReturn(ierr);
}


// ### _____________________________________________________________________ ___
// ### ///////////////// readPhiMesh /////////////////////////////////////// ###
PetscErrorCode readPhiMesh(std::vector<double> &centers, std::shared_ptr<NMisc> n_misc, std::string f, bool read_comp_data, std::vector<int> *comps, bool overwrite_sigma) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  std::ifstream file(f);
  std::stringstream ss;
  int np = 0, k = 0;
  double sigma = 0, spacing = 0;
  if (file.is_open()) {
    ss << " reading Gaussian centers from file " << f; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    centers.clear();
    if (read_comp_data && comps != nullptr) {(*comps).clear();}
    std::string line;
    std::getline(file, line); // sigma, spacing
    std::string token;
    size_t pos1 = line.find("=");
    size_t pos2 = line.find(",");
    if (pos1 != std::string::npos && pos2 != std::string::npos) {sigma = atof(line.substr(pos1+1, pos2).c_str());}
    line.erase(0, pos2+1); pos1 = line.find("=");
    if (pos1 != std::string::npos) {spacing = atof(line.substr(pos1+1, line.length()).c_str());}
    ss << " reading sigma="<<sigma<<", spacing="<<spacing; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    if (n_misc->phi_sigma_data_driven_ != sigma) {
      ss << " Warning: specified sigma="<<n_misc->phi_sigma_data_driven_<<" != sigma="<<sigma<<" (read from file).";
      if (overwrite_sigma) {
        ss << " Specified sigma overwritten.";
        n_misc->phi_sigma_data_driven_ = sigma;
      }
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }
    std::getline(file, line); // throw away;
    std::string t;
    int ii = 0;
    while (std::getline(file, line)) {
      if (line.rfind("]") != std::string::npos) break;   // end of file reached, exit out
      std::stringstream l(line);
      ii = 0;
      while (std::getline(l, t, ',')) {
        if (ii < 3) {
          centers.push_back(atof(t.c_str()));
        } else if (read_comp_data && comps != nullptr) {
          (*comps).push_back(atof(t.c_str()));
        }
        ii++;
      }
      np++;
    }
    file.close();
    n_misc->np_ = np;
    ss << " np=" << np << " centers read "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  } else {
    ss << " cannot open file " << f; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(ierr);
}


// ### _____________________________________________________________________ ___
// ### ///////////////// readPVec ////////////////////////////////////////// ###
PetscErrorCode readPVec(Vec* x, int size, int np, std::string f) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  double *x_ptr;
  std::string file, msg, path, ext, line;
  std::stringstream ss;

  TU_assert(!f.empty(), "filename not set");
  // get file name without path
  ierr = getFileName(path, file, ext, f);                         CHKERRQ(ierr);
  msg = "file " + file + " does not exist";
  TU_assert(fileExists(f), msg.c_str());

  if (strcmp(ext.c_str(),".bin") == 0) {
    ierr = readBIN(&(*x), size, f);                               CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // if not nullptr, clear memory
  if (*x != nullptr) {ierr = VecDestroy(x); CHKERRQ(ierr); *x = nullptr;}
  // create vec
  #ifdef SERIAL
      ierr = VecCreateSeq (PETSC_COMM_SELF, size, &(*x));        CHKERRQ (ierr);
  #else
      ierr = VecCreate (PETSC_COMM_WORLD, &(*x));                CHKERRQ (ierr);
      ierr = VecSetSizes (*x, PETSC_DECIDE, size);               CHKERRQ (ierr);
      ierr = VecSetFromOptions (*x);                             CHKERRQ (ierr);
  #endif
  ierr = VecSet (*x, 0.);                                        CHKERRQ (ierr);

  // read pvec from file
  std::ifstream pfile(f);
  int pi = 0;
  if (pfile.is_open()) {

    ss << " reading p_i values from file " << f; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    std::getline(pfile, line); // throw away (p = [);
    ierr = VecGetArray(*x, &x_ptr);                                CHKERRQ (ierr);
    while (std::getline(pfile, line)) {
      if (line.rfind("]") != std::string::npos) break;   // end of file reached, exit out
      TU_assert(pi < np, "index out of bounds reading p_vec from file.");
      x_ptr[pi] = atof(line.c_str());
      pi++;
    }
    ss << " ... success: " << pi << " values read, size of vector: " << size; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    TU_assert(pi == np, "number of read p_i values does not match with number of read Gaussian centers.");
    ierr = VecRestoreArray(*x, &x_ptr);                            CHKERRQ (ierr);
    pfile.close();
  } else {
    ss << " cannot open file " << f; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    PetscFunctionReturn(1);
  }
  if (procid == 0) {
    ierr = VecView (*x, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);
  }
  PetscFunctionReturn(0);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// readBIN /////////////////////////////////////////// ###
PetscErrorCode readBIN(Vec* x, int size, std::string f) {
  PetscFunctionBegin;
  int nprocs, procid;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  PetscErrorCode ierr = 0;
  PetscViewer viewer=nullptr;
  std::string file, msg;
  std::stringstream ss;

  ss << "reading p_i values from binary file " << f; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  TU_assert(!f.empty(), "filename not set");
  // get file name without path
  ierr = getFileName(file, f);                                    CHKERRQ(ierr);
  msg = "file " + file + " does not exist";
  TU_assert(fileExists(f), msg.c_str());
  if (*x != nullptr) {ierr = VecDestroy(x); CHKERRQ(ierr); *x = nullptr;}
  #ifdef SERIAL
      ierr = VecCreateSeq (PETSC_COMM_SELF, size, &(*x));        CHKERRQ (ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, f.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(ierr);
  #else
      ierr = VecCreate (PETSC_COMM_WORLD, &(*x));                CHKERRQ (ierr);
      ierr = VecSetSizes (*x, PETSC_DECIDE, size);               CHKERRQ (ierr);
      ierr = VecSetFromOptions (*x);                             CHKERRQ (ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, f.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(ierr);
  #endif
  TU_assert(viewer != NULL, "could not read binary file");
  ierr = PetscViewerBinarySetFlowControl(viewer, 2);              CHKERRQ(ierr);
  ierr = VecLoad(*x, viewer);                                     CHKERRQ(ierr);

  if (procid == 0) {
  ierr = VecView (*x, PETSC_VIEWER_STDOUT_SELF);                 CHKERRQ (ierr);
  }
  // clean up
  if (viewer!=nullptr) {
      ierr = PetscViewerDestroy(&viewer);                         CHKERRQ(ierr);
      viewer=nullptr;
  }
  PetscFunctionReturn(0);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// writeBIN ////////////////////////////////////////// ###
PetscErrorCode writeBIN(Vec x, std::string f) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  PetscViewer viewer = nullptr;
  PetscFunctionBegin;
  TU_assert(x != nullptr, "null pointer");
  TU_assert(!f.empty(), "filename not set");
  #ifdef SERIAL
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, f.c_str(), FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  #else
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, f.c_str(), FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  #endif
  TU_assert(viewer != nullptr, "could not write binary file");
  ierr = VecView(x, viewer);                                      CHKERRQ(ierr);
  // clean up
  if (viewer != nullptr) {
      ierr = PetscViewerDestroy(&viewer);                         CHKERRQ(ierr);
      viewer = nullptr;
  }
  PetscFunctionReturn(0);
}


// ### _____________________________________________________________________ ___
// ### ///////////////// readConCompDat /////////////////////////////////////// ###
PetscErrorCode readConCompDat(std::vector<double> &weights, std::vector<double> &centers, std::string f) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  std::ifstream file(f);
  std::stringstream ss;
  int ncomp = 0;
  if (file.is_open()) {
    ss << " reading concomp.dat file " << f; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    weights.clear();
    centers.clear();

    std::string line;
    std::getline(file, line); // #components
    std::getline(file, line); // #components (value)
    ncomp = atoi(line.c_str());
    std::getline(file, line); // center of mass:
    for(int i=0; i < ncomp; ++i){
      std::getline(file, line); // values
      std::stringstream l(line);
      std::string t;
        while (std::getline(l, t, ',')) {
          centers.push_back(atof(t.c_str()));
        }
    }
    std::getline(file, line); // relative mass:
    for(int i=0; i < ncomp; ++i){
      std::getline(file, line); // values
      weights.push_back(atof(line.c_str()));
    }

    file.close();
    ss << " ncomp=" << ncomp << " component read "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << " weights: ";
    for(int i=0; i < ncomp; ++i){
      ss << weights[i];
      if(i < ncomp) {ss << ", ";}
    }
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << " centers: ";
    for(int i=0; i < ncomp; ++i){
      ss << "(";
      for(int j=0; j < 3; ++j){
        ss << centers[3*i+j];
        if(j < 2) {ss << ",";}
      }
      ss << "); ";
    }
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  } else {
    ss << " cannot open file " << f; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(ierr);
}



// ### _____________________________________________________________________ ___
// ### ///////////////// getFileName /////////////////////////////////////// ###
PetscErrorCode getFileName(std::string& filename, std::string file) {
    PetscErrorCode ierr = 0;
    std::string path;
    size_t sep;
    PetscFunctionBegin;

    sep = file.find_last_of("\\/");
    if (sep != std::string::npos) {
        path=file.substr(0,sep);
        filename=file.substr(sep + 1);
    }
    if (filename == "") { filename = file; }
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// getFileName /////////////////////////////////////// ###
PetscErrorCode getFileName(std::string& path, std::string& filename,
                           std::string& extension, std::string file) {
    PetscErrorCode ierr = 0;
    int nprocs, procid;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    std::string::size_type idx;
    std::stringstream ss;
    PetscFunctionBegin;

    // get path
    idx = file.find_last_of("\\/");
    if (idx != std::string::npos) {
        path = file.substr(0,idx);
        filename = file.substr(idx + 1);
    }
    if (filename == "") {
        filename = file;
    }

    // get extension
    idx = filename.rfind(".");
    if (idx != std::string::npos) {
        extension = filename.substr(idx+1);

        // handle zipped files
        if (strcmp(extension.c_str(),"gz") == 0) {
            filename = filename.substr(0,idx);
            idx = filename.rfind(".");
            if(idx != std::string::npos) {
                extension = filename.substr(idx+1);
                extension = extension + ".gz";
            }
        }
        extension = "." + extension;
        filename  = filename.substr(0,idx);

    } else {
        ss << "ERROR: no extension found"; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }

    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// fileExists //////////////////////////////////////// ###
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

//TODO
//Rewrite variables according to standard conventions


int weierstrassSmoother (double * Wc, double *c, std::shared_ptr<NMisc> N_Misc, double sigma) {
	MPI_Comm c_comm = N_Misc->c_comm_;
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);

	int *N = N_Misc->n_;
	int istart[3], isize[3], osize[3], ostart[3];
	int alloc_max = accfft_local_size_dft_r2c(N, isize, istart, osize, ostart,
			c_comm);

	double self_exec_time = -MPI_Wtime();

	const int Nx = N_Misc->n_[0], Ny = N_Misc->n_[1], Nz = N_Misc->n_[2];
	const double pi = M_PI, twopi = 2.0 * pi, factor = 1.0 / (Nx * Ny * Nz);
	const double hx = twopi / Nx, hy = twopi / Ny, hz = twopi / Nz;
	accfft_plan * plan = N_Misc->plan_;

	Complex *c_hat = (Complex*) accfft_alloc(alloc_max);
	Complex *f_hat = (Complex*) accfft_alloc(alloc_max);
	double *f = (double*) accfft_alloc(alloc_max);
	if ((c_hat == NULL) || (f_hat == NULL) || (f == NULL)) {
		printf("Proc %d: Error allocating array\n", procid);
		exit(-1);
	}

	// Build the filter
	int num_th = omp_get_max_threads();
	double sum_th[num_th];
	for (int i = 0; i < num_th; i++)
		sum_th[i] = 0.;
	#pragma omp parallel num_threads(num_th)
	{
		int thid = omp_get_thread_num();
		double X, Y, Z, Xp, Yp, Zp;
		int64_t ptr;
	#pragma omp for
		for (int i = 0; i < isize[0]; i++)
			for (int j = 0; j < isize[1]; j++)
				for (int k = 0; k < isize[2]; k++) {
					X = (istart[0] + i) * hx;
					Xp = X - twopi;
					Y = (istart[1] + j) * hy;
					Yp = Y - twopi;
					Z = (istart[2] + k) * hz;
					Zp = Z - twopi;
					ptr = i * isize[1] * isize[2] + j * isize[2] + k;
					f[ptr] = std::exp((-X * X - Y * Y - Z * Z) / sigma / sigma / 2.0)
							+ std::exp((-Xp * Xp - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

					 f[ptr] += std::exp((-Xp * Xp - Y * Y - Z * Z) / sigma / sigma / 2.0)
					 		+ std::exp((-X * X - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

					 f[ptr] += std::exp((-X * X - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
					 		+ std::exp((-Xp * Xp - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

					 f[ptr] += std::exp((-Xp * Xp - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
					 		+ std::exp((-X * X - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

					if (f[ptr] != f[ptr])
						f[ptr] = 0.; // To avoid Nan
					sum_th[thid] += f[ptr];
				}
	}

	// Normalize the Filter
	double sum_f_local = 0., sum_f = 0;
	for (int i = 0; i < num_th; i++)
		sum_f_local += sum_th[i];

	MPI_Allreduce(&sum_f_local, &sum_f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double normalize_factor = 1. / (sum_f * hx * hy * hz);

	#pragma omp parallel for
	for (int i = 0; i < isize[0] * isize[1] * isize[2]; i++)
		f[i] = f[i] * normalize_factor;
	//PCOUT<<"sum f= "<<sum_f<<std::endl;
	//PCOUT<<"normalize factor= "<<normalize_factor<<std::endl;

	/* Forward transform */
	accfft_execute_r2c(plan, f, f_hat);
	accfft_execute_r2c(plan, c, c_hat);

	// Perform the Hadamard Transform f_hat=f_hat.*c_hat
	std::complex<double>* cf_hat = (std::complex<double>*) (double*) f_hat;
	std::complex<double>* cc_hat = (std::complex<double>*) (double*) c_hat;
	#pragma omp parallel for
	for (int i = 0; i < osize[0] * osize[1] * osize[2]; i++)
		cf_hat[i] *= (cc_hat[i] * factor * hx * hy * hz);


	/* Backward transform */
	accfft_execute_c2r(plan, f_hat, Wc);

	accfft_free(f);
	accfft_free(f_hat);
	accfft_free(c_hat);

	//PCOUT<<"\033[1;32m weierstrass_smoother } "<<"\033[0m"<<std::endl;
	//self_exec_time+= MPI_Wtime();

	return 0;
}

/// @brief computes difference diff = x - y
PetscErrorCode computeDifference(PetscScalar *sqrdl2norm,
	Vec diff_wm, Vec diff_gm, Vec diff_csf, Vec diff_glm, Vec diff_bg,
	Vec x_wm, Vec x_gm, Vec x_csf, Vec x_glm, Vec x_bg,
	Vec y_wm, Vec y_gm, Vec y_csf, Vec y_glm, Vec y_bg) {

	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
	// diff = x - y
	if(x_wm != nullptr) {
		ierr = VecWAXPY (diff_wm, -1.0, y_wm, x_wm);                 CHKERRQ (ierr);
		ierr = VecDot (diff_wm, diff_wm, &mis_wm);                   CHKERRQ (ierr);
	}
	if(x_gm != nullptr) {
		ierr = VecWAXPY (diff_gm, -1.0, y_gm, x_gm);                 CHKERRQ (ierr);
		ierr = VecDot (diff_gm, diff_gm, &mis_gm);                   CHKERRQ (ierr);
	}
	if(x_csf != nullptr) {
		ierr = VecWAXPY (diff_csf, -1.0, y_csf, x_csf);              CHKERRQ (ierr);
		ierr = VecDot (diff_csf, diff_csf, &mis_csf);                CHKERRQ (ierr);
	}
	if(x_glm != nullptr) {
		ierr = VecWAXPY (diff_glm, -1.0, y_glm, x_glm);              CHKERRQ (ierr);
		ierr = VecDot (diff_glm, diff_glm, &mis_glm);                CHKERRQ (ierr);
	}
	*sqrdl2norm  = mis_wm + mis_gm + mis_csf + mis_glm;
	//PetscPrintf(PETSC_COMM_WORLD," geometricCouplingAdjoint mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, \n", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
	PetscFunctionReturn(0);
	}

	/** @brief computes difference xi = m_data - m_geo
	 *  - function assumes that on input, xi = m_geo * (1-c(1))   */
PetscErrorCode geometricCouplingAdjoint(PetscScalar *sqrdl2norm,
	Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg,
	Vec m_geo_wm, Vec m_geo_gm, Vec m_geo_csf, Vec m_geo_glm, Vec m_geo_bg,
	Vec m_data_wm, Vec m_data_gm, Vec m_data_csf, Vec m_data_glm, Vec m_data_bg) {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
	if(m_geo_wm != nullptr) {
		ierr = VecAXPY (xi_wm, -1.0, m_data_wm);                     CHKERRQ (ierr);
		ierr = VecScale (xi_wm, -1.0);                               CHKERRQ (ierr);
		ierr = VecDot (xi_wm, xi_wm, &mis_wm);                       CHKERRQ (ierr);
	}
	if(m_geo_gm != nullptr) {
		ierr = VecAXPY (xi_gm, -1.0, m_data_gm);                     CHKERRQ (ierr);
		ierr = VecScale (xi_gm, -1.0);                               CHKERRQ (ierr);
		ierr = VecDot (xi_gm, xi_gm, &mis_gm);                       CHKERRQ (ierr);
	}
	if(m_geo_csf != nullptr) {
		ierr = VecAXPY (xi_csf, -1.0, m_data_csf);                   CHKERRQ (ierr);
		ierr = VecScale (xi_csf, -1.0);                              CHKERRQ (ierr);
		ierr = VecDot (xi_csf, xi_csf, &mis_csf);                    CHKERRQ (ierr);
	}
	if(m_geo_glm != nullptr) {
		ierr = VecAXPY (xi_glm, -1.0, m_data_glm);                   CHKERRQ (ierr);
		ierr = VecScale (xi_glm, -1.0);                              CHKERRQ (ierr);
		ierr = VecDot (xi_glm, xi_glm, &mis_glm);                    CHKERRQ (ierr);
	}
	*sqrdl2norm  = mis_wm + mis_gm + mis_csf + mis_glm;
	//PetscPrintf(PETSC_COMM_WORLD," geometricCouplingAdjoint mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, \n", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
	PetscFunctionReturn(0);
}

//Hoyer measure for sparsity of a vector
PetscErrorCode vecSparsity (Vec x, double &sparsity) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	int size;
	ierr = VecGetSize (x, &size);									CHKERRQ (ierr);
	double norm_1, norm_inf;
	ierr = VecNorm (x, NORM_1, &norm_1);							CHKERRQ (ierr);
	ierr = VecNorm (x, NORM_INFINITY, &norm_inf);					CHKERRQ (ierr);

	if (norm_inf == 0) {
		sparsity = 1.0;
		PetscFunctionReturn (0);
	}

	sparsity = (size - (norm_1 / norm_inf)) / (size - 1);

	PetscFunctionReturn (0);
}

/// @brief computes geometric tumor coupling m1 = m0(1-c(1))
PetscErrorCode geometricCoupling(
	Vec m1_wm, Vec m1_gm, Vec m1_csf, Vec m1_glm, Vec m1_bg,
	Vec m0_wm, Vec m0_gm, Vec m0_csf, Vec m0_glm, Vec m0_bg,
	Vec c1, std::shared_ptr<NMisc> nmisc) {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar *ptr_wm, *ptr_gm, *ptr_csf, *ptr_glm, *ptr_bg, *ptr_tu;
	PetscScalar *ptr_m1_wm, *ptr_m1_gm, *ptr_m1_csf, *ptr_m1_glm, *ptr_m1_bg;
	PetscScalar sum = 0;
  if(m0_wm  != nullptr) {ierr = VecGetArray(m0_wm,  &ptr_wm);     CHKERRQ(ierr);}
	if(m0_gm  != nullptr) {ierr = VecGetArray(m0_gm,  &ptr_gm);     CHKERRQ(ierr);}
	if(m0_csf != nullptr) {ierr = VecGetArray(m0_csf, &ptr_csf);    CHKERRQ(ierr);}
	if(m0_glm != nullptr) {ierr = VecGetArray(m0_glm, &ptr_glm);    CHKERRQ(ierr);}
	if(m0_bg  != nullptr) {ierr = VecGetArray(m0_bg,  &ptr_bg);     CHKERRQ(ierr);}
	if(m1_wm  != nullptr) {ierr = VecGetArray(m1_wm,  &ptr_m1_wm);  CHKERRQ(ierr);}
	if(m1_gm  != nullptr) {ierr = VecGetArray(m1_gm,  &ptr_m1_gm);  CHKERRQ(ierr);}
	if(m1_csf != nullptr) {ierr = VecGetArray(m1_csf, &ptr_m1_csf); CHKERRQ(ierr);}
	if(m1_glm != nullptr) {ierr = VecGetArray(m1_glm, &ptr_m1_glm); CHKERRQ(ierr);}
	if(m1_bg  != nullptr) {ierr = VecGetArray(m1_bg,  &ptr_m1_bg);  CHKERRQ(ierr);}
	if(c1     != nullptr) {ierr = VecGetArray(c1,     &ptr_tu);     CHKERRQ(ierr);}
  // m = m0(1-c(1))
	for (PetscInt j = 0; j < nmisc->n_local_; j++) {
		sum = 0;
    if(m0_gm   != nullptr) {ptr_m1_gm[j]  = ptr_gm[j]  * (1 - ptr_tu[j]); sum += ptr_m1_gm[j];}
		if(m0_csf  != nullptr) {ptr_m1_csf[j] = ptr_csf[j] * (1 - ptr_tu[j]); sum += ptr_m1_csf[j];}
		if(m0_glm  != nullptr) {ptr_m1_glm[j] = ptr_glm[j] * (1 - ptr_tu[j]); sum += ptr_m1_glm[j];}
		if(m0_bg   != nullptr) {ptr_m1_bg[j]  = ptr_bg[j]  * (1 - ptr_tu[j]); sum += ptr_m1_bg[j];}
		if(m0_wm   != nullptr) {ptr_m1_wm[j]  = 1. - (sum + ptr_tu[j]);}
	}
	if(m0_wm  != nullptr) {ierr = VecRestoreArray(m0_wm,  &ptr_wm);    CHKERRQ(ierr);}
	if(m0_gm  != nullptr) {ierr = VecRestoreArray(m0_gm,  &ptr_gm);    CHKERRQ(ierr);}
	if(m0_csf != nullptr) {ierr = VecRestoreArray(m0_csf, &ptr_csf);   CHKERRQ(ierr);}
	if(m0_glm != nullptr) {ierr = VecRestoreArray(m0_glm, &ptr_glm);   CHKERRQ(ierr);}
	if(m0_bg  != nullptr) {ierr = VecRestoreArray(m0_bg,  &ptr_bg);    CHKERRQ(ierr);}
	if(m1_wm  != nullptr) {ierr = VecRestoreArray(m1_wm,  &ptr_m1_wm); CHKERRQ(ierr);}
	if(m1_gm  != nullptr) {ierr = VecRestoreArray(m1_gm,  &ptr_m1_gm); CHKERRQ(ierr);}
	if(m1_csf != nullptr) {ierr = VecRestoreArray(m1_csf, &ptr_m1_csf);CHKERRQ(ierr);}
	if(m1_glm != nullptr) {ierr = VecRestoreArray(m1_glm, &ptr_m1_glm);CHKERRQ(ierr);}
	if(m1_bg  != nullptr) {ierr = VecRestoreArray(m1_bg,  &ptr_m1_bg); CHKERRQ(ierr);}
	if(c1     != nullptr) {ierr = VecRestoreArray(c1,     &ptr_tu);    CHKERRQ(ierr);}
  // go home
	PetscFunctionReturn(0);
}

PetscErrorCode vecSign (Vec x) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	double *x_ptr;
	int size;
	ierr = VecGetSize (x, &size);		CHKERRQ (ierr);
	ierr = VecGetArray (x, &x_ptr);		CHKERRQ (ierr);

	for (int i = 0; i < size; i++) {
		if (x_ptr[i] > 0) x_ptr[i] = 1.0;
		else if (x_ptr[i] == 0) x_ptr[i] = 0.0;
		else x_ptr[i] = -1.0;
	}

	ierr = VecRestoreArray (x, &x_ptr);	CHKERRQ (ierr);

	PetscFunctionReturn (0);
}

PetscErrorCode hardThreshold (Vec x, int sparsity_level, int sz, std::vector<int> &support, int &nnz) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    nnz = 0;

    std::priority_queue<std::pair<PetscReal, int>> q;
    double *x_ptr;
    ierr = VecGetArray (x, &x_ptr);   CHKERRQ (ierr);
    for (int i = 0; i < sz; i++) {
    q.push(std::pair<PetscReal, int>(x_ptr[i], i));   // Push values and idxes into a priiority queue
    }

    double tol = 0.0; // 1E-10; // tolerance for specifying if signal is present: We don't need to add signal components which
            // are (almost)zero to the support
    for (int i = 0; i < sparsity_level; i++) {
    if (std::abs(q.top().first) > tol) {
      nnz++;  // keeps track of how many non-zero (important) components of the signal there are
      support.push_back (q.top().second);
    } else {  // if top of the queue is not greater than tol, we are done since none of the elements
          // below it will every be greater than tol
      break;
    }
    q.pop ();
    }

    ierr = VecRestoreArray (x, &x_ptr);   CHKERRQ (ierr);

    PetscFunctionReturn (0);
}


PetscErrorCode hardThreshold (Vec x, int sparsity_level, int sz, std::vector<int> &support, std::vector<int> labels, std::vector<double> weights, int &nnz, int num_components, bool double_mode) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int nprocs, procid;
    MPI_Comm_rank(PETSC_COMM_WORLD, &procid);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

    nnz = 0;
    std::stringstream ss;
    std::priority_queue<std::pair<PetscReal, int>> q;
    double *x_ptr;
    double tol = 0.0; // 1E-10; // tolerance for specifying if signal is present: We don't need to add signal components which
                      // are (almost)zero to the support
    ierr = VecGetArray (x, &x_ptr);   CHKERRQ (ierr);

    if (double_mode) sparsity_level /= 2;

    std::vector<int> component_sparsity;
    int fin_spars;
    int sparsity;
    int ncc = 0;
    for (auto w : weights) if (w >= 1E-3) ncc++;
    for (int nc = 0; nc < num_components; nc++) {
    if (nc != num_components - 1) {
      // sparsity level in total is 5 * #nc (number components)
      // every component gets at 3 degrees of freedom, the remaining 2 * #nc degrees of freedom are distributed based on component weight
      sparsity = (weights[nc] > 1E-3) ? (3 + std::floor (weights[nc] * (sparsity_level - 3 * ncc - (num_components-ncc)))) : 1;
      if (double_mode) sparsity *= 2;
      component_sparsity.push_back (sparsity);
      ss << "sparsity of component " << nc << ": " << component_sparsity.at(nc); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    } else { // last component is the remaining support
      int used = 0;
      for (auto x : component_sparsity)  {used += x;}
      if (double_mode) sparsity_level *= 2;
      fin_spars = sparsity_level - used;
      component_sparsity.push_back (fin_spars);
      ss << "sparsity of component " << nc << ": " << fin_spars; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }

    for (int i = 0; i < sz; i++) {
      if (labels[i] == nc + 1) // push the current components into the priority queue
        q.push(std::pair<PetscReal, int>(x_ptr[i], i));   // Push values and idxes into a priiority queue
    }

    for (int i = 0; i < component_sparsity[nc]; i++) {
      if (q.size() > 0) {
        if (std::abs(q.top().first) > tol) {
          nnz++;  // keeps track of how many non-zero (important) components of the signal there are
          support.push_back (q.top().second);
        } else {  // if top of the queue is not greater than tol, we are done since none of the elements
                  // below it will ever be greater than tol
          ss << "  ... some DOF not used in comp " << nc << "; p_i = " << std::abs(q.top().first) << " < " << tol << " = tolerance"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
          break;
        }
        q.pop ();
      } else {
        ss << "  ... no DOF selected in comp. " << nc << "; no value in queue (component weight, w="<< weights[nc]  <<"). "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        break;
      }
    }
    q = std::priority_queue<std::pair<PetscReal, int>> (); // reset the queue
    }

    ierr = VecRestoreArray (x, &x_ptr); 	CHKERRQ (ierr);

    PetscFunctionReturn (0);
}



double myDistance (double *c1, double *c2) {
    return std::sqrt((c1[0] - c2[0]) * (c1[0] - c2[0]) + (c1[1] - c2[1]) * (c1[1] - c2[1]) + (c1[2] - c2[2]) * (c1[2] - c2[2]));
}

PetscErrorCode computeCenterOfMass (Vec x, int *isize, int *istart, double *h, double *cm) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int64_t ptr_idx;
	double X, Y, Z;
    double *data_ptr;
    double com[3], sum;
    for (int i = 0; i < 3; i++)
    	com[i] = 0.;
    sum = 0;
    ierr = VecGetArray (x, &data_ptr);                 CHKERRQ (ierr);
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

    double sm;
    MPI_Allreduce (&com, cm, 3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce (&sum, &sm, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

    for (int i = 0; i < 3; i++) {
    	cm[i] /= sm;
    }

    ierr = VecRestoreArray (x, &data_ptr);                 CHKERRQ (ierr);


	PetscFunctionReturn (0);
}
