#ifndef _IO_H
#define _IO_H

#include "Parameters.h"

//pnetcdf error handling
PetscErrorCode NCERRQ (int cerr);
PetscErrorCode throwErrorMsg(std::string msg, int line, const char *file);
PetscErrorCode myAssert(bool condition, std::string msg);

//Read/Write function prototypes
#ifdef NIFTIIO
	template<typename T> PetscErrorCode readNifti(nifti_image* image, ScalarType *data_global, std::shared_ptr<Parameters> params);
	PetscErrorCode writeNifti(nifti_image** image, ScalarType *p_x, std::shared_ptr<Parameters> params, const char* fname);
#endif
PetscErrorCode dataIn (ScalarType *A, std::shared_ptr<Parameters> params, const char *fname);
PetscErrorCode dataIn (Vec A, std::shared_ptr<Parameters> params, const char *fname);
PetscErrorCode dataOut (ScalarType *A, std::shared_ptr<Parameters> params, const char *fname);
PetscErrorCode dataOut (Vec A, std::shared_ptr<Parameters> params, const char *fname);
/// @reads in binary vector, serial
PetscErrorCode readBIN(Vec* x, int size2, std::string f);
/// @brief writes out vector im binary format, serial
PetscErrorCode writeBIN(Vec x, std::string f);
/// @reads in p_vec vector from txt file
PetscErrorCode readPVec(Vec* x, int size, int np, std::string f);
/// @brief reads in Gaussian centers from file
PetscErrorCode readPhiMesh(std::vector<ScalarType> &centers, std::shared_ptr<Parameters> params, std::string f, bool read_comp_data = false, std::vector<int> *comps = nullptr, bool overwrite_sigma = true);
/// @brief reads connected component data from file
PetscErrorCode readConCompDat(std::vector<ScalarType> &weights, std::vector<ScalarType> &centers, std::string f);
/// @brief write checkpoint for p-vector and Gaussian centers
PetscErrorCode writeCheckpoint(Vec p, std::shared_ptr<Phi> phi, std::string path, std::string suffix);
/// @brief returns only filename
PetscErrorCode getFileName(std::string& filename, std::string file);
/// @brief returns filename, extension and path
PetscErrorCode getFileName(std::string& path, std::string& filename, std::string& extension, std::string file);
/// @brief checks if file exists
bool fileExists(const std::string& filename);


#endif