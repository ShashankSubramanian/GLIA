## Output Description of TIL Solver
  **Input (directory) : provides resampled images of segmentation, data components and target data at 64, 128 and 256 resolution**
  
  **Inversion (directory) : inversion results for three resolutions 64, 128, 256**
  
    - nx256 (directory) : 
      > init (directory)           : Input for solver at each resolution
      > obs-1.0 (directory)        : Observed results with threshold 1.0 
        ++ bg.nc                   : Background image 
        ++ c0FinalGuess.nc         : Final guess of solver for tumor initial location (c0)
        ++ c0guess_csitr-1.nc      : First guess of solver for tumor initial location (c0)
        ++ c0_rec_256256256.nii.gz : Reconstructed tumor initial location at resolution 256 (nifti file)
        ++ c0_rec.nc               : Recostructed tumor initial location (netcdf file)
        ++ c1_rec_256256256.nii.gz : Recostructed tumor concentration at time = 1 with reaction diffusion solver using c0_rec.nc
        ++ c_pred_at_[t=1.2].nc    : Prediction of concentation at t = 1.2 using c0_rec.nc
        ++ c_pred_at_[t=1.5].nc    : Prediction of concentation at t = 1.5 using c0_rec.nc
        ++ csf.nc                  : Cerebrospinal fluid of given segmentation segmentation
        ++ data.nc                 : Given data at 256 resolution
        ++ dcomp.dat               : Data components including center of mass and relative mass 
        ++ gm.nc                   : Gray matter of given segmentation
        ++ obs.nc                  : Observation map in solver
        
