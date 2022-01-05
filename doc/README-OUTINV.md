## Description of Outputs for Mass Effect Solver 

 
    - atlas-list.txt                             : Provides list of adni atlases used for the patient
    - job0.sh                                    : Job file submitted to run the solver
    - reg (directory)                            : Provides images used for registration using CLAIRE, subdirectories are template images and outputs of CLAIRE
    - tu                                         : Provides outputs from GLIA 
      > [TEMPLATE] (directory).                  : Provides outputs for Mass Effect solver using this specific template 
        ++ bg.nc                                 : Background image of patient's segmentation in template space
        ++ csf.nc                                : Cerebrospinal fluid of patient's segmentation in template space
        ++ gm.nc                                 : Gray matter of patient's segmentation in template space
        ++ [PATIENT]_c0Recon_transported_160.nc  : Reconstructed tumor initial location (TIL) which is moved to template space at resolution 160
        ++ c0_input.nc                           : TIL used in the solver (same as above)
        ++ c0_rec.nc                             : Reconstructed TIL using gaussians (same as above)
        ++ c1_rec.nc                             : Estimated tumor concentration at time = 1 (time where the data is gived)
        ++ c_pred_at_[t=1.2].nc                  : Prediction of tumor concentration at time = 1.2 
        ++ c_pred_at_[t=1.5].nc                  : Prediction of tumor concentration at time = 1.5
        ++ c_rec_final.nc & c_pred_at_[t=1].nc   : Same as c1_rec.nc
        ++ csf_rec_final.nc                      : Cerebrospinal fluid of template segmentation at time = 1.0
        ++ gm_rec_final.nc                       : Gray matter of template segmentation at time = 1.0
        ++ c_t[1].nc                             : Same as c1_rec.nc
        ++ data.nc                               : Given tumor segmentation 
        ++ displacement_rec_final.nc             : Displacement field at time = 1
        ++ displacement_t[1].nc                  : Same as above 
        ++ disp_[XYZ]_x_pred_at_[t=1.5].nc       : [XYZ] component of displacement field prediction at time = 1.0
        ++ disp_[XYZ]_pred_at_[t=1.2].nc         : [XYZ] component of displacement field prediction at time = 1.2
        ++ disp_[XYZ]_x_pred_at_[t=1.5].nc       : [XYZ] component of displacement field prediction at time = 1.5
        ++ mri_rec_final.nc                      : Reconstructed MRI using template MRI (compare with patients MRI in template space)
        ++ obs.nc                                : Observation field for tumor concentration
        ++ p_csf.nc                              : Patient's CSF from transformed patients's segmentation to template space
        ++ p_gm.nc                               : Patient's gray matter from transformed patients's segmentation to template space
        ++ p_vt.nc                               : Patient's ventricle from transformed patients's segmentation to template space
        ++ p_wm.nc                               : Patient's white matter from transformed patients's segmentation to template space
        ++ reconstruction_info.dat               : Contain's parameters informations such as :
          ^ rho                                  : Reaction parameter 
          ^ k                                    : Diffusion parameter
          ^ gamma                                : Forcing paramters (Mass Effect)
          ^ max_disp                             : Maximum displacement field at time = 1 
          ^ norm_disp                            : Magnitude of displacement field at time t = 1
          ^ c1_rel                               : Relative error of tumor estimation from solver wrt data at time = 1
          ^ c0_rel                               : Relative error of tumor estimation from solver wrt given TIL (= 0 if TIL is used)
        ++ seg_pred_at_[t=*].nc                  : Prediction of segmentation at time = *
        ++ seg_rec_final.nc                      : Final estimated segmentation at time = 1
        ++ solver_config.txt                     : Paramters and data paths given to the solver 
        ++ solver_log.txt                        : Solver output including timings and convergence
        ++ vel_[XYZ]_rec_final.nc                : [XYZ] component of velocity field at time = 1
        ++ vt.nc                                 : Ventricles of patient's segmentation in template space
        ++ vt_rec_final.nc                       : Estimated ventricles at time = 1
        ++ wm.nc                                 : White matter of patient's segmentation in template space
        ++ wm_rec_final.nc                       : Estimated white matter at time = 1
      > stats (directory)                        : Contains analysis of paramters and fields 
        ++ stats.csv                             : Cotains inverted paramters and outputs of all templates 
          ^ atlas.                               : Atlas ID 
          ^ gam                                  : Forcing parameter (Mass Effect)
          ^ rho                                  : Reaction parameter 
          ^ k                                    : Diffusion parameter 
          ^ u                                    : 
