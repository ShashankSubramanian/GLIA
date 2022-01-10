## GLIA Run scripts 
This readme outlines how to run the solver and use python scripts to interface
with the different solver functionalities

## Running environment
* scripts/params.py outlines all input parameters and their default values used. Please do not change any values
* There are different scripts in this folder (each described below) that create a dict of user-defined values for any of the parameters in params.py to overwride their default values
* Running a script creates a config file (solver_config.txt) in the user-defined output directory
* To run the solver, use 
```bash
mpirun/ibrun -np $NUMPROCS build/last/tusolver -config /path/to/solver_config.txt
```
(for GPUs, use $NUMPROCS=1 for single gpu; for CPUs recommended use is $NUMPROCS=256 for 256x256x256 resolutions)
* Alternatively, on TACC machines, running any script with the submit variable set to true will submit batch jobs
  * For non-TACC machines, each script also creates a job.sh file in the output directory that can be submitted manually according to the user's compute cluster


## Testing scripts
These scripts run the solver for some sample low-resolution test data stores in testdata/
to test all core functionalities
1. forward.py: runs a forward tumor simulation with mass effect 
2. inverse.py: runs inverse solver to compute tumor initial location (TIL) and tumor growth parameters
3. inverse_masseffect.py: runs inverse solver using an ensemble procedure to estimate mass-effect in addition to TIL and growth parameters

Once the inversion solvers have reconstructed parameters, additional post-processing scripts can be used to compute biophysically derived features that can be used for downstream visulization, analysis, and machine learning tasks. 

There are also additional scripts that should be of interest to developers. 

Each individual script contains an output directory that logs the solver results in solver_log.txt and writes out important paraview data


## Running scripts overview
These are the running scripts to run the solver in its various modes. Each script outlines all the important parameters. The user can change any value or input data according to their needs.
To run any script:
```bash
python3 {name_of_script}.py
```
1. forward.py: runs a forward tumor simulation
2. inverse_nomasseffect.py: runs the inverse solver without masseffect on a list of patients to infer the TIL 
3. inverse_masseffect.py: runs an ensembled mass effect inversion on a list of patients

Others (for devs):
1. inverse_gridcont.py: runs the inverse TIL solver for a single patient (used for synthetic testing)

The subfolders are utilty functions for these run scripts

## Running scripts details
These provide important how-to details of each script (note that the description of all important input data and parameters is also in comments in each script): 
1. forward.py: 
  * Input data: A template (healthy/atlas) brain image segmented into its constituent tissues: white matter (wm), gray matter (gm), ventricles (vt), and cerebrospinal fluid (csf) and an optional T1 MRI for visualization purposes. These are the "atlas_labels" in the script
  * Input parameters: Tumor seed location(s) called "user_cms", parameter values for different coefficients (rho is reaction, k/kappa is diffusion, gamma is mass effect intensity). Other parameters are outlined in the script comments. User also defines an "output_dir" to store all results. 
  * Output: the log file solver_log.txt logs the course of the solver. The solver outputs the time history of each tissue, each tumor, and additional fields such as displacement, velocity, mass effect forcing function, screening and Lame coefficients, reaction and diffusion coefficients for each time step. Please refer the the forward solver [paper](https://link.springer.com/article/10.1007/s00285-019-01383-y) for all notations, models used, algorithms etc.
2. inverse_til.py
  * Input data:
    - A patient/diseased brain image(s) segmented into its constituent tissues: white matter (wm), gray matter (gm), ventricles (vt), cerebrospinal fluid (csf), enhancing tumor (en), necrotic tumor (nec), and edema (ed).  
    - These are the "segmentation_labels" in the script. We assume all imaging scans to be affinely registered to some template in this solver
    - The script can run on multiple patients simulataneously depending on the input resources (for example: number of gpus in each system). A patient directory structure is assumed for the input (we use the [BraTS](http://braintumorsegmentation.org/) structure) where each patient's data is in a separate folder with the segmentation file name as /path/to/all_patients/patient_name/aff2jakob/{patient_name}_seg_ants_aff2jakob.nii.gz. See our [example real data](https://drive.google.com/drive/folders/1QtC6R8b_sQoB0BGUumoz9NqWtfKXndre?usp=sharing) to understand the directory structure.
  * Input parameters: All parameters are chosen by the solver. User only sets the computational resources such as number of gpus.
  * Output: 
    - The script runs a grid continuation scheme and saves each mesh resolution results in output_dir/inversion/nx{mesh_size}/obs-1.0/. All final resolution results are in output_dir/inversion/nx256/obs-1.0/. 
    - See the solver_log.txt at any resolution to track the progress of inversion and important metrics, parameters, and errors. Alternatively, the job script log file can be tracked for a high-level progress status. 
    - The other important outputs are c0_rec.nii.gz and c1_rec.nii.gz which show the reconstructed tumor IC and final tumor concentration. All reconstructed parameter values and errors are in reconstruction_info.dat (and in the solver_log.txt additionally). See the inverse solver [paper](https://arxiv.org/abs/1907.06564) for all notations, models, inversion algorithms etc.
    - See [README-OUT-TIL.md](../doc/README-OUT-TIL.md) for full description of outputs
3. inverse_ensemble.py
  * Input data: 
    - This script is run after inverse_til.py to reconstruct mass effect deformations with a more complex model.
    - It requires the patient segmentation labels as before. Additionally, it requires a data directory of several healthy template brain images (we recommend using the [ADNI](http://adni.loni.usc.edu/data-samples/access-data/) dataset of normal control brains). The templates also must be segmented (and affinely registered like the patient). 
    - Our templates are named {ID}_seg_aff2jakob_ants.nii.gz and {ID}_t1_aff2jakob.nii.gz (for the optional T1 MRI).
  See our [example real data/templates](https://drive.google.com/drive/folders/1QtC6R8b_sQoB0BGUumoz9NqWtfKXndre?usp=sharing) to understand the template structure.
    - The script can run on multiple patients simulataneously depending on the input resources (for example: number of gpus in each system). The path to all patients is an input as before
    - It requires binaries of a registration solver. We use [CLAIRE](https://github.com/andreasmang/claire); see [install.md](../doc/install.md) for installation details
    - Last, it requires the path to the tumor TILs (the output dir from inverse_til) 
  * Input parameters: All parameters are chosen by the solver. User only sets the computational resources such as number of gpus.
  * Output: 
    - The script saves registration and tumor inversion results in output_dir/reg and output_dir/tu
    - The inversion on each template is saved within subfolders with template ID
    - See [README-OUT-INV-ENSEMBLE.md](../doc/README-OUT-INV-ENSEMBLE.md) for full description of outputs
    - See next section on scripts to extract important reconstruction statistics

## Post-solve analysis scripts
1. Reconstruction statistics for mass effect inversion: masseffect/extract_stats.py extracts important statistics and saves them in output_dir/stat. Use --help for all options. See [README-OUT-INV-ENSEMBLE.md](../doc/README-OUT-INV-ENSEMBLE.md) in *stats directory* for full description of outputs
2. (devs) Post-mass effect inversion analysis: masseffect/analysis.py performs additional analysis. Use --help for options
3. (devs) Visualzation helper scripts are in vis/
