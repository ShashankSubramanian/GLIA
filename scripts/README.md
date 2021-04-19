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
2. inverse.py: runs an inverse TIL (tumor initiation location) solver
3. inverse_masseffect.py: runs an inverse tumor parameter prediction for models with mass effect using the reconstructed TIL from inverse.py

Each individual script contains an output directory that logs the solver results in solver_log.txt and writes out important paraview data


## Running scripts overview
These are the running scripts to run the solver in its various modes. Each script outlines all the important parameters. The user can change any value or input data according to their needs.
To run any script:
```bash
python3 {name_of_script}.py
```
1. forward.py: runs a forward tumor simulation
2. inverse_til.py: runs the inverse TIL solver on a list of patients
3. inverse_ensemble.py: runs an ensembled mass effect inversion on a list of patients

Others (for developers):
1. inverse_gridcont.py: runs the inverse TIL solver for a single patient (used for synthetic testing)
2. inverse_alzh.py: runs the inverse solver for Alzheimer's disease modeling

The subfolders are utilty functions for these run scripts

## Running scripts details
These provide important how-to details of each script (note that the description of all important input data and parameters is also in comments in each script): 
1. forward.py: 
  * Input data: A template (healthy/atlas) brain image segmented into its constituent tissues: white matter (wm), gray matter (gm), ventricles (vt), and cerebrospinal fluid (csf) and an optional T1 MRI for visualization purposes. These are the "atlas_labels" in the script
  * Input config: Tumor seed location(s) called "user_cms", parameter values for different coefficients (rho is reaction, k/kappa is diffusion, gamma is mass effect intensity). Other parameters are outlined in the script comments. User also defines an "output_dir" to store all results. 
  * Output: the log file solver_log.txt logs the course of the solver. The solver outputs the time history of each tissue, each tumor, and additional fields such as displacement, velocity, mass effect forcing function, screening and Lame coefficients, reaction and diffusion coefficients for each time step. Please refer the the forward solver [paper](https://link.springer.com/article/10.1007/s00285-019-01383-y) for all notations, models used, algorithms etc.
2. inverse_til.py
3. inverse_ensemble.py

## Post-solve analysis scripts
visualization scripts are in vis/
extracting statistics from ensemble inversion is in masseffect/extract_stats.py
running futher analysis for ensemble inversion is in masseffect/analysis.py
