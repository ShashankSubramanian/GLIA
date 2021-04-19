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


## Running scripts
These are the running scripts to run the solver in its various modes. Each script outlines all the important parameters. The user can change any value or input data according to their needs.
1. forward.py: runs a forward tumor simulation
2. inverse_til.py: runs the inverse TIL solver on a list of patients
3. inverse_ensemble.py: runs an ensembled mass effect inversion on a list of patients

Other dev scripts:
4. inverse_gridcont.py: runs the inverse TIL solver for a single patient (used for synthetic testing)
5. inverse_alzh.py: runs the inverse solver for Alzheimer's disease modeling

The subfolders are utilty functions for these run scripts

## Post-solve analysis scripts
visualization scripts are in vis/
extracting statistics from ensemble inversion is in masseffect/extract_stats.py
running futher analysis for ensemble inversion is in masseffect/analysis.py
