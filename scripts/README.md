Run scripts for the solver

## TEST SCRIPTS ##
These scripts can be used to test the code using the saved test data in ../testdata/
forward.py --> runs a forward solver with mass effect
inverse.py --> runs a inverse TIL (tumor initial location) solver
inverse_masseffect.py --> runs a inverse masseffect solver

## MAIN SCRIPTS ##
forward.py --> runs a forward solver for tumor simulation
inverse_til.py --> runs the inverse TIL solver on list of patients (or single patient)
inverse_ensemble.py --> runs the ensemble mass effect inversion on list of patients (or single patient)

## OTHERS ##
inverse_gridcont.py --> runs the inverse TIL solver for a single patient (used for synthetic testing)
inverse_alzh.py --> [untested] runs the alzheimers inverse solver

The subfolders are utilty functions for these run scripts

### POST-ANALYSIS ##
visualization scripts are in vis/
extracting statistics from ensemble inversion is in masseffect/extract_stats.py
running futher analysis for ensemble inversion is in masseffect/analysis.py