#!/bin/bash
PAT_DIR=/scratch/05027/shas1693/penn_gbm_survival20/Data/
AT_DIR=/scratch/05027/shas1693/adni-nc/
scripts_path=/scratch/05027/shas1693/pglistr_tumor/scripts/masseffect
res_path=/scratch/05027/shas1693/pglistr_tumor/results/penn_masseffect/
sz=160
at_stat_path=/scratch/05027/shas1693/adni-nc/adni-atlas-stats.csv
pat_stat_path=/scratch/05027/shas1693/penn_gbm_survival20/Data/pat_stats.csv

python3 $scripts_path/extract_stats.py -n $sz -results_path $res_path -atlas_stat_path $at_stat_path -patient_stat_path $pat_stat_path -patient_dir $PAT_DIR
