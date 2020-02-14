#!/bin/bash

python3 tumor_gen.py -i /work/04716/naveen15/frontera/brats18/results/brats19-gridcont-concomp-cosamp \
                     -o /scratch1/04716/naveen15/brats19_syn_tumor/ \
                     -a /work/04716/naveen15/frontera/698_normal_brains_segmented/mri_images_nifti_256x256x256_aff2jakob \
                     -r /home1/04716/naveen15/pglistr_tumor/scripts/data-gen/brat18_features.csv \
                     -m /home1/04716/naveen15/pglistr_tumor/scripts/data-gen/name_mapping.csv \
                     -c /home1/04716/naveen15/pglistr_tumor/ \
                     -d /scratch1/04716/naveen15/brats19_syn_tumor/698_templates_netcdf_256x256x256/ \


