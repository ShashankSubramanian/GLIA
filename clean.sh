#!/bin/bash

#### this is a clean script with all default options ####
#### for more options, use scons ####
#### frontera is the default platform value. Use other platforms only if needed ####
#### multigpu is deprecated due to accfft-gpu and unnecessary for res <= 256^3  ####

scons build=release compiler=mpicxx niftiio=no platform=frontera single_precision=yes gpu=yes multi_gpu=no -j 20 -c
