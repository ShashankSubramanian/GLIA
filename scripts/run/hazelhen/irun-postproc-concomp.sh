#!/bin/bash
  
for DIR in $1/Brats*;
do
  python concomp.py -x $DIR -data -sol
done
