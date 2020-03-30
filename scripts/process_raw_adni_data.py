#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:58:46 2020

@author: biros

Create an empty  work directory and copy the script there. 

define adnidir and fsldir variables
define the museimg (for template)

subject parameters:
subject = '127_S_2234'    - subject folder in ADNI dir
pet_dir = '*Averaged'     - Pet directories (this shouldn't change)
t1_dir = 'Acc*_IR-*'      - T1 directories (this changes for each patient)

pet_years = [2016, 2017, 2018, 2019, 2020]  # how many  tau  scans, indexed by year

T1 directory labels; their number doesn't alayws match the PET scans; Length 
of t1_prflags must be equal to len(pet_years). If we don't have T1 matching a PET scan, we can  use empty string
t1_prflags = ['2016-12-15','','2018-01','2019','']   

Another example
subject = '022_S_6013'
pet_dir = '*Averaged'
t1_dir = '*MPRAGE'
pet_years = [2017, 2018, 2019]
t1_prflags = ['2017','2018','2019']

Key output files: (all rigidly registered to the first T1 scan)
Below %d is the scan index
t1_%ds.nii  - the stripped T1
t1_%ds_seg.nii - the segmented stripped T1
p_%ds.nii - the stripped PET 
p_%ds_scaled.nii - the stripped cerebellum scaled PET
muse.nii - template (this one is affinely registered to subject's first T1)
muselabel.nii - template labels (also affinely registered to subject's first T1)

"""

import os
#import nibabel
#import glob

# USER/PLATFORM-SPECIFIC PATHS
#adnidir = '~/data/brain/adni/'
adnidir = '/scratch/scheufele/alzh/ADNI'
fsldir = '/workspace/apps/fsl'
museimg = '/scratch/scheufele/data/muse//Template27'

os.environ["FSLOUTPUTTYPE"]="NIFTI"
os.environ["FSLDIR"]=fsldir
os.environ["FSLMULTIFILEQUIT"]="TRUE"
os.environ["PATH"]+=os.pathsep + os.path.join('%s/bin'%fsldir)

# su
subject = '127_S_2234'
pet_dir = '*Averaged'
t1_dir = 'Acc*_IR-*'
pet_years = [2016, 2017, 2018, 2019, 2020]
t1_prflags = ['2016-12-15','','2018-01','2019','']
# the assumption is t1_prflags[0] is always non-empty;
# The image corresponding to tr_prflags will become the rigid reg ref image.
assert(t1_prflags[0]) 

# Auxiliary function to execute an OS command.
def excmd(cmd,skip=False):
    print(cmd)
    if not skip:
        os.system(cmd)


print('\n\t GET FILES TO LOCAL DIRECTORY')
for i in range(len(pet_years)):
    pet_file = '%s%s/%s/%d*/*/*.nii' % (adnidir,subject,pet_dir,pet_years[i])
    excmd('cp %s p_%d.nii' % (pet_file,i))    
    if t1_prflags[i]:
        t1_file = '%s%s/%s/%s*/*/*.nii' % (adnidir,subject,t1_dir,t1_prflags[i])
        excmd('cp %s t1_%d.nii' % (t1_file,i))

print('\n\t RIGID REGISTRATION')
# rigid  registration of all images to t1_0.nii, the first time point
refimg = 't1_0.nii'
flirtoptions = lambda dof:\
    '-bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof %d -interp trilinear' \
    % dof
flirt = lambda inimg,refimg,outimg,outmat,dof:'flirt -in %s -ref %s -out %s -omat %s %s'\
        % (inimg,refimg,outimg,outmat,flirtoptions(dof))
rigid=6 # rigid registration; use 12 for affine registration
affine=12

for i in range(len(pet_years)):
    inimg = 'p_%d.nii'  % i
    outimg = 'p_%dr.nii' % i
    outmat = 'p_%dr.mat' % i
    flirtcmd = flirt(inimg,refimg,outimg,outmat,rigid)
    excmd(flirtcmd)
        
    if i>0 and t1_prflags[i]:
        inimg = 't1_%d.nii' % i
        outimg = 't1_%dr.nii' % i
        outmat = 't1_%dr.mat' % i
        flirtcmd = flirt(inimg,refimg,outimg,outmat,rigid)
        excmd(flirtcmd)
   
print('\n\t SKULL STRIPPING  (this phase is quite slow)')
bet = lambda imgin, imgout: 'bet %s %s -R -B' % (imgin,imgout)
betcmd = bet('t1_0.nii','t1_0s.nii')
excmd(betcmd)
t1_0_mask = 't1_0s_mask.nii'

#op one of : mul,div,add,sub,mas
fslbinop =  lambda imgin,op,imgop,imgout: 'fslmaths %s -%s %s %s' % (imgin,op,imgop,imgout)
fslmask = lambda imgin,imgmsk,imgout: fslbinop(imgin,'mas',imgmsk,imgout)

for i in range(len(pet_years)):
    imgin = 'p_%dr.nii' % i
    imgout= 'p_%ds.nii' % i    
    excmd(fslmask(imgin,t1_0_mask,imgout))
    
    if i>0 and t1_prflags[i]:
        inimg = 't1_%dr.nii' %  i
        outimg = 't1_%ds.nii' % i
        betcmd = bet(inimg,outimg)
        excmd(betcmd)
        
print('\t\n SEGMENTATION OF T1 IMAGES (this phase also takes some time)')        
fastoptions  = '-t 1 -n 3 -H 0.1 -I 4 -l 20.0'
fastseg = lambda inimg: 'fast %s -o %s' % (fastoptions,inimg)
for i in range(len(pet_years)):
    if t1_prflags[i]:
        inimg = 't1_%ds.nii' %  i
        excmd(fastseg(inimg))


print('\t\n NORMALIZATION with MUSE TEMPLATES')
refimg = 't1_0s.nii'    
inimg = '%s_str.nii' % museimg
outimg = 'muse_r.nii'
outmat = 'muse_r.mat'

print('First do rigid registration of the MUSE template to the t1_0 image')
excmd(flirt(inimg,refimg,'muse_r.nii','muse_r.mat',rigid))

print('Now do affine registration')
excmd('flirt -in muse_r.nii -ref %s -o muse.nii -omat muse.mat -dof 12 -interp nearestneighbour'% refimg)

print('Transfer the labels (in two steps, first rigit and then affine)')
excmd('flirt -in %s_label.nii.gz -ref %s -applyxfm -init muse_r.mat -out muselabel_r.nii' % (museimg,refimg))
excmd('flirt -in muselabel_r.nii -ref %s -applyxfm -init muse.mat -out muselabel.nii' % refimg)

print('Threshold labels to select cerebellum: values 38--41')
excmd('fslmaths muselabel.nii -thr 38 -uthr 41 cereb.nii')

print('Erode it because registration interpolation introduces false labels')
excmd('fslmaths cereb -ero cereb')

print('Finally extract the cerebellum white matter from the subject reference t1')
excmd('fslmaths t1_0s_seg -thr 3 whm')
excmd(fslmask('whm','cereb','whmcereb'))

print('And now scale all the PET scans with the average value of their white matter cerebellum')
for i in range(len(pet_years)):
    excmd(fslmask('p_%ds.nii'%i,'whmcereb','p_%ds_cereb'%i))
    cmd='fslstats p_%ds_cereb -M'%i
    print(cmd)
    sc =os.popen(cmd).readlines()
    sc = float(sc[0].splitlines()[0])
    print(sc)
    excmd('fslmaths p_%ds -div %f p_%ds_scaled' % (i,sc,i))
    



    
    

    
    
    
    
