import os,sys
import numpy as np
from netCDF4 import Dataset
import argparse
import scipy.ndimage as ndimage
import file_io as fio
import nibabel as nib
from nibabel import processing


###
### ------------------------------------------------------------------------ ###
def EnsurePartitionOfUnity(img):
    img[np.where(img<=0)]=0;
    img[np.where(img>=1)]=1;
    tot = np.sum(img,0);
    bg = 1-tot;
    bg[np.where(bg<=0)]=0;
    bg[np.where(bg>=1)]=1;
    tot += bg;
    return np.divide(img,tot);

###
### ------------------------------------------------------------------------ ###
def smoothBinaryMap(img):
    '''
    smoothes an image img with a gaussian kernel with uniform sigma=1
    '''
    return ndimage.filters.gaussian_filter(img, sigma=(1, 1, 1), order=0);

###
### ------------------------------------------------------------------------ ###
def ensurePartitionOfUnity(img):
    '''
    ensures partition of unity in the vector image img
    '''
    #img[np.where(img<=0)]=0;
    #img[np.where(img>=1)]=1;
    tot = np.sum(img,0);
    bg = 1-tot;
    bg[np.where(bg<=0)]=0;
    bg[np.where(bg>=1)]=1;
    tot += bg;
    return np.divide(img,tot);

###
### ------------------------------------------------------------------------ ###
def resizeImage(img, new_size, interp_order):
    '''
    resize image to new_size
    '''
    factor = tuple([float(x)/float(y) for x,y in zip(list(new_size), list(np.shape(img)))]);
    #print(factor);
    return ndimage.zoom(img, factor, order=interp_order);

###
### ------------------------------------------------------------------------ ###
def resizeNIIImage(img, new_size, interp_order):
    '''
    uses nifti img object to resize it to new_size
    '''
    # load the image
    old_voxel_size = img.header['pixdim'][1:4];
    old_size = img.header['dim'][1:4];
    new_size = np.asarray(new_size);
    new_voxel_size = np.multiply(old_voxel_size, np.divide(old_size, new_size));
    # print("old_size ",old_size)
    # print("new_size ",new_size)
    # print("old_voxel_size ", old_voxel_size)
    # print("new_voxel_size ", new_voxel_size)
    return nib.processing.resample_to_output(img, new_voxel_size, order=interp_order, mode='wrap');

###
### ------------------------------------------------------------------------ ###
def extractImageLabels(img, label_dict):
    '''
    extracts image labels from the segmented image img
    label_dict is a dict storing what label is what tissue
    '''    
    labelnums = list(label_dict.keys());
    numlabels = max(labelnums) - min(labelnums) + 1; # includes missing labels if any in the input
    newsize = (numlabels,) + np.shape(img);
    newimg = np.zeros(newsize,dtype=float);

    for label in labelnums:        
        newimg[label,:,:,:] = (img==label).astype(float)

    label_dict_rev = {v:k for k,v in label_dict.items()}
    if 'mask' in label_dict_rev:
        # for mask label
        en = label_dict_rev['en'];
        nec = label_dict_rev['nec'];
        tc = np.logical_or(img == en, img == nec);
        tc = tc.astype(float);
        mask = 1.0 - tc;
        newimg[-1,:,:,:] = mask;

    return newimg, labelnums, numlabels

###
### ------------------------------------------------------------------------ ###
def createProbabilityMaps(img, default_size, label_dict):
    '''
    creates probability maps given a segmented image img and saves them
    '''
    #create a stacked array of different labels
    label_img,labelnums,numlabels = extractImageLabels(img, label_dict);
    #print('labelnums', labelnums)
    smoothed_label_img = np.zeros((numlabels,) + tuple([int(x) for x in default_size]));
    #smooth the label images
    for i in labelnums:
        # smooth label maps
        smoothed_label_img[i,:,:,:] = smoothBinaryMap(label_img[i,:,:,:]);
    #ensure parition of unity
    return smoothed_label_img, label_img

###
### ------------------------------------------------------------------------ ###
def saveProbabilityMaps(img, labelimg, filename_prefix, ref_image, label_dict):
    '''
    saves probability maps and individual label maps as nifti and netcdf images
    BRATS {0:'bg', 1:'nec', 2:'ed', 3:'en', 4:'gm', 5:'wm', 6:'vt', 7:'csf'}
    wt = wt
    '''

    # reverse key-value in label_dict
    label_dict_rev = {v:k for k,v in label_dict.items()}

    # handle special cases
    # combine csf and vt
    # if 'csf' in label_dict_rev and 'vt' in label_dict_rev:
    #     csflabel = label_dict_rev['csf'];
    #     vtlabel = label_dict_rev['vt'];
    #     img[csflabel,:,:,:] += img[vtlabel,:,:,:];
    #     labelimg[csflabel,:,:,:] += labelimg[vtlabel,:,:,:];

    if 'wm' in label_dict_rev and 'ed' in label_dict_rev:
        wmlabel = label_dict_rev['wm'];
        edlabel = label_dict_rev['ed'];
        wmed = img[wmlabel,:,:,:] + img[edlabel,:,:,:];
        print('creating ', filename_prefix+'_ed_wm');
        fio.writeNII(wmed, filename_prefix +'_ed_wm.nii.gz', ref_image=ref_image);
        fio.createNetCDF(filename_prefix+'_ed_wm.nc', np.shape(wmed), np.swapaxes(wmed,0,2));

    if 'wm' in label_dict_rev and 'csf' in label_dict_rev:
        wmlabel = label_dict_rev['wm'];        
        csflabel = label_dict_rev['csf'];
        wmcsf = img[wmlabel,:,:,:] + img[csflabel,:,:,:];
        print('creating ', filename_prefix+'_wm_csf');
        fio.writeNII(wmcsf, filename_prefix +'_wm_csf.nii.gz', ref_image=ref_image);
        fio.createNetCDF(filename_prefix+'_wm_csf.nc', np.shape(wmcsf), np.swapaxes(wmcsf,0,2));


    if 'wm' in label_dict_rev and 'ed' in label_dict_rev and 'csf' in label_dict_rev:
        wmlabel = label_dict_rev['wm'];
        edlabel = label_dict_rev['ed'];
        csflabel = label_dict_rev['csf'];
        edwmcsf = img[wmlabel,:,:,:] + img[edlabel,:,:,:] + img[csflabel,:,:,:];
        print('creating ', filename_prefix+'_ed_wm_csf');
        fio.writeNII(edwmcsf, filename_prefix +'_ed_wm_csf.nii.gz', ref_image=ref_image);
        fio.createNetCDF(filename_prefix+'_ed_wm_csf.nc', np.shape(edwmcsf), np.swapaxes(edwmcsf,0,2));

    if 'en' in label_dict_rev and 'nec' in label_dict_rev:
        enlabel = label_dict_rev['en'];
        neclabel = label_dict_rev['nec'];
        tc = img[neclabel,:,:,:] + img[enlabel,:,:,:];
        tcl = labelimg[neclabel,:,:,:] + labelimg[enlabel,:,:,:];

        print('creating ', filename_prefix+'_tc');
        fio.writeNII(tc, filename_prefix +'_tc.nii.gz', ref_image=ref_image);
        fio.writeNII(tcl, filename_prefix +'_seg_tc.nii.gz', ref_image=ref_image);
        fio.createNetCDF(filename_prefix+'_tc.nc', np.shape(tc), np.swapaxes(tc,0,2));
        fio.createNetCDF(filename_prefix+'_seg_tc.nc', np.shape(tcl), np.swapaxes(tcl,0,2));

        wtl = tcl;
        wt = tc;
        if 'ed' in label_dict_rev:
            edlabel = label_dict_rev['ed'];
            # make whole tumor
            wtl = wtl + labelimg[edlabel,:,:,:];
            wt = wt + img[edlabel,:,:,:];

        #fio.writeNII(wt, filename_prefix +'_wt.nii.gz', ref_image=ref_image);
        fio.writeNII(wtl, filename_prefix +'_seg_wt.nii.gz', ref_image=ref_image);
        #fio.createNetCDF(filename_prefix+'_wt.nc', np.shape(wt), np.swapaxes(wt,0,2));
        fio.createNetCDF(filename_prefix+'_seg_wt.nc', np.shape(wtl), np.swapaxes(wtl,0,2));

        if 'wm' in label_dict_rev:
            wmlabel = label_dict_rev['wm'];
            wmwtl = labelimg[wmlabel,:,:,:] + wtl;
            wmwt = img[wmlabel,:,:,:] + wt;
            #fio.writeNII(wmwt, filename_prefix +'_wm_wt.nii.gz', ref_image=ref_image);
            fio.writeNII(wmwtl, filename_prefix +'_seg_wm_wt.nii.gz', ref_image=ref_image);
            #fio.createNetCDF(filename_prefix+'_wm_wt.nc', np.shape(wmwt), np.swapaxes(wmwt,0,2));
            fio.createNetCDF(filename_prefix+'_seg_wm_wt.nc', np.shape(wmwtl), np.swapaxes(wmwtl,0,2));


    # nosave_labels = ['vt','en', 'nec', 'bg'];
    nosave_labels = ['en', 'nec', 'bg'];
    for i in list(label_dict.keys()):
        # write both nifti and netcdf files
        # save probability maps for use in registration
        if label_dict[i] not in nosave_labels:
            print('creating ', filename_prefix+'_'+label_dict[i]);
            fio.writeNII(img[i,:,:,:],  filename_prefix+'_'+label_dict[i]+'.nii.gz', ref_image=ref_image)
            fio.createNetCDF(filename_prefix+'_'+label_dict[i]+'.nc', np.shape(img[i,:,:,:]), np.swapaxes(img[i,:,:,:],0,2));
            # save individual label images for use in tumor inversion
            if label_dict[i] != 'mask':
                fio.writeNII(labelimg[i,:,:,:],  filename_prefix+'_seg_'+label_dict[i]+'.nii.gz', ref_image=ref_image)
                fio.createNetCDF(filename_prefix+'_seg_'+label_dict[i]+'.nc', np.shape(labelimg[i,:,:,:]), np.swapaxes(labelimg[i,:,:,:],0,2));

###
### ------------------------------------------------------------------------ ###
def getNIIImageList(path):
    file_list = [];
    for dirpath,_,filenames in os.walk(path):
        for f in filenames:
            if f[-7:]=='.nii.gz' or f[-4:]=='.nii':
                file_list.append(os.path.abspath(os.path.join(dirpath,f)))
    return file_list

###
### ------------------------------------------------------------------------ ###
def getNetCDFImageList(path):
    file_list = [];
    for dirpath,_,filenames in os.walk(path):
        for f in filenames:
            if f[-3:]=='.nc':
                file_list.append(os.path.abspath(os.path.join(dirpath,f)))
    return file_list
