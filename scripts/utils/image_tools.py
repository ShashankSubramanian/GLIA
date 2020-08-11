import os,sys
import numpy as np
from netCDF4 import Dataset
import argparse
import scipy.ndimage as ndimage
#import utils.file_io as fio
import file_io as fio
import nibabel as nib


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
def smoothBinaryMap(img,s=None):
    '''
    smoothes an image img with a gaussian kernel with uniform sigma=1
    '''
    if s is None:
        return ndimage.filters.gaussian_filter(img, sigma=(2,2,2), order=0);
    else:
        return ndimage.filters.gaussian_filter(img, sigma=(s,s,s), order=0);

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
#    print("old v size: {}, old sze: {}, new vx size: {}, new size: {}".format(old_voxel_size, old_size, new_voxel_size, new_size))
    return nib.processing.resample_to_output(img, new_voxel_size, order=interp_order, mode='wrap');

###
### ------------------------------------------------------------------------ ###
def extractImageLabels(img, label_dict):
    '''
    extracts image labels from the segmented image img
    label_dict is a dict storing what label is what tissue e.g. {0:'bg', 1:'nec', ...}
    '''
    print(label_dict)
    
    # reverse the label dictionary {0:'bg', 1:'nec', ...} ==> {'bg':0, 'nec':1, ...}
    label_dict_rev = {v:k for k,v in label_dict.items()}
    # create a list of label numbers
    labels = list(label_dict.keys());
    # number of labels
    num_labels = len(labels)
    # create a new long array for each label image
    newsize = (num_labels,) + np.shape(img);
    newimg = np.zeros(newsize);
    
    print(np.unique(img))
    # new label dict
    new_label_dict = {}
    # loop over labels, extract them and store in newimg
    for label,i in zip(labels,np.arange(num_labels)):
        binimg = img==label
        newimg[i,:,:,:] = binimg.astype(float);
        print('index = {}, label = {}, label_id = {}, min = {}, max = {}'.format(i,label_dict[label], label, np.amin(newimg[i,:,:,:]), np.amax(newimg[i,:,:,:])))
        # assign new label numbers in label dict
        new_label_dict[i] = label_dict[label]
    
    print(new_label_dict)
    # if mask is a label, then create it
    if 'mask' in label_dict_rev:
        # for mask label
        en = label_dict_rev['en'];
        nec = label_dict_rev['nec'];
        tc = np.logical_or(img == en, img == nec);
        tc = tc.astype(float);
        mask = 1.0 - tc;
        newimg[-1,:,:,:] = mask;

    return newimg, new_label_dict

###
### ------------------------------------------------------------------------ ###
def createProbabilityMaps(img, default_size, label_dict):
    '''
    creates probability maps given a segmented image img and saves them
    '''
    #create a stacked array of different labels
    label_img, new_label_dict = extractImageLabels(img, label_dict);
    # number of labels = length of the new_label_dict
    num_labels = len(new_label_dict)
    # new array of smoothed label images aka probability maps
    smoothed_label_img = np.zeros_like(label_img)
    #smooth the label images and resize them
    for i in np.arange(num_labels):
        # smooth label maps
        if i==num_labels-1:
            smoothed_label_img[i,:,:,:] = smoothBinaryMap(label_img[i,:,:,:],s=1);
        else:
            smoothed_label_img[i,:,:,:] = smoothBinaryMap(label_img[i,:,:,:]);
    #ensure parition of unity
    return smoothed_label_img, label_img, new_label_dict

###
### ------------------------------------------------------------------------ ###
def saveProbabilityMaps(img, labelimg, filename_prefix, ref_image, label_dict):
    '''
    saves probability maps and individual label maps as nifti and netcdf images
    BRATS {0:'bg', 1:'nec', 2:'ed', 4:'en', 5:'gm', 6:'wm', 7:'vt', 8:'csf'}
    '''

    # reverse key-value in label_dict
    label_dict_rev = {v:k for k,v in label_dict.items()}
    print(label_dict_rev)

    # handle special cases
    if 'vt' in label_dict_rev:
        vtlabel = label_dict_rev['vt']
        print('creating ', filename_prefix+'_vt');
        fio.writeNII(img[vtlabel,:,:,:], filename_prefix +'_vt.nii.gz', ref_image=ref_image);
    
    # combine csf and vt
    if 'csf' in label_dict_rev and 'vt' in label_dict_rev:
        csflabel = label_dict_rev['csf'];
        print('creating ', filename_prefix+'_csf_no_vt');
        fio.writeNII(img[csflabel,:,:,:], filename_prefix +'_csf_no_vt.nii.gz', ref_image=ref_image);

        vtlabel = label_dict_rev['vt'];
        img[csflabel,:,:,:] += img[vtlabel,:,:,:];
        labelimg[csflabel,:,:,:] += labelimg[vtlabel,:,:,:];


    if 'wm' in label_dict_rev and 'ed' in label_dict_rev:
        wmlabel = label_dict_rev['wm'];
        edlabel = label_dict_rev['ed'];
        wmed = img[wmlabel,:,:,:] + img[edlabel,:,:,:];
        print('creating ', filename_prefix+'_ed_wm');
        fio.writeNII(wmed, filename_prefix +'_ed_wm.nii.gz', ref_image=ref_image);
        fio.createNetCDF(filename_prefix+'_ed_wm.nc', np.shape(wmed), np.swapaxes(wmed,0,2));
    
    if 'wm' in label_dict_rev and 'csf' in label_dict_rev:
        wmlabel = label_dict_rev['wm']
        csflabel = label_dict_rev['csf']
        wmcsf = img[wmlabel,:,:,:] + img[csflabel,:,:,:]
        print('creating ', filename_prefix+'_wm_csf');
        fio.writeNII(wmcsf, filename_prefix +'_wm_csf.nii.gz', ref_image=ref_image);
        if 'ed' in label_dict_rev:
            edlabel = label_dict_rev['ed']
            edwmcsf = img[edlabel,:,:,:] + wmcsf
            print('creating ', filename_prefix+'_ed_wm_csf');
            fio.writeNII(edwmcsf, filename_prefix +'_ed_wm_csf.nii.gz', ref_image=ref_image);

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


    nosave_labels = ['vt', 'en', 'nec', 'bg'];
    for label in list(label_dict_rev.keys()):
        # write both nifti and netcdf files
        # save probability maps for use in registration
        if label not in nosave_labels:
            print('creating ', filename_prefix+'_'+label);
            i = label_dict_rev[label]
            fio.writeNII(img[i,:,:,:],  filename_prefix+'_'+label+'.nii.gz', ref_image=ref_image)
            fio.createNetCDF(filename_prefix+'_'+label+'.nc', np.shape(img[i,:,:,:]), np.swapaxes(img[i,:,:,:],0,2));
            # save individual label images for use in tumor inversion
            if label != 'mask':
                fio.writeNII(labelimg[i,:,:,:],  filename_prefix+'_seg_'+label+'.nii.gz', ref_image=ref_image)
                fio.createNetCDF(filename_prefix+'_seg_'+label+'.nc', np.shape(labelimg[i,:,:,:]), np.swapaxes(labelimg[i,:,:,:],0,2));

###
### ------------------------------------------------------------------------ ###
def getNIIImageList(path, exclude=None):
    file_list = [];
    for dirpath,_,filenames in os.walk(path):
        for f in filenames:
            if f.split(".").count('nii') == 1:
                file_list.append(os.path.abspath(os.path.join(dirpath,f)))
    return file_list

###
### ------------------------------------------------------------------------ ###
def getNetCDFImageList(path, exclude=None):
    file_list = [];
    special_list = ["c0_rec", "data", "seg"]
    for dirpath,_,filenames in os.walk(path):
        for f in filenames:
            #if f.split(".").count('nc') == 1: 
            if f.split(".").count('nc') == 1 and any(substring in f for substring in special_list):
                file_list.append(os.path.abspath(os.path.join(dirpath,f)))
    return file_list
