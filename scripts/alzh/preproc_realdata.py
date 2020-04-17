#!/bin/python
import nibabel as nib
import nibabel.processing
import os, sys
import ntpath
import numpy as np
import scipy.ndimage as ndimage
import netCDF4
from netCDF4 import Dataset
import skimage
from skimage.util import random_noise
import scipy as sp
import argparse


###
### ------------------------------------------------------------------------ ###
def resizeImage(img, new_size, interp_order):
    '''
    resize image to new_size
    '''
    factor = tuple([float(x)/float(y) for x,y in zip(list(new_size), list(np.shape(img)))]);
    return ndimage.zoom(img, factor, order=interp_order);

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
def resizeNIIImage(img, new_size, interp_order):
    '''
    uses nifti img object to resize it to new_size
    '''
    # load the image
    old_voxel_size = img.header['pixdim'][1:4];
    old_size = img.header['dim'][1:4];
    new_size = np.asarray(new_size);
    new_voxel_size = np.multiply(old_voxel_size, np.divide(old_size, new_size));
    return nib.processing.resample_to_output(img, new_voxel_size, order=0, mode='wrap');

### ------------------------------------------------------------------------ ###
def createNetCDF(filename,dimensions,variable):
    '''
    function to write a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename,mode='w',format="NETCDF3_CLASSIC");
    x = imgfile.createDimension("x",dimensions[0]);
    y = imgfile.createDimension("y",dimensions[1]);
    z = imgfile.createDimension("z",dimensions[2]);
    data = imgfile.createVariable("data","f8",("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    imgfile.close();

###
### ------------------------------------------------------------------------ ###
def readNetCDF(filename):
    '''
    function to read a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename);
    img = imgfile.variables['data'][:]
    imgfile.close();
    return img

###
### ------------------------------------------------------------------------ ###
def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename);

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
        writeNII(img[vtlabel,:,:,:], filename_prefix +'_vt.nii.gz', ref_image=ref_image);

    # combine csf and vt
    if 'csf' in label_dict_rev and 'vt' in label_dict_rev:
        csflabel = label_dict_rev['csf'];
        print('creating ', filename_prefix+'_csf_no_vt');
        #writeNII(img[csflabel,:,:,:], filename_prefix +'_csf_no_vt.nii.gz', ref_image=ref_image);
        # createNetCDF(filename_prefix+'_csf_no_vt.nc', np.shape(img[csflabel,:,:,:]), np.swapaxes(img[csflabel,:,:,:],0,2));

        vtlabel = label_dict_rev['vt'];
        img[csflabel,:,:,:] += img[vtlabel,:,:,:];
        labelimg[csflabel,:,:,:] += labelimg[vtlabel,:,:,:];


    if 'wm' in label_dict_rev and 'ed' in label_dict_rev:
        wmlabel = label_dict_rev['wm'];
        edlabel = label_dict_rev['ed'];
        wmed = img[wmlabel,:,:,:] + img[edlabel,:,:,:];
        print('creating ', filename_prefix+'_ed_wm');
        writeNII(wmed, filename_prefix +'_ed_wm.nii.gz', ref_image=ref_image);
        createNetCDF(filename_prefix+'_ed_wm.nc', np.shape(wmed), np.swapaxes(wmed,0,2));
        #print('creating ', filename_prefix+'_wm_no_ed');
        # writeNII(img[wmlabel,:,:,:], filename_prefix +'_wm_no_ed.nii.gz', ref_image=ref_image);

    if 'wm' in label_dict_rev and 'csf' in label_dict_rev:
        wmlabel = label_dict_rev['wm']
        csflabel = label_dict_rev['csf']
        wmcsf = img[wmlabel,:,:,:] + img[csflabel,:,:,:]
        # print('creating ', filename_prefix+'_wm_csf');
        # writeNII(wmcsf, filename_prefix +'_wm_csf.nii.gz', ref_image=ref_image);
        if 'ed' in label_dict_rev:
            edlabel = label_dict_rev['ed']
            edwmcsf = img[edlabel,:,:,:] + wmcsf
            print('creating ', filename_prefix+'_ed_wm_csf');
            # writeNII(edwmcsf, filename_prefix +'_ed_wm_csf.nii.gz', ref_image=ref_image);

    if 'en' in label_dict_rev and 'nec' in label_dict_rev:
        enlabel = label_dict_rev['en'];
        neclabel = label_dict_rev['nec'];
        tc = img[neclabel,:,:,:] + img[enlabel,:,:,:];
        tcl = labelimg[neclabel,:,:,:] + labelimg[enlabel,:,:,:];

        # print('creating ', filename_prefix+'_tc');
        # writeNII(tc, filename_prefix +'_tc.nii.gz', ref_image=ref_image);
        # writeNII(tcl, filename_prefix +'_seg_tc.nii.gz', ref_image=ref_image);
        #createNetCDF(filename_prefix+'_tc.nc', np.shape(tc), np.swapaxes(tc,0,2));
        #createNetCDF(filename_prefix+'_seg_tc.nc', np.shape(tcl), np.swapaxes(tcl,0,2));

        wtl = tcl;
        wt = tc;
        if 'ed' in label_dict_rev:
            edlabel = label_dict_rev['ed'];
            # make whole tumor
            wtl = wtl + labelimg[edlabel,:,:,:];
            wt = wt + img[edlabel,:,:,:];

        # writeNII(wt, filename_prefix +'_wt.nii.gz', ref_image=ref_image);
        # writeNII(wtl, filename_prefix +'_seg_wt.nii.gz', ref_image=ref_image);
        # createNetCDF(filename_prefix+'_wt.nc', np.shape(wt), np.swapaxes(wt,0,2));
        # createNetCDF(filename_prefix+'_seg_wt.nc', np.shape(wtl), np.swapaxes(wtl,0,2));

        # if 'wm' in label_dict_rev:
            # wmlabel = label_dict_rev['wm'];
            # wmwtl = labelimg[wmlabel,:,:,:] + wtl;
            # wmwt = img[wmlabel,:,:,:] + wt;
            #writeNII(wmwt, filename_prefix +'_wm_wt.nii.gz', ref_image=ref_image);
            # writeNII(wmwtl, filename_prefix +'_seg_wm_wt.nii.gz', ref_image=ref_image);
            #createNetCDF(filename_prefix+'_wm_wt.nc', np.shape(wmwt), np.swapaxes(wmwt,0,2));
            #createNetCDF(filename_prefix+'_seg_wm_wt.nc', np.shape(wmwtl), np.swapaxes(wmwtl,0,2));


    nosave_labels = ['vt', 'en', 'nec', 'bg'];
    for label in list(label_dict_rev.keys()):
        # write both nifti and netcdf files
        # save probability maps for use in registration
        if label not in nosave_labels:
            print('creating ', filename_prefix+'_'+label);
            i = label_dict_rev[label]
            writeNII(img[i,:,:,:],  filename_prefix+'_'+label+'.nii.gz', ref_image=ref_image)
            createNetCDF(filename_prefix+'_'+label+'.nc', np.shape(img[i,:,:,:]), np.swapaxes(img[i,:,:,:],0,2));
            # save individual label images for use in tumor inversion
            if label != 'mask':
                writeNII(labelimg[i,:,:,:],  filename_prefix+'_seg_'+label+'.nii.gz', ref_image=ref_image)
                createNetCDF(filename_prefix+'_seg_'+label+'.nc', np.shape(labelimg[i,:,:,:]), np.swapaxes(labelimg[i,:,:,:],0,2));

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

    # new label dict
    new_label_dict = {}
    # loop over labels, extract them and store in newimg
    for label,i in zip(labels,np.arange(num_labels)):
        binimg = img==label
        newimg[i,:,:,:] = binimg.astype(float);
        print('index = {}, label = {}, label_id = {}, min = {}, max = {}'.format(i,label_dict[label], label, np.amin(newimg[i,:,:,:]), np.amax(newimg[i,:,:,:])))
        # assign new label numbers in label dict
        new_label_dict[i] = label_dict[label]

    return newimg, new_label_dict


###
### ------------------------------------------------------------------------ ###
def createProbabilityMaps(img, default_size, label_dict, smooth=False):
    '''
    creates probability maps given a segmented image img and saves them
    '''
    #create a stacked array of different labels
    label_img, new_label_dict = extractImageLabels(img, label_dict);
    # number of labels = length of the new_label_dict
    num_labels = len(new_label_dict)
    # new array of smoothed label images aka probability maps
    if smooth:
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
    else:
        return label_img, new_label_dict


###
### ------------------------------------------------------------------------ ###
def preprocImageFromSegmentation(image_path, output_path, resolution, labels, name):
    '''
    @short - read segmented image and split into probability maps (anatomy labels)
    '''
    print('preprocessing patient segmentation maps, converting to probability maps')
    if not os.path.exists(output_path):
        print('preprocessing output path does not exist, creating ', output_path);
        os.mkdir(output_path);

    img = nib.load(image_path);
    affine = img.affine;

    resolution = tuple([float(x) for x in resolution]);
    label_rev = {v:k for k,v in labels.items()}

    # do NN interpolation on the image and store in input/
    if not resolution == img.shape:
        #img_resized = resizeImage(img.get_fdata(), resolution, 0);
        img_resized = resizeNIIImage(img, resolution, interp_order=0)
    else:
        img_resized = img

    # new refernce image
    ref_image = img_resized
    img_resized = img_resized.get_fdata()
    writeNII(img_resized, output_path +'/' + name + '_seg.nii.gz', ref_image=ref_image)
    createNetCDF(output_path + '/' + name +'_seg.nc', np.shape(img_resized), np.swapaxes(img_resized,0,2));
    # create probability maps from segmentation
    probmaps, labelmaps, new_label_dict = createProbabilityMaps(img_resized, resolution, labels, smooth=True);

    if not os.path.exists(output_path):
        os.mkdirs(output_path)
    saveProbabilityMaps(probmaps, labelmaps, output_path + '/' + name, ref_image, new_label_dict);
    return ref_image



###
### ------------------------------------------------------------------------ ###
parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
parser.add_argument ('-subj',        type = str,          help = 'ADNI subject');
args = parser.parse_args();

label_dict = { 0: 'bg', 2 : 'gm', 3 : 'wm', 1 : 'csf'}

tp_dict = { 
    '127_S_4301' : {'t0' : 1, 't1' : 2},  # NO REG
    '127_S_2234' : {'t0' : 2, 't1' : 4},  # NO REG
    '022_S_6013' : {'t0' : 1, 't1' : 2},
    '023_S_1190' : {'t0' : 1, 't1' : 2},
    '012_S_6073' : {'t0' : 0, 't1' : 1}, # NO REG
    '941_S_4036' : {'t0' : 0, 't1' : 1}, #
    '033_S_4179' : {'t0' : 1, 't1' : 2}, # NO REG
    '032_S_5289' : {'t0' : 0, 't1' : 2}, # NO REG
    '035_S_4114' : {'t0' : 0, 't1' : 1}, #
    }
reg_dict = {'127_S_4301' : False, '127_S_2234' : False, '022_S_6013' : True, '023_S_1190' : True, '012_S_6073' : False, '941_S_4036' : True, '033_S_4179' : False,
'032_S_5289' : False, '035_S_4114' : True}

DIRS = os.listdir(args.x)
SUBJ = {}
for key in tp_dict.keys():
    max_val_tau_longitudinal = -1;
    max_tp = ''; 
    for tp in tp_dict[key].items():
        tp = tp[1]
        file_tau = 'p_'+str(tp)+'s_scaled.nii';
        file_t1  = 't1_'+str(tp)+'s.nii';
        file_seg = 't1_'+str(tp)+'s_seg.nii';
        case_out_dir = os.path.join(os.path.join('CASE_' + key, 'data'));
        case_out_dir_true = os.path.join(os.path.join('CASE_' + key, 'tc'));
        case_path_to_tau = os.path.join(args.x,os.path.join(key, file_tau));
        tau_img = nib.load(case_path_to_tau);
        max = np.amax(tau_img.get_fdata());
        if max > max_val_tau_longitudinal:
            max_val_tau_longitudinal = max;
            max_tp = key;

    wm_mask = np.zeros((256,256,256))
    wm_mask_resized = np.zeros((256,256,256))
    for time_point, tp in zip(tp_dict[key].items(), range(len(tp_dict[key].items()))):
        time_point = time_point[1]
        file_tau = 'p_'+str(time_point)+'s_scaled.nii';
        if reg_dict[key]:
            file_t1  = 't1_'+str(time_point)+'s.nii';
            file_seg = 't1_'+str(time_point)+'s_seg.nii';
        else:
            print("NO REGISTRATION, NO T1")
            file_t1  = 't1_'+str(tp_dict[key]['t0'])+'s.nii';
            file_seg = 't1_'+str(tp_dict[key]['t0'])+'s_seg.nii';

        print(" ... processing subject {} at time point {} ".format(key, time_point))
        case_out_dir = os.path.join(os.path.join('CASE_' + key, 'data'));
        case_out_dir_true = os.path.join(os.path.join('CASE_' + key, 'tc'));
        if not os.path.exists(case_out_dir):
            os.makedirs(case_out_dir);
        if not os.path.exists(case_out_dir_true):
            os.makedirs(case_out_dir_true);

        print("subject {} time point {}: using T1  file: {}".format(key, time_point, file_t1))
        print("subject {} time point {}: using SEG file: {}".format(key, time_point, file_seg))
        print("subject {} time point {}: using TAU file: {}".format(key, time_point, file_tau))
        case_path_to_seg = os.path.join(args.x,os.path.join(key, file_seg));
        case_path_to_t1  = os.path.join(args.x,os.path.join(key, file_t1));
        case_path_to_tau = os.path.join(args.x,os.path.join(key, file_tau));

        preprocImageFromSegmentation(case_path_to_seg, case_out_dir, [256,256,256], label_dict, "time_point_"+str(tp))

        res = tuple([256,256,256]);
        tau_img = nib.load(case_path_to_tau);
        t1_img  = nib.load(case_path_to_t1);
        seg_img  = nib.load(case_path_to_seg);

        print("normalizing tau concentration in longitudinal series. Max value is {} from time point {}".format(max_val_tau_longitudinal, max_tp))
        tau_data = tau_img.get_fdata()
        tau_data /= max_val_tau_longitudinal;

        if tp==0:
            wm_mask = (smoothBinaryMap((seg_img.get_fdata()==250).astype(float), s=1) > 1E-1).astype(float);
            writeNII(wm_mask, os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_wm_mask.nii.gz'), ref_image=tau_img);
            seg_resized = resizeNIIImage(seg_img, res, interp_order=0)
            wm_mask_resized = (smoothBinaryMap((seg_resized.get_fdata()==250).astype(float), s=1) > 1E-1).astype(float);

        writeNII(np.where(tau_data > 0.6, tau_data, 0), os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_tau_th06.nii.gz'), ref_image=tau_img);
        writeNII(np.where(tau_data > 0.4, tau_data, 0), os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_tau_th04.nii.gz'), ref_image=tau_img);
        writeNII(np.where(tau_data > 0.2, tau_data, 0), os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_tau_th02.nii.gz'), ref_image=tau_img);
        writeNII(tau_data, os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_tau.nii.gz'), ref_image=tau_img);
        writeNII(np.multiply(tau_data, wm_mask), os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_tau_wm.nii.gz'), ref_image=tau_img);
        writeNII(t1_img.get_fdata(), os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_t1.nii.gz'), ref_image=t1_img);
        writeNII(seg_img.get_fdata(), os.path.join(case_out_dir_true, "time_point_"+str(tp) + '_seg.nii.gz'), ref_image=t1_img);

        for name, img in zip(['tau'],[tau_img]):
            affine = img.affine;
            if not res == img.shape:
                img_resized = resizeNIIImage(img, res, interp_order=0)
            else:
                img_resized = img

            ref_image = img_resized
            img_resized = img_resized.get_fdata()
            img_tresh   = img_resized
            if 'tau' in name:
                print("normalizing tau concentration in longitudinal series. Max value is {} from time point {}".format(max_val_tau_longitudinal, max_tp))
                img_resized /= max_val_tau_longitudinal;
                img_thresh06 = np.where(img_resized > 0.6, img_resized, 0);
                img_thresh04 = np.where(img_resized > 0.4, img_resized, 0);
                img_thresh02 = np.where(img_resized > 0.2, img_resized, 0);
                writeNII(img_thresh06, os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_th06.nii.gz'), ref_image=ref_image)
                writeNII(img_thresh04, os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_th04.nii.gz'), ref_image=ref_image)
                writeNII(img_thresh02, os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_th02.nii.gz'), ref_image=ref_image)
                createNetCDF(os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_th06.nc'), np.shape(img_thresh06), np.swapaxes(img_thresh06,0,2));
                createNetCDF(os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_th04.nc'), np.shape(img_thresh04), np.swapaxes(img_thresh04,0,2));
                createNetCDF(os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_th02.nc'), np.shape(img_thresh02), np.swapaxes(img_thresh02,0,2));

            writeNII(img_resized, os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '.nii.gz'), ref_image=ref_image)
            createNetCDF(os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '.nc'), np.shape(img_resized), np.swapaxes(img_resized,0,2));
 
            writeNII(np.multiply(img_resized,wm_mask_resized), os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_wm.nii.gz'), ref_image=ref_image)
            createNetCDF(os.path.join(case_out_dir, "time_point_"+str(tp) + '_' + name + '_wm.nc'), np.shape(img_resized), np.swapaxes(np.multiply(img_resized,wm_mask_resized),0,2));



