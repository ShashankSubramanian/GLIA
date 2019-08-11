import os,argparse
import numpy as np
import nibabel as nib
import imageTools as imgtools
import file_io as fio
import ntpath
from netCDF4 import Dataset


###
### ------------------------------------------------------------------------ ###
def preprocPatientFromProbmaps(image_path, output_path, resolution, labels):
    """
    @short - read patient probamps, resize, and write out
    """
    return preprocImageFromProbmaps(image_path, output_path, resolution, labels, 'paient');

###
### ------------------------------------------------------------------------ ###
def preprocAtlasFromProbmaps(image_path, output_path, resolution, labels):
    """
    @short - read atlas probamps, resize, and write out
    """
    return preprocImageFromProbmaps(image_path, output_path, resolution, labels, 'atlas');

###
### ------------------------------------------------------------------------ ###
def preprocImageFromProbmaps(image_path, output_path, resolution, labels, name):
    """
    @short - read anatomy probamps, resize, and write out
    """
    print('preprocessing atlas probability maps')
    if not os.path.exists(output_path):
        print('preprocessing output path does not exist, creating ', output_path);
        os.mkdir(output_path);

    file_list = imgtools.getNIIImageList(image_path);
    resolution = tuple([float(x) for x in resolution]);
    # read images one by one and resize them
    for f in file_list:
        img = nib.load(f);
        affine = img.affine;
        data = img.get_fdata();
        newdata = imgtools.resizeImage(data, tuple(resolution), 1);

        # separate the filename and extension from the absolute path
        output_filename, extension  = os.path.splitext(ntpath.basename(f));
        if output_filename[-3:] == 'nii' or output_filename[-2:] == 'nc':
            output_filename, ext = os.path.splitext(output_filename);
        # get the label name by splitting at "_"
        labelname = output_filename.split("_")[-1];
        # setup new output filename
        output_filename_nii = os.path.join(output_path, name + '_' + labelname + '.nii.gz');
        print('creating ', output_filename_nii)
        output_filename_nc = os.path.join(output_path,  name + '_' + labelname + '.nc');
        print('creating ', output_filename_nc)
        # write NII
        fio.writeNII(newdata, output_filename_nii, affine);
        # write Netcdf
        fio.createNetCDF(output_filename_nc , np.shape(newdata), np.swapaxes(newdata,0,2));
    return affine;


###
### ------------------------------------------------------------------------ ###
def preprocPatientFromSegmentation(image_path, output_path, resolution, labels):
    """
    @short - read segmented patient and split into probability maps (anatomy labels)
    """
    return preprocImageFromSegmentation(image_path, output_path, resolution, labels, "patient");

###
### ------------------------------------------------------------------------ ###
def preprocAtlasFromSegmentation(image_path, output_path, resolution, labels):
    """
    @short - read segmented atlas and split into probability maps (anatomy labels)
    """
    return preprocImageFromSegmentation(image_path, output_path, resolution, labels, "atlas");

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

    img_img = img.get_fdata();
    resolution = tuple([float(x) for x in resolution]);
    label_rev = {v:k for k,v in labels.items()}
    if 'vt' in label_rev and 'csf' in label_rev:
        img_img[np.where(img_img == label_rev['vt'])] = label_rev['csf'];

    # do NN interpolation on the image and store in input/
    img_resized = imgtools.resizeImage(img_img, resolution, 0);
    fio.writeNII(img_resized, output_path +'/' + name + '_seg.nii.gz', affine);
    fio.createNetCDF(output_path + '/' + name +'_seg.nc', np.shape(img_resized), np.swapaxes(img_resized,0,2));

    # create probability maps from segmentation
    probmaps,labelmaps = imgtools.createProbabilityMaps(img_resized, resolution, labels);
    probmaps = imgtools.ensurePartitionOfUnity(probmaps);

    if not os.path.exists(output_path):
        os.mkdirs(output_path)
    imgtools.saveProbabilityMaps(probmaps, labelmaps, output_path + '/' + name, affine, labels);
    return affine

###
### ------------------------------------------------------------------------ ###
def postprocDisplacementField(args):
    '''
    @short - cocomputes norm of displacement field in aspace and pspace,
           - downsamples velocity
    '''
    reg_path = os.path.join(args.base_path,'registration');

    # compute displacement norms
    dispx1 = fio.readNetCDF(reg_path+"/displacement-field-x1.nc");
    dim = list(np.shape(dispx1));
    dispx2 = fio.readNetCDF(reg_path+"/displacement-field-x2.nc");
    dispx3 = fio.readNetCDF(reg_path+"/displacement-field-x3.nc");
    dispnorm = np.sqrt(np.square(dispx1) + np.square(dispx2) + np.square(dispx3));
    print('creating ',reg_path+"/displacement-field-norm.nc")
    fio.createNetCDF(reg_path+"/displacement-field-norm.nc",dim,dispnorm);

    dispx1 = fio.readNetCDF(reg_path+"/displacement-field-x1_in_Pspace.nc");
    dispx2 = fio.readNetCDF(reg_path+"/displacement-field-x2_in_Pspace.nc");
    dispx3 = fio.readNetCDF(reg_path+"/displacement-field-x3_in_Pspace.nc");
    dispnorm = np.sqrt(np.square(dispx1) + np.square(dispx2) + np.square(dispx3));
    print('creating ',reg_path+"/displacement-field-norm_in_Pspace.nc")
    fio.createNetCDF(reg_path+"/displacement-field-norm_in_Pspace.nc",dim,dispnorm);

    # resample velocity to 128^3 and
    v1 = fio.readNetCDF(reg_path+"/velocity-field-x1.nc");
    v2 = fio.readNetCDF(reg_path+"/velocity-field-x2.nc");
    v3 = fio.readNetCDF(reg_path+"/velocity-field-x3.nc");
    s = tuple([128,128,128]);
    v1 = imgtools.resizeImage(v1, s, 3);
    v2 = imgtools.resizeImage(v2, s, 3);
    v3 = imgtools.resizeImage(v3, s, 3);
    dim = [128, 128, 128];
    fio.createNetCDF(reg_path+"/128_velocity-field-x1.nc",dim,v1);
    fio.createNetCDF(reg_path+"/128_velocity-field-x2.nc",dim,v2);
    fio.createNetCDF(reg_path+"/128_velocity-field-x3.nc",dim,v3);


###
### ------------------------------------------------------------------------ ###
def extraxtLabelsFromDeformedAtlas(args, atlas_labels, patient_labels):
    """
    @short - separates segmented deformed atlas (wapred-to-pspace) img into
             wm, gm, csf anatomy labels, not smoothed.
    """
    reg_path = os.path.join(args.base_path,'registration');
    input_path = os.path.join(args.base_path,'input');
    return extraxtHealthyLabelsFromSegmentedImage(args, atlas_labels, patient_labels, reg_path, reg_path, 'atlas_in_Pspace_seg.nc');


###
### ------------------------------------------------------------------------ ###
def extraxtHealthyLabelsFromSegmentedImage(args, atlas_labels, patient_labels, inpath, outpath, name):
    """
    @short - separates segmented deformed atlas (wapred-to-pspace) img into
             wm, gm, csf anatomy labels, not smoothed.
    """
    aseg = fio.readNetCDF(inpath + "/" + name);
    atlas_labels = {v:k for k,v in atlas_labels.items()};
    acsf = (aseg == atlas_labels['csf']).astype(float);
    agm = (aseg == atlas_labels['gm']).astype(float);
    awm = (aseg == atlas_labels['wm']).astype(float);

    dim = list(awm.shape);
    print('\n\npost processing registration output')
    print('creating ',outpath+"/atlas_in_Pspace_seg_csf.nc")
    fio.createNetCDF(outpath+"/atlas_in_Pspace_seg_csf.nc", dim, acsf);

    print('creating ',outpath+"/atlas_in_Pspace_seg_gm.nc")
    fio.createNetCDF(outpath+"/atlas_in_Pspace_seg_gm.nc", dim, agm);

    print('creating ',outpath+"/atlas_in_Pspace_seg_wm.nc")
    fio.createNetCDF(outpath+"/atlas_in_Pspace_seg_wm.nc", dim, awm);


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='process input images')
    parser.add_argument ('-patient_image_path',  type = str, help = 'path to patient segmented image')
    parser.add_argument ('-patient_labels',      type=str, help = 'patient labels');
    parser.add_argument ('-atlas_image_path',    type = str, help = 'path to atlas segmented image (affinely registered to patient)')
    parser.add_argument ('-atlas_t1_image_path', type = str, help = 'path to atlas t1 image (affinely registered to patient)')
    parser.add_argument ('-atlas_labels',        type=str, help = 'atlas labels');
    parser.add_argument ('-N',                   type=int, help = 'compute resolution (uniform NxNxN)');
    parser.add_argument ('-output_path',         type=str, help = 'output folder');
    parser.add_argument ('--use_patient_segmentation',    action='store_true', help = 'patient image path is a segmentation and not probability maps, creates probability maps from segmentation');
    parser.add_argument ('--use_atlas_segmentation',      action='store_true', help = 'atlas image path is a segmentation and not probability maps, creates probability maps from segmentation');
    parser.add_argument ('--process_registration_output', action='store_true', help = 'reads transported label maps from registration output and separates labels for input to tumor solver');
    parser.add_argument ('-base_path',           type=str, help = 'base path to output');
    args = parser.parse_args();


    patient_labels = {}
    atlas_labels = {}
    resolution = (args.N, args.N, args.N);

    # patient labels
    if args.patient_labels is not None:
        for x in args.patient_labels.split(','):
            patient_labels[int(x.split('=')[0])] = x.split('=')[1];
        patient_label_rev = {v:k for k,v in patient_labels.items()};
        # add mask label if tumor present
        if 'en' in patient_label_rev and 'nec' in patient_label_rev:
            labelnums = list(patient_labels.keys());
            maxlabelnum = max(labelnums)
            patient_labels[maxlabelnum + 1] = 'mask';

    # atlas labels
    if args.atlas_labels is not None:
        for x in args.atlas_labels.split(','):
            atlas_labels[int(x.split('=')[0])] = x.split('=')[1];

    # ---- process registration output
    if args.process_registration_output:
        # post process registration
        postprocDisplacementField(args)
        extraxtLabelsFromDeformedAtlas(args, atlas_labels, patient_labels)

    else:
        # resize and write atlas T1 image
        if args.atlas_t1_image_path is not None:
            atlas_t1_image_path = args.atlas_t1_image_path
            atlas = nib.load(atlas_t1_image_path);
            affine = atlas.affine;
            atlas = atlas.get_fdata();
            atlas = imgtools.resizeImage(atlas, resolution, 1);
            fio.createNetCDF(args.output_path + '/atlas_t1.nc', np.shape(atlas), np.swapaxes(atlas,0,2));

        # preprocess atlas
        if args.use_atlas_segmentation:
            affine = preprocAtlasFromSegmentation(args.atlas_image_path, args.output_path, resolution, atlas_labels);
        else:
            affine = preprocAtlasFromProbmaps(args.atlas_image_path, args.output_path, resolution, atlas_labels);

        # preprocess patient
        if args.use_patient_segmentation:
            affine = preprocPatientFromSegmentation(args.patient_image_path, args.output_path, resolution, patient_labels);
        else:
            affine = preprocPatientFromProbmaps(args.patient_image_path, args.output_path, resolution);
