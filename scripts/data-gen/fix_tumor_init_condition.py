import nibabel as nib
import scipy.ndimage as ndimage
from scipy.stats import multivariate_normal
from netCDF4 import Dataset
import numpy as np
import os, math, sys, argparse
from TumorParams import *
import pandas as pd


TARGET_DIMS = (256, 256, 256)
LABEL_DICT = {"csf":10, "vt":50, "gm":150, "wm":250}
SIGMA = 2*math.pi/256
DEBUG=False

def moveCenter(source, target, offset):
    new_source = np.copy(source)
    direction = (target - source).astype(float)
    # normalize direction vector
    direction = direction/np.linalg.norm(direction)
    # quantize direction vector
    direction = np.where(direction > 0, np.ceil(direction), direction);
    direction = np.where(direction < 0, np.floor(direction), direction);
    # check for new source locations by using a scalar offset in direction of "direction"
    new_source = target + direction*offset
    return new_source, direction
    
def randomOffset(point, magnitude):
    new_point = np.zeros_like(point)
    for i in range(point.shape[0]):
        sign = 1 if np.random.randint(0,2) < 0.5 else -1
        new_point[i] = point[i] + sign*magnitude
    return new_point

def _find_border(data):
        from scipy.ndimage.morphology import binary_erosion
        eroded = binary_erosion(data)
        border = np.logical_and(data, np.logical_not(eroded))
        return border

def _get_coordinates(data, affine):
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    indices = np.vstack(np.nonzero(data))
    indices = np.vstack((indices, np.ones(indices.shape[1])))
    coordinates = np.dot(affine, indices)
    return coordinates[:3, :]

def _eucl_min(vol1, affine1, vol2, affine2):
    from scipy.spatial.distance import cdist, euclidean

    border1 = _find_border(vol1)
    border2 = _find_border(vol2)

    set1_coordinates = _get_coordinates(border1, affine1)
    set2_coordinates = _get_coordinates(border2, affine2)

    from scipy.spatial import cKDTree
    min_dists, min_dist_idxs = cKDTree(set1_coordinates.T).query(set2_coordinates.T, 1)
    min_dist_idx_set2 = np.argmin(min_dists);
    min_dist = min_dists[min_dist_idx_set2];
    min_dist_idx_set1 = min_dist_idxs[min_dist_idx_set2];

    point1 = set1_coordinates.T[min_dist_idx_set1, :]
    point2 = set2_coordinates.T[min_dist_idx_set2, :]

    return min_dist, point1, point2;

def computeDistance(nii1, nii2):
    vol1 = nii1.get_fdata().astype(np.bool)
    vol2 = nii2.get_fdata().astype(np.bool)
    distance, p1, p2 = _eucl_min(vol1, vol1.affine, vol2, vol2.affine);
    return distance, p1, p2;

def computeDistance(data1, affine1, data2, affine2):
    data1 = data1.astype(np.bool)
    data2 = data2.astype(np.bool)
    distance, p1, p2 = _eucl_min(data1, affine1, data2, affine2);
    return distance, p1, p2;

def resizeImage(img, new_size, interp_order):
    '''
    resize image to new_size
    '''
    factor = tuple([float(x)/float(y) for x,y in zip(list(new_size), list(np.shape(img)))]);
    #print(factor);
    return ndimage.zoom(img, factor, order=interp_order);

def resizeNIIImage(img, new_size, interp_order, new_filename):
    '''
    uses nifti img object to resize it to new_size
    '''
    # load the image
    img = nib.load(filename);
    old_voxel_size = img.header['pixdim'][1:4];
    old_size = img.header['dim'][1:4];
    new_size = np.asarray(new_size);
    new_voxel_size = np.multiply(old_voxel_size, np.divide(old_size, new_size));
    return nib.processing.resample_to_output(img, new_voxel_size, order=0, mode='wrap');

def readNetCDF(filename):
    '''
    function to read a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename);
    img = imgfile.variables['data'][:]
    imgfile.close();
    return img

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

def writeNII(img, filename, affine=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if affine is None:
        data = nib.Nifti1Image(img, np.eye(4));
    else:
        data = nib.Nifti1Image(img, affine);
    nib.save(data, filename);


def getCentersAndActivations(centers_path, activations_path):
    centers = []
    with open(centers_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0].isdigit():
                center = [float(num.strip()) for num in line.split(",")]
                centers.append(center)

    activations = []
    with open(activations_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0].isdigit():
                activations.append(float(line))

    return centers, activations


def getConnectedComponents(white_matter, thresh=0):
    '''
    Returns connected components with rel_mass > thresh
    '''
    print("Finding connected components")
    structure = np.ones((3, 3, 3), dtype=np.int)
    labeled, ncomponents = ndimage.measurements.label(white_matter, structure);
    total_mass = 0
    relmass = np.zeros((ncomponents,))

    for i in range(ncomponents):
        comp = (labeled == i+1)
        a, b = ndimage.measurements._stats(comp)
        total_mass += b
    for i in range(ncomponents):
        comp = (labeled == i+1)
        count, sums = ndimage.measurements._stats(comp)
        relmass[i] = sums/float(total_mass)
        # if rel mass is too low, remove the component
        if relmass[i] < thresh:
            labeled = np.where(labeled == i+1, 0, labeled)

    #print("total_mass={}".format(total_mass))
    # Do connected components again now that we've removed components that are too small
    labeled, ncomponents = ndimage.measurements.label(labeled, structure);
    print("valid ncomponents (relmass>{}) = {}".format(thresh, ncomponents))

    return labeled, ncomponents


def writeTissueSegToFiles(label, label_dict, path, scale=True, affine=np.eye(3)):
    # Write out gm, wm, etc as its own file
    print("Writing NetCDF files for each tissue type...")
    for tissue in label_dict:
        index = label_dict[tissue]
        data = np.where(label == index, 1, 0)
        new_path_nii = os.path.join(path, "atlas_" + tissue + ".nii")
        new_path_nc = os.path.join(path, "atlas_" + tissue + ".nc")
        writeNII(data, new_path_nii, affine)
        # should already have axes 02 swapped
        createNetCDF(new_path_nc , np.shape(data), data)


def getGaussian(center, sigma, shape):
    coords = np.indices(shape)
    coords = np.swapaxes(coords, 1,3) # Swap image dims 0, 2
    coords = coords * sigma
    coords = np.moveaxis(coords, 0, -1) # Move coords to end
    x = coords - center
    x = np.linalg.norm(x, 1, axis=-1)
    gaussian = np.exp(-1*np.square(x)/(2*sigma*sigma))

    return gaussian



def fixTumorInitialCondition(args):
    #patient_path = args.patient_path
    atlas_path = args.atlas
    centers_path = args.centers
    activations_path = args.activations
    output_path = args.output

    if centers_path is None:
        centers_path = os.path.join(patient_path, "tumor_inversion/nx256/obs-1.0/phi-mesh-scaled.txt")
    if activations_path is None:
        activations_path = os.path.join(patient_path, "tumor_inversion/nx256/obs-1.0/p-rec-scaled.txt")
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)

    #patient_nib = nib.load(patient_path)
    #patient = np.squeeze(patient_nib.get_fdata())
    #patient = resizeImage(patient, TARGET_DIMS, 1)
    #patient_shape = patient_nib.get_fdata().shape

    atlas_nib = nib.load(atlas_path)
    atlas = np.swapaxes(np.squeeze(atlas_nib.get_fdata()), 0, 2)
    atlas = resizeImage(atlas, TARGET_DIMS, 0)

    # Write each tissue type in atlas to seperate file as prob map
    #writeTissueSegToFiles(atlas, LABEL_DICT, output_path, scale=True, affine=atlas_nib.affine)
    # Read centers and activations from file
    centers, activations = getCentersAndActivations(centers_path, activations_path)
    
    # Get white matter mask
    wm_index = LABEL_DICT["wm"]
    wm_mask = np.where(atlas == wm_index, 1, 0)
    labeled, ncomponents = getConnectedComponents(wm_mask, thresh=1e-3)

    # Check if centers are in wm, else move them
    print("Checking each center location")
    verified_centers = []
    for center, act in zip(centers, activations):
        center = [int(x / SIGMA) for x in center]
        # keep a copy of the original center
        new_center = center.copy()
        # starting offset for moving centers from target location in WM
        offset = 2
        while True:
            comp_index = labeled[int(new_center[0]), int(new_center[1]), int(new_center[2])]
            # if center already in white matter, add it to list of centers, else
            # get loc of nearest white matter and move it there
            if comp_index > 0:
                verified_centers.append([x * SIGMA for x in new_center])
                break
            else:
                center_3d = np.zeros_like(labeled)
                # use old center to compute distances
                center_3d[int(center[0]), int(center[1]), int(center[2])] = 1
                distance, current, target = computeDistance(center_3d, np.eye(4), labeled > 0, np.eye(4))
                #center = target
                # Randomly move the target location 2 voxels away to prevent being on edge
                # This needs to be change, maybe use vector from current to target position
                # to determine how to move away from edge?
                new_center,direction = moveCenter(center, target, offset)
                if DEBUG:
                    print("old center {}\n target {}\n Trying new center at {}\n (in direction {}\n with offset {})".format(center, target, new_center, direction, offset))
                # new offset if old one didnt work
                offset += 1

    if DEBUG:
        print("Transported centers = {}".format(verified_centers))
    c0 = np.zeros(TARGET_DIMS)

    for center, act in zip(verified_centers, activations):
        print("Adding gaussian centered at {}".format(center))
        gaussian = getGaussian(center, SIGMA, TARGET_DIMS)
        c0 += gaussian * act

    # Check c0 values are in correct range, min should be ~= 0 and max
    # should be near 1
    print("c0: max={}, min={}".format(np.max(c0), np.min(c0)))
    assert abs(np.max(c0) - 1) <= 1e-6, "c0 max not close to 1"
    assert abs(np.min(c0)) < 1e-6, "c0 min = {}".format(np.min(c0))


    new_path_nii = os.path.join(output_path, "c0.nii.gz")
    new_path_nc = os.path.join(output_path, "c0.nc")
    #new_path_brats_nii = os.path.join(output_path, "c0_brats_affine.nii")

    writeNII(c0, new_path_nii, atlas_nib.affine)
    #writeNII(resizeImage(c0, patient_nib.get_fdata().shape, 1), new_path_brats_nii, patient_nib.affine)
    createNetCDF(new_path_nc , np.shape(c0), np.swapaxes(c0, 0, 2))        


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="fix tumor initial condition to fall only in WM in atlas")
    #parser.add_argument("--patient", help="Path to patient T1", required=True)
    parser.add_argument("--atlas", help="Path to atlas tissue label file", required=True)
    parser.add_argument("--centers", help="Path to phi-mesh-scaled.txt", required=True)
    parser.add_argument("--activations", help="Path to p-rec-scaled.txt", required=True)
    parser.add_argument("--output", help="Directory to save output files in [default=output/]", default="output")
    args = parser.parse_args()

    fixTumorInitialCondition(args)
