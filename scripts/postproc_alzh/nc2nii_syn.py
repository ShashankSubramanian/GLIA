from tools import *
import argparse

code_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
data_path = code_dir + 'syn_data1/data/'
results_dir = code_dir + 'd_nc-80/'
template = nib.load(os.path.join(code_dir, '0368Y01_seg_wm.nii.gz'))

def convert(d):
    for dir in os.listdir(d):
        if not '.nc' in dir:
           continue
        print("   ... converting {}".format(dir))
        dat = readNetCDF(os.path.join(d, dir));
        dat = np.swapaxes(dat,0,2);
        output_size = tuple(dat.shape)
        filename = ntpath.basename(dir);
        filename = filename.split('.nc')[0]
        newfilename = filename + '.nii.gz';
        writeNII(dat, os.path.join(d,newfilename), template.affine);




parser = argparse.ArgumentParser(description='reading files')
parser.add_argument ('-x', type = str, help = 'path to the reuslts folder')
args = parser.parse_args()

DIRS = os.listdir(results_dir + '/.')
TAB = []

for dir in DIRS:

    # Converting the nc files to nii 
    print(' [] converting ddir {}'.format(results_dir + dir))
    convert(results_dir+dir)

