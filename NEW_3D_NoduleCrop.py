import os
import glob
import pandas
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import array
import csv

luna_path = "/home/mustafa/project/LUNA16/"
all_nodules = pandas.read_csv(luna_path + "candidates_V3.csv")
nodules_save_path = '/home/mustafa/project/LUNA16/cropped_nodules/'
# window size is hyper parameter
window_size = 15


def read_mhd_file(filepath):
    """Read and load CT image"""
    # Read file
    #print("filepath for IMG = ", filepath)
    scan = sitk.ReadImage(filepath)
    # get the image to an array
    scan_array = sitk.GetArrayFromImage(scan)[0, :, :, :]
    # Read the origin of the image
    origin = np.array(list(reversed(scan.GetOrigin())))  # get [z, y, x] origin
    # Delete the first element from origin
    origin = np.delete(origin, 0)
    # Read spacing of the image
    old_spacing = np.array(list(reversed(scan.GetSpacing())))  # get [z, y, x] spacing
    # Delete the first element from spacing
    old_spacing = np.delete(old_spacing, 0)
    return scan_array, origin, old_spacing


def normalize(volume):
    """Normalize the CT image"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def plot_ct_scan(scan, num_column=4, jump=1):
    """Plot the CT scan in one figure with all slides"""
    num_slices = len(scan)
    num_row = (num_slices // jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column * 5, num_row * 5))
    for i in range(0, num_row * num_column):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
        plot.axis('off')
        if i < num_slices // jump:
            plot.imshow(scan[i * jump], cmap=plt.cm.bone)


## Define resample method to make images isomorphic, default spacing is [1, 1, 1]mm
# Learned from Guido Zuidhof
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    """Resample to uniform the spacing"""
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


# This is just for testing

def worldToVoxelCoord(worldCoord, origin, spacing):
    # There is no need for this function, the cropping will handle the voxel coords
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def crop_nodule(image, new_spacing, window_size, origin):
    """Cropping the nodule in 3D cube"""
    # Attention: Z, Y, X
    nodule_center = np.array([patient.coordZ, patient.coordY, patient.coordX])
    # You can use the following line to convert from world to voxel coords.
    # voxelCoord = worldToVoxelCoord(nodule_center, origin, new_spacing)
    # The following line will do the same math so no need for converting from world to voxel for the centre coords
    v_center = np.rint((nodule_center - origin) / new_spacing)
    v_center = np.array(v_center, dtype=int)
    # This is to creat the cube Z Y X
    zyx_1 = v_center - window_size  # Attention: Z, Y, X
    zyx_2 = v_center + window_size + 1
    # This will give you a [19, 19, 19] volume
    img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
    return img_crop


# uncomment if you need a MHD/.raw file format
def write_meta_header(filename, meta_dict):
    "Saving the MHD/raw meta header"
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType', 'NDims', 'BinaryData',
            'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
            'TransformMatrix', 'Offset', 'CenterOfRotation',
            'AnatomicalOrientation',
            'ElementSpacing',
            'DimSize',
            'ElementType',
            'ElementDataFile',
            'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n' % (tag, meta_dict[tag])
    f = open(filename, 'w')
    f.write(header)
    f.close()


def dump_raw_data(filename, data):
    " Write the data into a raw format file. Big endian is always used. "
    # Begin 3D fix
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    # End 3D fix
    rawfile = open(filename, 'wb')
    a = array.array('f')
    for o in data:
        a.fromlist(list(o))
    # if is_little_endian():
    #    a.byteswap()
    a.tofile(rawfile)
    rawfile.close()


def write_mhd_file(mhdfile, data, dsize):
    "Writing the MHD file"
    assert (mhdfile[-4:] == '.mhd')
    meta_dict = {}
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False'
    meta_dict['ElementType'] = 'MET_FLOAT'
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
    write_meta_header(mhdfile, meta_dict)
    # Change this if you want to save the file in a different place, I'll leave because I don't want the MHD format
    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd + '/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    dump_raw_data(data_file, data)


def save_nodule(nodule_crop, name_index):
    """Saving the nodules in .npy and MHD/.raw file"""
    np.save(os.path.join(nodules_save_path, str(name_index) + '.npy'), nodule_crop)
    # uncomment this if you want MHD/.raw file
    # write_mhd_file(str(name_index) + '.mhd', nodule_crop, nodule_crop.shape[::-1])
    print("nodulesaved ", name_index)


def process_scan(path):
    """Read the CT image, normalize, resample, and crop and save the nodules"""
    # Read scan
    volume, origin, old_spacing = read_mhd_file(path)
    # Normalize
    volume = normalize(volume)
    # Resample
    image, new_spacing = resample(volume, old_spacing)
    # plot one slide of the CT image, choose any slide number
    # plt.imshow(np.squeeze(image[10, :, :]), cmap="gray")
    # plt.show()
    return image, new_spacing, origin


row_iterator = all_nodules.iterrows()
x = 0
for index, patient in row_iterator:
    # If patient is the same then read the image process once and cut all nodules, if not then read the next image
    if patient.seriesuid != x:
        image, new_spacing, origin = process_scan(glob.glob(luna_path + "/**/" + patient.seriesuid + ".mhd", recursive=True))
        x = patient.seriesuid
    # Crop the nodule
    img_crop = crop_nodule(image, new_spacing, window_size, origin)
    print(index, " , ", patient.seriesuid, " , ", patient.state)
    # Save the nodule
    save_nodule(img_crop, index)
    # open csv file and save the truth table
    csv_file = open(luna_path + 'cropped_nodules_2.csv', "a")
    writer = csv.writer(csv_file)
    writer.writerow([index, patient.state])
csv_file.close()
    #Plot the nodule images
    #plot_ct_scan(img_crop)
    #plt.show()

