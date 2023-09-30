import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from medimage import image
myfirstimage = image("/home/mustafa/project/Testfiles/test/1.3.6.1.4.1.14519.5.2.1.6279.6001.113697708991260454310623082679.mhd")
img = np.array(myfirstimage)
print(img.shape)
plt.imshow(img[:, :, 20],cmap="gray")
plt.show()


data =sitk.ReadImage("/home/mustafa/project/Testfiles/test/1.3.6.1.4.1.14519.5.2.1.6279.6001.113697708991260454310623082679.mhd")
scan = sitk.GetArrayFromImage(data)



def plot_ct_scan(scan, num_column=4, jump=1):
    num_slices = len(scan)
    print("testttt")
    num_row = (num_slices//jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column*5, num_row*5))
    for i in range(0, num_row*num_column):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
        plot.axis('off')
        if i < num_slices//jump:
            plot.imshow(scan[i*jump], cmap=plt.cm.bone)





spacing = data.GetSpacing()
print('spacing: ', spacing)
print('# slice: ', len(scan))
plot_ct_scan(scan)

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import scipy.ndimage


def get_segmented_lungs(im, spacing, threshold=-400):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < threshold

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)

    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)

    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)

    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    return binary



print(scan.shape)

mask = np.array([get_segmented_lungs(slice.copy(), spacing) for slice in scan])
scan[~mask] = 0
plt.imshow(scan[5])
plot_ct_scan(scan, jump=1)



