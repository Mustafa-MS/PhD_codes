import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

def world_to_voxel(world_coords, origin, spacing):
    stretched_voxel_coords = np.absolute(world_coords - origin)
    voxel_coords = stretched_voxel_coords / spacing
    return voxel_coords.astype(int)

def mark_nodule_in_mask(mask, center_voxel, diameter, spacing):
    radius = np.ceil(diameter / 2 / spacing).astype(int)
    # Create a spherical region
    for x in range(-radius[0], radius[0] + 1):
        for y in range(-radius[1], radius[1] + 1):
            for z in range(-radius[2], radius[2] + 1):
                if x**2 + y**2 + z**2 <= np.max(radius)**2:
                    coord = center_voxel + np.array([x, y, z])
                    if np.all(coord >= 0) and np.all(coord < mask.shape):
                        mask[coord[0], coord[1], coord[2]] = 1
    return mask

def create_mask_for_scan(ct_scan, nodules_info, origin, spacing):
    mask = np.zeros(ct_scan.shape, dtype=np.int)
    for index, nodule in nodules_info.iterrows():
        world_coords = np.array([nodule['coordZ'], nodule['coordY'], nodule['coordX']])
        voxel_coords = world_to_voxel(world_coords, origin, spacing)
        mask = mark_nodule_in_mask(mask, voxel_coords, nodule['diameter_mm'], spacing)
    return mask
def process_and_save_masks(base_path, annotations_path, mask_save_path):
    annotations = pd.read_csv(annotations_path)

    # Check if the mask save directory exists, create it if not
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    for i in range(10):  # Iterate through each subset
        subset_folder = os.path.join(base_path, f'subset{i}')
        ct_files = [f for f in os.listdir(subset_folder) if f.endswith('.mhd')]

        for ct_file in ct_files:
            seriesuid = os.path.splitext(ct_file)[0]
            ct_image_path = os.path.join(subset_folder, ct_file)
            ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(ct_image_path))
            origin = np.array(sitk.ReadImage(ct_image_path).GetOrigin())[::-1]
            spacing = np.array(sitk.ReadImage(ct_image_path).GetSpacing())[::-1]

            nodules_info = annotations[annotations['seriesuid'] == seriesuid]

            mask = create_mask_for_scan(ct_scan, nodules_info, origin, spacing)
            # Convert mask to uint8
            mask = mask.astype(np.uint8)
            mask_filename = seriesuid + '_mask.npy'
            np.savez_compressed(os.path.join(mask_save_path, mask_filename), mask)

            print(f"Processed and saved compressed mask for: {seriesuid}")

# Example usage
base_path = '/home/mustafa/project/LUNA16/'
annotations_path = '/home/mustafa/project/LUNA16/annotations.csv'
mask_save_path = '/home/mustafa/project/mask_folder'
process_and_save_masks(base_path, annotations_path, mask_save_path)
