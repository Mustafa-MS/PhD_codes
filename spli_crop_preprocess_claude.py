import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split


class LungNodulePreprocessor:
    def __init__(self, base_path, output_path, window_size=(31, 31, 31)):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.window_size = np.array(window_size)
        self.half_size = self.window_size // 2

        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split).mkdir(parents=True, exist_ok=True)

    def load_candidates(self, candidates_path):
        df = pd.read_csv(candidates_path)
        return df

    def preprocess_ct_scan(self, image):
        image_array = sitk.GetArrayFromImage(image)

        # Apply windowing (Lung window: center=-600, width=1500)
        min_bound = -600 - (1500 / 2)
        max_bound = -600 + (1500 / 2)
        image_array = np.clip(image_array, min_bound, max_bound)

        # Normalize to [0,1]
        image_array = (image_array - min_bound) / (max_bound - min_bound)

        return image_array

    def extract_patch(self, image_array, center, patch_size):
        """
        Extract a patch of given size around a center point

        Args:
            image_array: 3D numpy array of image
            center: tuple of (z,y,x) coordinates
            patch_size: tuple of (depth,height,width)

        Returns:
            3D numpy array of patch
        """
        patch = np.zeros(patch_size, dtype=np.float32)
        half_size = np.array(patch_size) // 2

        # Calculate start and end coordinates for both patch and image
        start_patch = np.array([0, 0, 0])
        end_patch = np.array(patch_size)

        start_img = np.array(center) - half_size
        end_img = start_img + patch_size

        # Clip image coordinates to valid range
        valid_start_img = np.maximum(start_img, 0)
        valid_end_img = np.minimum(end_img, image_array.shape)

        # Calculate corresponding patch coordinates
        start_patch = valid_start_img - start_img
        end_patch = patch_size - (end_img - valid_end_img)

        # Copy the valid region
        slices_patch = tuple(slice(s, e) for s, e in zip(start_patch, end_patch))
        slices_img = tuple(slice(s, e) for s, e in zip(valid_start_img, valid_end_img))

        patch[slices_patch] = image_array[slices_img]

        return patch

    def crop_nodule(self, image_array, coord_z, coord_y, coord_x, spacing):
        # Convert world coordinates to voxel coordinates
        voxel_z = int(round(coord_z / spacing[2]))
        voxel_y = int(round(coord_y / spacing[1]))
        voxel_x = int(round(coord_x / spacing[0]))

        center = np.array([voxel_z, voxel_y, voxel_x])
        patch = self.extract_patch(image_array, center, tuple(self.window_size))

        return patch

    def split_dataset(self, candidates_df, test_size=0.2, val_size=0.2, random_state=42):
        series_uids = candidates_df['seriesuid'].unique()
        positive_series = candidates_df[candidates_df['class'] == 1]['seriesuid'].unique()

        train_val_series, test_series = train_test_split(
            series_uids,
            test_size=test_size,
            random_state=random_state,
            stratify=[1 if uid in positive_series else 0 for uid in series_uids]
        )

        train_series, val_series = train_test_split(
            train_val_series,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=[1 if uid in positive_series else 0 for uid in train_val_series]
        )

        return train_series, val_series, test_series

    def process_and_save(self):
        candidates_df = self.load_candidates(self.base_path / 'candidates_V2.csv')
        train_series, val_series, test_series = self.split_dataset(candidates_df)

        splits = {
            'train': train_series,
            'val': val_series,
            'test': test_series
        }

        for split_name, series_list in splits.items():
            print(f"Processing {split_name} split...")

            with h5py.File(self.output_path / f"{split_name}_nodules.h5", 'w') as f:
                # Create initial datasets
                max_expected_samples = len(candidates_df[candidates_df['seriesuid'].isin(series_list)])

                patches_dataset = f.create_dataset(
                    'patches',
                    shape=(max_expected_samples, *self.window_size),
                    dtype=np.float32,
                    chunks=True,
                    compression='gzip'
                )

                labels_dataset = f.create_dataset(
                    'labels',
                    shape=(max_expected_samples,),
                    dtype=np.int32,
                    chunks=True,
                    compression='gzip'
                )

                current_idx = 0

                for series_uid in tqdm(series_list):
                    series_candidates = candidates_df[candidates_df['seriesuid'] == series_uid]

                    # Find the correct subset folder
                    mhd_path = None
                    for subset in range(10):
                        potential_path = self.base_path / f"subset{subset}" / f"{series_uid}.mhd"
                        if potential_path.exists():
                            mhd_path = potential_path
                            break

                    if mhd_path is None:
                        continue

                    try:
                        ct_image = sitk.ReadImage(str(mhd_path))
                        spacing = ct_image.GetSpacing()

                        if split_name in ['train', 'val']:
                            image_array = self.preprocess_ct_scan(ct_image)
                        else:
                            image_array = sitk.GetArrayFromImage(ct_image)

                        for _, candidate in series_candidates.iterrows():
                            try:
                                crop = self.crop_nodule(
                                    image_array,
                                    candidate['coordZ'],
                                    candidate['coordY'],
                                    candidate['coordX'],
                                    spacing
                                )

                                if crop.shape == tuple(self.window_size):
                                    patches_dataset[current_idx] = crop
                                    labels_dataset[current_idx] = candidate['class']
                                    current_idx += 1

                            except Exception as e:
                                print(f"Error processing candidate in {series_uid}: {e}")
                                continue

                    except Exception as e:
                        print(f"Error processing series {series_uid}: {e}")
                        continue

                # Resize datasets to remove unused space
                if current_idx < max_expected_samples:
                    patches_dataset.resize(current_idx, axis=0)
                    labels_dataset.resize(current_idx, axis=0)

                # Store metadata
                f.attrs['n_positive'] = np.sum(labels_dataset[:] == 1)
                f.attrs['n_negative'] = np.sum(labels_dataset[:] == 0)


if __name__ == "__main__":
    preprocessor = LungNodulePreprocessor(
        base_path="/home/mustafa/project/dataset",
        output_path="/home/mustafa/project/processed_dataset",
        window_size=(31, 31, 31)
    )
    preprocessor.process_and_save()