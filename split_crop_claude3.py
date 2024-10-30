import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
import warnings
import csv
import scipy.ndimage

warnings.filterwarnings('ignore')


class LungNoduleDatasetPreparation:
    def __init__(self, base_path, candidates_file, output_path):
        self.base_path = Path(base_path)
        self.candidates_df = pd.read_csv(candidates_file)
        self.output_path = Path(output_path)
        self.window_size = 15  # This will give 31x31x31 volume (15 + 1 + 15)
        self.create_output_directories()

    def create_output_directories(self):
        """Create directories for train, validation, and test sets"""
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'samples').mkdir(parents=True, exist_ok=True)

    def resample(self, image, old_spacing, new_spacing=[1, 1, 1]):
        """Resample image to new spacing"""
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor

        image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing

    def process_and_save_split(self, series_ids, split_name, preprocess=True):
        """Process and save data for a specific split"""
        samples_dir = self.output_path / split_name / 'samples'
        csv_path = self.output_path / split_name / f'{split_name}_info.csv'

        # Create CSV file with headers
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'series_id', 'class', 'coord_x', 'coord_y', 'coord_z'])

        for series_id in tqdm(series_ids, desc=f'Processing {split_name} set'):
            try:
                # Find the mhd file
                mhd_files = list(self.base_path.glob(f'**/{series_id}.mhd'))
                if not mhd_files:
                    print(f"Warning: Could not find mhd file for {series_id}")
                    continue
                mhd_file = mhd_files[0]

                # Get candidates for this series
                series_candidates = self.candidates_df[self.candidates_df['seriesuid'] == series_id]

                if split_name == 'test':
                    # For test set, just save metadata
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([str(mhd_file), series_id, 'test',
                                         str(mhd_file).replace('.mhd', '.raw')])
                    continue

                # Read and process image
                ct_image = sitk.ReadImage(str(mhd_file))
                full_scan = sitk.GetArrayFromImage(ct_image)

                # Get origin and spacing
                origin = np.array(ct_image.GetOrigin())[::-1]  # get [z, y, x] origin
                old_spacing = np.array(ct_image.GetSpacing())[::-1]  # get [z, y, x] spacing

                if preprocess:
                    # Resample to 1mm spacing
                    image, new_spacing = self.resample(full_scan, old_spacing)
                else:
                    image = full_scan
                    new_spacing = old_spacing

                # Process each candidate
                for idx, candidate in series_candidates.iterrows():
                    try:
                        # Get nodule center
                        nodule_center = np.array([candidate.coordZ, candidate.coordY, candidate.coordX])
                        v_center = np.rint((nodule_center - origin) / new_spacing)
                        v_center = np.array(v_center, dtype=int)

                        # Crop the nodule
                        zyx_1 = v_center - self.window_size
                        zyx_2 = v_center + self.window_size + 1

                        # Ensure within bounds
                        zyx_1 = np.maximum(zyx_1, 0)
                        zyx_2 = np.minimum(zyx_2, image.shape)

                        img_crop = image[zyx_1[0]:zyx_2[0],
                                   zyx_1[1]:zyx_2[1],
                                   zyx_1[2]:zyx_2[2]]

                        # Save cropped nodule
                        file_name = f'{series_id}_{idx}.npy'
                        save_path = samples_dir / file_name
                        np.save(save_path, img_crop.astype(np.float32))  # Save as float32 for efficiency

                        # Write to CSV
                        with open(csv_path, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                file_name,
                                series_id,
                                candidate['class'],
                                v_center[2],  # x
                                v_center[1],  # y
                                v_center[0]  # z
                            ])

                    except Exception as e:
                        print(f"Error processing candidate {idx} from series {series_id}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing series {series_id}: {str(e)}")
                continue

    def split_dataset(self, test_size=0.1, val_size=0.1, random_state=42):
        """Split dataset ensuring each split has positive samples"""
        unique_series = self.candidates_df.groupby('seriesuid')['class'].max().reset_index()

        train_series, temp_series, train_labels, temp_labels = train_test_split(
            unique_series['seriesuid'].values,
            unique_series['class'].values,
            test_size=(test_size + val_size),
            random_state=random_state,
            stratify=unique_series['class'].values
        )

        val_series, test_series, val_labels, test_labels = train_test_split(
            temp_series,
            temp_labels,
            test_size=0.5,
            random_state=random_state,
            stratify=temp_labels
        )

        print(f"Dataset split statistics:")
        print(f"Total scans: {len(unique_series)}")
        print(f"Training scans: {len(train_series)} ({len(train_series) / len(unique_series) * 100:.1f}%)")
        print(f"Validation scans: {len(val_series)} ({len(val_series) / len(unique_series) * 100:.1f}%)")
        print(f"Testing scans: {len(test_series)} ({len(test_series) / len(unique_series) * 100:.1f}%)")

        return train_series, val_series, test_series

    def prepare_dataset(self):
        """Main function to prepare the complete dataset"""
        print("Splitting dataset...")
        train_series, val_series, test_series = self.split_dataset()

        print("Processing training set...")
        self.process_and_save_split(train_series, 'train', preprocess=True)

        print("Processing validation set...")
        self.process_and_save_split(val_series, 'val', preprocess=True)

        print("Processing test set...")
        self.process_and_save_split(test_series, 'test', preprocess=False)

        print("Dataset preparation completed!")


# Usage
if __name__ == "__main__":
    base_path = "/home/mustafa/project/dataset"
    candidates_file = "/home/mustafa/project/dataset/candidates_V2.csv"
    output_path = "/home/mustafa/project/processed_dataset2"

    processor = LungNoduleDatasetPreparation(base_path, candidates_file, output_path)
    processor.prepare_dataset()