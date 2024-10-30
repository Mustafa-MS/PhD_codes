import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import SimpleITK as sitk
from tqdm import tqdm
import h5py
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
import csv

warnings.filterwarnings('ignore')


class LungNoduleDatasetPreparation:
    def __init__(self, base_path, candidates_file, output_path):
        self.base_path = Path(base_path)
        self.candidates_df = pd.read_csv(candidates_file)
        self.output_path = Path(output_path)
        self.window_size = (31, 31, 31)

        # Create output directories
        self.create_output_directories()

    def create_output_directories(self):
        """Create directories for train, validation, and test sets"""
        self.output_path.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test']:
            #(self.output_path / split).mkdir(parents=True, exist_ok=True)
            split_dir = self.output_path / split
            split_dir.mkdir(exist_ok=True)

            samples_dir = split_dir / 'samples'
            samples_dir.mkdir(exist_ok=True)

            print(f"Created directory: {samples_dir}")  # Debug print

    def preprocess_ct_scan(self, ct_image):
        """Preprocess CT scan"""
        # Normalize to Hounsfield units
        image_array = sitk.GetArrayFromImage(ct_image)
        # Clip to lung window
        min_bound = -1000.0
        max_bound = 400.0
        image_array = np.clip(image_array, min_bound, max_bound)
        # Normalize to [0,1]
        image_array = (image_array - min_bound) / (max_bound - min_bound)
        return image_array

    def crop_nodule(self, ct_array, center_coord, window_size):
        """Crop a window around the nodule"""
        half_size = np.array(window_size) // 2
        center = np.array(center_coord, dtype=int)

        # Calculate boundaries
        start = center - half_size
        end = center + half_size + 1

        # Ensure within bounds
        pad_before = np.maximum(0, -start)
        pad_after = np.maximum(0, end - ct_array.shape)
        start = np.maximum(0, start)
        end = np.minimum(ct_array.shape, end)

        # Extract and pad if necessary
        crop = ct_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        if np.any(pad_before > 0) or np.any(pad_after > 0):
            crop = np.pad(crop, list(zip(pad_before, pad_after)), mode='constant')

        return crop

    def split_dataset(self, test_size=0.1, val_size=0.1, random_state=42):
        """Split dataset ensuring each split has positive samples"""
        # Get unique series IDs and their corresponding classes
        unique_series = self.candidates_df.groupby('seriesuid')['class'].max().reset_index()

        # First split: training vs (validation + test)
        train_series, temp_series, train_labels, temp_labels = train_test_split(
            unique_series['seriesuid'].values,
            unique_series['class'].values,
            test_size=(test_size + val_size),
            random_state=random_state,
            stratify=unique_series['class'].values
        )

        # Second split: split temp into validation and test
        val_series, test_series, val_labels, test_labels = train_test_split(
            temp_series,
            temp_labels,
            test_size=0.5,  # Split remaining data equally between val and test
            random_state=random_state,
            stratify=temp_labels
        )

        print(f"Dataset split statistics:")
        print(f"Total scans: {len(unique_series)}")
        print(f"Training scans: {len(train_series)} ({len(train_series) / len(unique_series) * 100:.1f}%)")
        print(f"Validation scans: {len(val_series)} ({len(val_series) / len(unique_series) * 100:.1f}%)")
        print(f"Testing scans: {len(test_series)} ({len(test_series) / len(unique_series) * 100:.1f}%)")

        # Verify class distribution in each split
        def print_class_distribution(series_ids, name):
            classes = unique_series[unique_series['seriesuid'].isin(series_ids)]['class']
            pos_ratio = (classes == 1).mean() * 100
            print(f"{name} set - Positive samples: {pos_ratio:.1f}%")

        print("\nClass distribution:")
        print_class_distribution(train_series, "Training")
        print_class_distribution(val_series, "Validation")
        print_class_distribution(test_series, "Testing")

        return train_series, val_series, test_series

    # def process_and_save_split(self, series_ids, split_name, preprocess=True):
    #     """Process and save data for a specific split"""
    #     output_file = self.output_path / split_name / f'{split_name}_data.h5'
    #
    #     with h5py.File(output_file, 'w') as h5f:
    #         for series_id in tqdm(series_ids, desc=f'Processing {split_name} set'):
    #             # Get candidates for this series
    #             series_candidates = self.candidates_df[self.candidates_df['seriesuid'] == series_id]
    #
    #             # Load CT scan
    #             mhd_file = list(self.base_path.glob(f'**/{series_id}.mhd'))[0]
    #             ct_image = sitk.ReadImage(str(mhd_file))
    #
    #             # Preprocess if required
    #             ct_array = sitk.GetArrayFromImage(ct_image)
    #             if preprocess:
    #                 ct_array = self.preprocess_ct_scan(ct_image)
    #
    #             # Create groups in h5 file
    #             series_group = h5f.create_group(series_id)
    #
    #             # Process each candidate
    #             for _, candidate in series_candidates.iterrows():
    #                 # Convert world coordinates to voxel coordinates
    #                 world_coord = np.array([candidate.coordZ, candidate.coordY, candidate.coordX])
    #                 voxel_coord = np.round(ct_image.TransformPhysicalPointToIndex(world_coord)).astype(int)
    #
    #                 # Crop nodule
    #                 crop = self.crop_nodule(ct_array, voxel_coord, self.window_size)
    #
    #                 # Save to h5 file
    #                 candidate_id = f'candidate_{len(series_group)}'
    #                 candidate_group = series_group.create_group(candidate_id)
    #                 candidate_group.create_dataset('data', data=crop)
    #                 candidate_group.create_dataset('label', data=candidate['class'])
    #                 candidate_group.create_dataset('coordinates', data=voxel_coord)

    # def process_and_save_split(self, series_ids, split_name, preprocess=True):
    #     """Process and save data for a specific split"""
    #     metadata = []
    #
    #     for series_id in tqdm(series_ids, desc=f'Processing {split_name} set'):
    #         series_candidates = self.candidates_df[self.candidates_df['seriesuid'] == series_id]
    #         mhd_file = list(self.base_path.glob(f'**/{series_id}.mhd'))[0]
    #         ct_image = sitk.ReadImage(str(mhd_file))
    #
    #         if split_name == 'test':
    #             # For test set, just save metadata
    #             metadata.append({
    #                 'series_id': series_id,
    #                 'mhd_path': str(mhd_file),
    #                 'raw_path': str(mhd_file).replace('.mhd', '.raw'),
    #                 'candidates': series_candidates.to_dict('records')
    #             })
    #             continue
    #
    #         ct_array = sitk.GetArrayFromImage(ct_image)
    #         if preprocess:
    #             ct_array = self.preprocess_ct_scan(ct_image)
    #
    #         for idx, candidate in series_candidates.iterrows():
    #             world_coord = np.array([candidate.coordZ, candidate.coordY, candidate.coordX])
    #             voxel_coord = np.round(ct_image.TransformPhysicalPointToIndex(world_coord)).astype(int)
    #
    #             crop = self.crop_nodule(ct_array, voxel_coord, self.window_size)
    #
    #             # Save as .npy file
    #             label_dir = 'nodules' if candidate['class'] == 1 else 'non_nodules'
    #             save_path = self.output_path / split_name / label_dir / f'{series_id}_{idx}.npy'
    #             np.save(save_path, crop)
    #
    #             metadata.append({
    #                 'file_path': str(save_path),
    #                 'series_id': series_id,
    #                 'coordinates': voxel_coord.tolist(),
    #                 'class': candidate['class']
    #             })
    #
    #     # Save metadata
    #     metadata_path = self.output_path / split_name / f'{split_name}_metadata.npy'
    #     np.save(metadata_path, metadata)

    def process_and_save_split(self, series_ids, split_name, preprocess=True):
        """Process and save data for a specific split"""
        samples_dir = self.output_path / split_name / 'samples'
        csv_path = self.output_path / split_name / f'{split_name}_info.csv'

        # Create CSV file with headers
        csv_headers = ['file_name', 'series_id', 'class', 'coord_x', 'coord_y', 'coord_z']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)

        for series_id in tqdm(series_ids, desc=f'Processing {split_name} set'):
            series_candidates = self.candidates_df[self.candidates_df['seriesuid'] == series_id]
            try:
                mhd_file = list(self.base_path.glob(f'**/{series_id}.mhd'))[0]
            except IndexError:
                print(f"Warning: Could not find mhd file for {series_id}")
                continue

            ct_image = sitk.ReadImage(str(mhd_file))

            if split_name == 'test':
                # For test set, just save metadata
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        str(mhd_file),
                        series_id,
                        'test',
                        str(mhd_file).replace('.mhd', '.raw')
                    ])
                continue

            ct_array = sitk.GetArrayFromImage(ct_image)
            if preprocess:
                ct_array = self.preprocess_ct_scan(ct_image)

            for idx, candidate in series_candidates.iterrows():
                try:
                    world_coord = np.array([candidate.coordZ, candidate.coordY, candidate.coordX])
                    voxel_coord = np.round(ct_image.TransformPhysicalPointToIndex(world_coord)).astype(int)

                    crop = self.crop_nodule(ct_array, voxel_coord, self.window_size)

                    # Save as .npy file
                    file_name = f'{series_id}_{idx}.npy'
                    save_path = samples_dir / file_name
                    np.save(save_path, crop)

                    # Save info to CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            file_name,
                            series_id,
                            candidate['class'],
                            voxel_coord[2],  # x
                            voxel_coord[1],  # y
                            voxel_coord[0]  # z
                        ])

                except Exception as e:
                    print(f"Error processing candidate {idx} from series {series_id}: {str(e)}")
                    continue


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

        # Save split information
        split_info = {
            'train': train_series.tolist(),
            'val': val_series.tolist(),
            'test': test_series.tolist()
        }
        np.save(self.output_path / 'split_info.npy', split_info)

        print("Dataset preparation completed!")


# Usage
if __name__ == "__main__":
    base_path = "/home/mustafa/project/dataset"
    candidates_file = "/home/mustafa/project/dataset/candidates_V2.csv"
    output_path = "/home/mustafa/project/processed_dataset2"
    try:
        print(f"Starting dataset preparation...")
        print(f"Base path: {base_path}")
        print(f"Output path: {output_path}")

        processor = LungNoduleDatasetPreparation(base_path, candidates_file, output_path)
        processor.prepare_dataset()

    except Exception as e:
        print(f"Fatal error occurred:")
        print(f"Error details: {str(e)}")
    #processor = LungNoduleDatasetPreparation(base_path, candidates_file, output_path)
    #processor.prepare_dataset()

# Created/Modified files during execution:
# /home/processed_dataset/train/train_data.h5
# /home/processed_dataset/val/val_data.h5
# /home/processed_dataset/test/test_data.h5
# /home/processed_dataset/split_info.npy