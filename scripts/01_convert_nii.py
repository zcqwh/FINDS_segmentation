# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:34:53 2024

@author: Chenqi Zhang
"""
import numpy as np
import os
import csv
from concurrent.futures import ProcessPoolExecutor
import h5py
from skimage import io
from skimage.filters import threshold_otsu
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

"""
Python script for preprocessing 3D image data
Threshold_otsu was performed on the image to remove the background. threshold_range: (100, 1000).
Keep the largest region in the image and remove the rest.
"""

def apply_otsu(image, lower_limit, upper_limit):
    subset_image = image[(image > lower_limit) & (image < upper_limit)]
    if subset_image.size > 0:
        threshold = threshold_otsu(subset_image)
    else:
        threshold = lower_limit

    image_copy = image.copy()
    image_copy[image_copy < threshold] = 0

    return image_copy

def process_image(image_path, sigma=2):
    """
        Process an image from an HDF5 file.

        Parameters:
        image_path (str): The path to the HDF5 file containing the image data.
        sigma (float): The standard deviation for the Gaussian filter (default is 2).

        Returns:
        numpy.ndarray: The processed image.
        numpy.ndarray: max_region_mask.
    """

    with h5py.File(image_path, 'r') as file:
        image = np.array(file['dataset_1'])
        image = apply_otsu(image, 100, 1000)

    smoothed_image = gaussian_filter(image, sigma=sigma)
    threshold_image = np.where(smoothed_image > 0, 1, 0)

    labeled_image, _ = label(threshold_image)

    region_sizes = np.bincount(labeled_image.ravel())
    max_region_label = region_sizes[1:].argmax() + 1

    max_region_mask = (labeled_image == max_region_label)
    max_region_mask = binary_fill_holes(max_region_mask)
    image = image * max_region_mask

    return image, max_region_mask

def process_single_item(image_path, index, save_path, save_mask_path):
    # Process the image and mask
    processed_image, mask = process_image(image_path)
    image_save_path = os.path.join(save_path, f"FINDS_{index:04d}.nii.gz")
    mask_save_path = os.path.join(save_mask_path, f"FINDS_{index:04d}.nii.gz")

    # Save the processed image and mask
    sitk_image = sitk.GetImageFromArray(processed_image.astype(np.float32))
    sitk.WriteImage(sitk_image, image_save_path)

    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk.WriteImage(sitk_mask, mask_save_path)

    print(f'Processed dataset {index}: {os.path.basename(image_path)}')

    return image_path, image_save_path

def process_file_wrapper(args):
    return process_single_item(*args)

# Main function to orchestrate the multiprocessing workflow
def main():
    target_root = r"Z:\\Nana\\FINDS_task\\data\\DATASET" # Root folder to save the processed data

    # Dictionary to store the mapping between data folders and target folders
    folder_mapping = {
        r"Y:\20240702_kobayashi\h5_affined": "Task516_Kobayashi20240702", # Example entry
        # Add more mappings as needed
    }

    for data_folder, target_folder in folder_mapping.items():
        target_path = os.path.join(target_root, target_folder)

        save_label_path = os.path.join(target_path, "labelsTr")
        save_train_path = os.path.join(target_path, "imagesTr")
        save_test_path = os.path.join(target_path, "imagesTs")
        save_train_mask_path = os.path.join(target_path, "imagesTr_mask")
        save_test_mask_path = os.path.join(target_path, "imagesTs_mask")

        # Check if each path exists, if not, create it
        for path in [save_label_path, save_train_path, save_test_path, save_train_mask_path, save_test_mask_path]:
            os.makedirs(path, exist_ok=True)

        mappings = []  # This will store the mappings

        # Prepare arguments for each file to be processed in parallel
        file_paths = [(os.path.join(data_folder, file), i, save_test_path, save_test_mask_path) for i, file in
                      enumerate(os.listdir(data_folder))]

        with ThreadPoolExecutor() as executor:
            # Submit the tasks and collect the results
            results = executor.map(process_file_wrapper, file_paths)

            # Append results to mappings
            mappings.extend(results)

        # Convert the list to a DataFrame and save as Excel
        df_mappings = pd.DataFrame(mappings, columns=['Source Path', 'Processed Image Path'])
        excel_path = f'processed_mappings_{target_folder}.xlsx'
        df_mappings.to_excel(excel_path, index=False)

        print(f"Excel file with mappings saved to {excel_path}")

if __name__ == "__main__":
    main()
