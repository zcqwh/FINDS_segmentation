# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:48:33 2023

@author: Nana
"""
import os
from skimage.measure import regionprops
import h5py
import numpy as np
import SimpleITK as sitk
import threading
from scipy.spatial import ConvexHull
import skimage.morphology as morphology
import time
import pandas as pd

properties_save = ['area',
                   'area_convex',
                   'area_filled',
                   'axis_major_length',
                   'axis_minor_length',
                   # 'bbox',
                   # 'centroid',
                   # 'centroid_local',
                   # 'centroid_weighted',
                   # 'centroid_weighted_local',
                   # 'coords',
                   # 'coords_scaled',
                   # 'equivalent_diameter_area',
                   # 'euler_number',
                   # 'extent',
                   'feret_diameter_max',
                   'filename',
                   # 'inertia_tensor',
                   # 'inertia_tensor_eigvals',
                   'intensity_max',
                   'intensity_mean',
                   'intensity_min',
                   # 'label',
                   # 'moments',
                   # 'moments_central',
                   # 'moments_normalized',
                   # 'moments_weighted',
                   # 'moments_weighted_central',
                   # 'moments_weighted_normalized',
                   # 'num_pixels',
                   'solidity'
                   # 'image',
                   # 'image_convex',
                   # 'image_filled',
                   # 'image_intensity',
                   ]
# pickle_file = 'region_features.pickle'
pixel_size = (0.008, 0.008, 0.008)  # unit: mm
pixel_volume = pixel_size[0] * pixel_size[1] * pixel_size[2]
pixel_length = pixel_size[0]

# Define the data groups to process
data_groups = [
    {
        "disc_mask_folder": r"Y:\kobayashi\prediction\Dataset505_KobayashiPTFE\3d_fullres_pp",
        "intensity_folder": r"Y:\kobayashi\processed\Task505_KobayashiPTFE",
        "h5_file_path": "features_Task505_KobayashiPTFE.h5"
    },
    {
        "disc_mask_folder": r"Y:\kobayashi\prediction\Dataset506_KobayashiPP0.5\3d_fullres_pp",
        "intensity_folder": r"Y:\kobayashi\processed\Task506_KobayashiPP0.5",
        "h5_file_path": "features_Task506_KobayashiPP0.5.h5"
    },
    {
        "disc_mask_folder": r"Y:\kobayashi\prediction\Dataset507_KobayashiPMMA\3d_fullres_pp",
        "intensity_folder": r"Y:\kobayashi\processed\Task507_KobayashiPMMA",
        "h5_file_path": "features_Task507_KobayashiPMMA.h5"
    },
    {
        "disc_mask_folder": r"Y:\kobayashi\prediction\Dataset508_KobayashiPP4\3d_fullres_pp",
        "intensity_folder": r"Y:\kobayashi\processed\Task508_KobayashiPP4",
        "h5_file_path": "features_Task508_KobayashiPP4.h5"
    },

    # Add more groups as needed
]

for group in data_groups:
    disc_mask_folder = group["disc_mask_folder"]
    intensity_folder = group["intensity_folder"]
    h5_file_path = '7class' + group["h5_file_path"]

    # Create a dictionary to store properties by label
    label_props_dict = {}

    # Initialize locks
    print_lock = threading.Lock()
    progress_lock = threading.Lock()

    # Shared variables to track the number of processed files
    processed_files = 0
    total_files = 0


    def process_file(filenames):
        global processed_files
        for filename in filenames:
            # Load NII file
            disc_image = sitk.ReadImage(os.path.join(disc_mask_folder, filename))
            disc_image = sitk.GetArrayFromImage(disc_image)
            unique_labels = np.unique(disc_image.astype('uint8'))
            expected_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            # Check if unique_labels contains all elements of expected_labels
            if np.array_equal(np.sort(unique_labels), np.sort(expected_labels)):

                intensity_image = sitk.ReadImage(os.path.join(intensity_folder, filename))
                intensity_image = sitk.GetArrayFromImage(intensity_image)

                # Process properties for each region
                disc_props = regionprops(disc_image, intensity_image, spacing=pixel_size)
                # Create a dictionary to store properties by label

                for i, region in enumerate(disc_props):
                    region_label = i + 1

                    if region_label not in label_props_dict:
                        label_props_dict[region_label] = {prop: [] for prop in properties_save}
                        label_props_dict[region_label]['filename'] = []

                    for prop in properties_save:
                        if prop == 'filename':
                            label_props_dict[region_label]['filename'].append(filename)
                        else:
                            try:
                                value = getattr(region, prop)
                                if isinstance(value, np.ndarray):
                                    label_props_dict[region_label][prop].append(value)

                                elif isinstance(value, tuple):
                                    label_props_dict[region_label][prop].append(list(value))
                                else:
                                    label_props_dict[region_label][prop].append(value)
                            except:
                                label_props_dict[region_label][prop].append(None)

            else:
                with print_lock:
                    print(f"{filename}:{unique_labels}")

            # Update the counter for processed files
            with progress_lock:
                processed_files += 1
                progress = (processed_files / total_files) * 100
                print(f"Processed {filename} {processed_files}/{total_files} files. Progress: {progress:.2f}%")


    # Define number of threads
    num_threads = 64  # Change the number of threads as needed
    # Get all files in the mask folder
    file_list = [filename for filename in os.listdir(disc_mask_folder) if filename.endswith('.nii.gz')]
    total_files = len(file_list)

    # Create and start multiple threads to process files
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=process_file, args=(file_list[i::num_threads],))
        thread.start()
        threads.append(thread)
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Open HDF5 file to write data
    with h5py.File(h5_file_path, 'w') as h5file:
        h5file.attrs['unit'] = 'mm'
        h5file.attrs['pixel_size'] = pixel_size
        # h5file.attrs['label_3'] = 'left wing disc'
        # h5file.attrs['label_4'] = 'right wing disc'

        # Iterate over each label in the dictionary
        for region_label, props in label_props_dict.items():
            # Create a group for each label
            group = h5file.create_group(f'Label_{region_label}')

            # Traverse each attribute list of the current label
            for prop_name, prop_values in props.items():
                if prop_name in ['coords', 'coords_scaled']:
                    pass
                else:
                    if isinstance(prop_values, np.ndarray):
                        group.create_dataset(prop_name, data=prop_values)
                    elif isinstance(prop_values, (int, float, np.int32, np.float64)):
                        group.create_dataset(prop_name, data=prop_values)
                    elif isinstance(prop_values, (tuple, list)):
                        # Assuming all elements in the tuple/list are of the same type, and if they're strings,
                        # it's handled by converting to an array of dtype object
                        if all(isinstance(item, str) for item in prop_values):
                            dt = h5py.special_dtype(vlen=str)  # Define variable-length string datatype
                            string_array = np.array(prop_values,
                                                    dtype=object)  # Convert list of strings to an object array
                            group.create_dataset(prop_name, data=string_array, dtype=dt)
                        else:
                            # If the list/tuple contains non-string data, handle as numerical data
                            group.create_dataset(prop_name, data=np.array(prop_values).astype('float64'))
                    elif isinstance(prop_values, str):
                        dt = h5py.special_dtype(vlen=str)  # Define variable-length string datatype
                        group.create_dataset(prop_name, data=prop_values, dtype=dt)
                    else:
                        # For other data types, you might want to convert them to string or handle them specifically
                        group.create_dataset(prop_name, data=str(prop_values))

    print("The data has been successfully saved to the HDF5 file.")

    # Prepare a list to hold all rows of the final table
    all_data = []

    # Iterate over each label and its properties in the dictionary
    for region_label, props in label_props_dict.items():
        # Each entry in props will become a row in the DataFrame
        for index in range(len(props['filename'])):  # Assuming 'filename' is always present
            row_data = {'Label': region_label}
            for prop_name, prop_values in props.items():
                # Handle each property accordingly, assuming they all have the same length
                row_data[prop_name] = prop_values[index] if index < len(prop_values) else None
            all_data.append(row_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_data)

    # Define the Excel file path
    excel_file_path = f'{h5_file_path[:-3]}.xlsx'

    # Save the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)

    print(f"Data has been successfully saved to {excel_file_path}.")
