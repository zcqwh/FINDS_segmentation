# from nnunet.dataset_conversion.utils import generate_dataset_json
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[5:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!",
                          dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


labels = {
    "0": 'background',
    "1": 'salivary glands_L',
    "2": 'salivary glands_R',
    "3": 'wing disc_L',
    "4": 'wing disc_R',
    "5": 'haltere disc_L',
    "6": 'haltere disc_R',
    "7": 'esophagus'
}
dataset_name = "FINDS"
modalities = tuple(['FINDS-pfa'])

# Change the task number as needed
tasks = ["Task516_Kobayashi20240702"]

for task in tasks:
    output_file = f'Z:\\Nana\\FINDS_task\\data\\DATASET\\{task}\\dataset.json'
    imagesTr_dir = f'Z:\\Nana\\FINDS_task\\data\\DATASET\\{task}\\imagesTr'
    imagesTs_dir = f'Z:\\Nana\\FINDS_task\\data\\DATASET\\{task}\\imagesTs'
    generate_dataset_json(output_file=output_file, imagesTr_dir=imagesTr_dir, imagesTs_dir=imagesTs_dir, labels=labels,
                          dataset_name=dataset_name, modalities=modalities)
    print(f"Generated dataset.json for {task}.")

# from nnunet.dataset_conversion.convert_MSD_dataset import convert_msd_dataset
