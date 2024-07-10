# FINDS_segmentation

This project targets the tissue segmentation of #63148 *Drosophila* larva.  
More details about the data acquisition can be found in our paper [Flow zoometry of *Drosophila*](https://www.biorxiv.org/content/10.1101/2024.04.04.588032v1).


## Installation

### Environment Setup

To set up the environment for the FINDS_segmentation project, follow these steps:

1. **Create a Conda Environment**: Create a new Conda environment with Python 3.9 using the following command:
    ```sh
    conda create --name finds_env python=3.9
    ```
2. **Activate the Conda Environment**:
    ```sh
    conda activate finds_env
    ```
3. **Install torch**: 
    ```sh
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
   Note: `pytorch==2.2.0` is also compatible, but `pytorch==2.3.0` and above are not compatible.
4. **Install nnU-Net**: Install the nnU-Net package using the following command:
    ```sh
    pip install nnunetv2==2.3
    ```

This will set up the required environment for the project.
## Usage Instructions

### Setting up environment variables

First, add the following environment variables to your system:

* `nnUNet_raw`: Path to the raw data for nnU-Net.
* `nnUNet_preprocessed`: Path to the preprocessed data for nnU-Net.
* `nnUNet_results`: Path to store the results of nnU-Net.

### Installing pretrained model

To install the pretrained model, follow these steps:

*  Execute the `install_model.py` script to complete the pretrained model installation.

### Running Segmentation and Feature Extraction Scripts

To perform tissue segmentation and feature extraction, follow these steps in order:

* **Convert NIfTI Files**: Run the `01_convert_nii.py` script to convert the necessary files into NIfTI format.
* **Generate JSON Configuration**: Run the `02_gen_json.py` script to generate the required JSON configuration files.
* **Convert to MSD Dataset Format**: Run the `03_convert_msd_dataset.py` script to convert the data into the MSD dataset format.
* **Perform Predictions**: Run the `04_predict.py` script to perform the segmentation predictions.
* **Post-Processing**: Run the `05_post_process.py` script to post-process the prediction results.
* **Extract Features for 7 Classes**: Finally, run the `06_features_7classes.py` script to extract features for the 7 distinct classes.

Follow these steps in sequence to complete the tissue segmentation and feature extraction process for the FINDS_segmentation project.


