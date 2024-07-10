import subprocess

def run_nnUNet_prediction(input_folder, output_folder, dataset_id, cascade_mode, num_parts, part_id,
                          prev_stage_predictions):
    command = (
        f'nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} '
        f'-c {cascade_mode} -num_parts {num_parts} -part_id {part_id} '
        f'--save_probabilities -prev_stage_predictions {prev_stage_predictions}'
    )
    try:
        subprocess.run(command, check=True, shell=True)
        print(f'Successfully ran prediction for dataset {dataset_id}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while running prediction for dataset {dataset_id}: {e}')


# pre-trained model
model_id = 503  # pretrained model id
config = '3d_fullres'  # pretrained model config

# input
input_folder = r'..\DATASET\nnUNet\nnUNet_raw\Dataset500_FINDS\imagesTs'
output_folder = f'dataset\data_20240702\prediction\{config}'

run_nnUNet_prediction(input_folder, output_folder, model_id, config)
