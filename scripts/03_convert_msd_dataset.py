import subprocess

def run_nnUNet_conversion(task_folder):
    command = f'nnUNetv2_convert_MSD_dataset -i {task_folder}'
    try:
        subprocess.run(command, check=True, shell=True)
        print(f'Successfully converted dataset at {task_folder}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while converting dataset at {task_folder}: {e}')

# Call the function
task_folder = r'Z:\Nana\FINDS_task\data\DATASET\Task516_Kobayashi20240702'
run_nnUNet_conversion(task_folder)
