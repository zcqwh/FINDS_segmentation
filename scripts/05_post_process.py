import subprocess
import os

nnUNet_results = os.environ.get('nnUNet_results')

def run_nnUNet_postprocessing(input_folder, output_folder, postprocessing_pkl, num_threads, plans_json):
    command = (
        f'nnUNetv2_apply_postprocessing -i {input_folder} -o {output_folder} '
        f'-pp_pkl_file {postprocessing_pkl} -np {num_threads} -plans_json {plans_json}'
    )
    try:
        subprocess.run(command, check=True, shell=True)
        print(f'Successfully applied postprocessing for dataset')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while applying postprocessing: {e}')

# 示例调用
input_folder = r''
output_folder = f'{input_folder}_pp'
postprocessing_pkl = f'{nnUNet_results}\Dataset503_FINDS\nnUNetTrainer__nnUNetPlans__3d_fullres\crossval_results_folds_0_1_2_3_4\postprocessing.pkl'
plans_json = f'{nnUNet_results}\Dataset503_FINDS\nnUNetTrainer__nnUNetPlans__3d_fullres\crossval_results_folds_0_1_2_3_4\plans.json'

run_nnUNet_postprocessing(input_folder, output_folder, postprocessing_pkl, plans_json)
