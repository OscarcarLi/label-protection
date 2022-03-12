import os
import subprocess
import time

# print(shlex.split('python main_wdl.py --gpu_option --device-number 5 --num_epochs 100 --batch_size 1024 --lr 1e-4--input_file ~/deep_ctr/criteo/train_0.001.csv --noise_layer_function sumKL --p pos_frac --dynamic --error_prob_lower_bound 0.45 --init_scale 5.0'))

FNULL = open(os.devnull, 'w')
common_options = [
    'python3',
    '../main_isic_convmlp.py',
    '--gpu_option',
    '--num_epochs', '1000',
    '--batch_size', '128', #'32768',
    '--lr_schedule', '1e-5', #'2e-4',for adam 
    '--train_folder', 'dataset/ISIC-2020/train',
    '--test_folder', 'dataset/ISIC-2020/test',
    '--image_size', '84',
    '--period', '1000', # this will not be used
    '--num_hints', '1', '3', '5',
]

model_config = [
    '--model_config',
            'conv64', 'conv64', 'conv64', 'conv64', '-1', 'conv64', 'conv64',
            'flatten', 'fc64',
    '--l2_regularization_weight', '0.002'
]
common_options += model_config



gpus = {}
for gpu_id in range(8):
    gpus[gpu_id] = ['--device_number', str(gpu_id),]

###################################
######## white gaussian ###########
###################################

# ratio, gpu_idx
white_gaussian_configs = [
    # (0, 0),
    # (0.5, 3),
    # (1.0, 3),
    # (1.25, 3),
    # (1.5, 4),
    # (1.75, 4),
    # (2, 3),
    # (2.5, 4),
    # (3, 4),
    # (3.5, 5),
    # (4.0, 5),
    (4.5, 0),
    # (5, 6),
    (6.0, 0),
    # (7, 5),
    (9, 0),
    (11, 1),
    (13, 1),
    # (0.25, 5),
    (15.0, 1),
    # (25.0, 6),
    # (0.75, 6),
    # (2.25, 6),
    # (2.75, 7),
]

for ratio, gpu_idx in white_gaussian_configs:
    white_gaussian_options = [
        '--noise_layer_function', 'white_gaussian',
        '--ratio', str(ratio)]
    subprocess.Popen(args=common_options + gpus[gpu_idx] + white_gaussian_options, stdout=FNULL)
    

####################
######## KL ########
####################

# init_scale, gpu_index

KL_configs = [
    # (0.01, 5),
    (0.05, 2),
    # (0.1, 6),
    (0.15, 2),
    # (0.2, 6),
    (0.25, 2),
    # (0.3, 6),
    (0.4, 3),
    # (0.5, 7),
#     (1.0, 3),
    (1.5, 3),
    (1.75, 3),
#     (2.0, 4),
    # (2.5, 4),
#     (3.0, 4),
#     (4.0, 5),
#     (5.0, 5),
#     (6.0, 5),
    # (7.0, 7),
#     (8.0, 6),
    # (9.0, 7),
#     (10.0, 7),
#     (20.0, 7),
#     (25.0, 7)
]

for init_scale, gpu_idx in KL_configs:
    KL_specific_options = [
        '--noise_layer_function', 'sumKL',
        '--p_frac', 'pos_frac',
        # '--dynamic',
        # '--error_prob_lower_bound', '0.3',
        '--init_scale', str(init_scale),
        '--uv_choice', 'uv',
    ]
    subprocess.Popen(args=common_options + gpus[gpu_idx] + KL_specific_options, stdout=FNULL)

################################
########### Max Norm ###########
################################

#### max_options
max_norm_options = [
    '--noise_layer_function', 'expectation',
]
subprocess.Popen(args=common_options + gpus[2] + max_norm_options, stdout=FNULL)

FNULL.close()
