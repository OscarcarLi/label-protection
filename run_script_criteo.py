import os
import subprocess
import time

# print(shlex.split('python main_wdl.py --gpu_option --device-number 5 --num_epochs 100 --batch_size 1024 --lr 1e-4--input_file ~/deep_ctr/criteo/train_0.001.csv --noise_layer_function sumKL --p pos_frac --dynamic --error_prob_lower_bound 0.45 --init_scale 5.0'))

FNULL = open(os.devnull, 'w')
common_options = [
    'python3',
    'main_wdl_criteo.py',
    '--gpu_option',
    '--num_epochs', '5',
    '--batch_size', '1024', #'32768',
    '--lr_schedule', '1e-4',
    '--train_file', 'dataset/criteo/new_0.005train_0.9.csv',
    '--test_file', 'dataset/criteo/new_0.005test_0.1.csv',
    '--model_config', '128', '128', '128', '-1', '128', '128', '128',
    '--period', '100',
    '--num_hints', '1', '3', '5',
]

gpus = {}
for gpu_id in range(8):
    gpus[gpu_id] = ['--device_number', str(gpu_id),]

##### KL
# init_scale, gpu_index
KL_configs = [
    # (0.01, 0),
    # (0.05, 0),
    (0.1, 0),
    # (0.15, 1),
    # (0.2, 1),
    # (0.25, 1),
    # (0.3, 2),
    # (0.4, 2),
    # (0.5, 2),
    # (1.0, 3),
    # (1.5, 3),
    # (1.75, 3),
    # (2.0, 4),
    # (2.5, 4),
    # (3.0, 4),
    # (4.0, 5),
    # (5.0, 5),
    # (6.0, 5),
    # (7.0, 6),
    # (8.0, 6),
    # (9.0, 6),
    # (10.0, 7),
    # (20.0, 7),
    # (25.0, 7)
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

# ratio, gpu_idx
white_gaussian_configs = [
    # (0, 0),
    # (0.5, 0),
    (1.0, 1),
    # (1.25, 1),
    # (1.5, 2),
    # (1.75, 2),
    # (2, 3),
    # (2.5, 3),
    # (3, 0),
    # (3.5, 0),
    # (4.0, 1),
    # (4.5, 1),
    # (5, 2),
    # (7, 2),
    # (9, 3),
    # (11, 3),
    # (0.25, 4),
    # (15.0, 4),
    # (25.0, 5),
    # (0.75, 5),
    # (2.25, 6),
    # (2.75, 6),
    # (6.0, 7)
]

for ratio, gpu_idx in white_gaussian_configs:
    white_gaussian_options = [
        '--noise_layer_function', 'white_gaussian',
        '--ratio', str(ratio)]
    subprocess.Popen(args=common_options + gpus[gpu_idx] + white_gaussian_options, stdout=FNULL)

#### max_options
max_norm_options = [
    '--noise_layer_function', 'expectation',
]
subprocess.Popen(args=common_options + gpus[2] + max_norm_options, stdout=FNULL)

FNULL.close()
