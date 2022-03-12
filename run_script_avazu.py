import os
import subprocess
import time

# print(shlex.split('python main_wdl.py --gpu_option --device-number 5 --num_epochs 100 --batch_size 1024 --lr 1e-4--input_file ~/deep_ctr/criteo/train_0.001.csv --noise_layer_function sumKL --p pos_frac --dynamic --error_prob_lower_bound 0.45 --init_scale 5.0'))

FNULL = open(os.devnull, 'w')
common_options = [
    'python3',
    '../main_wdl_avazu.py',
    '--gpu_option',
    '--num_epochs', '5',
    '--batch_size', '32768',
    '--lr_schedule', '1e-4',
    '--train_file', 'dataset/avazu/new_train_full_0.9.csv',
    '--test_file', 'dataset/avazu/new_test_full_0.1.csv', # 'new_test_1.0_0.1.csv'
    '--dictionary_file', 'dataset/avazu/vocab_size.pickle', 
    '--model_config', '128', '128', '128', '-1', '128', '128', '128',
    '--period', '100',
    '--num_hints', '1', '3', '5',
]

gpus = {}
for gpu_id in range(8):
    gpus[gpu_id] = ['--device_number', str(gpu_id),]

##### baseline with no noise added
no_noise_options = ['--noise_layer_function','identity']
subprocess.Popen(args=common_options + gpus[7] + no_noise_options,
                stdout=FNULL)

#### KL
# init_scale, gpu_index
KL_configs = [
    # (0.01, 0),
    # (0.05, 0),
    # (0.1, 1),
    # (0.15, 1),
    # (0.2, 2),
    # (0.25, 2),
    # (0.3, 3),
    # (0.4, 3),
    # (0.5, 4),
    (1.0, 4),
    # (1.5, 5),
    # (1.75, 5),
    # (2.0, 6),
    # (2.5, 6),
    # (3.0, 7),
    (4.0, 7),
    # (5.0, 0),
    # (6.0, 1),
    # (7.0, 2),
    # (8.0, 3),
    # (9.0, 4),
    # (10.0, 5),
    # (20.0, 6),
    # (25.0, 7)
]
# KL_configs = [
    # (0.01, 0),
    # (0.05, 4),
    # (0.25, 1),
    # (0.5, 2),
    # (1.0, 7),
    # (5.0, 6),
    # (10.0, 5),
    # (25.0, 7)
# ]
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
    # (0.5, 1),
    # (1.0, 2),
    # (1.25, 3),
    # (1.5, 4),
    # (1.75, 5),
    # (2, 6),
    # (2.5, 7),
    # (3, 0),
    # (3.5, 1),
    # (4.0, 2),
    # (4.5, 3),
    # (5, 4),
    # (7, 5),
    # (9, 6),
    # (11, 7),
    # (0.25, 5),
    # (15.0, 5),
    (25.0, 0),
    # (0.75, 6),
    # (2.25, 6),
    (2.75, 1),
    # (6.0, 7)
]
#
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
