#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # remove tensorflow INFO messages
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
import datetime
from pytz import timezone
from collections import defaultdict, OrderedDict

from model import ConvMLP
from custom_gradients_masking import no_noise, gradient_masking, gradient_perp_masking_function_creator, gradient_gaussian_noise_masking_function_creator, KL_gradient_perturb_function_creator
from utils import sigmoid_cross_entropy, learning_rate_string
from train_and_test import train, test
from feature_tag import SparseFeat, DenseFeat
import shared_var

from resource_setup import setup_gpu

import tensorflow as tf
tf.random.set_seed(1234)
tf.get_logger().setLevel('WARNING')
from tensorflow import keras


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_option', action='store_true', help='whether to use gpu')
parser.add_argument('--device_number', type=int, default=0)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=300)
parser.add_argument("--lr_schedule", nargs='*', type=float, default=[1e-3])
parser.add_argument("--lr_drop_step", nargs='*', type=int, default=[], help='after how many SGD updates to use the next learning rate; should have one fewer elements than lr_schedule')
parser.add_argument("--train_folder", type=str,
                    default="",
                    help="directory of the dataset for training")
parser.add_argument("--test_folder", type=str,
                    default="",
                    help="directory of the dataset for testing")
parser.add_argument("--image_size", type=int,)
parser.add_argument("--period", type=int, default=3000, help='how many batches per test evaluation, will also average the training loss over this period, ')
parser.add_argument("--num_hints", nargs="*", type=int, default=None)

parser.add_argument(
  "--model_config",  # model config for the deep part of the model
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=['conv64', 'conv64', 'conv64', 'conv64',
            'avgpool5', 'flatten', '-1', 'fc64', 'fc64', 'fc64'],  # default if nothing is provided
)
parser.add_argument(
    "--l2_regularization_weight", type=float, default=0.0,
)

parser.add_argument("--noise_layer_function", type=str,
                    default="identity",
                    help="choose between 1) perp 2) expectation 3) identity 4) sumKL")
parser.add_argument("--lower", type=float, default=1.0, help='for gradient_perp')
parser.add_argument("--upper", type=float, default=5.0, help='for gradient_perp')
parser.add_argument("--p_frac", type=str, default='pos_frac', help='for KL')
parser.add_argument("--dynamic", action='store_true', help='wheter to use dynamic changing of power constraint based on KL')
parser.add_argument("--error_prob_lower_bound", type=float, default=None, help='minimum allowed error probability of worst case passive party')
parser.add_argument("--sumKL_threshold", type=float, default=None, help='maximum allowed sumKL,\
                                                                         do not need to specify this if using dynamic=True')
parser.add_argument("--init_scale", type=float, default=1.0, help='the initial value of P is scale * g')
parser.add_argument("--uv_choice", type=str, default='uv', help='uv, same, or zero')
parser.add_argument("--ratio", type=float, help='ratio to max_norm for white gaussian')

# parser.add_argument('--name', type=str, default='', help='save folder name')

args = parser.parse_args()

#############################
########CPU/GPU Setup########
#############################
setup_gpu(gpu_option=args.gpu_option, device_number=args.device_number)
gpu_list = tf.config.experimental.list_logical_devices('GPU') # this gpu numbering is not the same as the absolute numbering

######################################
########## Dataset Creation ##########
######################################
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
def normalize(x, y):
    return normalization_layer(x), y

batch_size_train = args.batch_size
batch_size_test = int(128)

train_set = \
    tf.keras.preprocessing.image_dataset_from_directory(
        directory=args.train_folder,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size_train,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        seed=32,
).map(normalize)
test_set = \
    tf.keras.preprocessing.image_dataset_from_directory(
        directory=args.test_folder,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size_test,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        seed=32,
).map(normalize)

print('training and set construction finishes')

###################################
######### model creation ##########
###################################
print('model construction starts')
num_outputs = 1

hidden_units = ['noise_layer' if a == '-1' else a for a in args.model_config] # [128, 128, 128, 128, -1] => [128, 128, 128, 128, 'noise_layer']
print('layers', hidden_units)

# gradient_gaussian_noise_masking = gradient_gaussian_noise_masking_function_creator(ratio=10.0)

###### construct the noise layer function ######
print('noise_layer_function', args.noise_layer_function)
if args.noise_layer_function == 'perp':
    print('lower', args.lower, 'upper', args.upper)
    noise_layer_function = gradient_perp_masking_function_creator(lower=args.lower, upper=args.upper)
elif args.noise_layer_function == 'expectation':
    noise_layer_function = gradient_masking
elif args.noise_layer_function == 'identity':
    noise_layer_function = no_noise
elif args.noise_layer_function == 'sumKL':
    noise_layer_function = KL_gradient_perturb_function_creator(p_frac=args.p_frac,
                                                                dynamic=args.dynamic,
                                                                error_prob_lower_bound=args.error_prob_lower_bound,
                                                                sumKL_threshold=args.sumKL_threshold,
                                                                init_scale=args.init_scale,
                                                                uv_choice=args.uv_choice)
elif args.noise_layer_function == 'white_gaussian':
    noise_layer_function = gradient_gaussian_noise_masking_function_creator(ratio=args.ratio)
else:
    assert False, 'noise layer function not found'

# model = MLP(config=hidden_units + [num_outputs], noise_layer_function=noise_layer_function)
# model = WDL(wide_feature_tags=wide_feature_tags,
#             deep_feature_tags=deep_feature_tags,
#             config=hidden_units + [num_outputs],
#             noise_layer_function=noise_layer_function)
model = ConvMLP(config=hidden_units + [num_outputs],
                noise_layer_function=noise_layer_function,
                l2_regularization_weight=args.l2_regularization_weight)

print('model construction finishes')

regularization_weight = 1

###########################################################
######## learning rate and optimizer construction #########
###########################################################

lr_schedule = args.lr_schedule
lr_drop_step = args.lr_drop_step
print('lr_schedule', lr_schedule)
print('lr_drop_step', lr_drop_step)

if len(lr_schedule) > 1:
    assert len(lr_schedule) == len(lr_drop_step) + 1
    learning_rate_w_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_drop_step,
                                                                                    values=lr_schedule)
    # learning_rate_w_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[500, 2500, 5000, 10000],
    #                                                                                 values=[1e-3,5e-4,1e-4,5e-5,1e-5])
    trainer = tf.keras.optimizers.Adam(learning_rate=learning_rate_w_schedule)
    # trainer = tf.keras.optimizers.SGD(learning_rate=learning_rate_w_schedule,)
else:
    trainer = tf.keras.optimizers.Adam(learning_rate=lr_schedule[0])
    # trainer = tf.keras.optimizers.SGD(learning_rate=lr_schedule[0])
# 
# trainer = tf.keras.optimizers.Adagrad(learning_rate=lr)

############################################
######## Set up tensorboard logging ########
############################################
west_tz = timezone('US/Eastern')
stamp = datetime.datetime.now(tz=west_tz).strftime("%Y%m%d-%H:%M:%S")

if args.noise_layer_function == 'identity':
    folder_name = 'no_noise'
    file_name = folder_name
elif args.noise_layer_function == 'perp':
    folder_name = 'perp_noise'
    file_name = folder_name + '_lower{}_upper{}'.format(args.lower, args.upper)
elif args.noise_layer_function == 'expectation':
    folder_name = 'expectation_align'
    file_name = folder_name
elif args.noise_layer_function == 'sumKL':
    folder_name = 'KL'
    file_name = folder_name
    if args.dynamic:
        file_name += '_dynamic_errorLB{}'.format(args.error_prob_lower_bound)
    else:
        file_name += '_fixed_scale{}'.format(args.init_scale)
    file_name += '_uv{}'.format(args.uv_choice)
elif args.noise_layer_function == 'white_gaussian':
    if args.ratio == 0:
        folder_name = 'no_noise'
        file_name = 'no_noise'
    else:
        folder_name = 'white_gaussian'
        file_name = folder_name + '_ratio{}'.format(args.ratio)
else:
    assert False, 'noise layer: {}'.format(args.noise_layer_function)
    
file_name = file_name + ''.join([str(a) for a in model.config]) \
                            + '_batchsize{}_reg{}_lr{}_data{}'.format(args.batch_size, 
                                                          args.l2_regularization_weight,
                                                        learning_rate_string(lr_schedule=lr_schedule,                         
                                                                             lr_drop_step=lr_drop_step),
                                                        f'ISIC{args.image_size}')

path_name = os.path.join(folder_name, file_name)
logdir = 'logs/isic_logs/{}_{}'.format(path_name, stamp)
print(logdir)

writer = tf.summary.create_file_writer(logdir)
shared_var.writer = writer
shared_var.logdir = logdir

t_s = datetime.datetime.now()
print("gpu_option: {}, train_batch_size: {}, regularization_weight: {}, lr: {}".format(args.gpu_option, args.batch_size, regularization_weight, learning_rate_string(lr_schedule=lr_schedule, lr_drop_step=lr_drop_step)))

####################################
######### train the model ##########
####################################
num_epochs = args.num_epochs

print('start training')
with tf.device(gpu_list[0].name):
    train(
        model=model,
        train_set=train_set, test_set=test_set,
        loss_function=sigmoid_cross_entropy, num_epochs=num_epochs,
        trainer=trainer, 
        writer=writer,
        regularization_weight=regularization_weight,
        period=args.period,
        num_hints=args.num_hints
    )
# test(test_set=test_set, model=model,
#      loss_function=sigmoid_cross_entropy, regularization_weight=regularization_weight)


print("gpu_option: {}, train_batch_size: {}, regularization_weight: {}, lr: {}".format(args.gpu_option, args.batch_size, regularization_weight, learning_rate_string(lr_schedule=lr_schedule, lr_drop_step=lr_drop_step)))
t_e = datetime.datetime.now()
# print("total_training_instances: {}, total_test_instances: {}, num_batchs: {}, training used: {}".format(total_training_instances,
#                                                                                       total_test_instances, num_batchs, t_e - t_s))