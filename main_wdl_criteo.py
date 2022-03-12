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

from model import MLP, WDL
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
parser.add_argument("--lr_drop_step", nargs='*', type=int, default=[],
                    help='after how many SGD updates to use the next learning rate; should have one fewer elements than lr_schedule')
parser.add_argument("--train_file", type=str,
                    default="/dataset/criteo/new_0.005train_0.9.csv",
                    help="directory of the dataset for training")
parser.add_argument("--test_file", type=str,
                    default="/dataset/criteo/new_0.005test_0.1.csv",
                    help="directory of the dataset for testing")
parser.add_argument("--period", type=int, default=3000,
                    help='how many batches per test evaluation, will also average the training loss over this period, ')
parser.add_argument("--num_hints", nargs="*", type=int, default=None)

parser.add_argument(
  "--model_config",  # model config for the deep part of the model
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[128, 128, 128, 128, -1],  # default if nothing is provided
)

parser.add_argument("--noise_layer_function", type=str,
                    default="identity",
                    help="choose between 1) perp 2) expectation 3) identity 4) sumKL")
parser.add_argument("--lower", type=float, default=1.0, help='for gradient_perp')
parser.add_argument("--upper", type=float, default=5.0, help='for gradient_perp')
parser.add_argument("--p_frac", type=str, default='pos_frac', help='for KL')
parser.add_argument("--dynamic", action='store_true',
                    help='wheter to use dynamic changing of power constraint based on KL')
parser.add_argument("--error_prob_lower_bound", type=float, default=None,
                    help='minimum allowed error probability of worst case passive party')
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
embedding_dimension = 4
with open('dataset/criteo/vocab_size.pkl', 'rb') as file:
    vocab_size = pickle.load(file=file)

sparse_feature_names = ['C' + str(i) for i in range(1, 27)]
dense_feature_names = ['I' + str(i) for i in range(1, 14)]

# label encoding for sparse features,and do simple Transformation for dense features

feature_tags = [SparseFeat(name=feat, vocabulary_size=vocab_size[feat], embedding_dim=embedding_dimension)
                        for i, feat in enumerate(sparse_feature_names)] \
                            + [DenseFeat(name=feat, dimension=1,)
                        for feat in dense_feature_names] # name tags

deep_feature_tags = feature_tags
wide_feature_tags = feature_tags

batch_size_train = args.batch_size
batch_size_test = 1024



# total_training_instances = len(train_data)
# total_test_instances = len(test_data)
# num_batchs = total_training_instances // batch_size_train
# print("total_training_instances: {}, total_test_instances: {}, num_batchs: {}".format(total_training_instances,
#                                                                                       total_test_instances, num_batchs))

select_columns = ['Label'] + dense_feature_names + sparse_feature_names
print(select_columns)
# currently replacing NA values with 0 and empty string
column_defaults = [tf.int32] + [tf.zeros(1) for d in dense_feature_names] \
                             + [tf.int32 for s in sparse_feature_names]
train_set = tf.data.experimental.make_csv_dataset(
    file_pattern=args.train_file,
    batch_size=batch_size_train,
    column_defaults=column_defaults,
    label_name='Label',
    select_columns=select_columns,
    header=True,
    num_epochs=1,
    shuffle=True,
    shuffle_buffer_size=8 * batch_size_train,
    shuffle_seed=28,
    prefetch_buffer_size=1 * batch_size_train,
    num_parallel_reads=32,
    sloppy=True,
)
test_set = tf.data.experimental.make_csv_dataset(
    file_pattern=args.test_file,
    batch_size=batch_size_test,
    column_defaults=column_defaults,
    label_name='Label',
    select_columns=select_columns,
    header=True,
    num_epochs=1,
    shuffle=False,
    num_parallel_reads=32,
    sloppy=True,
)

print('training and set construction finishes')

###################################
######### model creation ##########
###################################
print('model construction starts')
num_outputs = 1

hidden_units = ['noise_layer' if a == -1 else a for a in args.model_config] # [128, 128, 128, 128, -1] => [128, 128, 128, 128, 'noise_layer']
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
model = WDL(wide_feature_tags=wide_feature_tags,
            deep_feature_tags=deep_feature_tags,
            config=hidden_units + [num_outputs],
            noise_layer_function=noise_layer_function)

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
else:
    trainer = tf.keras.optimizers.Adam(learning_rate=lr_schedule[0])
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
                            + '_batchsize{}_reg{}_lr{}_datafrac{}'.format(args.batch_size, 
                                                          regularization_weight,
                                                        learning_rate_string(lr_schedule=lr_schedule,                         
                                                                             lr_drop_step=lr_drop_step),
                                                         args.train_file[args.train_file.find('train_')+6: args.train_file.rfind('.')])

path_name = os.path.join(folder_name, file_name)
logdir = 'logs/{}_{}'.format(path_name, stamp)
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