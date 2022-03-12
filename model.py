import time
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
 
from collections import OrderedDict
from feature_tag import SparseFeat, DenseFeat
# from custom_gradients_masking import gradient_perp_masking

import shared_var

class MLP(tf.keras.Model):
    
    def __init__(self, config, noise_layer_function):
        """construct the layers of the model

        Args:
            config (list): a list of numbers and strings that describes the layers of the model
                           an integer means a fully connected layer's hidden size
                           string noise_mask_layer means a gradient_masking layer
                           the last layer is the output layer and we don't have relu as the activation function
        """
        super(MLP, self).__init__()

        self.config = config
        self.noise_layer_function = noise_layer_function
        self.layer_list = []
        for item in config[:-1]:
            if item == 'noise_layer':
                layer = self.noise_layer_function
            elif isinstance(item, int):
                layer = tf.keras.layers.Dense(units=item,
                                            activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer=None,
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None)
            else:
                assert False, 'layer not defined'
            self.layer_list.append(layer)
        # add the last layer
        # the last layer is a linear layer without relu activation
        if isinstance(config[-1], int):
            logit_layer = tf.keras.layers.Dense(units=config[-1],
                                                activation='linear',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                bias_constraint=None)
        else:
            assert False, 'layer not defined'
        self.layer_list.append(logit_layer)
        self.set_up_layer_names()


    def call(self, X, no_noise=False):
        """ feedforward through the model
            return all the hidden layer activations
        """
        hidden_activation = tf.reshape(X, shape=(X.shape[0], -1))
        activations_by_layer = []
        for layer in self.layer_list:
            if layer == self.noise_layer_function and no_noise:
                hidden_activation = tf.identity(hidden_activation)
            else:
                hidden_activation = layer(hidden_activation)
            activations_by_layer.append(hidden_activation)
        return activations_by_layer

    def predict(self, X):
        """ feedforward through the model
            return all the hidden layer activations
        """
        hidden_activation = tf.reshape(X, shape=(X.shape[0], -1))
        activations_by_layer = []
        for layer in self.layer_list:
            hidden_activation = layer(hidden_activation)
        return hidden_activation

    @property
    def trainable_variables(self):
        params_list = []
        for layer in self.layer_list:
            if hasattr(layer, 'trainable_variables'):
                params_list.extend(layer.trainable_variables) # this could still be an empty list but that's ok
        return params_list
    
    def regularization_losses(self):
        r_loss = 0
        for layer in self.layer_list:
            if layer != self.noise_layer_function:
                r_loss += tf.math.reduce_sum(layer.losses)
        return r_loss

    def set_up_layer_names(self):
        self.layer_names = []
        layer_index = 0
        noise_index = self.config.index('noise_layer')
        for i in range(len(self.config)):
            if i == len(self.config) - 1:
                self.layer_names.append('logits_unit_'+str(self.config[i]))
            elif isinstance(self.config[i], int):
                layer_index += 1
                if i < noise_index:
                    key = 'mask_'
                else:
                    key = 'non_mask_'
                self.layer_names.append(key+'layer_'+str(layer_index)+'_unit_'+str(self.config[i]))
            elif self.config[i] == 'noise_layer':
                self.layer_names.append('non_mask_layer_'+str(layer_index)+'_unit_'+str(self.config[i-1])) # here i-1 might not be safe against BN added layers
            else:
                assert False

    def leak_auc_dict(self):
        auc_dict = OrderedDict()
        for layer_name in self.layer_names:
            auc_dict[layer_name] = tf.keras.metrics.Mean()
        return auc_dict


class ConvMLP(tf.keras.Model):

    @staticmethod
    def conv_block(n_out_channel, l2_regularization_weight=0.0):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(
            filters=n_out_channel,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularization_weight)
        ))
        result.add(
            tf.keras.layers.MaxPool2D(
                pool_size=(2,2),
            )
        )
        return result

    def __init__(self, config, noise_layer_function, l2_regularization_weight=0.0):
        """construct the layers of the model

        Args:
            config (list): a list of numbers and strings that describes the layers of the model
                           'conv64' means a model with a conv block with 64 output channels
                           'fc128' means a fully connected layer's hidden size is 128
                           'flatten' is a flatten layer with no parameters
                           'avgpool5' applies 5 by 5 pooling over h,w dimension
                           string noise_mask_layer means a gradient_masking layer
                           the last layer is the output layer and we don't have relu as the activation function
                           and is provided as an int
        """
        super(ConvMLP, self).__init__()

        self.config = config
        self.noise_layer_function = noise_layer_function
        self.layer_list = []
        for item in config[:-1]:
            if item == 'noise_layer':
                layer = self.noise_layer_function
            elif 'conv' in item:
                layer = ConvMLP.conv_block(
                    n_out_channel=int(item[4:]), l2_regularization_weight=l2_regularization_weight)

            elif 'fc' in item:
                layer = tf.keras.layers.Dense(units=int(item[2:]),
                                            activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer=None,
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None)
            elif 'avgpool' in item:
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=(int(item[7:]), int(item[7:])))

            elif 'flatten' == item:
                layer = tf.keras.layers.Flatten()

            else:
                assert False, 'layer not defined'
            self.layer_list.append(layer)
        # add the last layer
        # the last layer is a linear layer without relu activation
        if isinstance(config[-1], int):
            logit_layer = tf.keras.layers.Dense(units=config[-1],
                                                activation='linear',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                bias_constraint=None)
        else:
            assert False, 'layer not defined'
        self.layer_list.append(logit_layer)
        self.set_up_layer_names()


    def call(self, X, no_noise=False):
        """ feedforward through the model
            return all the hidden layer activations
        """
        hidden_activation = X 
        activations_by_layer = []
        for layer in self.layer_list:
            if layer == self.noise_layer_function and no_noise:
                hidden_activation = tf.identity(hidden_activation)
            else:
                hidden_activation = layer(hidden_activation)

            activations_by_layer.append(hidden_activation)
        return activations_by_layer
    
    def predict(self, X):
        """ feedforward through the model
            return all the hidden layer activations
        """
        hidden_activation = X
        activations_by_layer = []
        for layer in self.layer_list:
            hidden_activation = layer(hidden_activation)
        return hidden_activation

    @property
    def trainable_variables(self):
        params_list = []
        for layer in self.layer_list:
            if hasattr(layer, 'trainable_variables'):
                params_list.extend(layer.trainable_variables) # this could still be an empty list but that's ok
        return params_list
    
    def regularization_losses(self):
        r_loss = 0
        for layer in self.layer_list:
            if layer != self.noise_layer_function:
                r_loss += tf.math.reduce_sum(layer.losses)
        return r_loss

    def set_up_layer_names(self):
        self.layer_names = []
        layer_index = 0
        noise_index = self.config.index('noise_layer')
        for i in range(len(self.config)):
            if i == len(self.config) - 1:
                self.layer_names.append('logits_unit_'+str(self.config[i]))
                
            elif 'conv' in self.config[i] or \
                    'fc' in self.config[i] or \
                        'avgpool' in self.config[i] or \
                            self.config[i] in ['flatten']:
                layer_index += 1
                if i < noise_index:
                    key = 'mask_'
                else:
                    key = 'non_mask_'
                self.layer_names.append(key+'layer_'+str(layer_index)+'_'+str(self.config[i]))

            elif self.config[i] == 'noise_layer':
                self.layer_names.append('non_mask_layer_'+str(layer_index)+'_'+str(self.config[i-1])) # here i-1 might not be safe against BN added layers
            else:
                assert False

    def leak_auc_dict(self, attack_method='leak_norm'):
        auc_dict = OrderedDict()
        for layer_name in self.layer_names:
            auc_dict[layer_name+'_'+attack_method] = tf.keras.metrics.Mean()
        return auc_dict

class WDL(tf.keras.Model):
    
    def __init__(self, wide_feature_tags, deep_feature_tags, config, noise_layer_function):
        """construct the layers of the model

        Args:
            linear_feature_columns: a list of feature object (SparseFeat, DenseFeat, etc.) objects with fields 
                                    containing information of feature dimension/embedding dimension, name
            deep_feature_columns: this list could be nonintersecting with linear_feature_columns
            config (list): a list of numbers and strings that describes the layers of the model
                           an integer means a fully connected layer's hidden size
                           string noise_mask_layer means a gradient_masking layer
                           the last layer is the output layer and we don't have relu as the activation function
        """
        super(WDL, self).__init__()

        self.config = config
        self.noise_layer_function = noise_layer_function

        self.wide_feature_tags = wide_feature_tags
        self.deep_feature_tags = deep_feature_tags

        ##### embedding construction #####
        self.deep_categorical_embedding_layers = OrderedDict()
        for feat in self.deep_feature_tags:
            if isinstance(feat, SparseFeat):
                self.deep_categorical_embedding_layers[feat.name] = \
                        tf.keras.layers.Embedding(input_dim=feat.vocabulary_size,
                                                output_dim=feat.embedding_dim,
                                                embeddings_initializer=None,
                                                embeddings_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                name='deep_emb_' + feat.embedding_name)
        self.wide_categorical_embedding_layers = OrderedDict()
        for feat in self.wide_feature_tags:
            if isinstance(feat, SparseFeat):
                self.wide_categorical_embedding_layers[feat.name] = \
                        tf.keras.layers.Embedding(input_dim=feat.vocabulary_size,
                                                # output_dim=feat.embedding_dim,
                                                output_dim=1, # the embedding is 1 for the wide part
                                                embeddings_initializer=None,
                                                embeddings_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                name='wide_emb_' + feat.embedding_name)

        ##### layer construction #####
        self.deep_layer_list = []
        for item in config[:-1]:
            if item == 'noise_layer':
                layer = self.noise_layer_function
            elif isinstance(item, int):
                layer = tf.keras.layers.Dense(units=item,
                                            activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer=None,
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None)
            else:
                assert False, 'layer not defined'
            self.deep_layer_list.append(layer)

        # add the last layer
        # the last layer is a linear layer without relu activation
        if isinstance(config[-1], int):
            # the last linear layer of the deep part
            logit_layer = tf.keras.layers.Dense(units=config[-1],
                                                activation='linear',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                bias_constraint=None)
            self.deep_layer_list.append(logit_layer)
            # the linear layer of the wide part
            self.wide_layer = tf.keras.layers.Dense(units=config[-1],
                                                activation='linear',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                bias_constraint=None)
        else:
            assert False, 'last layer not defined'
        self.set_up_layer_names()


    def call(self, raw_input_dict, no_noise=False):
        """ feedforward through the model
            return all the hidden layer activations
        """
        # retrieve the embedding of categorical variables for the deep part
        deep_retrieved_embedding_list = self.embedding_lookup(embedding_layer_dict=self.deep_categorical_embedding_layers,
                                                        raw_input_dict=raw_input_dict,
                                                        feature_columns=self.deep_feature_tags)
        # get the continuous variables for the deep part
        deep_continuous_feature_list = self.get_dense_input(raw_input_dict=raw_input_dict,
                                                           feature_columns=self.deep_feature_tags)
        # form the dnn input by concatenating embeddings and continuous variables
        deep_input = tf.keras.layers.concatenate(inputs=deep_retrieved_embedding_list + deep_continuous_feature_list, axis=-1)

        hidden_activation = deep_input
        deep_activations_by_layer = []
        for layer in self.deep_layer_list:
            if layer == self.noise_layer_function and no_noise:
                hidden_activation = tf.identity(hidden_activation)
            else:
                hidden_activation = layer(hidden_activation)
            deep_activations_by_layer.append(hidden_activation)
        deep_logit = deep_activations_by_layer[-1]

        # retrieve the embedding of categorical variables for the wide part
        wide_retrieved_embedding_list = self.embedding_lookup(embedding_layer_dict=self.wide_categorical_embedding_layers,
                                                raw_input_dict=raw_input_dict,
                                                feature_columns=self.wide_feature_tags)
        # get the continuous variables for the wide part
        wide_continuous_feature_list = self.get_dense_input(raw_input_dict=raw_input_dict,
                                                           feature_columns=self.wide_feature_tags)
        wide_input = tf.keras.layers.concatenate(inputs=wide_retrieved_embedding_list + wide_continuous_feature_list, axis=-1)

        wide_logit = self.wide_layer(wide_input)

        # add deep and wide part for the final logit
        final_logit = wide_logit + deep_logit

        # returned activations are ordered by dnn layers first, then by parameters
        activations_by_layer = deep_activations_by_layer + [wide_logit, final_logit]
        

        return activations_by_layer

    @staticmethod
    def embedding_lookup(embedding_layer_dict, raw_input_dict, feature_columns):
        # retrieved_group_embedding_dict = defaultdict(list)
        retrieved_embedding_list = []
        for fc in feature_columns:
            if isinstance(fc, SparseFeat):
                feature_name = fc.name
                raw_input = raw_input_dict[feature_name]
                embedding_layer = embedding_layer_dict[feature_name]
                retrieved_embedding_list.append(embedding_layer(raw_input))
                # print(retrieved_embedding_list[-1].shape)

        return retrieved_embedding_list
        #     retrieved_group_embedding_dict[fc.group_name].append(embedding_func(raw_input))
        # return retrieved_group_embedding_dict

    @staticmethod
    def get_dense_input(raw_input_dict, feature_columns):
        # get the DenseFeat input tensors contained the feature columns
        dense_input_list = []
        for fc in feature_columns:
            if isinstance(fc, DenseFeat):
                dense_input_list.append(tf.reshape(raw_input_dict[fc.name], shape=(raw_input_dict[fc.name].shape[0], -1)))
        return dense_input_list
    
    def predict(self, raw_input_dict):
        """ feedforward through the model
            return the logits
        """
        # retrieve the embedding of categorical variables for the deep part
        deep_retrieved_embedding_list = self.embedding_lookup(embedding_layer_dict=self.deep_categorical_embedding_layers,
                                                        raw_input_dict=raw_input_dict,
                                                        feature_columns=self.deep_feature_tags)
        # get the continuous variables for the deep part
        deep_continuous_feature_list = self.get_dense_input(raw_input_dict=raw_input_dict,
                                                           feature_columns=self.deep_feature_tags)
        # form the dnn input by concatenating embeddings and continuous variables
        deep_input = tf.keras.layers.concatenate(inputs=deep_retrieved_embedding_list + deep_continuous_feature_list, axis=-1)

        hidden_activation = deep_input
        for layer in self.deep_layer_list:
            hidden_activation = layer(hidden_activation)
        deep_logit = hidden_activation

        # retrieve the embedding of categorical variables for the wide part
        wide_retrieved_embedding_list = self.embedding_lookup(embedding_layer_dict=self.wide_categorical_embedding_layers,
                                                raw_input_dict=raw_input_dict,
                                                feature_columns=self.wide_feature_tags)
        # get the continuous variables for the wide part
        wide_continuous_feature_list = self.get_dense_input(raw_input_dict=raw_input_dict,
                                                           feature_columns=self.wide_feature_tags)
        wide_input = tf.keras.layers.concatenate(inputs=wide_retrieved_embedding_list + wide_continuous_feature_list, axis=-1)

        wide_logit = self.wide_layer(wide_input)

        # add deep and wide part for the final logit
        final_logit = wide_logit + deep_logit

        return final_logit

    @property
    def trainable_variables(self):
        params_list = []
        # add embedding parameters
        for feature_name, embedding in self.deep_categorical_embedding_layers.items():
            if hasattr(embedding, 'trainable_variables'):
                params_list.extend(embedding.trainable_variables)
        for feature_name, embedding in self.wide_categorical_embedding_layers.items():
            if hasattr(embedding, 'trainable_variables'):
                params_list.extend(embedding.trainable_variables)
        # add deep part neural network parameters
        for layer in self.deep_layer_list:
            if hasattr(layer, 'trainable_variables'):
                params_list.extend(layer.trainable_variables) # this could still be an empty list but that's ok
        # add wide part linear parameters
        params_list.extend(self.wide_layer.trainable_variables)

        return params_list
    
    def regularization_losses(self):
        r_loss = 0.0
        for embedding in self.deep_categorical_embedding_layers.values():
            if isinstance(embedding, tf.keras.layers.Layer):
                r_loss += tf.math.reduce_sum(embedding.losses)
        for embedding in self.wide_categorical_embedding_layers.values():
            if isinstance(embedding, tf.keras.layers.Layer):
                r_loss += tf.math.reduce_sum(embedding.losses)
        for layer in self.deep_layer_list:
            if isinstance(layer, tf.keras.layers.Layer):
                r_loss += tf.math.reduce_sum(layer.losses)
        r_loss += tf.math.reduce_sum(self.wide_layer.losses)
        return r_loss

    def set_up_layer_names(self):
        """[Return a list of layer names in the order of the layer activations returned by self.call()]
        """        
        self.layer_names = []
        layer_index = 0
        try:
            noise_index = self.config.index('noise_layer')
        except ValueError:
            noise_index = -1
        for i in range(len(self.config)):
            if i == len(self.config) - 1:
                self.layer_names.append('deep_non_mask_logits_unit_'+str(self.config[i]))
            elif isinstance(self.config[i], int):
                layer_index += 1
                if i < noise_index:
                    key = 'mask'
                else:
                    key = 'non_mask'
                self.layer_names.append('deep_' + key + '_layer_'+str(layer_index)+'_unit_'+str(self.config[i]))
            elif self.config[i] == 'noise_layer':
                self.layer_names.append('deep_non_mask_layer_'+str(layer_index)+'_unit_'+str(self.config[i-1])) # here i-1 might not be safe against BN added layers
            else:
                assert False
        self.layer_names.append('wide_non_mask_leak_auc_layer_1_unit_'+str(self.config[-1]))
        self.layer_names.append('final_non_mask_leak_auc_layer_1_unit_'+str(self.config[-1]))

    def leak_auc_dict(self, attack_method='leak_norm'):
        auc_dict = OrderedDict()
        for layer_name in self.layer_names:
            auc_dict[layer_name+'_'+attack_method] = tf.keras.metrics.Mean()
        return auc_dict
    
    def num_params(self):
        n_params = 0
        # add embedding parameters
        for feature_name, embedding in self.deep_categorical_embedding_layers.items():
            if hasattr(embedding, 'trainable_variables'):
                for var in embedding.trainable_variables:
                    n_params += np.prod(var.shape)
        for feature_name, embedding in self.wide_categorical_embedding_layers.items():
            if hasattr(embedding, 'trainable_variables'):
                for var in embedding.trainable_variables:
                    n_params += np.prod(var.shape)
        # add deep part neural network parameters
        for layer in self.deep_layer_list:
            if hasattr(layer, 'trainable_variables'):
                for var in layer.trainable_variables: # this could still be an empty list but that's ok
                    n_params += np.prod(var.shape)
        
        # add wide part linear parameters
        for var in self.wide_layer.trainable_variables: # this could still be an empty list but that's ok
            n_params += np.prod(var.shape)
        return n_params


def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of
    B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances
    """
    # with tf.variable_scope('pairwise_dist'):
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb,
                           0.0))
    return D

def cosine_similarity(A, B):
    row_normalized_A = tf.math.l2_normalize(A, axis=1)
    row_normalized_B = tf.math.l2_normalize(B, axis=1)
    cosine_matrix = tf.linalg.matmul(
              row_normalized_A,
              row_normalized_B,
              adjoint_b=True) # transpose second matrix)
    # average_abs_cosine = tf.reduce_mean(tf.math.abs(cosine_matrix), axis=1)
    # average_abs_cosine = tf.reduce_mean(cosine_matrix, axis=1)
    return cosine_matrix

def projection_similarity(A, B):
    row_normalized_B = tf.math.l2_normalize(B, axis=1)
    matrix = tf.linalg.matmul(
              A,
              row_normalized_B,
              adjoint_b=True)
    return matrix

def inner_product(A, B):
    return tf.linalg.matmul(A, B, adjoint_b=True)

def update_auc(y, predicted_value, m_auc):
    auc = compute_auc(y, predicted_value)
    if auc:
        m_auc.update_state(auc)
    return auc


def compute_auc(y, predicted_value):
    # get rid of the 2nd dimension in  [n, 1]
    predicted_value = tf.reshape(predicted_value, shape=(-1))
    if tf.reduce_sum(y) == 0: # no positive examples in this batch
        return None
    # m_auc.update_state(0.5) # currently set as 0.5 in some sense this is not well defined
    val_max = tf.math.reduce_max(predicted_value)
    val_min = tf.math.reduce_min(predicted_value)
    pred = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)
    # create this is to avoid putting all different batches of examples in the same epoch together
    # auc_calculator = tf.keras.metrics.AUC()
    # auc_calculator.reset_states()
    # auc_calculator.update_state(y, pred)
    # auc = auc_calculator.result()
    auc = roc_auc_score(y_true=y.numpy(), y_score=pred.numpy())
    # if display and auc > 0.0:
    #     print('Alert')
    #     print(tf.reduce_max(predicted_value[y==1]))
    #     print(tf.reduce_min(predicted_value[y==0]))
    #     assert False
    return auc


def update_all_norm_leak_auc(norm_leak_auc_dict, grad_list, y):
    for (key, grad) in zip(norm_leak_auc_dict.keys(), grad_list):
        # flatten each example's grad to one-dimensional
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))

        if grad.shape[1] == 1: # the last layer's logit
            grad = tf.reshape(grad, shape=[-1])
            auc = update_auc(y=y,
                predicted_value=grad,
                m_auc=norm_leak_auc_dict[key])

        else:
            auc = update_auc(y=y,
                       predicted_value=tf.norm(grad, axis=-1, keepdims=False),
                       m_auc=norm_leak_auc_dict[key])
        # not only update the epoch average above
        # also log this current batch value on the tensorboard
        if auc:
            with shared_var.writer.as_default():
                tf.summary.scalar(name=key+'_batch',
                                  data=auc,
                                  step=shared_var.counter)

def update_all_ip_leak_auc(ip_leak_auc_dict, grad_list, pos_grad_list, y):
    for (key, grad, pos_grad) in zip(ip_leak_auc_dict.keys(), grad_list, pos_grad_list):
        # flatten each example's grad to one-dimensional
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))
        # there should only be one positive example's gradient in pos_grad
        pos_grad = tf.reshape(pos_grad, shape=(pos_grad.shape[0], -1))

        auc = update_auc(
                    y=y,
                    predicted_value=inner_product(grad, pos_grad),
                    m_auc=ip_leak_auc_dict[key])
        # not only update the epoch average above
        # also log this current batch value on the tensorboard
        if auc:
            with shared_var.writer.as_default():
                tf.summary.scalar(name=key+'_batch',
                                  data=auc,
                                  step=shared_var.counter)


def update_all_cosine_leak_auc(cosine_leak_auc_dict, grad_list, pos_grad_list, y):
    for (key, grad, pos_grad) in zip(cosine_leak_auc_dict.keys(), grad_list, pos_grad_list):
        # flatten each example's grad to one-dimensional
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))
        # there should only be one positive example's gradient in pos_grad
        pos_grad = tf.reshape(pos_grad, shape=(pos_grad.shape[0], -1))

        auc = update_auc(
                    y=y,
                    predicted_value=cosine_similarity(grad, pos_grad),
                    m_auc=cosine_leak_auc_dict[key])
        # not only update the epoch average above
        # also log this current batch value on the tensorboard
        if auc:
            with shared_var.writer.as_default():
                tf.summary.scalar(name=key+'_batch',
                                  data=auc,
                                  step=shared_var.counter)


def update_all_hint_norm_attack_leak_auc(hint_attack_auc_dict, grad_list, y, num_hints=1):
    for (key, grad) in zip(hint_attack_auc_dict.keys(), grad_list):
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))

        labels = tf.reshape(y, shape=[tf.shape(grad)[0]])
        pos_grads = tf.boolean_mask(grad, labels)
        pos_grads = tf.random.shuffle(pos_grads)
        # print("g: {}".format(grad))
        # print("labels: {}".format(labels))
        # print("pos_grads: {}".format(pos_grads))
        # print("num_hints: {}".format(tf.minimum(num_hints, tf.shape(pos_grads)[0])))
        selected_grads = tf.slice(pos_grads, [0, 0], [tf.minimum(num_hints, tf.shape(pos_grads)[0]), -1])
        # print("selected_grads: {}".format(selected_grads))
        dist_res = pairwise_dist(grad, selected_grads)
        # print("dist_res_pairwise: {}".format(dist_res))
        # print("dist_res shape: {}".format(tf.shape(dist_res)))
        dist_res = tf.math.reduce_min(dist_res, axis = 1)
        # print("dist_res reduce : {}".format(dist_res))
        dist_res = tf.reshape(dist_res, shape=[tf.shape(grad)[0]])
        dist_res = -1.0 * dist_res
        # print("dist_res: {}".format(dist_res))
        # print("label: {}".format(y))
        auc = update_auc(y=y, predicted_value=dist_res, m_auc=hint_attack_auc_dict[key])
        # print("hint_attack_auc: {}".format(auc))
        # also log this current batch value on the tensorboard
        if auc:
            with shared_var.writer.as_default():
                tf.summary.scalar(name=key+'_batch',
                                  data=auc,
                                  step=shared_var.counter)

def update_all_hint_inner_product_attack_leak_auc(hint_attack_auc_dict, grad_list, y, num_hints=1):
    for (key, grad) in zip(hint_attack_auc_dict.keys(), grad_list):
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))

        labels = tf.reshape(y, shape=[tf.shape(grad)[0]])
        pos_grads = tf.boolean_mask(grad, labels)
        pos_grads = tf.random.shuffle(pos_grads)
        # print("g: {}".format(grad))
        # print("labels: {}".format(labels))
        # print("pos_grads: {}".format(pos_grads))
        # print("num_hints: {}".format(tf.minimum(num_hints, tf.shape(pos_grads)[0])))
        selected_grads = tf.slice(pos_grads, [0, 0], [tf.minimum(num_hints, tf.shape(pos_grads)[0]), -1])
        selected_grads_norm = tf.norm(selected_grads, axis=1, keepdims=False)
        longest_norm_index = tf.math.argmax(selected_grads_norm)
        
        # print("selected_grads: {}".format(selected_grads))
        project_sim = inner_product(grad, selected_grads[longest_norm_index: longest_norm_index + 1])
        # print("dist_res_pairwise: {}".format(dist_res))
        # print("dist_res shape: {}".format(tf.shape(dist_res)))
        # print("dist_res reduce : {}".format(dist_res))
        # print("dist_res: {}".format(dist_res))
        # print("label: {}".format(y))
        auc = update_auc(y=y, predicted_value=project_sim, m_auc=hint_attack_auc_dict[key])
        # print("hint_attack_auc: {}".format(auc))
        # also log this current batch value on the tensorboard
        if auc:
            with shared_var.writer.as_default():
                tf.summary.scalar(name=key+'_batch',
                                  data=auc,
                                  step=shared_var.counter)

# def update_all_hint_inner_product_attack_leak_auc(hint_attack_auc_dict, grad_list, y, num_hints=1):
#     for (key, grad) in zip(hint_attack_auc_dict.keys(), grad_list):
#         # print("g: {}".format(grad))
#         # print("labels: {}".format(labels))
#         # print("pos_grads: {}".format(pos_grads))
#         # print("num_hints: {}".format(tf.minimum(num_hints, tf.shape(pos_grads)[0])))
        
#         # print("selected_grads: {}".format(selected_grads))
#         cos_sim = cosine_similarity(grad, grad)
#         same_direction_counts = \
#             tf.math.reduce_sum(tf.cast(tf.math.greater(
#                                         cos_sim,
#                                         tf.zeros(shape=cos_sim.shape)),
#                                        tf.float32),
#                                axis=1)
#         negative_example_index = tf.math.argmax(same_direction_counts)
#         print(y[negative_example_index], same_direction_counts[negative_example_index], tf.math.reduce_min(same_direction_counts))
#         project_sim = inner_product(grad, grad[negative_example_index:negative_example_index + 1])
#         # print("dist_res_pairwise: {}".format(dist_res))
#         # print("dist_res shape: {}".format(tf.shape(dist_res)))
#         # print("dist_res reduce : {}".format(dist_res))
#         # print("dist_res: {}".format(dist_res))
#         # print("label: {}".format(y))
#         auc = update_auc(y=y, predicted_value=-project_sim, m_auc=hint_attack_auc_dict[key])
#         # print("hint_attack_auc: {}".format(auc))
#         # also log this current batch value on the tensorboard
#         if auc:
#             with shared_var.writer.as_default():
#                 tf.summary.scalar(name=key+'_batch',
#                                   data=auc,
#                                   step=shared_var.counter)

# def update_all_hint_cosine_attack_leak_auc(hint_attack_auc_dict, grad_list, y, num_hints=1):
#     for (key, grad) in zip(hint_attack_auc_dict.keys(), grad_list):
#         labels = tf.reshape(y, shape=[tf.shape(grad)[0]])
#         pos_grads = tf.boolean_mask(grad, labels)
#         pos_grads = tf.random.shuffle(pos_grads)
#         # print("g: {}".format(grad))
#         # print("labels: {}".format(labels))
#         # print("pos_grads: {}".format(pos_grads))
#         # print("num_hints: {}".format(tf.minimum(num_hints, tf.shape(pos_grads)[0])))
#         selected_grads = tf.slice(pos_grads, [0, 0], [tf.minimum(num_hints, tf.shape(pos_grads)[0]), -1])
#         # print("selected_grads: {}".format(selected_grads))
#         avg_abs_consine = cosine_similarity(grad, selected_grads)
#         # print("dist_res_pairwise: {}".format(dist_res))
#         # print("dist_res shape: {}".format(tf.shape(dist_res)))
#         # print("dist_res reduce : {}".format(dist_res))
#         # print("dist_res: {}".format(dist_res))
#         # print("label: {}".format(y))
#         auc = update_auc(y=y, predicted_value=avg_abs_consine, m_auc=hint_attack_auc_dict[key])
#         # print("hint_attack_auc: {}".format(auc))
#         # also log this current batch value on the tensorboard
#         if auc:
#             with shared_var.writer.as_default():
#                 tf.summary.scalar(name=key+'_batch',
#                                   data=auc,
#                                   step=shared_var.counter)

def reset_all_leak_auc(leak_auc_dict):
    for key in leak_auc_dict.keys():
        leak_auc_dict[key].reset_states()

def print_all_leak_auc(leak_auc_dict):
    for key in leak_auc_dict.keys():
        print('{:<40s}{:>10f}'.format(key, leak_auc_dict[key].result().numpy()))

def tf_summary_all_leak_auc(leak_auc_dict, step):
    for key in leak_auc_dict.keys():
        tf.summary.scalar(key, leak_auc_dict[key].result(), step=step)
