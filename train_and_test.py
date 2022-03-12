from collections import defaultdict, OrderedDict
import tensorflow as tf
import datetime
import os
import numpy as np
import time

from model import (
    update_all_norm_leak_auc,
    update_all_ip_leak_auc,
    update_all_cosine_leak_auc,
    update_all_hint_norm_attack_leak_auc,
    update_all_hint_inner_product_attack_leak_auc,
    print_all_leak_auc,
    reset_all_leak_auc,
    tf_summary_all_leak_auc
)

from utils import compute_gradient_norm, compute_sampled_inner_product, compute_sampled_cosine
import shared_var


def train(model, train_set, test_set, loss_function, num_epochs, writer,
          trainer=None, regularization_weight=0.1, period=None, num_hints=None):
    """[summary]

    Args:
        model ([type]): [description]
        train_set ([type]): [description]
        test_set ([type]): [description]
        loss_function ([type]): [description]
        num_epochs ([type]): [description]
        writer ([type]): [description]
        trainer ([type], optional): [description]. Defaults to None.
        regularization_weight (float, optional): [description]. Defaults to 0.1.
        period ([type], optional): [description]. Defaults to None.
        num_hints (list, optional): specifies how many hints to use for the hint attack.
                            Defaults to None which means no evaluation using hint

    Returns:
        [type]: [description]
    """
    best_test_auc = 0
    best_epoch = 0

    train_loss = tf.keras.metrics.Mean()
    train_accu = tf.keras.metrics.BinaryAccuracy()
    train_auc = tf.keras.metrics.AUC()

    # p_norm = tf.keras.metrics.Mean()
    # n_norm = tf.keras.metrics.Mean()
    norm_leak_auc_dict = model.leak_auc_dict(attack_method='norm_leak')
    ip_leak_auc_dict = model.leak_auc_dict(attack_method='ip_leak')
    cosine_leak_auc_dict = model.leak_auc_dict(attack_method='cosine_leak')
    if num_hints is not None:
        hint_attack_norm_leak_auc_dicts = {}
        hint_attack_ip_leak_auc_dicts = {}
        for n_hint in num_hints:
            hint_attack_norm_leak_auc_dicts[n_hint] = model.leak_auc_dict(attack_method=f"{n_hint}hint_norm")
            hint_attack_ip_leak_auc_dicts[n_hint] = model.leak_auc_dict(attack_method=f"{n_hint}hint_inner_product")

    # global_batch_idx = 0

    # @tf.function
    def train_step(model, X, y):
        """[the feedforward and backprop for one training update
            tf.function constructs the computation graph and avoids memory leaks?
            defined inside avoids passing trainer as an argument]

        Args:
            model (tf.keras.Model): the model to feedforward and backprop
            X ([type]): input
            y ([type]): target output

        Returns:
            loss, logits, activation_grad_list
        """    
        with tf.GradientTape(persistent=False) as tape:
            # start = time.time()
            activations_by_layer = model(X, no_noise=False)
            # model.save_weights('model.ckpt')
            # print(model.num_params())
            # print(time.time() - start)

            logits = activations_by_layer[-1]
            loss = tf.math.reduce_mean(loss_function(y_hat=logits, y=y)) + regularization_weight * model.regularization_losses()
        
        params = model.trainable_variables
        grad_list = tape.gradient(loss, params + activations_by_layer)

        # apply grad to only the parameters but not the activations
        parameter_grad_list = grad_list[:len(params)]
        activation_grad_list = grad_list[len(params):]
        # print('logit gradient')
        # print(activation_grad_list[-1])
        # print('pos_prob')
        # print(tf.math.sigmoid(logits))
        trainer.apply_gradients(zip(parameter_grad_list, params))

        return loss, logits, activation_grad_list

    for epoch in range(num_epochs):
        print("epoch {}:".format(epoch))

        # gradients_over_training_set = defaultdict(list)
        # labels_over_training_set = []

        e_s = datetime.datetime.now()

        for (batch_idx, (X, y)) in enumerate(train_set, 1):

            # print('number of positive examples', tf.math.reduce_sum(y))
            if tf.math.reduce_sum(y).numpy() == 0:
                # if the batch has no positive examples, continue
                continue


            # store the batch label in shared_var for the custom gradient noise_layer_function to access
            shared_var.batch_y = y
            # global_batch_idx += 1
            b_s = datetime.datetime.now()

            ###########################################################
            ######## preparation for direction attack begins ##########
            ###########################################################
            # get a positive example now that there is at least one positive example in the batch
            # do forward and backward on this single variable, store the grad_activation_list in the shared_var
            pos_idx = np.random.choice(np.where(y.numpy() == 1)[0], size=1)[0]
            with tf.GradientTape(persistent=False) as tape:
                if isinstance(X, OrderedDict):
                    pos_X = {
                        key: value[pos_idx:pos_idx+1] for key, value in X.items()
                    }
                elif isinstance(X, tf.Tensor):
                    pos_X = X[pos_idx: pos_idx+1]
                else:
                    assert False, 'unsupported X type'

                pos_activations_by_layer = model(pos_X, no_noise=True)
                pos_logits = pos_activations_by_layer[-1]
                pos_loss = tf.math.reduce_mean(loss_function(y_hat=pos_logits, y=y[pos_idx:pos_idx+1])) + regularization_weight * model.regularization_losses()
            pos_activation_grad_list = tape.gradient(pos_loss, pos_activations_by_layer)

            ###########################################################
            ######## preparation for direction attack ends ############
            ###########################################################


            # start = time.time()
            loss, logits, activation_grad_list = train_step(model, X, y)

            # save these for generating the plots
            # if shared_var.counter < 2000:
            #     for layer_name, grad in zip(model.layer_names, activation_grad_list):
            #         np.save(file=os.path.join(shared_var.logdir, layer_name + '_' + 'itr' + str(shared_var.counter) + '.npy'),
            #                     arr=grad.numpy())
            #     np.save(file=os.path.join(shared_var.logdir, 'y' + '_' + 'itr' + str(shared_var.counter) + '.npy'),
            #             arr=y.numpy())

            # print(time.time() - start)

            # update training statistics
            start = time.time()
            train_loss.update_state(loss.numpy())
            train_accu.update_state(y_true=y, 
                                    y_pred=tf.math.sigmoid(logits))
            train_auc.update_state(tf.reshape(y, [-1, 1]), tf.reshape(tf.math.sigmoid(logits), [-1, 1]))

            layer_idx = model.config.index('noise_layer')

            # p_norm.update_state(tf.norm(activation_grad_list[layer_idx][y==1], axis=1))
            # n_norm.update_state(tf.norm(activation_grad_list[layer_idx][y==0], axis=1))
            update_all_norm_leak_auc(
                norm_leak_auc_dict=norm_leak_auc_dict,
                grad_list=activation_grad_list,
                y=y)
            update_all_ip_leak_auc(
                ip_leak_auc_dict=ip_leak_auc_dict,
                grad_list=activation_grad_list,
                pos_grad_list=pos_activation_grad_list,
                y=y)
            update_all_cosine_leak_auc(
                cosine_leak_auc_dict=cosine_leak_auc_dict,
                grad_list=activation_grad_list,
                pos_grad_list=pos_activation_grad_list,
                y=y)

            if num_hints is not None:
                for n_hint in num_hints:
                    update_all_hint_norm_attack_leak_auc(
                        hint_attack_norm_leak_auc_dicts[n_hint],
                        activation_grad_list,
                        y,
                        num_hints=int(n_hint))
                    update_all_hint_inner_product_attack_leak_auc(
                        hint_attack_ip_leak_auc_dicts[n_hint],
                        activation_grad_list,
                        y,
                        num_hints=int(n_hint))


            # records the gradient for each layer over this batch
            # for layer_name, layer_grad in zip(model.layer_names, activation_grad_list):
            #     gradients_over_training_set[layer_name].append(layer_grad)
            # record the labels
            # labels_over_training_set.append(y)

            with writer.as_default():
                tf.summary.scalar(name='p_norm_mean',
                                  data=tf.reduce_mean(tf.norm(activation_grad_list[layer_idx][y==1], axis=1, keepdims=False)),
                                  step=shared_var.counter)
                                #   step=global_batch_idx)
                tf.summary.scalar(name='n_norm_mean',
                                  data=tf.reduce_mean(tf.norm(activation_grad_list[layer_idx][y==0], axis=1, keepdims=False)),
                                  step=shared_var.counter)
                                #   step=global_batch_idx)
            # print('logging', time.time() - start)
            

            # clear out memory manually
            # del params # is this correct?
            # del loss
            # del activations_by_layer
            # del grad_list

            '''
            if epoch == num_epochs - 1:
                # checking the predicted probabilities after the last epoch's update
                batch_predicted_probability_positive_class = tf.math.sigmoid(model.predict(X))

                print("pos example pos prob: min: {:.4f}, mean: {:.4f}, max: {:.4f}".format(tf.reduce_min(batch_predicted_probability_positive_class[y==1]), 
                                                                                            tf.reduce_mean(batch_predicted_probability_positive_class[y==1]),
                                                                                            tf.reduce_max(batch_predicted_probability_positive_class[y==1])))
                print("neg example pos prob: min: {:.4f}, mean: {:.4f}, max: {:.4f}".format(tf.reduce_min(batch_predicted_probability_positive_class[y==0]),
                                                                                           tf.reduce_mean(batch_predicted_probability_positive_class[y==0]),
                                                                                           tf.reduce_max(batch_predicted_probability_positive_class[y==0])))
                print()
            '''

            b_e = datetime.datetime.now()
        
            if tf.data.experimental.cardinality(train_set).numpy() == -2:
                # for make_csv_dataset or datasets that have used filter in general
                # where the total number of examples is unknown
                predicate = (batch_idx == 1) or (batch_idx % period == 0)
            else:
                # for standard dataset
                predicate = batch_idx == len(train_set)

            # print(tf.data.experimental.cardinality(train_set).numpy())
            # print('predicate', predicate)
            if predicate:
                # log statistics in terminal
            
                e_e = datetime.datetime.now()
                print("train loss: {:.4f}\ntrain accu: {:.4f}\ntrain auc: {:.4f}\ntime used: {}s\n".format(train_loss.result(), train_accu.result(), train_auc.result(), e_e - e_s))
                # print("pos norm: {:.4f}\nneg norm {:.4f}".format(p_norm.result(), n_norm.result()))
                print_all_leak_auc(leak_auc_dict=norm_leak_auc_dict)
                print_all_leak_auc(leak_auc_dict=ip_leak_auc_dict)
                print_all_leak_auc(leak_auc_dict=cosine_leak_auc_dict)
                if num_hints is not None:
                    for n_hint in num_hints:
                        print_all_leak_auc(leak_auc_dict=hint_attack_norm_leak_auc_dicts[n_hint])
                        print_all_leak_auc(leak_auc_dict=hint_attack_ip_leak_auc_dicts[n_hint])

                # records the gradient for each layer over this batch
                # for layer_name, layer_grad in zip(model.layer_names, activation_grad_list):
                #     gradients_over_training_set[layer_name].append(layer_grad)
                # record the labels
                # labels_over_training_set.append(y)

                # concatenate batchs of gradients into one matrix for every layer
                # for layer_name in model.layer_names:
                #     gradients_over_training_set[layer_name] = tf.concat(gradients_over_training_set[layer_name], axis=0)
                # one array for the labels of the entire training set
                # labels_over_training_set = tf.concat(labels_over_training_set, axis=0)

                gradient_norm_by_layer = {}
                gradient_inner_product_by_layer = {}
                gradient_cosine_by_layer = {}

                for layer_name, layer_grad in zip(model.layer_names, activation_grad_list):
                    gradient_norm_by_layer[layer_name] = compute_gradient_norm(layer_grad, y)
                    gradient_inner_product_by_layer[layer_name] = compute_sampled_inner_product(layer_grad, y, sample_ratio=0.02)
                    gradient_cosine_by_layer[layer_name] = compute_sampled_cosine(layer_grad, y, sample_ratio=0.02)

                test_loss, test_accu, test_auc = test(test_set=test_set,
                                                    model=model, 
                                                    loss_function=loss_function,
                                                    regularization_weight=regularization_weight)

                # log statisitcs on tensorboard
                with writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=shared_var.counter)
                    tf.summary.scalar('train_auc',  train_auc.result(), step=shared_var.counter)
                    tf.summary.scalar('train_accu', train_accu.result(), step=shared_var.counter)
                    tf_summary_all_leak_auc(norm_leak_auc_dict, step=shared_var.counter)
                    tf_summary_all_leak_auc(ip_leak_auc_dict, step=shared_var.counter)
                    tf_summary_all_leak_auc(cosine_leak_auc_dict, step=shared_var.counter)
                    if num_hints is not None:
                        for n_hint in num_hints:
                            tf_summary_all_leak_auc(hint_attack_norm_leak_auc_dicts[n_hint], step=shared_var.counter)
                            tf_summary_all_leak_auc(hint_attack_ip_leak_auc_dicts[n_hint], step=shared_var.counter)

                    tf.summary.scalar('test_loss', test_loss, step=shared_var.counter)
                    tf.summary.scalar('test_accu', test_accu, step=shared_var.counter)
                    tf.summary.scalar('test_auc', test_auc, step=shared_var.counter)

                    for layer_name, info in gradient_norm_by_layer.items():
                        for name, item in info.items():
                            tf.summary.histogram(layer_name+'_norm/'+name, item, step=shared_var.counter)

                        '''
                        if shared_var.counter < 5000:
                            for name, item in info.items():
                                if name in ['pos_grad_norm', 'neg_grad_norm']:
                                    np.save(file=os.path.join(shared_var.logdir, layer_name + '_' + name + '_' + 'batch' + str(batch_idx)),
                                            arr=item.numpy())
                        '''


                    for layer_name, info in gradient_inner_product_by_layer.items():
                        for name, item in info.items():
                            tf.summary.histogram(layer_name+'_inner_product/'+name, item, step=shared_var.counter)
                    for layer_name, info in gradient_cosine_by_layer.items():
                        if 'logits' not in layer_name:
                            for name, item in info.items():
                                tf.summary.histogram(layer_name+'_cosine/'+name, item, step=shared_var.counter)

                if test_auc > best_test_auc:
                    best_test_auc = max(test_auc, best_test_auc)
                    best_epoch = epoch, batch_idx, shared_var.counter
                    print("current best test auc: {:.4f}".format(best_test_auc))
                    print("current best model: {:d}".format(shared_var.counter))
                
                ################################################
                ############ reset all the statistics ##########
                ################################################
                train_loss.reset_states()
                train_accu.reset_states()
                train_auc.reset_states()

                # p_norm.reset_states()
                # n_norm.reset_states()
                reset_all_leak_auc(norm_leak_auc_dict)
                reset_all_leak_auc(ip_leak_auc_dict)
                reset_all_leak_auc(cosine_leak_auc_dict)
                if num_hints is not None:
                    for n_hint in num_hints:
                        reset_all_leak_auc(hint_attack_norm_leak_auc_dicts[n_hint])
                        reset_all_leak_auc(hint_attack_ip_leak_auc_dicts[n_hint])

                e_s = datetime.datetime.now()

            shared_var.counter += 1 # count how many batches (iterations) done so far. 
        print()

    print("best test auc: {:.4f}".format(best_test_auc))
    print("best epoch: ", best_epoch)


def test(test_set, model, loss_function, regularization_weight):
    test_loss = tf.keras.metrics.Mean()
    test_accu = tf.keras.metrics.BinaryAccuracy()
    test_auc = tf.keras.metrics.AUC()

    start = time.time()
    for (idx, (X, y)) in enumerate(test_set):
        logits = model.predict(X)
        loss = tf.math.reduce_mean(loss_function(y_hat=logits, y=y)) + regularization_weight * model.regularization_losses()
        test_loss.update_state(loss.numpy())
        y = tf.cast(y, dtype=tf.float32)
        test_accu.update_state(y_true=tf.reshape(y, [-1, 1]), 
                               y_pred=tf.math.sigmoid(logits))
        test_auc.update_state(tf.reshape(y, [-1,1]), tf.math.sigmoid(logits))
    end = time.time()
    print('test takes {}s'.format(end - start))

    print("test loss: {:4f}\ntest accu: {:4f}\ntest auc: {:4f}".format(test_loss.result(), test_accu.result(), test_auc.result()))

    return test_loss.result(), test_accu.result(), test_auc.result()