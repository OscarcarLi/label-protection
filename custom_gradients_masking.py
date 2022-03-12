import math
import numpy as np
import tensorflow as tf

import time

import shared_var
from solver import solve_isotropic_covariance, symKL_objective
import os

@tf.custom_gradient
def no_noise(x):
    def grad_fn(g):
        return g
    return x, grad_fn

def gradient_perp_masking_function_creator(lower=1.0, upper=10.0):
    """
    return a gradient_perp_masking layer function that aligns the 
    each example's gradient norm to a Uniform distribution [max_norm * lower, max_norm * upper]
    Args:
        lower (float, optional): lower ratio relative to max norm. Defaults to 1.0.
        upper (float, optional): upper ratio relative to max norm. Defaults to 10.0.

    Returns:
        function: gradient_perp_masking to be used for noise layer
    """    
    @tf.custom_gradient
    def gradient_perp_masking(x):
        # add scalar noise with aligning the maximum norm
        def grad_fn(g):
            # if np.all(np.isnan(g)):
            #     assert False
            g_original_shape = g.shape
            g = tf.reshape(g, shape=(g_original_shape[0], -1))

            g_norm = tf.norm(g, axis=1, keepdims=True)
            tf.debugging.check_numerics(g_norm, 'g_norm', name=None)
            max_norm = tf.reduce_max(g_norm)
            std_gaussian_noise = tf.random.normal(shape=g.shape, mean=0.0, stddev=1.0)
            inner_product = tf.reduce_sum(tf.math.multiply(std_gaussian_noise, g), axis=1, keepdims=True)
            init_perp = std_gaussian_noise - tf.math.divide(inner_product, tf.math.square(g_norm) + 1e-16) * g
            tf.debugging.check_numerics(init_perp, 'init_perp', name=None)
            unit_perp = tf.math.l2_normalize(init_perp, axis=1)
            tf.debugging.check_numerics(unit_perp, 'unit_perp', name=None)
            norm_to_align = tf.random.uniform(shape=g_norm.shape,
                                            minval=lower * max_norm, maxval=upper * max_norm,
                                            dtype=tf.dtypes.float32)
            perp = tf.math.sqrt(tf.math.square(norm_to_align) - tf.math.square(g_norm) + 1e-8) * unit_perp
            tf.debugging.check_numerics(perp, 'perp', name=None)

            # if shared_var.counter < 2000:
            #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_unperturbed',
            #             arr=g.numpy())
            #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_perturbed',
            #             arr=(g + perp).numpy())
            #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_label',
            #             arr=shared_var.batch_y.numpy())

            return tf.reshape(g + perp, shape=g_original_shape)

        return x, grad_fn
    
    return gradient_perp_masking


def KL_gradient_perturb_function_creator(p_frac='pos_frac', dynamic=False, error_prob_lower_bound=None,
                                         sumKL_threshold=None, init_scale=1.0, uv_choice='uv'):
    """construct a noise perturbation layer that uses sumKL to determine the type of perturbation

    Args:
        p_frac (str, optional): The weight for the trace of the covariance of the
                                positive example's gradient noise. Defaults to 'pos_frac'.
                                The default value will be computed from the fraction of
                                positive examples in the batch of gradients.
        dynamic (bool, optional): whether to adjust the power constraint hyperparameter P
                                to satisfy the error_prob_lower_bound or sumKL_threshold.
                                Defaults to False.
        error_prob_lower_bound (float, optional): The lower bound of the passive party's 
                                detection error (the average of FPR and FNR). Given this
                                value, we will convert this into a corresponding sumKL_threshold,
                                where if the sumKL of the solution is lower than sumKL_threshold,
                                then the passive party detection error will be lower bounded
                                by error_prob_lower_bound.
                                Defaults to None.
        sumKL_threshold ([type], optional): Give a perturbation that have the sum of
                                KL divergences between the positive perturbed distribution
                                and the negative perturbed distribution upper bounded by 
                                sumKL_threshold. Defaults to None.
        init_scale (float, optional): Determines the first value of the power constraint P.
                                P = init_scale * g_diff_norm**2. If dynamic, then this
                                init_scale could be increased geometrically if the solution 
                                does not satisfy the requirement. Defaults to 1.0.
        uv_choice (str, optional): A string of three choices.
                                'uv' model the distribution of positive gradient and negative
                                gradient with individual isotropic gaussian distribution.
                                'same': average the computed empirical isotropic gaussian distribution.
                                'zero': assume positive gradient distribution and negative 
                                gradient distribution to be dirac distributions. Defaults to 'uv'.

    Returns:
        [type]: [description]
    """

    print('p_frac', p_frac)
    print('dynamic', dynamic)
    if dynamic and (error_prob_lower_bound is not None):
        '''
        if using dynamic and error_prob_lower_bound is specified, we use it to 
        determine the sumKL_threshold and overwrite what is stored in it before.
        '''
        sumKL_threshold = (2 - 4 * error_prob_lower_bound)**2
        print('error_prob_lower_bound', error_prob_lower_bound)
        print('implied sumKL_threshold', sumKL_threshold)
    elif dynamic:
        print('using sumKL_threshold', sumKL_threshold)
    
    print('init_scale', init_scale)
    print('uv_choice', uv_choice)

    @tf.custom_gradient
    def KL_gradient_perturb(x):
        # scale = 5.0
        def grad_fn(g):
            # the batch label was stored in shared_var.batch_y in train_and_test
            # print('start')
            # start = time.time()
            g_original_shape = g.shape
            g = tf.reshape(g, shape=(g_original_shape[0], -1))

            y = shared_var.batch_y
            pos_g = g[y==1]
            pos_g_mean = tf.math.reduce_mean(pos_g, axis=0, keepdims=True) # shape [1, d]
            pos_coordinate_var = tf.reduce_mean(tf.math.square(pos_g - pos_g_mean), axis=0) # use broadcast
            neg_g = g[y==0]
            neg_g_mean = tf.math.reduce_mean(neg_g, axis=0, keepdims=True) # shape [1, d]
            neg_coordinate_var = tf.reduce_mean(tf.math.square(neg_g - neg_g_mean), axis=0)

            avg_pos_coordinate_var = tf.reduce_mean(pos_coordinate_var)
            avg_neg_coordinate_var = tf.reduce_mean(neg_coordinate_var)
            # print('pos', avg_pos_coordinate_var)
            # print('neg', avg_neg_coordinate_var)

            g_diff = pos_g_mean - neg_g_mean
            g_diff_norm = float(tf.norm(tensor=g_diff).numpy())
            # if g_diff_norm ** 2 > 1:
            #     print('pos_g_mean', pos_g_mean.shape)
            #     print('neg_g_mean', neg_g_mean.shape)
            #     assert g_diff_norm

            if uv_choice == 'uv':
                u = float(avg_neg_coordinate_var)
                v = float(avg_pos_coordinate_var)
                if u == 0.0:
                    print('neg_g')
                    print(neg_g)
                if v == 0.0:
                    print('pos_g')
                    print(pos_g)

            elif uv_choice == 'same':
                u = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
                v = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
            elif uv_choice == 'zero':
                u, v = 0.0, 0.0

            d = float(g.shape[1])

            if p_frac == 'pos_frac':
                p = float(tf.reduce_sum(y) / len(y)) # p is set as the fraction of positive in the batch
            else:
                p = float(p_frac)

            # print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g_diff_norm**2,p,P))


            scale = init_scale

            # print('compute problem instance', time.time() - start)
            # start = time.time()

            lam10, lam20, lam11, lam21 = None, None, None, None
            while True:
                P = scale * g_diff_norm**2
                # print('g_diff_norm ** 2', g_diff_norm ** 2)
                # print('P', P)
                # print('u, v, d, p', u, v, d, p)
                lam10, lam20, lam11, lam21, sumKL = \
                    solve_isotropic_covariance(
                        u=u,
                        v=v,
                        d=d,
                        g=g_diff_norm ** 2,
                        p=p,
                        P=P,
                        lam10_init=lam10,
                        lam20_init=lam20,
                        lam11_init=lam11,
                        lam21_init=lam21)
                # print('sumKL', sumKL)
                # print()

                # print(scale)
                if not dynamic or sumKL <= sumKL_threshold:
                    break

                scale *= 1.5 # loosen the power constraint
            
            # print('solving time', time.time() - start)
            # start = time.time()

            with shared_var.writer.as_default():
                tf.summary.scalar(name='solver/u',
                                data=u,
                                step=shared_var.counter)
                tf.summary.scalar(name='solver/v',
                                data=v,
                                step=shared_var.counter)
                tf.summary.scalar(name='solver/g',
                                data=g_diff_norm ** 2,
                                step=shared_var.counter)
                tf.summary.scalar(name='solver/p',
                                data=p,
                                step=shared_var.counter)
                tf.summary.scalar(name='solver/scale',
                                  data=scale,
                                  step=shared_var.counter)
                tf.summary.scalar(name='solver/P',
                                data=P,
                                step=shared_var.counter)
                tf.summary.scalar(name='solver/lam10',
                                  data=lam10,
                                  step=shared_var.counter)
                tf.summary.scalar(name='solver/lam20',
                                  data=lam20,
                                  step=shared_var.counter)
                tf.summary.scalar(name='solver/lam11',
                                  data=lam11,
                                  step=shared_var.counter)
                tf.summary.scalar(name='solver/lam21',
                                  data=lam21,
                                  step=shared_var.counter)
                # tf.summary.scalar(name='sumKL_before',
                #                 data=symKL_objective(lam10=0.0,lam20=0.0,lam11=0.0,lam21=0.0,
                #                                     u=u, v=v, d=d, g=g_diff_norm**2),
                #                 step=shared_var.counter)
                # even if we didn't use avg_neg_coordinate_var for u and avg_pos_coordinate_var for v, we use it to evaluate the sumKL_before
                tf.summary.scalar(name='sumKL_before',
                                data=symKL_objective(lam10=0.0,lam20=0.0,lam11=0.0,lam21=0.0,
                                                    u=float(avg_neg_coordinate_var),
                                                    v=float(avg_pos_coordinate_var),
                                                    d=d, g=g_diff_norm**2),
                                step=shared_var.counter)
                tf.summary.scalar(name='sumKL_after',
                                data=sumKL,
                                step=shared_var.counter)
                tf.summary.scalar(name='error prob lower bound',
                                data=0.5 - math.sqrt(sumKL) / 4,
                                step=shared_var.counter)
            

            # print('tb logging', time.time() - start)
            # start = time.time()

            perturbed_g = g
            y_float = tf.cast(y, dtype=tf.float32)

            # positive examples add noise in g1 - g0
            perturbed_g += tf.reshape(tf.multiply(x=tf.random.normal(shape=y.shape),
                                    y=y_float), shape=(-1, 1)) * g_diff * (math.sqrt(lam11-lam21)/g_diff_norm)

            # add spherical noise to positive examples
            if lam21 > 0.0:
                perturbed_g += tf.random.normal(shape=g.shape) * tf.reshape(y_float, shape=(-1, 1)) * math.sqrt(lam21)

            # negative examples add noise in g1 - g0
            perturbed_g += tf.reshape(tf.multiply(x=tf.random.normal(shape=y.shape),
                                    y=1-y_float), shape=(-1, 1)) * g_diff * (math.sqrt(lam10-lam20)/g_diff_norm)

            # add spherical noise to negative examples
            if lam20 > 0.0:
                perturbed_g += tf.random.normal(shape=g.shape) * tf.reshape(1-y_float, shape=(-1, 1)) * math.sqrt(lam20)

            # print('noise adding', time.time() - start)

            # print('a')
            # print(perturbed_g)
            # print('b')
            # print(perturbed_g[y==1])
            # print('c')
            # print(perturbed_g[y==0])

            '''
            pos_cov = tf.linalg.matmul(a=g[y==1] - pos_g_mean, b=g[y==1] - pos_g_mean, transpose_a=True) / g[y==1].shape[0]
            print('pos_var', pos_coordinate_var)
            print('pos_cov', pos_cov)
            print('raw svd', tf.linalg.svd(pos_cov, compute_uv=False))
            print('diff svd', tf.linalg.svd(pos_cov - tf.linalg.tensor_diag(pos_coordinate_var), compute_uv=False))
            # assert False
            '''
            # if shared_var.counter < 2000:
            #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_unperturbed',
            #             arr=g.numpy())
            #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_perturbed',
            #             arr=perturbed_g.numpy())
            #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_label',
            #             arr=shared_var.batch_y.numpy())

            return tf.reshape(perturbed_g, shape=g_original_shape)

        return x, grad_fn

    return KL_gradient_perturb


def gradient_gaussian_noise_masking_function_creator(ratio=1.0):
    @tf.custom_gradient
    def gradient_gaussian_noise_masking(x):
        # add scalar noise with aligning the maximum norm
        def grad_fn(g):
            # print(np.linalg.svd(g.numpy(), compute_uv=False))
            g_original_shape = g.shape
            g = tf.reshape(g, shape=(g_original_shape[0], -1))

            g_norm = tf.norm(g, axis=1, keepdims=False)
            max_norm = tf.reduce_max(g_norm)
            gaussian_noise = tf.random.normal(shape=g.shape, mean=0.0,
                                              stddev=ratio * max_norm / tf.math.sqrt(tf.cast(g.shape[1], dtype=tf.float32)))

            return tf.reshape(g + gaussian_noise, shape=g_original_shape)
        return x, grad_fn

    return gradient_gaussian_noise_masking


@tf.custom_gradient
def gradient_masking(x):
    # add scalar noise with aligning the maximum norm
    def grad_fn(g):
        g_original_shape = g.shape
        g = tf.reshape(g, shape=(g_original_shape[0], -1))
        
        g_norm = tf.reshape(tf.norm(g, axis=1, keepdims=True), [-1, 1])
        max_norm = tf.reduce_max(g_norm)
        stds = tf.sqrt(tf.maximum(max_norm ** 2 / (g_norm ** 2 + 1e-32) - 1.0, 0.0))
        standard_gaussian_noise = tf.random.normal(shape = (tf.shape(g)[0], 1), mean=0.0, stddev=1.0)
        gaussian_noise = standard_gaussian_noise * stds
        res = g * (1 + gaussian_noise)
        
        return tf.reshape(res, shape=g_original_shape)
    return x, grad_fn

