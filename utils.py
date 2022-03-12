from operator import le
import numpy as np
import tensorflow as tf

def learning_rate_string(lr_schedule, lr_drop_step):
    assert len(lr_schedule) == len(lr_drop_step) + 1
    result = []
    if len(lr_drop_step) == 0:
        return str(lr_schedule[0])
    for i in range(len(lr_schedule) - 1):
        result.append(str(lr_schedule[i]))
        result.append('^')
        result.append(str(lr_drop_step[i]))
        result.append('w')
    result.append(str(lr_schedule[-1]))
    return ''.join(result)


def softmax(logits, axis=-1):
    return tf.nn.softmax(logits, axis=axis)

def cross_entropy(y_hat, y):
    # return tf.nn.softmax_cross_entropy_with_logits(y, y_hat)
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8)

def sigmoid_cross_entropy(y_hat, y):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat)
    return loss

def boolean_to_keep(y, ratio=10):
    # True/False array of which examples to keep
    # only keep 1/ratio of the examples of class 1
    num_to_keep = (y[y == 1]).shape[0] // ratio
    indices_to_throw = [i for i in range(y.shape[0]) if y[i] == 1][num_to_keep:]
    print('remove', len(indices_to_throw), 'examples')
    boolean_to_keep = np.array([True] * y.shape[0])
    boolean_to_keep[indices_to_throw] = False
    return boolean_to_keep

def change_label(y):
    def condition(x):
        return 1 if x==1 else 0
    condition_vec = np.vectorize(condition)
    res = condition_vec(y)
    print("positive ratio: {}".format(sum(res) / len(res)))
    return res

def change_label_half_and_half(y):
    def condition2(x):
        return 0 if x in [0,1,2,3,4] else 1
    condition_vec = np.vectorize(condition2)
    res = condition_vec(y)
    print("positive ratio: {}".format(sum(res) / len(res)))
    return res


def random_change_label(y, ratio=0.1):
    def condition(x):
        return 1 if x==1 and np.random.rand(1) <= ratio else 0
    condition_vec = np.vectorize(condition)
    res = condition_vec(y)
    print("positive ratio: {}".format(sum(res) / len(res)))
    return res

def random_change_label_half_and_half(y, ratio=0.1):
    def condition(x):
        return 1 if x in list(range(5)) and np.random.rand(1) <= ratio else 0
    condition_vec = np.vectorize(condition)
    res = condition_vec(y)
    print("positive ratio: {}".format(sum(res) / len(res)))
    return res


def compute_gradient_norm(gradient, label):
    gradient = tf.reshape(gradient, shape=(gradient.shape[0], -1))

    g_norm = tf.norm(gradient, axis=1, keepdims=False)
    pos_g_norm = g_norm[label==1]
    neg_g_norm = g_norm[label==0]
    # print("g_norm: {}".format(g_norm))
    # print("pos_norm: {}".format(pos_g_norm))
    # print("neg_norm: {}".format(neg_g_norm))
    # print(g_norm.shape)
    # print(pos_g_norm.shape)
    # print(neg_g_norm.shape)
    return {'grad_norm':g_norm,
            'pos_grad_norm': pos_g_norm,
            'neg_grad_norm': neg_g_norm}


def lower_triangle_entries(matrix, k=-1):
    """return an array of entires of matrix in the lower half of the matrix

    Args:
        matrix ([numpy array]): [square matrix]
        k (int, optional): [the offset from the diagonal]. Defaults to -1.
                           k=-1 means not including the diagonal.
                           k=0 means including the diagonal.

    Returns:
        numpy array: entries of the lower half of matrix
    """    
    # assert matrix.ndim == 2, matrix.ndim
    # assert matrix.shape[0] == matrix.shape[1]
    return matrix[np.tril_indices(n=matrix.shape[0], k=k)]


def compute_sampled_inner_product(gradient, label, sample_ratio=0.1, divide_by=1e-4):
    gradient = tf.reshape(gradient, shape=(gradient.shape[0], -1))

    # select only a subset to compute inner product
    size = int(1e3)
    if gradient.shape[0] * sample_ratio < size:
        chosen_mask = np.random.rand(gradient.shape[0]) < sample_ratio
    else:
        chosen_mask = np.random.choice(a=gradient.shape[0], size=size, replace=False)
    gradient = gradient.numpy()[chosen_mask]
    label = label.numpy()[chosen_mask]
    inner_product = np.matmul(gradient, gradient.T)
    complete_inner_product = lower_triangle_entries(inner_product)
    pos_inner_product = inner_product[label==1,:][:,label==1]
    pos_inner_product = lower_triangle_entries(pos_inner_product)
    pos_neg_inner_product = np.reshape(inner_product[label==1,:][:,label==0], [-1])
    neg_inner_product = inner_product[label==0,:][:,label==0]
    neg_inner_product = lower_triangle_entries(neg_inner_product)

    return {'every_inner_product': complete_inner_product / divide_by,
            'pos_inner_product': pos_inner_product / divide_by,
            'pos_neg_inner_product': pos_neg_inner_product / divide_by,'neg_inner_product': neg_inner_product / divide_by}


def compute_sampled_cosine(gradient, label, sample_ratio=0.1):
    gradient = tf.reshape(gradient, shape=(gradient.shape[0], -1))

    # select only a subset to compute cosine
    size = int(1e3)
    if gradient.shape[0] * sample_ratio < size:
        chosen_mask = np.random.rand(gradient.shape[0]) < sample_ratio
    else:
        chosen_mask = np.random.choice(a=gradient.shape[0], size=size, replace=False)
    gradient = gradient.numpy()[chosen_mask]
    label = label.numpy()[chosen_mask]
    grad_norm = np.linalg.norm(gradient, axis=1, keepdims=False)
    cosine = np.divide(np.matmul(gradient, gradient.T) + 1e-16, np.outer(grad_norm, grad_norm) + 1e-16)
    complete_cosine = lower_triangle_entries(cosine)
    pos_cosine = cosine[label==1,:][:,label==1]
    pos_cosine = lower_triangle_entries(pos_cosine)
    pos_neg_cosine = np.reshape(cosine[label==1,:][:,label==0], [-1])
    neg_cosine = cosine[label==0,:][:,label==0]
    neg_cosine = lower_triangle_entries(neg_cosine)

    return {'every_cosine': complete_cosine,
            'pos_cosine': pos_cosine,
            'pos_neg_cosine': pos_neg_cosine,
            'neg_cosine': neg_cosine}


def normalized_norm(vector):
    norm_g = tf.norm(vector, axis=-1, keepdims=False)
    norm_g = norm_g / tf.math.reduce_max(norm_g)
    return norm_g


def get_fashion_mnist_label(label):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels[int(label)]


def get_fashion_mnist_labels(labels):
    return [get_fashion_mnist_label(i) for i in labels]