import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import random
import math
from keras import datasets, layers, models
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras.datasets import cifar10
import pickle
import json
import os



# MARK:  Plotting Helpers

def plot_loss(pnts1, pnts2):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('loss:', color='C0')
    ax.plot(pnts1, 'C1', label='loss on training data')
    ax.plot(pnts2, 'C2', label='loss on test data')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    ax.legend()
    plt.show()


def plot_loss_custom(pnts1, title1, pnts2, title2, x_title, y_title):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('loss:', color='C0')
    ax.plot(pnts1, 'C1', label= title1)
    ax.plot(pnts2, 'C2', label= title2)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    ax.legend()
    plt.show()




# MARK:  Data Read

def read_data(class_):
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_inds = np.where(train_labels == class_)[0]
    test_inds = np.where(test_labels == class_)[0]
    y_train = np_utils.to_categorical(train_labels, 10)
    y_test = np_utils.to_categorical(test_labels, 10)
    x_train = train_images[train_inds]
    x_test = test_images[test_inds]
    y_train = y_train[train_inds]
    y_test = y_test[test_inds]
    return  x_train, y_train, x_test, y_test


def read_all_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    y_train = np_utils.to_categorical(train_labels, 10)
    y_test = np_utils.to_categorical(test_labels, 10)
    return  train_images, y_train, test_images, y_test



# MARK:  TensorFlow Helpers.


def var(name, shape, init=None, std=None, regularizer=None):
    if init is None:
        if std is None:
            std = (2./shape[0])**0.5
        init = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name=name, shape=shape,
                           dtype=tf.float32, initializer=init, regularizer=regularizer)


def conv(X, f, strides=[1, 1, 1, 1], padding='VALID'):
    return tf.nn.conv2d(X, f, strides, padding, )

def conv2(X, f, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(X, f, strides, padding, )

def max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
    return tf.nn.max_pool(X, ksize, strides, padding)

def max_pool2(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(X, ksize, strides, padding)

def accuracy(logits, Y):
    pred = tf.argmax(logits, axis=1)
    truth = tf.argmax(Y, axis=1)
    match = tf.cast(tf.equal(pred, truth), tf.float32)
    acc = tf.reduce_mean(match, name='acc')
    return acc

def conv_layer(input, shape, layer, strides=[1, 1, 1, 1], output_layer=False, batch_norm=False, is_train=True):
    W = var('W{0}'.format(layer), shape)
    b = var('b{0}'.format(layer), [shape[-1]], tf.constant_initializer(0.1))
    if not output_layer:
        out = tf.nn.relu(conv(input, W, strides= strides) + b)
    else:
        out =  tf.nn.sigmoid(conv(input, W, strides= strides) + b)
    if batch_norm:
        out = tf.layers.batch_normalization(out, is_train)
    return out


def deconv_layer(input, filter_size, stride, kernel, output_layer=False, batch_norm=False, is_train=True):
    out = tf.layers.conv2d_transpose(input, filter_size, stride, kernel, padding='valid')
    if not output_layer:
        out = tf.nn.relu(out)
    else:
        out = tf.nn.sigmoid(out)
    if batch_norm:
        out = tf.layers.batch_normalization(out, training=is_train)
    return out


def dense_adv(input_org, input_gen, shape, layer, output_layer=False):
    W = var('W{0}'.format(layer), shape)
    b = var('b{0}'.format(layer), [shape[1]], tf.constant_initializer(0.1))
    if not output_layer:
        return tf.nn.relu(tf.matmul(input_org, W) + b), tf.nn.relu(tf.matmul(input_gen, W) + b)
    else:
        return tf.matmul(input_org, W) + b, tf.matmul(input_gen, W) + b

def conv_layer_adv(input_org, input_gen, shape, layer, strides=[1, 1, 1, 1], output_layer=False, batch_norm=False, is_train=True):
    W = var('W{0}'.format(layer), shape)
    b = var('b{0}'.format(layer), [shape[-1]], tf.constant_initializer(0.1))
    if not output_layer:
        out_org, out_gen = tf.nn.relu(conv(input_org, W, strides= strides) + b), tf.nn.relu(conv(input_gen, W, strides= strides) + b)
    else:
        out_org, out_gen =  tf.nn.sigmoid(conv(input_org, W, strides= strides) + b), tf.nn.sigmoid(conv(input_gen, W, strides= strides) + b)
    if batch_norm:
        out_org, out_gen = tf.layers.batch_normalization(out_org, is_train), tf.layers.batch_normalization(out_gen, is_train)
    return out_org, out_gen

def maxpool_layer_adv(input_org, input_gen):
    return max_pool(input_org), max_pool(input_gen)

def flatten_adv(input_org, input_gen, shape):
    return tf.reshape(input_org, shape), tf.reshape(input_gen, shape)

def load_dgn_robust_network():
    '''
    loads DGN robust network which trained by Goodfellow's adversarial training method
    '''
    if not os.path.exists('{0}/dgnRobustModel'.format(os.getcwd())):
        raise FileNotFoundError()
    else:
        os.chdir('{0}/dgnRobustModel'.format(os.getcwd()))
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph('dgn_robust_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        # get tensors
        X = graph.get_tensor_by_name('X:0')
        X_adv = graph.get_tensor_by_name('X_adv:0')
        Y = graph.get_tensor_by_name('Y:0')
        acc = graph.get_tensor_by_name('acc:0')
        # change directory back to main project folder
        os.chdir('..')
        return sess, X, X_adv, Y, acc

def dgn_robust_model_evaluate(x, y):
    sess, X, X_adv, Y, acc = load_dgn_robust_network()
    return sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})

def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.
    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Mgaust be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'indices\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])


def save_dgn_network(class_, saver, session):
    if not os.path.exists('Models/{0}'.format(class_)):
        try:
            os.makedirs('Models/{0}'.format(class_))
        except OSError as err:
            print('[Helpers]: {0}'.format(err))
    try:
        saver.save(session, 'Models/{0}/generator_net_{0}'.format(class_))
        print('DGN network is successfully saved.')
    except Exception as err:
        print('[Helpers]: {0}'.format(err))





# MARK:   Image Helpers.


def clip_pixels(original_image, generated_image, eps):
    # diff = np.subtract(original_image, generated_image)
    diff = original_image - generated_image
    for i in range(original_image.shape[0]):
        # pixel_value = original_image[i]
        diff_value = diff[i]
        diff_abs = abs(diff_value)
        if diff_abs > eps and diff_value > 0.0:
            #print('Clipped diff: {0} to {1}'.format(
            #     diff_value, eps
            #))
            #diff[i] = -eps
            diff[i] = eps
        elif diff_abs > eps and diff_value < 0.0:
            #print('Clipped diff: {0} to {1}'.format(
            #    diff_value, -eps
            #))
            #diff[i]= eps
            diff[i] = -eps
        else:
            #print('else case:  diff_abs: {0},  diff_value: {1}'.format(diff_abs, diff_value))
            pass  # do nothing
    # return np.add(original_image, diff)
    return original_image - diff

def round_next_hundred(x):
    return int(math.ceil(x / 100.0) - 1) * 100


def scale_range(input_, min_, max_, threshold=-1.0):
    """
    scale the image in the range between min_ and max_.
    If given, the pixel values lower than threshold will
    be eliminated. This elimination might not be a good
    option if the goal is generating adversarial inputs.
    """
    input_ += -(np.min(input_))
    input_ /= np.max(input_) / (max_ - min_)
    input_ += min_
    if threshold > 0.0:
        modified_scaled_image = np.zeros_like(input_)
        for i in range(len(input_[0])):
            modified_value = input_[0][i]
            modified_scaled_image[0][i] = modified_value
            if modified_value < threshold:
                modified_scaled_image[0][i] = 0.0
        input_ = modified_scaled_image
    return input_


def show(image):
    image = image.reshape(28,28)
    plt.imshow(image)
    plt.show()


def show_multiple(images):
    for image in images:
        plt.imshow(image)
        plt.show()




# MARK: Distance Metrics

#  What is the total number of pixels that differ in their value between image X and image Z?
def L_zero(im1, im2):
    im1, im2 = im1.reshape(784, ), im2.reshape(784, )
    num_diff = 0
    for i in range(len(im1)):
        if not im1[i] == im2[i]:
            num_diff += 1
    return num_diff

# What is the summed absolute value difference between image X and image Z?
def L_one(im1, im2):
    im1, im2 = im1.reshape(784, ), im2.reshape(784, )
    total_diff = 0
    for i in range(len(im1)):
        total_diff += abs(im1[i] - im2[i])
    return total_diff

# What is the squared difference between image X and image Z?
# for each pixel, take the distance between image X and Z, square it, and sum that over all pixels
def L_two(im1, im2):
    im1, im2 = im1.reshape(784, ), im2.reshape(784, )
    total_diff = 0
    for i in range(len(im1)):
        total_diff += np.square(im1[i] - im2[i])
    return total_diff

def L_inf(im1, im2):
    l_inf = np.amax(np.abs(im1 - im2))
    return l_inf


# MARK: model helpers
def load_model(model_name):
    with open('{0}.json'.format(model_name), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('{0}.h5'.format(model_name))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_trainable_layers(model, include_output_layer=False):
    trainable_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            trainable_layers.append(model.layers.index(layer))
        except:
            pass

    if not include_output_layer:
        trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers

def get_layer_outs(model, test_input):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  #  evaluation functions
    layer_outs = [func([test_input]) for func in functors]
    return layer_outs

def get_layer_weights(model, layer_no):
    return model.layers[layer_no].get_weights()

def predict(model, sample):
    prediction = model.predict_classes(sample)[0]
    return prediction

def is_predicted(model, sample, label):
    return predict(model, sample) == label

def get_activations(model, sample):
    trainable_layers = get_trainable_layers(model= model)
    layer_outs = get_layer_outs(model= model, test_input= sample)
    return layer_outs

def get_layer_type(model, layer_no):
    config = model.get_config()['layers']
    layer_config = config[layer_no]
    return layer_config['class_name']

def get_config():
    base_dir = os.path.dirname(__file__)
    # read configuration file
    with open('{0}/config.json'.format(base_dir)) as config_file:
        config = json.load(config_file)
    return config

def load_attack_robust_model(attack_type):
    config = get_config()
    if attack_type == 'pgd':
        load_path = '../../OtherAttacks/robust_models/{0}/cifar/pgd_step{1}'.format(config['other_attacks_adv_training_type'], config['defense_pgd_step'])
    else:
        load_path = '../../OtherAttacks/robust_models/{0}/cifar/fgsm'.format(config['other_attacks_adv_training_type'])
    if not os.path.exists(load_path):
        raise FileNotFoundError()
    else:
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph('{0}/{1}_robust.meta'.format(load_path, attack_type))
        saver.restore(sess, tf.train.latest_checkpoint('{0}/'.format(load_path)))
        # get tensors
        X = graph.get_tensor_by_name('X:0')
        X_adv = graph.get_tensor_by_name('X_adv:0')
        Y = graph.get_tensor_by_name('Y:0')
        acc = graph.get_tensor_by_name('acc:0')
        # change directory back to main project folder
        return sess, X, X_adv, Y, acc

def evaluate_attack_robust_model(x, y, attack_type):
    sess, X, X_adv, Y, acc = load_attack_robust_model(attack_type)
    return sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})
