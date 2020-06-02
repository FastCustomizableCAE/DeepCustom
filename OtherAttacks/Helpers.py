from keras.datasets import mnist, cifar10
from keras.models import model_from_json
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import json
import os

def load_mnist():
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    mnist_x_train, mnist_x_test = mnist_x_train / 255.0, mnist_x_test / 255.0
    y_train = tf.keras.utils.to_categorical(mnist_y_train, 10)
    y_test = tf.keras.utils.to_categorical(mnist_y_test, 10)
    x_train = mnist_x_train.reshape(len(mnist_x_train), 784)
    x_test = mnist_x_test.reshape(len(mnist_x_test), 784)
    return x_train, y_train, x_test, y_test

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


# loads original models
def build_model(model_name):
    file = open('../models/{0}.json'.format(model_name) , 'r')
    loaded_model_json = file.read()
    file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('../models/{0}.h5'.format(model_name))
    loaded_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model

def save_model(model, dataset, model_type, attack_type):
    if not os.path.exists('{0}_models/{1}'.format(model_type, attack_type)):
        os.makedirs('{0}_models/{1}'.format(model_type, attack_type))
    if dataset == 'mnist':
        model_name = get_config()['mnist_model_name']
    else:
        model_name = get_config()['cifar_model_name']
    model.save('{0}_models/{1}/{2}_{0}.h5'.format(model_type, attack_type, model_name))
    model.save_weights('{0}_models/{1}/{2}_{0}.h5'.format(model_type, attack_type, model_name))
    with open('{0}_models/{1}/{2}_{0}.json'.format(model_type, attack_type, model_name), 'w') as f:
        f.write(model.to_json())

def save_generated_data(generated_data, attack_type, dataset, set_type='train'):
    if not os.path.exists('generated_data/{0}/{1}'.format(attack_type, dataset)):
        os.makedirs('generated_data/{0}/{1}'.format(attack_type, dataset))
    np.save('generated_data/{0}/{1}/x_{2}_adv.npy'.format(attack_type, dataset, set_type), generated_data)


def get_config():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config


# Tensorflow Adversarial Training Helpers


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

def accuracy(logits, Y):
    pred = tf.argmax(logits, axis=1)
    truth = tf.argmax(Y, axis=1)
    match = tf.cast(tf.equal(pred, truth), tf.float32)
    acc = tf.reduce_mean(match, name='acc')
    return acc

def load_robust_network(dataset, attack, adv_training_type='gf'):
    load_path = 'robust_models/{0}/{1}/{2}'.format(adv_training_type, dataset, attack)
    if not os.path.exists(load_path):
        raise FileNotFoundError()
    else:
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph('{0}/ign_robust.meta'.format(load_path))
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        # get tensors
        X = graph.get_tensor_by_name('X:0')
        X_adv = graph.get_tensor_by_name('X_adv:0')
        Y = graph.get_tensor_by_name('Y:0')
        acc = graph.get_tensor_by_name('acc:0')
        # change directory back to main project folder
        os.chdir('../../..')
        return sess, X, X_adv, Y, acc

def evaluate_robust_model(x, y, dataset, attack, adv_training_type='gf'):
    sess, X, X_adv, Y, acc = load_robust_network(dataset, attack, adv_training_type)
    return sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})

