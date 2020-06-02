'''

This script evaluates IGN networks against each other.

Experiments with other attack types (FGSM, PGD) can be found in each projects' Evaluation folder.

'''
import tensorflow as tf
from keras.datasets import mnist
from keras import datasets
from keras.utils import np_utils
import numpy as np
import os




dataset = 'cifar'
attack_metrics = ['LRP', 'Suspicious', 'OriginalLoss']
defense_metric = 'OriginalLoss'


def load_ign_robust_network(project_path):
    if not os.path.exists('{0}/Evaluation/ign_robust_model'.format(project_path)):
        raise FileNotFoundError()
    else:
        #os.chdir('Evaluation/ign_robust_model')
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph('{0}/Evaluation/ign_robust_model/ign_robust.meta'.format(project_path))
        saver.restore(sess, tf.train.latest_checkpoint('{0}/Evaluation/ign_robust_model'.format(project_path)))
        # get tensors
        X = graph.get_tensor_by_name('X:0')
        X_adv = graph.get_tensor_by_name('X_adv:0')
        Y = graph.get_tensor_by_name('Y:0')
        acc = graph.get_tensor_by_name('acc:0')
        # change directory back to main project folder
        #os.chdir('../../..')
        return sess, X, X_adv, Y, acc

def ign_robust_model_evaluate(x, y, project_path):
    sess, X, X_adv, Y, acc = load_ign_robust_network(project_path)
    return sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})

def load_mnist():
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    mnist_x_train, mnist_x_test = mnist_x_train / 255.0, mnist_x_test / 255.0
    label_test = mnist_y_test
    label_train = mnist_y_train
    y_train = tf.keras.utils.to_categorical(mnist_y_train, 10)
    y_test = tf.keras.utils.to_categorical(mnist_y_test, 10)
    x_train = mnist_x_train.reshape(len(mnist_x_train), 784)
    x_test = mnist_x_test.reshape(len(mnist_x_test), 784)
    return x_train, y_train, x_test, y_test

def load_cifar():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    y_train = np_utils.to_categorical(train_labels, 10)
    y_test = np_utils.to_categorical(test_labels, 10)
    return  train_images, y_train, test_images, y_test


if __name__ == '__main__':
    if dataset == 'mnist':
        _, _, _, y_test = load_mnist()
        project_name = 'RobustMNIST'
    else:
        _, _, _, y_test = load_cifar()
        project_name = 'RobustCIFAR'
    defense_project_path = '{0}/{1}'.format(defense_metric, project_name)
    for attack_metric in attack_metrics:
        attack_project_path = '{0}/{1}'.format(attack_metric, project_name)
        generated_test_data = np.load('{0}/adversarial_test_data/gen_data.npy'.format(attack_project_path))
        robust_test_acc = ign_robust_model_evaluate(x=generated_test_data, y=y_test, project_path=defense_project_path)
        print('Robust test accuracy of {0}-IGN-network on {1}-IGN-attack is: {2}'.format(defense_metric, attack_metric,
                                                                                         robust_test_acc))








