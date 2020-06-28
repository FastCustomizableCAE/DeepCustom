import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from CustomLosses import *
from Helpers import *
import logging

logging.basicConfig(level=logging.DEBUG)


class DataGeneratorNetwork(object):

    # random seed settings
    np.random.seed(216105420)


    def __init__(self, class_):
        # read configuration file
        with open('config.json') as config_file:
            self.config = json.load(config_file)

        self.class_ = class_

        self.build_model()

        self.sess = tf.InteractiveSession(graph=self.g)
        tf.global_variables_initializer().run()


    def build_model(self):

        initial_learning_rate = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        decay_step = self.config['decay_step']
        decay_rate = self.config['decay_rate']
        batch_size = self.config['batch_size']


        # Image Generator Network
        self.g = tf.Graph()
        with self.g.as_default():

            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])


            self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

            self.is_train = tf.placeholder(tf.bool, name="is_train")

            # Encoding #

            W1 = var('W1', [3, 3, 1, 32])
            b1 = var('b1', [32], tf.constant_initializer(0.1))
            out1 = tf.nn.relu(conv2(self.X, W1) + b1)
            out1 = max_pool(out1)

            W2 = var('W2', [3, 3, 32, 64])
            b2 = var('b2', [64], tf.constant_initializer(0.1))
            out2 = tf.nn.relu(conv2(out1, W2) + b2)
            out2 = max_pool(out2)

            W3 = var('W3', [3, 3, 64, 128])
            b3 = var('b3', [128], tf.constant_initializer(0.1))
            out3 = tf.nn.relu(conv(out2, W3) + b3)
            encoded = max_pool(out3)

            # Decoding #

            # input, filters, stride, kernel
            out4 = tf.nn.relu(tf.layers.conv2d_transpose(encoded, 64, 3, 2, padding='valid'))
            out4 = tf.layers.batch_normalization(out4, training= self.is_train)


            out5 = tf.nn.relu(tf.layers.conv2d_transpose(out4, 32, 2, 2, padding='valid'))
            out5 = tf.layers.batch_normalization(out5, training=self.is_train)

            generate_images = tf.nn.sigmoid(tf.layers.conv2d_transpose(out5, 1, 2, 2, padding='valid'))

            original_images = tf.reshape(self.X, [-1, 784])
            generate_images = tf.reshape(generate_images, [-1, 784])

            self.loss = custom_loss(original_image=original_images, generated_image=generate_images)

            # learning rate decay
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate,  # Base learning rate.
                self.global_step,
                decay_step,
                decay_rate,
                staircase=True)

            self.step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step)
            self.saver = tf.train.Saver()



    def train(self):

        num_epochs = self.config['num_epochs']
        batch_size = self.config['batch_size']

        x_train, y_train, x_test, y_test = read_data(class_= self.class_)

        num_iter = len(x_train) // batch_size

        # store loss values to plot them later
        train_losses, test_losses = [], []

        for epoch in range(num_epochs):

            epoch_train_loss, epoch_test_loss = 0, 0

            for I in range(num_iter):

                # train batch
                x = x_train[I * batch_size: (I + 1) * batch_size]
                x = x.reshape((batch_size, 28, 28, 1))

                # test batch
                random_test_inds = np.random.choice(len(x_test), batch_size)
                x_t = x_test[random_test_inds]
                x_t = x_t.reshape((batch_size, 28, 28, 1))

                feed_train = {self.X: x, self.is_train: True}

                feed_test = {self.X: x_t, self.is_train: True}

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                # train step
                _, loss_train, _ = self.sess.run([self.step, self.loss, extra_update_ops], feed_dict=feed_train)

                # test loss
                loss_test, _ = self.sess.run([self.loss, extra_update_ops], feed_dict=feed_test)

                # update
                epoch_train_loss += loss_train
                epoch_test_loss += loss_test

            epoch_train_loss /= num_iter
            epoch_test_loss /= num_iter

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)

            if epoch % 25 == 0:
                logging.info('[DGN]: Epoch ({0}): Training Loss: {1},  Test Loss: {2}'.format(
                    epoch,
                    epoch_train_loss,
                    epoch_test_loss,
                ))

        plot_loss(train_losses, test_losses)
        save_dgn_network(class_=self.class_, saver=self.saver, session=self.sess)
        self.sess.close()





