import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from Helpers import *
from CustomLosses import *
import logging

logging.basicConfig(level=logging.DEBUG)


class ImageGeneratorNetwork(object):

    # random seed settings
    #tf.set_random_seed(451760341)
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

            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='X')
            X_reshaped = tf.reshape(self.X, [-1, 32 * 32 * 3])

            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

            # Encoding
            out1 = conv_layer(input=self.X, shape=[3, 3, 3, 32], layer=1)
            out2 = conv_layer(input=out1, shape=[3, 3, 32, 64], layer=2, strides=[1, 2, 2, 1])
            out3 = conv_layer(input=out2, shape=[3, 3, 64, 128], layer=3)
            encoded = conv_layer(input=out3, shape=[3, 3, 128, 128], layer=4, strides=[1, 2, 2, 1])

            # Decoding
            out4 = deconv_layer(input=encoded, filter_size=128, stride=3, kernel=2, batch_norm=True, is_train=self.is_train)
            out5 = deconv_layer(input=out4, filter_size=64, stride=4, kernel=1, batch_norm=True, is_train=self.is_train)
            out6 = deconv_layer(input=out5, filter_size=32, stride=4, kernel=2, batch_norm=True, is_train=self.is_train)
            generated_images = deconv_layer(input=out6, filter_size=3, stride=3, kernel=1, output_layer=True)

            original_images = tf.reshape(self.X, [-1, 32 * 32 * 3])
            generated_images = tf.reshape(generated_images, [-1, 32 * 32 * 3])

            self.loss = custom_loss(original_image= original_images, generated_image= generated_images, class_=  self.class_)

            # learning rate decay
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate,  # Base learning rate.
                self.global_step,
                decay_step,
                decay_rate,
                staircase=True)

            self.step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

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
                # test batch
                random_test_inds = np.random.choice(len(x_test), batch_size)
                x_t = x_test[random_test_inds]

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
                logging.info('[IGN]: Epoch ({0}): Training Loss: {1},  Test Loss: {2}'.format(
                    epoch,
                    epoch_train_loss,
                    epoch_test_loss,
                ))

        plot_loss(train_losses, test_losses)
        save_ign_network(class_=self.class_, saver=self.saver, session=self.sess)
        self.sess.close()





