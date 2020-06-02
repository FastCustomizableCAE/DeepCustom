from Helpers import *
import numpy as np
import tensorflow as tf
import json

class OriginalMNIST:

    # read configuration file
    with open('config.json') as config_file:
        configuration = json.load(config_file)

    #neuron_counts = configuration["neuron_counts"]
    #num_layers = configuration["num_dense_layers"]

    def __init__(self):
        self.read_meta()
        #self.save_weights()

    def read_meta(self):
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        self.graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph("./mnist_5x30.meta")
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        self.variables = tf.trainable_variables()


    def get_weight(self, layer):
        result = self.sess.run(self.variables[0])
        for i in range(len(self.variables)):
            variable = self.variables[i]
            desired_weight_name = 'W{0}'.format(layer)
            if desired_weight_name in variable.name:
                result = self.sess.run(variable)
        return result

    def get_bias(self, layer):
        result = self.sess.run(self.variables[0])
        for i in range(len(self.variables)):
            variable = self.variables[i]
            desired_bias_name = 'b{0}'.format(layer)
            if desired_bias_name in variable.name:
                result = self.sess.run(variable)
        return result

    def predict(self, x, y):
        X = self.graph.get_tensor_by_name("Placeholder:0")
        Y = self.graph.get_tensor_by_name("Placeholder_1:0")
        pred = self.graph.get_tensor_by_name("ArgMax:0")
        feed = {X: x, Y: y}
        prediction = self.sess.run(pred, feed_dict= feed)
        return prediction

    def isPredicted(self, x, y):
        prediction = self.predict(x= x, y= y)
        #print('prediction shape: {0}'.format(prediction.shape))
        #print("y shape: {0}".format(y.shape))
        #print("check if {0} == {1}".format(prediction[0], np.argmax(y[0])))
        return prediction[0] == np.argmax(y[0])

    def get_activations(self, x, y):
        X = self.graph.get_tensor_by_name("Placeholder:0")
        Y = self.graph.get_tensor_by_name("Placeholder_1:0")
        mul1 = self.graph.get_tensor_by_name("add:0")
        mul2 = self.graph.get_tensor_by_name("add_1:0")
        mul3 = self.graph.get_tensor_by_name("add_2:0")
        mul4 = self.graph.get_tensor_by_name("add_3:0")
        mul5 = self.graph.get_tensor_by_name("add_4:0")
        feed = {X: x, Y: y}
        mul1_, mul2_, mul3_, mul4_, mul5_ = self.sess.run(
            [mul1, mul2, mul3, mul4, mul5], feed_dict= feed)
        activations = np.hstack((mul1_, mul2_, mul3_, mul4_, mul5_))
        return activations

    #
    #def save_weights(self):
    #    save_path = 'mnist_weights'
    #    for i in range(self.num_layers):
    #        w = self.get_weight(i + 1)
    #        b = self.get_bias(i + 1)
    #        np.save('{0}/w{1}.npy'.format(save_path, i + 1), w)
    #        np.save('{0}/b{1}.npy'.format(save_path, i + 1), b)
    #    print('MNIST network weights are saved.')

    def evaluate(self, x, y):
        X = self.graph.get_tensor_by_name("Placeholder:0")
        Y = self.graph.get_tensor_by_name("Placeholder_1:0")
        acc = self.graph.get_tensor_by_name("Mean:0")
        feed = {X: x, Y: y}
        accuracy = self.sess.run(acc, feed_dict= feed)
        return accuracy


    def fit(self, x, y):
        X = self.graph.get_tensor_by_name("Placeholder:0")
        Y = self.graph.get_tensor_by_name("Placeholder_1:0")
        feed = {X: x, Y: y}
        self.sess.run()
        acc = self.graph.get_tensor_by_name("Mean:0")
