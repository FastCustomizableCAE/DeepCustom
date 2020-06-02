from Helpers import *
from matplotlib import pyplot as plt
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.DEBUG)

class DataGenerator():

    def __init__(self, original_model_name, class_=None, data_type='training', num_classes=10, eps=.3, clipping=True, mis_classification_check=False):
        self.original_model_name = original_model_name
        self.class_ = class_
        self.data_type = data_type
        self.num_classes = num_classes
        self.eps = eps
        self.clipping = clipping
        self.mis_classification_check = mis_classification_check
        self.info()


    def info(self):
        logging.info('[DataGenerator]: DataGenerator is created with the following configurations:')
        logging.info('[DataGenerator]: model: {0}, class: {1}, data type: {2}, number of classes: {3}, eps: {4}, clipping: {5}, misclassification_check: {6}'.format(
            self.original_model_name, self.class_, self.data_type, self.num_classes, self.eps, self.clipping, self.mis_classification_check
        ))


    def generate_data(self):
        if not self.class_ == None:
            self.generate_data_for_class()
        else:
            self.generate_data_for_all_classes()


    def generate_data_for_class(self):
        if not self.data_type == 'training' and not self.data_type == 'test':
            logging.error("[DataGenerator]: Invalid data type argument is given. Please either use 'training' or 'test'.")
            raise Exception('[DataGenerator]: Invalid data type argument.')
        x, y, _, _ = read_data(class_=self.class_)
        if self.data_type == 'test':
            _, _, x, y = read_data(class_=self.class_)
        generated_data = self.generate_data_using_IGN(class_= self.class_, x= x)
        # ensure pixel values are in between 0 and 1
        np.clip(generated_data, 0.0, 1.0, out=generated_data)
        if self.mis_classification_check:
            mis_classified_original_samples, mis_classified_generated_samples, mis_classified_target = self.check_mis_classification(x, generated_data, y)
            self.save_generated_data(generated_samples=mis_classified_generated_samples,
                                     original_samples=mis_classified_original_samples,
                                     target=mis_classified_target,
                                     class_= self.class_)
        else:
            self.save_generated_data(generated_samples= generated_data, class_= self.class_)


    def generate_data_for_all_classes(self):
        if not self.data_type == 'training' and not self.data_type == 'test':
            logging.error(
                "[DataGenerator]: Invalid data type argument is given. Please either use 'training' or 'test'.")
            raise Exception('[DataGenerator]: Invalid data type argument.')
        if self.data_type == 'training':
            x, y, _, _ = read_all_data()
        else:
            _, _, x, y = read_all_data()
        all_generated_data = np.zeros_like(x)
        for class_ in range(self.num_classes):
            if self.data_type == 'training':
                x_c, y_c, _, _ = read_data(class_=class_)
            else:
                _, _, x_c, y_c = read_data(class_=class_)
            labels = list(map(lambda x: np.argmax(x), y))
            class_indices = np.where(np.array(labels) == class_)
            generated_data = self.generate_data_using_IGN(class_= class_, x=x_c)
            # ensure pixel values are in between 0 and 1
            np.clip(generated_data, 0.0, 1.0, out=generated_data)
            all_generated_data[class_indices] = generated_data

        if self.mis_classification_check:
            mis_classified_original_samples, mis_classified_generated_samples, mis_classified_target = self.check_mis_classification(x, all_generated_data, y)
            self.save_generated_data(generated_samples=mis_classified_generated_samples,
                                    original_samples=mis_classified_original_samples,
                                    target=mis_classified_target,
                                    class_=None)
        else:
            self.save_generated_data(generated_samples= all_generated_data, class_= None)


    def check_mis_classification(self, original_data, generated_data, target):
        model = load_model(self.original_model_name)
        original_predictions = model.predict_classes(original_data)
        generated_predictions = model.predict_classes(generated_data)
        num_correct_classifications = np.sum(original_predictions == generated_predictions)
        logging.info('[DataGenerator]: Number of different classifications: {0} out of {1}.'.format(len(original_predictions) - num_correct_classifications, len(original_predictions)))
        correct_classified_indices = np.where(original_predictions == generated_predictions)
        mis_classified_indices = np.setdiff1d(np.array(range(len(original_predictions))), correct_classified_indices)
        mis_classified_original_samples = original_data[mis_classified_indices]
        mis_classified_generated_samples = generated_data[mis_classified_indices]
        mis_classified_target = target[mis_classified_indices]
        return mis_classified_original_samples, mis_classified_generated_samples, mis_classified_target


    def generate_data_using_IGN(self, class_, x):
        sess, X, is_train, logits = self.load_generator_network(class_=class_)
        x = x.reshape((len(x), 28, 28, 1))
        feed = {X: x, is_train: False}

        generated_data = sess.run(logits, feed_dict=feed)
        if self.clipping:
            clipped_generated_data = np.zeros((len(generated_data), 784))
            for index in range(len(generated_data)):
                clipped_data = clip_pixels(original_image=x[index].reshape(784, ),
                            generated_image=generated_data[index].reshape(784, ),
                            eps=self.eps)
                clipped_generated_data[index] = clipped_data
            return clipped_generated_data
        return generated_data


    def load_generator_network(self, class_):
        # TODO: check for path existence
        # change directory to load IGN model for given class
        os.chdir('Models/{0}/'.format(class_))
        # read IGN model
        meta_path = 'generator_net_{0}.meta'.format(class_)
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        # get tensors
        X = graph.get_tensor_by_name("Placeholder:0")
        is_train = graph.get_tensor_by_name("is_train:0")
        logits = graph.get_tensor_by_name("Sigmoid:0")
        logging.info('[DataGenerator]: IGN model is loaded from the following path: {0}'.format(os.getcwd()))
        # change directory back to main project folder
        os.chdir('../..')
        return sess, X, is_train, logits


    def save_generated_data(self, generated_samples, original_samples=None, target=None, class_=None):
        save_path = ''
        if self.data_type == 'training':
            save_path = 'adversarial_training_data'
        elif self.data_type == 'test':
            save_path = 'adversarial_test_data'
        else:
            logging.error(
                "[DataGenerator]: Invalid data type argument is given. Please either use 'training' or 'test'.")
            raise Exception('[DataGenerator]: Invalid data type argument.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.mis_classification_check:
            if not class_ == None:
                np.save('{0}/org_data_class_{1}.npy'.format(save_path, class_), original_samples)
                np.save('{0}/gen_data_class_{1}.npy'.format(save_path, class_), generated_samples)
                np.save('{0}/target_class_{1}.npy'.format(save_path, class_), target)
            else:
                np.save('{0}/org_data.npy'.format(save_path), original_samples)
                np.save('{0}/gen_data.npy'.format(save_path), generated_samples)
                np.save('{0}/target.npy'.format(save_path), target)
        else:
            if not class_ == None:
                np.save('{0}/gen_data_class_{1}.npy'.format(save_path, class_), generated_samples)
            else:
                np.save('{0}/gen_data.npy'.format(save_path), generated_samples)




