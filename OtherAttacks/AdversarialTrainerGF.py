'''
Based on paper: https://arxiv.org/pdf/1412.6572.pdf
'''

import numpy as np
import tensorflow as tf
from Helpers import *
import os


class AdversarialTrainerGF:

    def __init__(self, dataset, attack_type):
        self.config = get_config()
        if attack_type == 'pgd':
            max_iterations = self.config['max_iterations_pgd_{0}'.format(dataset)]
            train_data_path = 'generated_data/{0}_step{1}/{2}/x_train_adv.npy'.format(attack_type, max_iterations, dataset)
            test_data_path = 'generated_data/{0}_step{1}/{2}/x_test_adv.npy'.format(attack_type, max_iterations, dataset)
        else:
            train_data_path = 'generated_data/{0}/{1}/x_train_adv.npy'.format(attack_type, dataset)
            test_data_path = 'generated_data/{0}/{1}/x_test_adv.npy'.format(attack_type, dataset)
        self.x_train_adv = np.load(train_data_path)
        self.x_test_adv = np.load(test_data_path)
        self.attack_type = attack_type
        self.dataset = dataset

        if dataset == 'mnist':
            self.x_train, self.y_train, self.x_test, self.y_test = load_mnist()
            self.num_epochs = self.config['adv_num_epochs_mnist']
            self.batch_size = self.config['adv_batch_size_mnist']
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = load_cifar()
            if attack_type == 'fgsm':
                self.num_epochs = self.config['adv_num_epochs_cifar_fgsm']
            else:
                self.num_epochs = self.config['adv_num_epochs_cifar_pgd']
            self.batch_size = self.config['adv_batch_size_cifar']


    def adversarial_train(self, graph, tensors):
        self.sess = tf.InteractiveSession(graph=graph)
        tf.global_variables_initializer().run()
        step, loss, X, X_adv, Y, acc, saver = tensors

        # train
        num_iter = len(self.x_train) // self.batch_size
        for epoch in range(self.num_epochs):
            train_loss, test_loss = 0, 0
            for I in range(num_iter):
                # train batch
                x, y, x_gen = self.x_train[I * self.batch_size: (I + 1) * self.batch_size], self.y_train[I * self.batch_size: (
                                                                                                                 I + 1) * self.batch_size], self.x_train_adv[
                                                                                                                                       I * self.batch_size: (
                                                                                                                                                                   I + 1) * self.batch_size]
                # x = x.reshape(batch_size, 784)
                # x_gen.reshape(batch_size, 784)
                # test batch
                random_test_inds = np.random.choice(len(self.x_test), self.batch_size)
                x_t, y_t, x_t_gen = self.x_test[random_test_inds], self.y_test[random_test_inds], self.x_test_adv[random_test_inds]
                # x_t = x_t.reshape(batch_size, 784)
                # x_t_gen = x_t_gen.reshape(batch_size, 784)
                # train step
                _, loss_ = self.sess.run([step, loss], feed_dict={X: x, X_adv: x_gen, Y: y})
                # test loss
                loss_t = self.sess.run(loss, feed_dict={X: x_t, X_adv: x_t_gen, Y: y_t})
                # update
                train_loss += loss_
                test_loss += loss_t
            train_loss /= num_iter
            test_loss /= num_iter
            print('Epoch [{0}]:  Train Loss: {1},   Test Loss: {2}'.format(epoch, train_loss, test_loss))

        self.evaluate(sess= self.sess, acc= acc, X= X, X_adv= X_adv, Y= Y)
        self.save(saver= saver)


    def evaluate(self, sess, acc, X, X_adv, Y):
        num_iter = len(self.x_test) // self.batch_size
        acc_total = 0
        for i in range(num_iter):
            x, y = self.x_test[i * self.batch_size: (i + 1) * self.batch_size], self.y_test[i * self.batch_size: (i + 1) * self.batch_size]
            acc_total += sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})
        print('Natural Test accuracy: {0}'.format(acc_total / num_iter))
        num_iter = len(self.x_test_adv) // self.batch_size
        acc_total = 0
        for i in range(num_iter):
            x, y = self.x_test_adv[i * self.batch_size: (i + 1) * self.batch_size], self.y_test[i * self.batch_size: (i + 1) * self.batch_size]
            acc_total += sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})
        print('Robust Test accuracy: {0}'.format(acc_total / num_iter))


    def save(self, saver):
        # save the new model
        save_folder_path = 'robust_models/gf/{0}/{1}'.format(self.dataset, self.attack_type)
        if self.attack_type == 'pgd':
            max_steps = self.config['max_iterations_pgd_{0}'.format(self.dataset)]
            save_folder_path = 'robust_models/gf/{0}/{1}_step{2}'.format(self.dataset, self.attack_type, max_steps)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        saver.save(self.sess, '{0}/{1}_robust'.format(save_folder_path, self.attack_type))


