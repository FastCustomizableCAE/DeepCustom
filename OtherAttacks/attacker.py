from Helpers import *
import numpy as np
from art.classifiers import KerasClassifier
from art.defences import AdversarialTrainer

class Attacker():

    def __init__(self, dataset):
        self.dataset = dataset
        self.config = get_config()

    def get_classifier_and_data(self):
        if self.dataset == 'mnist':
            x_train, y_train, x_test, y_test = load_mnist()
            model = build_model(self.config['mnist_model_name'])
        else:
            x_train, y_train, x_test, y_test = load_cifar()
            model = build_model(self.config['cifar_model_name'])
        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        self.evaluate(classifier= classifier, x= x_test, y= y_test)
        return classifier, x_train, y_train, x_test, y_test

    def adversarial_train(self, attack):
        classifier, x_train, y_train, x_test, y_test = self.get_classifier_and_data()
        adv_trainer = AdversarialTrainer(classifier, attacks=attack)
        if self.dataset == 'mnist':
            nb_epochs = self.config['adv_num_epochs_mnist']
        else:
            nb_epochs = self.config['adv_num_epochs_cifar']
        adv_trainer.fit(x= x_train, y= y_train, nb_epochs= nb_epochs)
        # return adversarially trained robust model
        return classifier

    def evaluate(self, classifier, x, y, evaluation_type='Test'):
        preds = np.argmax(classifier.predict(x), axis=1)
        acc = np.sum(preds == np.argmax(y, axis=1)) / y.shape[0]
        print("{0} accuracy: {1}".format(evaluation_type, (acc * 100)))


    def adversarial_train_gf(self, x_train_av, x_test_adv):
        _, x_train, y_train, x_test, y_test = self.get_classifier_and_data()



