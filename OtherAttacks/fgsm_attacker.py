from art.attacks import FastGradientMethod
from attacker import Attacker
from Helpers import *
import time

class FgsmAttacker(Attacker):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.config = get_config()

    def attack(self):
        classifier, x_train, y_train, x_test, y_test = super().get_classifier_and_data()
        if self.dataset == 'mnist':
            eps = self.config['epsilon_mnist']
        else:
            eps =  self.config['epsilon_cifar']
        fgsm = FastGradientMethod(classifier, eps= eps)
        x_train_adv = fgsm.generate(x=x_train)
        start_time = time.time()
        x_test_adv = fgsm.generate(x=x_test)
        end_time = time.time()
        elapsed_time = end_time - start_time
        super().evaluate(classifier= classifier, x= x_test_adv, y= y_test, evaluation_type='Robust Test')
        return fgsm, x_train_adv, x_test_adv, elapsed_time











