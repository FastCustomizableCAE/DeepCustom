from art.attacks import ProjectedGradientDescent
from attacker import Attacker
from Helpers import *
import time

class PgdAttacker(Attacker):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.config = get_config()

    def attack(self):
        classifier, x_train, y_train, x_test, y_test = super().get_classifier_and_data()
        if self.dataset == 'mnist':
            eps = self.config['epsilon_mnist']
            eps_step = self.config['eps_steps_pgd_mnist']
            max_iter = self.config['max_iterations_pgd_mnist']
            num_random_init = self.config['num_random_init_pgd_mnist']
        else:
            eps =  self.config['epsilon_cifar']
            eps_step = self.config['eps_steps_pgd_cifar']
            max_iter = self.config['max_iterations_pgd_cifar']
            num_random_init = self.config['num_random_init_pgd_cifar']
        pgd = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, num_random_init=num_random_init)
        x_train_adv = pgd.generate(x=x_train)
        start_time = time.time()
        x_test_adv = pgd.generate(x=x_test)
        end_time = time.time()
        elapsed_time = end_time - start_time
        super().evaluate(classifier= classifier, x= x_test_adv, y= y_test, evaluation_type='Robust Test')
        return pgd, x_train_adv, x_test_adv, elapsed_time
