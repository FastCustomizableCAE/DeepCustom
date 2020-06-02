from keras import datasets, layers, models
from keras.utils import np_utils
import matplotlib.pyplot as plt
from Helpers import *
from keras.models import model_from_json
import numpy as np
import logging
import os

logging.basicConfig(level=logging.DEBUG)


class Evaluator():

    def __init__(self):
        os.chdir('..')
        self.config = get_config()


    def load_data(self):
        # original data
        _, _, x_test, y_test = read_all_data()
        logging.info('[evaluate]: Original data is loaded.')

        # IGN generated data
        x_test_ign = np.load('adversarial_test_data/gen_data.npy')
        logging.info('[evaluate]: IGN generated data is loaded.')

        # FGSM generated data
        # x_test_fgsm = np.load('attacks_data/FGSM/x_test_fgsm.npy')
        x_test_fgsm = np.load('../../OtherAttacks/generated_data/fgsm/cifar/x_test_adv.npy')
        logging.info('[evaluate]: FGSM generated data is loaded.')

        # PGD generated data
        # x_test_pgd = np.load('attacks_data/PGD/x_test_pgd_step10.npy')
        x_test_pgd = np.load(
            '../../OtherAttacks/generated_data/pgd_step{0}/cifar/x_test_adv.npy'.format(self.config ['attack_pgd_step']))
        logging.info('[evaluate]: PGD generated data is loaded for step size: {0}'.format(self.config ['attack_pgd_step']))
        return  x_test, x_test_ign, x_test_fgsm, x_test_pgd, y_test


    def evaluate(self, model_type):
        x_test, x_test_ign, x_test_fgsm, x_test_pgd, y_test = self.load_data()
        if model_type == 'original':
            y_test = np.array(list(map(lambda x: np.argmax(x), y_test)))
            logging.info('[evaluator]: Evaluating the original model')
            # load original model
            original_model = load_model(self.config['model_name'])
            _, natural_test_acc = original_model.evaluate(x_test, y_test, verbose=2)
            _, original_model_on_ign_test_acc = original_model.evaluate(x_test_ign, y_test, verbose=2)
            _, original_model_on_fgsm_test_acc = original_model.evaluate(x_test_fgsm, y_test, verbose=2)
            _, original_model_on_pgd_test_acc = original_model.evaluate(x_test_pgd, y_test, verbose=2)
            logging.info('[evaluate]: (original model on original test data): {0}'.format(natural_test_acc))
            logging.info('[evaluate]: (original model on IGN generated test data): {0}'.format(original_model_on_ign_test_acc))
            logging.info( '[evaluate]: (original model on FGSM generated test data): {0}'.format(original_model_on_fgsm_test_acc))
            logging.info( '[evaluate]: (original model on step {0}-PGD generated test data): {1}'.format(get_config()['attack_pgd_step'], original_model_on_pgd_test_acc))
        elif model_type == 'fgsm_robust':
            logging.info('[evaluator]: Evaluating the FGSM robust model')
            fgsm_robust_test_acc = evaluate_attack_robust_model(x=x_test, y=y_test, attack_type='fgsm')
            fgsm_robust_on_ign_test_acc = evaluate_attack_robust_model(x=x_test_ign, y=y_test, attack_type='fgsm')
            fgsm_robust_on_fgsm_test_acc = evaluate_attack_robust_model(x=x_test_fgsm, y=y_test, attack_type='fgsm')
            fgsm_robust_on_pgd_test_acc = evaluate_attack_robust_model(x=x_test_pgd, y=y_test, attack_type='fgsm')
            logging.info('[evaluate]: (FGSM-robust model on original test data): {0}'.format(fgsm_robust_test_acc))
            logging.info('[evaluate]: (FGSM-robust model on IGN generated test data): {0}'.format(fgsm_robust_on_ign_test_acc))
            logging.info('[evaluate]: (FGSM-robust model on FGSM generated test data): {0}'.format(fgsm_robust_on_fgsm_test_acc))
            logging.info('[evaluate]: (FGSM-robust model on step {0}-PGD generated test data): {1}'.format(get_config()['attack_pgd_step'], fgsm_robust_on_pgd_test_acc))
        elif model_type == 'pgd_robust':
            logging.info('[evaluator]: Evaluating the PGD robust model')
            pgd_robust_test_acc = evaluate_attack_robust_model(x=x_test, y=y_test, attack_type='pgd')
            pgd_robust_on_ign_test_acc = evaluate_attack_robust_model(x=x_test_ign, y=y_test, attack_type='pgd')
            pgd_robust_on_fgsm_test_acc = evaluate_attack_robust_model(x=x_test_fgsm, y=y_test, attack_type='pgd')
            pgd_robust_on_pgd_test_acc = evaluate_attack_robust_model(x=x_test_pgd, y=y_test, attack_type='pgd')
            logging.info('[evaluate]: (PGD-robust model on original test data): {0}'.format(pgd_robust_test_acc))
            logging.info('[evaluate]: (PGD-robust model on IGN generated test data): {0}'.format(pgd_robust_on_ign_test_acc))
            logging.info('[evaluate]: (PGD-robust model on FGSM generated test data): {0}'.format(pgd_robust_on_fgsm_test_acc))
            logging.info('[evaluate]: ({0} step-PGD-robust model on step {1}-PGD generated test data): {2}'.format(get_config()['defense_pgd_step'],get_config()['attack_pgd_step'], pgd_robust_on_pgd_test_acc))









