from keras import datasets, layers, models
from keras.utils import np_utils
import matplotlib.pyplot as plt
from Helpers import *
from keras.models import model_from_json
from Evaluation.evaluator import Evaluator
import numpy as np
import logging
import os

logging.basicConfig(level=logging.DEBUG)

_, x_test_ign, x_test_fgsm, x_test_pgd, _ = Evaluator().load_data()
sess, X, X_adv, Y, acc = load_ign_robust_network()

def evaluate(x, y):
    return sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})

# original data
_, _, x_test, y_test = read_all_data()
logging.info('[evaluate]: Original data is loaded.')

os.chdir('RobustCIFAR')
attack_pgd_step = get_config()['attack_pgd_step']

logging.info('[evaluate]: IGN generated data is loaded.')
logging.info('[evaluate]: FGSM generated data is loaded.')
logging.info('[evaluate]: {0} step-PGD generated data is loaded.'.format(attack_pgd_step))


# y_test = np.array(list(map(lambda x: np.argmax(x), y_test)))


logging.info('[evaluate]: Evaluation process has been started for Goodfellow AT model...')

ign_robust_test_goodfellow_acc = evaluate(x_test, y_test)
ign_robust_test_goodfellow_on_ign_acc = evaluate(x_test_ign, y_test)
ign_robust_test_goodfellow_on_fgsm_acc = evaluate(x_test_fgsm, y_test)
ign_robust_test_goodfellow_on_pgd_acc = evaluate(x_test_pgd, y_test)

logging.info('[evaluateGoodfellowAT]: (IGN-robust model on original test data): {0}'.format(ign_robust_test_goodfellow_acc))
logging.info('[evaluateGoodfellowAT]: (IGN-robust model on IGN generated test data): {0}'.format(ign_robust_test_goodfellow_on_ign_acc))
logging.info('[evaluateGoodfellowAT]: (IGN-robust model on FGSM generated test data): {0}'.format(ign_robust_test_goodfellow_on_fgsm_acc))
logging.info('[evaluateGoodfellowAT]: (IGN-robust model on {0} step-PGD generated test data): {1}'.format(attack_pgd_step,ign_robust_test_goodfellow_on_pgd_acc))
