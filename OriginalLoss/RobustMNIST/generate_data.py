from DataGenerator import DataGenerator
from Helpers import *
import logging
import time

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    logging.info('[generate_data]: Data generation process is started.')

    # read configuration file
    with open('config.json') as config_file:
       config = json.load(config_file)
    num_classes = config['num_classes']
    model_name = config['model_name']
    eps = config['epsilon']
    mis_classification_check = config['mis_classification_check']
    clipping = config['clipping']

    logging.info('[generate_data]: Configurations are loaded.')


    # generate training data using IGNs
    dg = DataGenerator(original_model_name=model_name,
                       data_type='training',
                       num_classes=num_classes,
                       eps=eps,
                       clipping=clipping,
                       mis_classification_check=mis_classification_check)

    logging.info('[generate_data]: Data generation started for the training data.')
    dg.generate_data()


    # generate test data using IGNs
    dg = DataGenerator(original_model_name=model_name,
                       data_type='test',
                       num_classes=num_classes,
                       eps=eps,
                       clipping=clipping,
                       mis_classification_check=mis_classification_check)

    logging.info('[generate_data]: Data generation started for the test data.')
    dg.generate_data()

