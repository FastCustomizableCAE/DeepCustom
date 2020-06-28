from DataGeneratorNetwork import DataGeneratorNetwork
from DataGenerator import DataGenerator
from Helpers import *
import logging
import time

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    logging.info('[train]: DGN training process is started.')

    # read configuration file
    with open('config.json') as config_file:
       config = json.load(config_file)
    num_classes = config['num_classes']

    logging.info('[train]: Configurations are loaded.')

    # train DGN networks for all classes
    for class_ in range(num_classes):
        logging.info('[train]: DGN training started for class {0}.'.format(class_))
        start_time = time.time()
        DGN = DataGeneratorNetwork(class_= class_)
        DGN.train()
        logging.info('[train]: DGN training has finished for class {0}. Elapsed time: {1} secs.'.format(
            class_, (time.time() - start_time)
        ))









