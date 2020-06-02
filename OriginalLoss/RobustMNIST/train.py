from ImageGeneratorNetwork import ImageGeneratorNetwork
from DataGenerator import DataGenerator
from Helpers import *
import logging
import time

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    logging.info('[train]: IGN training process is started.')

    # read configuration file
    with open('config.json') as config_file:
       config = json.load(config_file)
    num_classes = config['num_classes']

    logging.info('[train]: Configurations are loaded.')

    # train IGN networks for all classes
    for class_ in range(num_classes):
    #for class_ in range(1):
        logging.info('[train]: IGN training started for class {0}.'.format(class_))
        start_time = time.time()
        IGN = ImageGeneratorNetwork(class_= class_)
        IGN.train()
        logging.info('[train]: IGN training has finished for class {0}. Elapsed time: {1} secs.'.format(
            class_, (time.time() - start_time)
        ))









