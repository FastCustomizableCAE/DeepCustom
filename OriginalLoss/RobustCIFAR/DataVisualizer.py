from Helpers import *
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.DEBUG)


# TODO: implement a customizable DataVisualizer class. No time for this currently.
#class DataVisualizer():
#
#    def __init__(self):
#        pass

# read configuration file
with open('config.json') as config_file:
    config = json.load(config_file)


eps = config['epsilon']



CLASS = 0
original_data, _, _, _ = read_data(class_= CLASS)
generated_data = np.load('adversarial_training_data/gen_data_class_{0}.npy'.format(CLASS))

original_data = original_data.reshape((len(original_data), 3072))
generated_data = generated_data.reshape((len(generated_data), 3072))

print('original_data.shape: {0}  generated_data.shape: {1}'.format(
    original_data.shape,
    generated_data.shape
))

for i in range(3072):
    org_px = original_data[0][i]
    gen_px = generated_data[0][i]
    diff = abs(org_px - gen_px)
    if diff > (eps + 0.001):
        print('diff: ', diff)

for i in range(5):
    plt.imshow(original_data[i].reshape((32, 32, 3)))
    plt.show()
    plt.imshow(generated_data[i].reshape((32, 32, 3)))
    plt.show()