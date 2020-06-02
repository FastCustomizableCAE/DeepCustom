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

CLASS = 0
original_data, _, _, _ = read_all_data()
generated_data = np.load('adversarial_training_data/gen_data.npy')

for i in range(784):
    org_px = original_data[0][i]
    gen_px = generated_data[0][i]
    diff = abs(org_px - gen_px)
    if diff > (0.3 + 0.001):
        print('diff: ', diff)

for i in range(5):
    plt.imshow(original_data[i].reshape(28, 28))
    plt.show()
    plt.imshow(generated_data[i].reshape(28,28))
    plt.show()