import tensorflow as tf
import numpy as np
from Helpers import *


# MARK: Prerequisites for custom loss
# read configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

k = config['k']
epsilon = config['epsilon']
alpha_parameter = config['custom_loss_constant']



# defining LRP Custom Loss
def custom_loss(original_image, generated_image, class_, name='lrp_custom_loss'):
    '''
    :param original_image:
    :param generated_image:
    :param name:
    :return:
    '''
    with tf.op_scope([original_image, generated_image, class_], name, 'lrp_custom_loss') as scope:

        original_image = tf.convert_to_tensor(original_image, name='original_image')
        generated_image = tf.convert_to_tensor(generated_image, name='generated_image')

        # Relevant pixels
        relevant_pixels_original = gather_cols(original_image, relevant_pixels(for_digit= class_, k= k))
        relevant_pixels_generated = gather_cols(generated_image, relevant_pixels(for_digit= class_, k= k))

        # Non-relevant pixels
        non_relevant_pixels_original = gather_cols(original_image, non_relevant_pixels(for_digit= class_, k= k))
        non_relevant_pixels_generated = gather_cols(generated_image, non_relevant_pixels(for_digit= class_, k= k))

        eps = tf.constant(epsilon, dtype=tf.float32)

        non_relevant_pixels_custom_loss = tf.reduce_mean(tf.squared_difference(
            tf.add(non_relevant_pixels_original, eps),
            non_relevant_pixels_generated
        ))

        relevant_pixels_custom_loss = tf.reduce_mean(tf.squared_difference(
            tf.subtract(relevant_pixels_original, eps),
            relevant_pixels_generated
        ))

        alpha = tf.constant(alpha_parameter, dtype=tf.float32)

        loss = tf.add(
            tf.math.multiply(alpha, relevant_pixels_custom_loss),
            tf.math.multiply(tf.subtract(tf.constant(1.0), alpha), non_relevant_pixels_custom_loss))

        return loss



