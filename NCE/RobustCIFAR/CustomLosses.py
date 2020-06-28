import tensorflow as tf
import numpy as np
from Helpers import *
import os


# read configuration file
with open('config.json') as config_file:
    config = json.load(config_file)
model_name = config['model_name']

# load original model and its weights
model = load_model(model_name)
trainable_layer_indices = get_trainable_layers(model= model, include_output_layer=True)
W1, b1 = get_layer_weights(model=model, layer_no=trainable_layer_indices[0]) # conv
W2, b2 = get_layer_weights(model=model, layer_no=trainable_layer_indices[1]) # conv
W3, b3 = get_layer_weights(model=model, layer_no=trainable_layer_indices[2]) # conv
W4, b4 = get_layer_weights(model=model, layer_no=trainable_layer_indices[3]) # dense
W5, b5 = get_layer_weights(model=model, layer_no=trainable_layer_indices[4]) # dense


# defining Custom Loss
def custom_loss(original_images, generated_images, target_batch, name='custom_loss'):

    with tf.op_scope([original_images, generated_images, target_batch], name, 'custom_loss') as scope:

        original_images = tf.convert_to_tensor(original_images, name='original_images')
        generated_images = tf.convert_to_tensor(generated_images, name='generated_images')
        target_batch = tf.convert_to_tensor(target_batch, name='target_batch')

        out1 = tf.nn.relu(conv(generated_images, W1) + b1)
        out1 = max_pool(out1)
        out2 = tf.nn.relu(conv(out1, W2) + b2)
        out2 = max_pool(out2)
        out3 = tf.nn.relu(conv(out2, W3) + b3)
        flatten = tf.reshape(out3, [-1, 4 * 4 * 64])
        out4 = tf.nn.relu(tf.matmul(flatten, W4) + b4)
        logits = tf.matmul(out4, W5) + b5

        # Original network's categorical cross-entropy loss
        xentropy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target_batch, logits, from_logits=True))

        # Mean Squared Error between generated images and their original versions
        mse = tf.reduce_mean(tf.squared_difference(generated_images, original_images))

        # Loss = MSE - L_orig
        loss = tf.subtract(mse, xentropy)

        return loss



