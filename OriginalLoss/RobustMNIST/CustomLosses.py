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
W1, b1 = get_layer_weights(model=model, layer_no=trainable_layer_indices[0]) # dense
W2, b2 = get_layer_weights(model=model, layer_no=trainable_layer_indices[1]) # dense
W3, b3 = get_layer_weights(model=model, layer_no=trainable_layer_indices[2]) # dense
W4, b4 = get_layer_weights(model=model, layer_no=trainable_layer_indices[3]) # dense
W5, b5 = get_layer_weights(model=model, layer_no=trainable_layer_indices[4]) # dense
W6, b6 = get_layer_weights(model=model, layer_no=trainable_layer_indices[5]) # output


# defining Custom Loss
def custom_loss(original_images, generated_images, target_batch, name='custom_loss'):

    with tf.op_scope([original_images, generated_images, target_batch], name, 'custom_loss') as scope:

        original_images = tf.convert_to_tensor(original_images, name='original_images')
        generated_images = tf.convert_to_tensor(generated_images, name='generated_images')
        target_batch = tf.convert_to_tensor(target_batch, name='target_batch')



        # simulate original network for generated image
        mul1_gen = tf.matmul(generated_images, W1) + b1
        out1_gen = tf.nn.relu(mul1_gen)
        mul2_gen = tf.matmul(out1_gen, W2) + b2
        out2_gen = tf.nn.relu(mul2_gen)
        mul3_gen = tf.matmul(out2_gen, W3) + b3
        out3_gen = tf.nn.relu(mul3_gen)
        mul4_gen = tf.matmul(out3_gen, W4) + b4
        out4_gen = tf.nn.relu(mul4_gen)
        mul5_gen = tf.matmul(out4_gen, W5) + b5
        out5_gen = tf.nn.relu(mul5_gen)
        logits = tf.matmul(out5_gen, W6) + b6

        # Original network's categorical cross-entropy loss
        xentropy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target_batch, logits, from_logits=True))

        # Mean Squared Error between generated images and their original versions
        mse = tf.reduce_mean(tf.squared_difference(generated_images, original_images))

        # Loss = MSE - L_orig
        loss = tf.subtract(mse, xentropy)

        return loss



