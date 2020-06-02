import tensorflow as tf
import numpy as np
from Helpers import *

# MARK: Prerequisites for custom loss

# load original model and its weights
model = load_model('mnist_5x30')
trainable_layer_indices = get_trainable_layers(model= model, include_output_layer=True)

W1, b1 = get_layer_weights(model=model, layer_no=trainable_layer_indices[0]) # dense
W2, b2 = get_layer_weights(model=model, layer_no=trainable_layer_indices[1]) # dense
W3, b3 = get_layer_weights(model=model, layer_no=trainable_layer_indices[2]) # dense
W4, b4 = get_layer_weights(model=model, layer_no=trainable_layer_indices[3]) # dense
W5, b5 = get_layer_weights(model=model, layer_no=trainable_layer_indices[4]) # dense


# for dense layers (no convolutional layer for this network)
suspicious_neuron_indices = suspicious_neurons()
normal_neuron_indices = normal_neurons()

# custom loss parameter configurations (optional. related to the custom loss implemented here)
parameter_theta = 0.5
parameter_gamma = 0.5


# defining suspicious custom loss
def custom_loss(original_image, generated_image, name='suspicious_custom_loss'):

    with tf.op_scope([original_image, generated_image], name, 'suspicious_custom_loss') as scope:

        original_image = tf.convert_to_tensor(original_image, name='original_image')
        generated_image = tf.convert_to_tensor(generated_image, name='generated_image')


        # simulate original network for original image
        mul1_org = tf.matmul(original_image, W1) + b1
        out1_org = tf.nn.relu(mul1_org)
        mul2_org = tf.matmul(out1_org, W2) + b2
        out2_org = tf.nn.relu(mul2_org)
        mul3_org = tf.matmul(out2_org, W3) + b3
        out3_org = tf.nn.relu(mul3_org)
        mul4_org = tf.matmul(out3_org, W4) + b4
        out4_org = tf.nn.relu(mul4_org)
        mul5_org = tf.matmul(out4_org, W5) + b5

        # simulate original network for generated image
        mul1_gen = tf.matmul(generated_image, W1) + b1
        out1_gen = tf.nn.relu(mul1_gen)
        mul2_gen = tf.matmul(out1_gen, W2) + b2
        out2_gen = tf.nn.relu(mul2_gen)
        mul3_gen = tf.matmul(out2_gen, W3) + b3
        out3_gen = tf.nn.relu(mul3_gen)
        mul4_gen = tf.matmul(out3_gen, W4) + b4
        out4_gen = tf.nn.relu(mul4_gen)
        mul5_gen = tf.matmul(out4_gen, W5) + b5


        # Define op.

        org_activations = tf.concat([mul1_org, mul2_org, mul3_org, mul4_org, mul5_org], 1)
        gen_activations = tf.concat([mul1_gen, mul2_gen, mul3_gen, mul4_gen, mul5_gen], 1)

        susp_org_dense_activations = gather_cols(org_activations, suspicious_neuron_indices)
        susp_gen_dense_activations = gather_cols(gen_activations, suspicious_neuron_indices)

        diff_susp_dense = tf.subtract(susp_org_dense_activations, susp_gen_dense_activations)
        diff_susp_dense_sum = tf.reduce_sum(diff_susp_dense, 1)
        dense_custom_loss = tf.reduce_mean(diff_susp_dense_sum)

        normal_org_dense_activations = gather_cols(org_activations, normal_neuron_indices)
        normal_gen_dense_activations = gather_cols(gen_activations, normal_neuron_indices)

        diff_normal_dense = tf.abs(tf.subtract(normal_org_dense_activations, normal_gen_dense_activations))
        diff_normal_dense_sum = tf.reduce_sum(diff_normal_dense, 1)
        dense_custom_normal_loss = tf.reduce_mean(diff_normal_dense_sum)

        # parameters to regularize loss value between suspicious and non-suspicious losses
        theta = tf.constant(parameter_theta, dtype=tf.float32)
        gamma = tf.constant(parameter_gamma, dtype=tf.float32)

        dense_loss = tf.add(
            tf.multiply(
                theta, dense_custom_loss
            ),
            tf.multiply(
                gamma, dense_custom_normal_loss
            )
        )

        # parameter to regularize suspicious and MSE loss
        alpha = tf.constant(get_config()['custom_loss_constant'], dtype=tf.float32)

        mse = tf.reduce_mean(tf.squared_difference(generated_image, original_image))

        loss = tf.add(
            tf.math.multiply(alpha, dense_loss),
            tf.math.multiply(tf.subtract(tf.constant(1.0), alpha), mse))

        return  loss








