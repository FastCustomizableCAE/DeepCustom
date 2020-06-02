import tensorflow as tf
import numpy as np
from Helpers import *
from OriginalNetwork import *

# MARK: Prerequisites for custom loss

# load original model and its weights
model = load_model('cifar_keras')
trainable_layer_indices = get_trainable_layers(model= model, include_output_layer=True)
W1, b1 = get_layer_weights(model=model, layer_no=trainable_layer_indices[0]) # conv
W2, b2 = get_layer_weights(model=model, layer_no=trainable_layer_indices[1]) # conv
W3, b3 = get_layer_weights(model=model, layer_no=trainable_layer_indices[2]) # conv
W4, b4 = get_layer_weights(model=model, layer_no=trainable_layer_indices[3]) # dense
W5, b5 = get_layer_weights(model=model, layer_no=trainable_layer_indices[4]) # output


# for convolutional layers
suspicious_feature_map_indices = suspicious_fmap_inds()
normal_feature_map_indices = normal_fmaps_inds()

# for dense layers
suspicious_neuron_indices = suspicious_neurons()
normal_neuron_indices = normal_neurons()

# custom loss parameter configurations (optional. related to the custom loss implemented here)
parameter_alpha =  0.5
parameter_beta =   1e-2
parameter_theta =  0.5
parameter_gamma =  1e-2


# defining suspicious custom loss
def custom_loss(original_image, generated_image, name='suspicious_custom_loss'):

    with tf.op_scope([original_image, generated_image], name, 'suspicious_custom_loss') as scope:

        susp_inds1 = suspicious_feature_map_indices[0]
        susp_inds2 = suspicious_feature_map_indices[1]
        susp_inds3 = suspicious_feature_map_indices[2]

        norm_inds1 = normal_feature_map_indices[0]
        norm_inds2 = normal_feature_map_indices[1]
        norm_inds3 = normal_feature_map_indices[2]

        original_image = tf.convert_to_tensor(original_image, name='original_image')
        generated_image = tf.convert_to_tensor(generated_image, name='generated_image')

        susp_inds1 = tf.convert_to_tensor(susp_inds1, name='susp_inds1')
        susp_inds2 = tf.convert_to_tensor(susp_inds2, name='susp_inds2')
        susp_inds3 = tf.convert_to_tensor(susp_inds3, name='susp_inds3')

        norm_inds1 = tf.convert_to_tensor(norm_inds1, name='norm_inds1')
        norm_inds2 = tf.convert_to_tensor(norm_inds2, name='norm_inds2')
        norm_inds3 = tf.convert_to_tensor(norm_inds3, name='norm_inds3')


        # simulate original network for original image
        mul1_org = conv(original_image, W1) + b1
        out1_org = tf.nn.relu(mul1_org)
        out1_org = max_pool2(out1_org)
        mul2_org = conv(out1_org, W2) + b2
        out2_org = tf.nn.relu(mul2_org)
        out2_org = max_pool2(out2_org)
        mul3_org = conv(out2_org, W3) + b3
        out3_org = tf.nn.relu(mul3_org)
        flatten_org = tf.reshape(out3_org, [-1, 4 * 4 * 64])
        mul4_org = tf.matmul(flatten_org, W4) + b4

        # simulate original network for generated image
        mul1_gen = conv(generated_image, W1) + b1
        out1_gen = tf.nn.relu(mul1_gen)
        out1_gen = max_pool2(out1_gen)
        mul2_gen = conv(out1_gen, W2) + b2
        out2_gen = tf.nn.relu(mul2_gen)
        out2_gen = max_pool2(out2_gen)
        mul3_gen = conv(out2_gen, W3) + b3
        out3_gen = tf.nn.relu(mul3_gen)
        flatten_gen = tf.reshape(out3_gen, [-1, 4 * 4 * 64])
        mul4_gen = tf.matmul(flatten_gen, W4) + b4

       # assertions
        try:
            susp_inds1.get_shape().assert_has_rank(2)
            susp_inds2.get_shape().assert_has_rank(2)
            susp_inds3.get_shape().assert_has_rank(2)
        except:
            raise ValueError('Suspicious Feature Map Indices Must Be In 2D.')

        # Define op.

        # convolutional custom loss
        susp_fmaps_org1 = tf.gather(mul1_org, susp_inds1, axis=3)
        susp_fmaps_org2 = tf.gather(mul2_org, susp_inds2, axis=3)
        susp_fmaps_org3 = tf.gather(mul3_org, susp_inds3, axis=3)

        susp_fmaps_gen1 = tf.gather(mul1_gen, susp_inds1, axis=3)
        susp_fmaps_gen2 = tf.gather(mul2_gen, susp_inds2, axis=3)
        susp_fmaps_gen3 = tf.gather(mul3_gen, susp_inds3, axis=3)

        diff1 = tf.subtract(susp_fmaps_org1, susp_fmaps_gen1)
        diff2 = tf.subtract(susp_fmaps_org2, susp_fmaps_gen2)
        diff3 = tf.subtract(susp_fmaps_org3, susp_fmaps_gen3)

        diff1_avg = tf.reduce_mean(tf.reduce_mean(diff1, axis= 1), axis= 1)
        diff2_avg = tf.reduce_mean(tf.reduce_mean(diff2, axis= 1), axis= 1)
        diff3_avg = tf.reduce_mean(tf.reduce_mean(diff3, axis= 1), axis= 1)

        diff_avg_sum1 = tf.reduce_mean(diff1_avg, 1)
        diff_avg_sum2 = tf.reduce_mean(diff2_avg, 1)
        diff_avg_sum3 = tf.reduce_mean(diff3_avg, 1)

        convolutional_custom_loss = tf.add_n([
            tf.reduce_sum(diff_avg_sum1),
            tf.reduce_sum(diff_avg_sum2),
            tf.reduce_sum(diff_avg_sum3)
        ])


        # convolutional custom normal loss
        norm_fmaps_org1 = tf.gather(mul1_org, norm_inds1, axis=3)
        norm_fmaps_org2 = tf.gather(mul2_org, norm_inds2, axis=3)
        norm_fmaps_org3 = tf.gather(mul3_org, norm_inds3, axis=3)

        norm_fmaps_gen1 = tf.gather(mul1_gen, norm_inds1, axis=3)
        norm_fmaps_gen2 = tf.gather(mul2_gen, norm_inds2, axis=3)
        norm_fmaps_gen3 = tf.gather(mul3_gen, norm_inds3, axis=3)

        diff1_norm = tf.abs(tf.subtract(norm_fmaps_org1, norm_fmaps_gen1))
        diff2_norm = tf.abs(tf.subtract(norm_fmaps_org2, norm_fmaps_gen2))
        diff3_norm = tf.abs(tf.subtract(norm_fmaps_org3, norm_fmaps_gen3))

        diff1_avg_norm = tf.reduce_mean(tf.reduce_mean(diff1_norm, axis=1), axis=1)
        diff2_avg_norm = tf.reduce_mean(tf.reduce_mean(diff2_norm, axis=1), axis=1)
        diff3_avg_norm = tf.reduce_mean(tf.reduce_mean(diff3_norm, axis=1), axis=1)

        diff_avg_sum1_norm = tf.reduce_mean(diff1_avg_norm, 1)
        diff_avg_sum2_norm = tf.reduce_mean(diff2_avg_norm, 1)
        diff_avg_sum3_norm = tf.reduce_mean(diff3_avg_norm, 1)

        convolutional_custom_normal_loss = tf.add_n([
            tf.reduce_sum(diff_avg_sum1_norm),
            tf.reduce_sum(diff_avg_sum2_norm),
            tf.reduce_sum(diff_avg_sum3_norm)
        ])


        susp_org_dense_activations = gather_cols(mul4_org, suspicious_neuron_indices)
        susp_gen_dense_activations = gather_cols(mul4_gen, suspicious_neuron_indices)

        diff_susp_dense = tf.subtract(susp_org_dense_activations, susp_gen_dense_activations)
        diff_susp_dense_sum = tf.reduce_sum(diff_susp_dense, 1)

        dense_custom_loss = tf.reduce_sum(diff_susp_dense_sum)

        normal_org_dense_activations = gather_cols(mul4_org, normal_neuron_indices)
        normal_gen_dense_activations = gather_cols(mul4_gen, normal_neuron_indices)

        diff_normal_dense = tf.abs(tf.subtract(normal_org_dense_activations, normal_gen_dense_activations))
        diff_normal_dense_sum = tf.reduce_sum(diff_normal_dense, 1)

        dense_custom_normal_loss = tf.reduce_mean(diff_normal_dense_sum)

        # parameters for optimal loss value
        alpha = tf.constant(parameter_alpha, dtype=tf.float32)
        beta = tf.constant(parameter_beta, dtype=tf.float32)
        theta = tf.constant(parameter_theta, dtype=tf.float32)
        gamma = tf.constant(parameter_gamma, dtype=tf.float32)

        convolutional_loss = tf.add(
            tf.multiply(
                alpha, convolutional_custom_loss
            ),
            tf.multiply(
                beta, convolutional_custom_normal_loss
            )
        )

        dense_loss = tf.add(
            tf.multiply(
                theta, dense_custom_loss
            ),
            tf.multiply(
                gamma, dense_custom_normal_loss
            )
        )

        suspiciousness_loss = tf.add(convolutional_loss, dense_loss)

        mse =  tf.reduce_mean(tf.squared_difference(generated_image, original_image))

        alpha = tf.constant(get_config()['custom_loss_constant'], dtype=tf.float32)

        loss = tf.add(
                tf.math.multiply(alpha, suspiciousness_loss),
                tf.math.multiply(tf.subtract(tf.constant(1.0), alpha), mse))

        return  loss








