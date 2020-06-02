from suspicious_detection.CNNLayer import CNNLayer as CL
from suspicious_detection.DenseLayer import DenseLayer as DL
from suspicious_detection.FeatureMap import FeatureMap
from suspicious_detection.Layer import Layer
from suspicious_detection.Neuron import Neuron
from OriginalNetwork import *
from Helpers import *
import tensorflow as tf
import numpy as np
import pickle
import json
import os

# initialize layers
dense_layers, conv_layers = [], []


def setup(model):
    # num_conv_layers, num_dense_layers = calculate_layer_types(model= model)
    trainable_layers = get_trainable_layers(model= model)
    for trainable_layer in trainable_layers:
        layer_properties = get_layer_properties(model= model, layer= trainable_layer)
        if get_layer_type(model= model, layer_no= trainable_layer) == 'Conv2D':
            cl = CL(layer_id= trainable_layer, im_h= layer_properties[1], im_w= layer_properties[0], depth= layer_properties[2])
            conv_layers.append(cl)
        elif get_layer_type(model= model, layer_no= trainable_layer) == 'Dense':
            dl = DL(id= trainable_layer, num_neurons= layer_properties)
            dense_layers.append(dl)
        else:
            print('[setup]: Invalid layer type.')


def suspicious_analyze(model, class_):

    print('Suspicious Neuron Analyzing Process Started for class-{0}...'.format(class_))
    x_train, y_train, x_test, y_test = read_data(digit= class_)
    print('Data Loaded. Test set size for class-{0}: {1}'.format(class_, len(x_test)))

    for i in range(len(x_test)):
    #for i in range(301):
        sample = x_test[i]
        sample = sample.reshape(1, 784)
        label = np.argmax(y_test[i].reshape(1, 10))
        isPredicted = is_predicted(model= model, sample= sample, label= label)
        activations = get_activations(model= model, sample= sample)
        update_properties(model= model, predicted= isPredicted, acts= activations)
        if i % 100 == 0:
            print('Test sample ({0}) analyze completed.'.format(i))


    # save most suspicious feature maps (for convolutional layers)
    most_suspicious_k_feature_maps = find_suspicious_feature_maps(k=  config['k'])
    print('[suspicious_analyze]: most suspicious {0} feature maps: {1}'.format(
        config['k'], most_suspicious_k_feature_maps))
    save_name = 'suspicious_detection/suspicious_feature_maps/suspicious_feature_maps_{0}.pkl'.format(config['class'])
    with open(save_name, 'wb') as output:
        pickle.dump(most_suspicious_k_feature_maps, output, pickle.HIGHEST_PROTOCOL)

    # save most suspicious neurons (for dense layers)
    most_suspicious_k_neurons = find_suspicious_neurons(k= config['k'])
    print('[suspicious_analyze]: most suspicious {0} neurons: {1}'.format(
        config['k'], most_suspicious_k_neurons))
    save_name = 'suspicious_detection/suspicious_dense_neurons/suspicious_dense_neurons_{0}.pkl'.format(config['class'])
    with open(save_name, 'wb') as output:
        pickle.dump(most_suspicious_k_neurons, output, pickle.HIGHEST_PROTOCOL)


def update_properties(model, predicted, acts):
    threshold = config['activation_threshold']
    num_conv_layers, num_dense_layers = calculate_layer_types(model= model)
    # assumption: Convolutional layers come before dense layers.
    # Hence, the first members of acts array are from convolutional layers.

    # update properties of convolutional layers
    for i in range(num_conv_layers):
        layer_index = get_convolutional_layer_index(model= model, order= i)
        w, h, depth = get_layer_properties(model= model, layer= layer_index)
        layer_activations = acts[layer_index][0].reshape(1, h, w, depth)
        for  d in range(depth):
            fmap_acts = layer_activations[:, :, : , d].reshape(h, w)
            # print('fmap_acts type: {0}, shape: {1}'.format(type(fmap_acts), fmap_acts.shape))
            for j in range(h):
                for k in range(w):
                    is_active = fmap_acts[j][k] > threshold
                    if predicted:
                        if is_active:
                            conv_layers[i].update_neuron(depth= d, h= j, w= k, property= 'NCS')
                        else:
                            conv_layers[i].update_neuron(depth=d, h=j, w=k, property='NUS')
                    else:
                        if is_active:
                            conv_layers[i].update_neuron(depth=d, h=j, w=k, property='NCF')
                        else:
                            conv_layers[i].update_neuron(depth=d, h=j, w=k, property='NUF')

    # update properties of dense layers
    for i in range(num_dense_layers):
        layer_index = get_dense_layer_index(model=model, order= i)
        # n: number of neurons
        n = get_layer_properties(model= model, layer= layer_index)
        layer_activations = acts[layer_index][0].reshape(1, n)
        for j in range(n):
            is_active = layer_activations[0][j] > threshold
            if predicted:
                if is_active:
                    dense_layers[i].update_neuron(rank=j, property='NCS')
                else:
                    dense_layers[i].update_neuron(rank=j, property='NUS')
            else:
                if is_active:
                    dense_layers[i].update_neuron(rank=j, property='NCF')
                else:
                    dense_layers[i].update_neuron(rank=j, property='NUF')




def find_suspicious_feature_maps(k):
    # stores a list of most suspicious feature maps
    suspicious_feature_maps, most_suspicious_feature_maps = [], []
    num_convolution_layers = len(conv_layers)
    # collect all k-most suspicious feature maps from all convolution layers
    for i in range(num_convolution_layers):
        suspicious_feature_maps.extend(conv_layers[i].most_suspicious_fmaps(k= k))
    most_suspicious_feature_maps = sorted(suspicious_feature_maps, key=lambda x: x.average_score(), reverse=True)
    # this returns most suspicious k feature maps among all suspicious  ( k )
    # return most_suspicious_feature_maps[: k]
    # this return k-most suspicious feature maps of all convolution layers.  ( num_convolution_layer *  k )
    return suspicious_feature_maps


def find_suspicious_neurons(k):
    suspicious_neurons = []
    num_dense_layers = len(dense_layers)
    for i in range(num_dense_layers):
        suspicious_neurons.extend(dense_layers[i].most_suspicious_neurons(k= k))
    most_suspicious_neurons = sorted(suspicious_neurons, key=lambda x: x.tarantula(), reverse=True)
    #return suspicious_neurons
    return most_suspicious_neurons[:k]


if __name__ == '__main__':

    # go one step back in folder structure
    os.chdir('..')
    print('Working directory: {0}'.format(os.getcwd()))

    # load original model
    model = load_model('mnist_5x30')

    # read configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)


    setup(model= model)
    suspicious_analyze(model= model, class_= config['class'])







    # # choose k-most suspicious feature maps among collected feature maps
    # for i in range(len(suspicious_feature_maps)):
    #     f = suspicious_feature_maps[i]
    #     if len(most_suspicious_feature_maps) < k:
    #         most_suspicious_feature_maps.append(f)
    #     else:
    #         avg_scores = list(map(lambda x: x.average_score(), most_suspicious_feature_maps))
    #         while None in avg_scores:
    #             index = avg_scores.index(None)
    #             avg_scores[index] = 0.0
    #         min_score = np.min(avg_scores)
    #         score = f.average_score()
    #         if not score == None:
    #             if score > min_score:
    #                 index = avg_scores.index(min_score)
    #                 most_suspicious_feature_maps[index] = f

    ###############################################################
    # this will be deleted. Only for test purpose
    # all_avg_scores = list(map(lambda x: x.average_score(), suspicious_feature_maps))
    # print('All suspicious feature maps count: {0}.  Their average scores: {1}'.format(
    #     len(suspicious_feature_maps), all_avg_scores
    # ))
    # most_suspicious_avg_scores = list(map(lambda x: x.average_score(), most_suspicious_feature_maps))
    # print('Most suspicious feature maps count: {0}. Their average scores: {1}'.format(
    #     len(most_suspicious_feature_maps), most_suspicious_avg_scores
    # ))
    ##############################################################