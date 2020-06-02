#from Helpers import *
from keras.models import model_from_json
from keras import backend as K
import json
import numpy as np
import os


def load_model(model_name):
    with open('{0}.json'.format(model_name), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('{0}.h5'.format(model_name))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_trainable_layers(model, include_output_layer=False):
    trainable_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            trainable_layers.append(model.layers.index(layer))
        except:
            pass

    if not include_output_layer:
        trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers

def get_layer_outs(model, test_input):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  #  evaluation functions
    layer_outs = [func([test_input]) for func in functors]
    return layer_outs

def get_layer_weights(model, layer_no):
    return model.layers[layer_no].get_weights()

def predict(model, sample):
    prediction = model.predict_classes(sample)[0]
    return prediction

def is_predicted(model, sample, label):
    return predict(model, sample) == label

def get_activations(model, sample):
    trainable_layers = get_trainable_layers(model= model)
    layer_outs = get_layer_outs(model= model, test_input= sample)
    return layer_outs

def get_layer_type(model, layer_no):
    config = model.get_config()['layers']
    layer_config = config[layer_no]
    return layer_config['class_name']

def calculate_layer_types(model):
    conv_counter, dense_counter = 0, 0
    trainable_layers = get_trainable_layers(model=model)
    config = model.get_config()['layers']
    for trainable_layer in trainable_layers:
        layer_type = config[trainable_layer]['class_name']
        if layer_type == 'Conv2D':
            conv_counter += 1
        elif layer_type == 'Dense':
            dense_counter += 1
        else:
            print('[calculate_layer_types]: Invalid layer type.')
    return conv_counter, dense_counter

def get_layer_properties(model, layer):
    config = model.get_config()['layers']
    layer_type = config[layer]['class_name']
    activations = get_activations(model= model, sample= np.zeros((1, 32, 32, 3)))
    if layer_type == 'Conv2D':
        w = activations[layer][0].shape[1]
        h = activations[layer][0].shape[2]
        d = activations[layer][0].shape[3]
        return (w, h, d)
    elif layer_type == 'Dense':
        n = activations[layer][0].shape[1]
        return n
    else:
        print('[get_layer_properties]: Invalid layer type.')

def get_convolutional_layer_index(model, order):
    config = model.get_config()['layers']
    trainable_layers = get_trainable_layers(model= model)
    conv_layer_inds = []
    for index in trainable_layers:
        layer_type = config[index]['class_name']
        if layer_type == 'Conv2D':
            conv_layer_inds.append(index)
    return conv_layer_inds[order]

def get_dense_layer_index(model, order):
    #print('[get_dense_layer_index]: ..called..')
    config = model.get_config()['layers']
    trainable_layers = get_trainable_layers(model= model)
    dense_layer_inds = []
    for index in trainable_layers:
        layer_type = config[index]['class_name']
        if layer_type == 'Dense':
            dense_layer_inds.append(index)
    return dense_layer_inds[order]

def get_dense_neuron_index(model, layer_no, neuron_no):
    num_conv_layers, num_dense_layers = calculate_layer_types(model= model)
    #print('num dense layers: ', num_dense_layers)
    num_total_layers = len(model.layers)
    starting_dense_layer_index = num_total_layers - num_dense_layers - 1 # -1 for last output layer
    # count all number of dense layer neurons before given dense layer
    neuron_count = 0
    #print('num_dense_layers: ', num_dense_layers)
    #print('num_conv_layers: ', num_conv_layers)
    #print('num_total_layers: ', num_total_layers)
    #print('starting layer index: ', starting_dense_layer_index)
    end_index = layer_no
    #if starting_dense_layer_index == end_index:
    #    end_index += 1
    #print('starting index: {0},   end index: {1}'.format(
    #    starting_dense_layer_index, end_index
    #))
    for layer_index in range(starting_dense_layer_index, end_index):
        #print('layer_index: ', layer_index)
        #layer_index = get_dense_layer_index(model= model, order= i)
        num_neurons = get_layer_properties(model= model, layer= layer_index)
        #print('num_neurons: ', num_neurons)
        neuron_count += num_neurons

    #print('neuron_count: ', neuron_count)
    # add neuron_no on counted neurons
    neuron_index = neuron_count + neuron_no
    return neuron_index

#def total_number_of_dense_neurons(model):
#    total_number_of_neurons = 0
#    _, dense_counter = calculate_layer_types(model= model)
#    print('dense_counter: ', dense_counter)
#    for i in range(dense_counter - 1):  # -2 for last output layer and current layer
#        layer_index = get_dense_layer_index(model= model, order= i)
#        num_neurons = get_layer_properties(model= model, layer= layer_index)
#        total_number_of_neurons += num_neurons
#    return total_number_of_neurons



def total_number_of_dense_neurons(model):
    total_number_of_neurons = 0
    num_conv_layers, num_dense_layers = calculate_layer_types(model= model)
    print('num dense layers: ', num_dense_layers)
    num_total_layers = len(model.layers)
    starting_dense_layer_index = num_total_layers - num_dense_layers - 1
    end_index = num_total_layers - 1 # -1 for indexing, -2 for dropping output layer
    #print('starting index: {0},   end index: {1}'.format(
    #    starting_dense_layer_index, end_index
    #))
    for layer_index in range(starting_dense_layer_index, end_index):
        num_neurons = get_layer_properties(model=model, layer=layer_index)
        total_number_of_neurons += num_neurons
    return total_number_of_neurons
