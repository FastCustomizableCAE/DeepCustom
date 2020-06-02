from suspicious_detection.Neuron import  Neuron
import numpy as np


class FeatureMap:


    def __init__(self, layer, id, im_h, im_w):
        self.layer = layer
        self.id = id
        self.im_h = im_h
        self.im_w = im_w
        self.neurons = []
        self.setup()


    def get_layer(self):
        return self.layer

    def get_depth(self):
        return self.id

    def setup(self):
        # print(self.im_h, self.im_w)
        for i in range(self.im_h):
            neurons_in_row = []
            for j in range(self.im_h):
                # NC_1_10_15_20: Neuron (15,20) at F.Map of depth 10 at conv layer-1
                neuron_id = 'NC_{0}_{1}_{2}_{3}'.format(self.layer, self.id, i, j)
                # neuron_no is normally (i, j), but we give only "i" index, but this does not effect calculations for
                # convolutional layers. For dense layers, it will be important.
                neurons_in_row.append(Neuron(neuron_id= neuron_id, layer_no= self.layer, neuron_no= i))
            self.neurons.append(neurons_in_row)

    def update_neuron(self, h, w, property):
        if property == 'NCS':
            self.neurons[h][w].addNCS()
        elif property == 'NUS':
            self.neurons[h][w].addNUS()
        elif property == 'NCF':
            self.neurons[h][w].addNCF()
        elif property == 'NUF':
            self.neurons[h][w].addNUF()
        else:
            print('Invalid property given.')


    def most_suspicious_neurons(self, k):
        suspicious_neurons_list = []
        for i in range(self.im_h):
            # self.print_scores(suspicious_neurons_list)
            for j in range(self.im_h):
                n = self.neurons[i][j]
                if len(suspicious_neurons_list) < k:
                    suspicious_neurons_list.append(n)
                else:
                    suspicious_neurons_list = self.find_replace(n, suspicious_neurons_list)
        return suspicious_neurons_list


    def average_score(self):
        total_sum = 0
        for i in range(self.im_h):
            for j in range(self.im_h):
                score = self.neurons[i][j].tarantula()
                if score == None:
                    score = 0.0
                total_sum += score
        average = total_sum / (self.im_h * self.im_h)
        return average

    def find_replace(self, neuron, suspicious_neurons_list):
        value = neuron.tarantula()
        if value == None:
            value = 0.0
        scores = list(map(lambda x: x.tarantula(), suspicious_neurons_list))
        # convert None's to 0.0
        while None in scores:
            index = scores.index(None)
            scores[index] = 0.0
        # print(scores)
        min_score = np.min(scores)
        if value > min_score:
            index = scores.index(min_score)
            suspicious_neurons_list[index] = neuron
        return suspicious_neurons_list


    def print_scores(self, suspicious_neurons_list):
        scores = list(map(lambda x: x.tarantula(), suspicious_neurons_list))
        print(scores)




    def __repr__(self):
        return 'FMap(layer:{0}, id:{1}, im_h:{2},  im_w:{3}, avg_score:{4})'\
            .format( self.layer, self.id, self.im_h, self.im_w, self.average_score())








