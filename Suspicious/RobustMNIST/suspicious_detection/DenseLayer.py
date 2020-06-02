from suspicious_detection.Neuron import Neuron
import numpy as np

class DenseLayer:

    def __init__(self, id, num_neurons):
        self.id = id
        self.num_neurons = num_neurons
        self.neurons = []
        self.setup()


    def setup(self):
        for i in range(self.num_neurons):
            # ND_1_10:  10th neuron in dense layer-1
            neuron_id = 'ND_{0}_{1}'.format(self.id, i)
            self.neurons.append(Neuron(neuron_id= neuron_id, layer_no= self.id, neuron_no= i))

    def most_suspicious_neurons(self, k):
        most_suspicious_k_neurons = []
        for i in range(len(self.neurons)):
            n = self.neurons[i]
            if len(most_suspicious_k_neurons) < k:
                most_suspicious_k_neurons.append(n)
            else:
                scores = list(map(lambda x: x.tarantula(), most_suspicious_k_neurons))
                while None in scores:
                    index = scores.index(None)
                    scores[index] = 0.0
                min_score = np.min(scores)
                score = n.tarantula()
                if score == None:
                    score = 0.0
                if score > min_score:
                    index = scores.index(min_score)
                    most_suspicious_k_neurons[index] = n
        return most_suspicious_k_neurons

    def update_neuron(self, rank, property):
        if property == 'NCS':
            self.neurons[rank].addNCS()
        elif property == 'NUS':
            self.neurons[rank].addNUS()
        elif property == 'NCF':
            self.neurons[rank].addNCF()
        elif property == 'NUF':
            self.neurons[rank].addNUF()
        else:
            print('Invalid property given.')