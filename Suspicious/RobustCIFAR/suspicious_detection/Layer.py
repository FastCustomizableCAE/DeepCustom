
class Layer:

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.neurons = []


    def addNeuron(self, neuron):
        self.neurons.append(neuron)


    def numNeurons(self):
        return len(self.neurons)

    def __str__(self):
        return "Layer_{0}: consists of {1} neurons".format(self.layer_id, self.numNeurons())

    def getNeurons(self):
        return self.neurons

    def getNeuron(self, neuron_idx):
        return self.neurons[neuron_idx]

