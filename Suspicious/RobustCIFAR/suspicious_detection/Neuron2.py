import math

class Neuron2:

    def __init__(self, neuron_id, layer_no, neuron_no):
        self.neuron_id = neuron_id
        self.layer_no = layer_no
        self.neuron_no = neuron_no
        self.NCF = 0
        self.NUF = 0
        self.NUS = 0
        self.NCS = 0

    def get_layer(self):
        return self.layer_no

    def get_neuron_no(self):
        return self.neuron_no

    def addNCF(self):
        self.NCF += 1

    def addNUF(self):
        self.NUF += 1

    def addNUS(self):
        self.NUS += 1

    def addNCS(self):
        self.NCS += 1

    def getNCF(self):
        return self.NCF

    def getNUF(self):
        return self.NUF

    def getNUS(self):
        return self.NUS

    def getNCS(self):
        return self.NCS

    def getNF(self):
        return self.getNCF() + self.getNUF()

    def getNS(self):
        return self.getNUS() + self.getNCS()

    def tarantula(self):
        if not self.getNF() == 0 and not self.getNS() == 0:
            if not (self.getNCF() / self.getNF() + self.getNCS() / self.getNS()) == 0.0:
                return (self.getNCF() / self.getNF()) / (self.getNCF() / self.getNF() + self.getNCS() / self.getNS())
        else:
            return 0.0

    def ochiai(self):
        if not math.sqrt(self.getNF() * (self.getNCF() + self.getNCS())) == 0.0:
            return self.getNCF() / math.sqrt(self.getNF() * (self.getNCF() + self.getNCS()))
        else:
            return 0.0

    def __repr__(self):
        return 'Neuron(id: {0}, tarantula: {1})'.format(self.neuron_id, self.tarantula())

