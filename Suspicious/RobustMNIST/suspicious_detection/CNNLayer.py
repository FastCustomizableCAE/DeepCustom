from suspicious_detection.FeatureMap import FeatureMap
import numpy as np

class CNNLayer:

    def __init__(self, layer_id, im_h, im_w, depth):
        self.layer_id = layer_id
        self.im_h = im_h
        self.im_w = im_w
        self.depth = depth
        self.fmaps = []
        self.setup()


    def setup(self):
        # fill out feature maps with neurons
        for d in range(self.depth):
            self.fmaps.append(FeatureMap(self.layer_id, d, self.im_h, self.im_w))


    def update_neuron(self, depth, h, w, property):
        self.fmaps[depth].update_neuron(h, w, property)


    def get_fmap(self, at_depth):
        return self.fmaps[at_depth]


    # def most_suspicious_neurons(self, k):
    #     # each feature map gives k most suspicious neurons of it
    #     most_suspicious_neurons = []
    #     for d in range(self.depth):
    #         most_suspicious_neurons.extend(self.get_fmap(at_depth= d).most_suspicious_neurons(k= k))
    #     most_suspicious_k_neurons = []
    #     for i in range(len(most_suspicious_neurons)):
    #         if len(most_suspicious_k_neurons) < k:
    #             most_suspicious_k_neurons.append(most_suspicious_neurons[i])
    #         else:
    #             scores = list(map(lambda x: x.tarantula(), most_suspicious_k_neurons))
    #             while None in scores:
    #                 index = scores.index(None)
    #                 scores[index] = 0.0
    #             min_score = np.min(scores)
    #             n = most_suspicious_neurons[i]
    #             score = n.tarantula()
    #             if not score == None:
    #                 if n.tarantula() > min_score:
    #                     index = scores.index(min_score)
    #                     most_suspicious_k_neurons[index] = n
    #     return most_suspicious_k_neurons


    def most_suspicious_fmaps(self, k):
        # each feature map gives average metric score
        most_suspicious_feature_maps = []
        for d in range(self.depth):
            current_fmap = self.get_fmap(at_depth=d)
            if len(most_suspicious_feature_maps) < k:
                most_suspicious_feature_maps.append(current_fmap)
            else:
                avg_scores = list(map(lambda x: x.average_score(), most_suspicious_feature_maps))
                while None in avg_scores:
                    index = avg_scores.index(None)
                    avg_scores[index] = 0.0
                min_score = np.min(avg_scores)
                score = current_fmap.average_score()
                if not score == None:
                    if score > min_score:
                        index = avg_scores.index(min_score)
                        most_suspicious_feature_maps[index] = current_fmap
        return most_suspicious_feature_maps


    def __repr__(self):
        return 'CL(id:{0}, im_h:{1},  im_w:{2},  depth:{3})'\
            .format(self.layer_id, self.im_h, self.im_w, self.depth)



















