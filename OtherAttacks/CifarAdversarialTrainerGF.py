from AdversarialTrainerGF import AdversarialTrainerGF
from Helpers import *


class CifarAdversarialTrainerGF(AdversarialTrainerGF):

    def __init__(self, attack_type):
        super().__init__('cifar', attack_type)
        self.config = get_config()

    def graph(self):
        # network
        g = tf.Graph()
        with g.as_default():
            X = tf.placeholder(dtype=tf.float32, name='X', shape=[None, 32, 32, 3])
            X_adv = tf.placeholder(dtype=tf.float32, name='X_adv', shape=[None, 32, 32, 3])
            Y = tf.placeholder(dtype=tf.float32, name='Y', shape=[None, 10])

            out1_org, out1_gen = conv_layer_adv(input_org=X, input_gen=X_adv, shape=[3, 3, 3, 32], layer=1)
            out1_org, out1_gen = maxpool_layer_adv(input_org=out1_org, input_gen=out1_gen)
            out2_org, out2_gen = conv_layer_adv(input_org=out1_org, input_gen=out1_gen, shape=[3, 3, 32, 64], layer=2)
            out2_org, out2_gen = maxpool_layer_adv(input_org=out2_org, input_gen=out2_gen)
            out3_org, out3_gen = conv_layer_adv(input_org=out2_org, input_gen=out2_gen, shape=[3, 3, 64, 64], layer=3)
            flatten_org, flatten_gen = flatten_adv(input_org=out3_org, input_gen=out3_gen, shape=[-1, 1024])
            out4_org, out4_gen = dense_adv(input_org=flatten_org, input_gen=flatten_gen, shape=[1024, 64], layer=4)
            logits_org, logits_gen = dense_adv(input_org=out4_org, input_gen=out4_gen, shape=[64, 10], layer=5,
                                               output_layer=True)

            J_org = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_org, labels=Y))
            J_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_gen, labels=Y))

            # as suggested in the paper
            alpha = tf.constant(self.config['adv_ratio_cifar'])

            loss = tf.add(tf.multiply(alpha, J_org), tf.multiply(tf.subtract(tf.constant(1.0), alpha), J_gen))

            # no matter which logits used since weights are same.
            acc = accuracy(logits_org, Y)

            step = tf.train.AdamOptimizer(learning_rate=self.config['adv_learning_rate_cifar']).minimize(loss)
            saver = tf.train.Saver()

        return g, step, loss, X, X_adv, Y, acc, saver

    def train(self):
        g, step, loss, X, X_adv, Y, acc, saver = self.graph()
        super().adversarial_train(graph= g, tensors=(step, loss, X, X_adv, Y, acc, saver))
        super().save(saver= saver)

