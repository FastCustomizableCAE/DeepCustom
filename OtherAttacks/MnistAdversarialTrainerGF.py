from AdversarialTrainerGF import AdversarialTrainerGF
from Helpers import *


class MnistAdversarialTrainerGF(AdversarialTrainerGF):

    def __init__(self, attack_type):
        super().__init__('mnist', attack_type)
        self.config = get_config()

    def graph(self):
        # network
        g = tf.Graph()
        with g.as_default():
            X = tf.placeholder(dtype=tf.float32, name='X', shape=[None, 28 * 28])
            X_adv = tf.placeholder(dtype=tf.float32, name='X_adv', shape=[None, 28 * 28])
            Y = tf.placeholder(dtype=tf.float32, name='Y', shape=[None, 10])

            out1_org, out1_gen = dense_adv(input_org=X, input_gen=X_adv, shape=[28 * 28, 30], layer=1)
            out2_org, out2_gen = dense_adv(input_org=out1_org, input_gen=out1_gen, shape=[30, 30], layer=2)
            out3_org, out3_gen = dense_adv(input_org=out2_org, input_gen=out2_gen, shape=[30, 30], layer=3)
            out4_org, out4_gen = dense_adv(input_org=out3_org, input_gen=out3_gen, shape=[30, 30], layer=4)
            out5_org, out5_gen = dense_adv(input_org=out4_org, input_gen=out4_gen, shape=[30, 30], layer=5)
            logits_org, logits_gen = dense_adv(input_org=out5_org, input_gen=out5_gen, shape=[30, 10], layer=6,
                                               output_layer=True)

            J_org = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_org, labels=Y))
            J_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_gen, labels=Y))

            # as suggested in the paper
            alpha = tf.constant(self.config['adv_ratio_mnist'])

            loss = tf.add(tf.multiply(alpha, J_org), tf.multiply(tf.subtract(tf.constant(1.0), alpha), J_gen), name='loss')

            # no matter which logits used since weights are same.
            acc = accuracy(logits_org, Y)

            step = tf.train.AdamOptimizer(learning_rate=self.config['adv_learning_rate_mnist']).minimize(loss, name='step')
            saver = tf.train.Saver(name='saver')
        return g, step, loss, X, X_adv, Y, acc, saver

    def train(self):
        g, step, loss, X, X_adv, Y, acc, saver = self.graph()
        super().adversarial_train(graph= g, tensors=(step, loss, X, X_adv, Y, acc, saver))
        #super().save(saver= saver)

