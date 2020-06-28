'''
Based on paper: https://arxiv.org/pdf/1412.6572.pdf
'''

import numpy as np
import tensorflow as tf
from Helpers import *
import os

# load original data
x_train, y_train, x_test, y_test = read_all_data()

# load DGN generated data
x_train_dgn = np.load('adversarial_training_data/gen_data.npy')
x_test_dgn = np.load('adversarial_test_data/gen_data.npy')

configuration = get_config()
batch_size = configuration['adv_batch_size']
num_epochs = configuration['adv_num_epochs']
learning_rate = configuration['learning_rate']

# network
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(dtype=tf.float32, name='X', shape=[None, 32, 32, 3])
    X_adv = tf.placeholder(dtype=tf.float32, name='X_adv', shape=[None, 32, 32, 3])
    Y = tf.placeholder(dtype=tf.float32, name='Y', shape=[None, 10])

    out1_org, out1_gen = conv_layer_adv(input_org= X, input_gen= X_adv, shape=[3, 3, 3, 32], layer=1)
    out1_org, out1_gen = maxpool_layer_adv(input_org= out1_org, input_gen= out1_gen)
    out2_org, out2_gen = conv_layer_adv(input_org= out1_org, input_gen= out1_gen, shape=[3, 3, 32, 64], layer=2)
    out2_org, out2_gen = maxpool_layer_adv(input_org=out2_org, input_gen=out2_gen)
    out3_org, out3_gen = conv_layer_adv(input_org= out2_org, input_gen= out2_gen, shape=[3, 3, 64, 64], layer=3)
    flatten_org, flatten_gen = flatten_adv(input_org= out3_org, input_gen= out3_gen, shape=[-1, 1024])
    out4_org, out4_gen = dense_adv(input_org= flatten_org, input_gen= flatten_gen, shape=[1024, 64], layer=4)
    logits_org, logits_gen = dense_adv(input_org= out4_org, input_gen= out4_gen, shape=[64, 10], layer=5, output_layer=True)

    J_org = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_org, labels=Y))
    J_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_gen, labels=Y))

    # as suggested in the paper
    alpha = tf.constant(0.5)

    loss = tf.add(tf.multiply(alpha, J_org), tf.multiply(tf.subtract(tf.constant(1.0), alpha), J_gen))

    # no matter which logits used since weights are same.
    acc = accuracy(logits_org, Y)

    step = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss)
    saver = tf.train.Saver()

sess = tf.InteractiveSession(graph=g)
tf.global_variables_initializer().run()

# train
num_iter = len(x_train) // batch_size
for epoch in range(num_epochs):
    train_loss, test_loss = 0, 0
    for I in range(num_iter):
        # train batch
        x, y, x_gen = x_train[I * batch_size: (I + 1) * batch_size],  y_train[I * batch_size: (I + 1) * batch_size], x_train_dgn[I * batch_size: (I + 1) * batch_size]
        #x = x.reshape(batch_size, 784)
        #x_gen.reshape(batch_size, 784)
        # test batch
        random_test_inds = np.random.choice(len(x_test), batch_size)
        x_t, y_t, x_t_gen = x_test[random_test_inds], y_test[random_test_inds], x_test_dgn[random_test_inds]
        #x_t = x_t.reshape(batch_size, 784)
        #x_t_gen = x_t_gen.reshape(batch_size, 784)
        # train step
        _, loss_ = sess.run([step, loss], feed_dict={X: x, X_adv: x_gen, Y: y})
        # test loss
        loss_t = sess.run(loss, feed_dict={X: x_t, X_adv: x_t_gen, Y: y_t})
        # update
        train_loss += loss_
        test_loss += loss_t
    train_loss /= num_iter
    test_loss /= num_iter
    print('Epoch [{0}]:  Train Loss: {1},   Test Loss: {2}'.format(epoch, train_loss, test_loss))

print('\n')


batch_size = 100
num_iter = len(x_test) // batch_size
acc_total = 0
for i in range(num_iter):
    x, y = x_test[i*batch_size : (i+1)*batch_size], y_test[i*batch_size : (i+1)*batch_size]
    acc_total += sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})
print('Natural Test accuracy: {0}'.format(acc_total / num_iter))


batch_size = 100
num_iter = len(x_test_dgn) // batch_size
acc_total = 0
for i in range(num_iter):
    x, y = x_test_dgn[i*batch_size : (i+1)*batch_size], y_test[i*batch_size : (i+1)*batch_size]
    acc_total += sess.run(acc, feed_dict={X: x, X_adv: x, Y: y})
print('Robust Test accuracy: {0}'.format(acc_total / num_iter))


# save the new model
if not os.path.exists('dgnRobustModel/dgn_robust_model'):
    os.makedirs('dgnRobustModel')

saver.save(sess, 'dgnRobustModel/dgn_robust_model')
