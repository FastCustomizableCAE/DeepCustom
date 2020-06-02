from keras import datasets, layers, models
from keras.utils import np_utils
import matplotlib.pyplot as plt
from Helpers import *
from keras.models import model_from_json
import numpy as np


# load data
x_train, y_train,  x_test, y_test = read_all_data()

# load adversarial training data
x_train_adv = np.load('adversarial_train_data/adv_images.npy')
y_train_adv = np.load('adversarial_train_data/targets.npy')
x_train_org = np.load('adversarial_train_data/org_images.npy')
x_train_adv = x_train_adv.astype('float32')
x_train_org = x_train_org.astype('float32')

# load adversarial test data
x_test_adv = np.load('adversarial_test_data/adv_images.npy')
y_test_adv = np.load('adversarial_test_data/targets.npy')
x_test_org = np.load('adversarial_test_data/org_images.npy')
x_test_adv = x_test_adv.astype('float32')
x_test_org = x_test_org.astype('float32')

# reshaping for training
# reshaping for evaluation
x_test_adv = x_test_adv.reshape((len(x_test_adv), 784))
x_test_org = x_test_org.reshape((len(x_test_org), 784))
y_test_adv = y_test_adv.reshape((len(y_test_adv), 10))
x_train_adv = x_train_adv.reshape((len(x_train_adv), 784))
x_train_org = x_train_org.reshape((len(x_train_org), 784))
y_train_adv = y_train_adv.reshape((len(y_train_adv), 10))
y_train_adv = np.array(list(map(lambda  x: np.argmax(x), y_train_adv)))
y_test_adv = np.array(list(map(lambda  x: np.argmax(x), y_test_adv)))

print('min: ', np.min(x_test_adv - x_test_org))
print('max: ', np.max(x_test_adv - x_test_org))

# Model reconstruction from JSON file
with open('mnist_5x30.json', 'r') as f:
    original_model = model_from_json(f.read())

# Load weights into the new model
original_model.load_weights('mnist_5x30.h5')

original_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

label_test = np.array(list(map(lambda x: np.argmax(x), y_test)))
label_train = np.array(list(map(lambda x: np.argmax(x), y_train)))

print('Evaluating the original model...')

_, natural_test_acc = original_model.evaluate(x_test,  label_test, verbose=2)
_, natural_train_acc = original_model.evaluate(x_train,  label_train, verbose=2)
_, robust_test_acc = original_model.evaluate(x_test_adv,  y_test_adv, verbose=2)
_, robust_train_acc = original_model.evaluate(x_train_adv,  y_train_adv, verbose=2)


print('[Original Model]  Natural training acc: {0},   test acc: {1}'.format(natural_train_acc, natural_test_acc))
print('[Original Model]   Robust training acc: {0},   test acc: {1}'.format(robust_train_acc, robust_test_acc))

# Model reconstruction from JSON file
with open('mnist_5x30_robust.json', 'r') as f:
    robust_model = model_from_json(f.read())

# Load weights into the new model
robust_model.load_weights('mnist_5x30_robust.h5')

robust_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Evaluating the robust model...')

_, natural_test_acc_robust = robust_model.evaluate(x_test,  label_test, verbose=2)
_, natural_train_acc_robust = robust_model.evaluate(x_train,  label_train, verbose=2)
_, robust_test_acc_robust = robust_model.evaluate(x_test_adv,  y_test_adv, verbose=2)
_, robust_train_acc_robust = robust_model.evaluate(x_train_adv,  y_train_adv, verbose=2)

print('[Robust Model]  Natural training acc: {0},   test acc: {1}'.format(natural_train_acc_robust, natural_test_acc_robust))
print('[Robust Model]   Robust training acc: {0},   test acc: {1}'.format(robust_train_acc_robust, robust_test_acc_robust))