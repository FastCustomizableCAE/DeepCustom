from Helpers import *
import tensorflow as tf
import numpy as np
import logging
import os

logging.basicConfig(level=logging.DEBUG)



if __name__ == '__main__':

    logging.info('[adversarial_train] Adversarial Training process is started.')

    # read configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)
    num_classes = config['num_classes']
    model_name = config['model_name']
    mis_classification_check= config['mis_classification_check']
    adversarial_train_num_epochs = config['adv_num_epochs']
    adversarial_train_batch_size = config['adv_batch_size']

    logging.info('[adversarial_train]: Configurations are loaded.')

    model = load_model(model_name)
    model.summary()


    logging.info('[adversarial_train]: Original model is loaded.')

    if not os.path.exists('adversarial_training_data') or not os.path.exists('adversarial_test_data'):
        logging.error('[adversarial_train]: Adversarial training or test folder does not exist.')
        raise FileExistsError('[adversarial_train]: Adversarial training or test folder does not exist.')

    if mis_classification_check:
        x_org_train = np.load('adversarial_training_data/org_data.npy')
        x_gen_train = np.load('adversarial_training_data/gen_data.npy')
        y_train = np.load('adversarial_training_data/target.npy')
        x_org_test = np.load('adversarial_test_data/org_data.npy')
        x_gen_test = np.load('adversarial_test_data/gen_data.npy')
        y_test = np.load('adversarial_test_data/target.npy')
    else:
        x_org_train, y_train, x_org_test, y_test = read_all_data()
        x_gen_train = np.load('adversarial_training_data/gen_data.npy')
        x_gen_test = np.load('adversarial_test_data/gen_data.npy')

    y_test = np.array(list(map(lambda x: np.argmax(x), y_test)))
    y_train = np.array(list(map(lambda x: np.argmax(x), y_train)))

    # evaluations before continue adversarial training
    test_loss, test_acc = model.evaluate(x_org_test, y_test, verbose=2)
    robust_test_loss, robust_test_acc = model.evaluate(x_gen_test, y_test, verbose=2)
    logging.info('[adversarial_train]: natural test accuracy : {0}, before adversarial train.'.format(test_acc))
    logging.info('[adversarial_train]: robust test accuracy : {0}, before adversarial train.'.format(robust_test_acc))

    # continue to training with adversarial samples
    model.fit(x_gen_train, y_train,
              epochs=adversarial_train_num_epochs,
              batch_size=adversarial_train_batch_size)

    # evaluations after adversarial training
    _, test_acc = model.evaluate(x_org_test, y_test, verbose=2)
    _, robust_test_acc = model.evaluate(x_gen_test, y_test, verbose=2)
    logging.info('[adversarial_train]: natural test accuracy : {0}, after adversarial train.'.format(test_acc))
    logging.info('[adversarial_train]: robust test accuracy : {0}, before adversarial train.'.format(robust_test_acc))

    # save the new model
    model.save('mnist_5x30_robust.h5')
    model.save_weights('mnist_5x30_robust.h5')
    with open('mnist_5x30_robust.json', 'w') as f:
        f.write(model.to_json())

    logging.info('[adversarial_train]: robust model is saved.')
