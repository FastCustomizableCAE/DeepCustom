import logging
from fgsm_attacker import FgsmAttacker
from pgd_attacker import PgdAttacker
from MnistAdversarialTrainerGF import MnistAdversarialTrainerGF
from CifarAdversarialTrainerGF import CifarAdversarialTrainerGF
from Helpers import save_model, get_config, save_generated_data
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    elapsed_time_information = []

    config = get_config()

    # get configurations
    attacks, datasets, mode = config['attacks'], config['datasets'], config['mode']

    for attack in attacks:
        if not attack in ['fgsm', 'pgd']:
            logging.error('[Main]: Given attack is not supported in the project. Please give fgsm, pgd or both.')
        else:
            for dataset in datasets:
                if not dataset in ['mnist', 'cifar']:
                    logging.error('[Main]: Given dataset is not supported in the project. Please give mnist, cifar or both.')
                else:
                    if not mode in ['only_adv_training', 'only_attack', 'both']:
                        logging.error('[Main]: Given mode is not supported in the project. Please give adv_training, only_attack or "both".')
                    else:
                        if not mode == 'only_adv_training':
                            if attack == 'fgsm':
                                attacker = FgsmAttacker(dataset=dataset)
                            else:
                                attacker = PgdAttacker(dataset=dataset)
                            logging.info('[Main]: Attacking with {0} on {1} network...'.format(attack, dataset))
                            attack_model, x_train_adv, x_test_adv, elapsed_time = attacker.attack()
                            elapsed_time_information.append('Generating {0} {1} test data using {2} attack takes {3} secs.'.format(
                                len(x_test_adv), dataset, attack, elapsed_time))
                            attack_type = attack
                            if attack == 'pgd':
                                attack_type = 'pgd_step{0}'.format(config['max_iterations_pgd_{0}'.format(dataset)])
                            save_generated_data(generated_data=x_train_adv, attack_type= attack_type, dataset=dataset,set_type='train')
                            save_generated_data(generated_data= x_test_adv, attack_type= attack_type, dataset= dataset, set_type='test')
                            if not mode == 'only_attack' and config['adv_training_type'] == 'default':  # use art adversarial training
                                logging.info('[Main]: Default adversarial training with {0} on {1} network...'.format(attack, dataset))
                                robust_model = attacker.adversarial_train(attack=attack_model)
                                save_model(model=robust_model._model, dataset=dataset, model_type='robust',
                                           attack_type=attack)

                        if not mode == 'only_attack' and config['adv_training_type'] == 'gf':
                            logging.info('[Main]: GF adversarial training with {0} on {1} network...'.format(attack,dataset))
                            # use adversarial training technique applied in ign (gf)
                            if dataset == 'mnist':
                                MnistAdversarialTrainerGF(attack).train()
                            else:
                                CifarAdversarialTrainerGF(attack).train()


    # prints time elapsed information for generating adversarial test data
    for info in elapsed_time_information:
        print(info)



