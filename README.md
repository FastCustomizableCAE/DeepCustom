# Fast and Customizable Adversarial Data Generation Using Convolutional Autoencoders

For any questions, please contact via email: deepCustomCAE@gmail.com.


## Overview of Code

The code consists of seven sub-projects. We studied three custom loss functions (LRP, NCE, Suspiciousness) and experiment on MNIST and CIFAR datasets. Since DGN architecture differs based on dataset is used, we decided to split the project into sub-projects to avoid complexity. For each custom loss, we have two sub-projects, namely RobustMNIST and RobustCIFAR. Hence, we have six sub-projects related to CAE based adversarial data generation. One sub-project is for generating FGSM and PGD attack data and FGSM, PGD adversarial training using IBM Robustness Toolbox. 

Each of six CAE based adversarial data generation project, has same Python scripts. 

* `CustomLosses.py` : contains custom loss function that will be used as the loss function of DGN models
* `train.py` :  trains DGN models and save them into automatically generated Models folder.
* `generate_data.py` : generates adversarial training and adversarial test data using DGN models
* `adversarial_train_goodfellow.py` : applies adversarial training using DGN generated adversarial training data 
* `Evaluation/evaluateGoodfellowAT.py` : evaluates DGN robust model obtained after adversarial training
* `Evaluation/evaluate2.py` : evaluates other robust models against eachother and against DGN attacks


## Configurations

Each of the six sub-projets, contains a configuration file called `config.json`. One can change DGN training, evalution, adversarial training and evaluation parameters from `config.json` file. Morever, problem-specfic parameters can be changed using the same file. Except problem specific configurations, the following parameters are common in configuration file of each sub-projects:

* `model_name` : the name of target model 

DGN training configurations:

* `learning_rate` : learning rate for DGN training
* `decay_rate` : learning rate decay rate for DGN training (if used)
* `momentum` : momentum value for DGN training (if used)
* `batch_size` : batch size for DGN training
* `num_epochs` : epoch size for DGN training
* `weight_decay` : weight decay for DGN training (if used)
* `custom_loss_constant` : the main parameter that regularizes custom loss used
* `num_classes` : the number of classes in the dataset


Adversarial Attack and Data Generation configurations:

* `epsilon` : the maximum L_infinity distance between original and generated samples
* `mis_classification_check` : a boolean value. If it is set to true, only generated samples which is misclassified by the target model will be included in adversarial training dataset. (default: false)
* `clipping` : a boolean value. If it is set to true, pixel clipping will be applied to keep perturbations in L_infinity norm-ball. (default: true)

Adversarial Training configurations:

* `adv_num_epochs` : epoch size for adversarial training
* `adv_batch_size` : batch size for adversarial training
* `adv_learning_rate` : learning rate for adversarial training
* `adv_decay_step` : learning rate decay step for adversarial training (if used)
* `adb_decay_rate` : learning rate decay rate for adversarial training (if used)
* `adb_momentum` : momentum value for adversarial training (if used)
* `adb_weight_decay` : weight decay value for adversarial training (if used)

Evalation configurations:

* `attack_pgd_step` : step size of PGD attack that will be used in evaluation.
* `defense_pgd_step` : step size of PGD attack that is used for PGD adversarial training.
* `other_attacks_adv_training_type` : type of adversarial training. (default: gf which is Goodfellow's adversarial training) 


## Additional Resources

For additional resources including target, DGN and robust models, please download "Additional Files.zip" file from the following link:
[Additional Files](https://drive.google.com/file/d/1P4fXZ-g3gIrWuXmMJknrLFg7pkjR49rS/view?usp=sharing).

DGN models can be found under **Models** folder of each project. For instance, LRP-MNIST sub-project (DeepCustom/LRP) should contain a **Models** folder which can be found in (Additional Files/LRP/MNIST). There are 10 DGN models for each sub-project since number of classes for MNIST and CIFAR is 10. DGN Robust model can be found under **ign_robust_model** folder of each sub-project. LRP and Suspiciousness folder also contain problem-specific properties. For LRP, **relevent_pixels**  and for Suspiciosness **suspicious_detection** folders gives problem-specific properties of each project which are used in custom loss functions. 


**TargetModels** folder contains MNIST and CIFAR target models that trained using Keras. Finally, **OtherAttacks** folder contains FGSM and PGD generated data and robust models, which are obtained using IBM Robustness Toolbox. In other attacks, pgd_step20 represents PGD attack with 10 random restart and 20 steps. pgd_step2000 represent PGD attacks with 20 steps and 1 random restart.  If you do not want to train DGN models or run FGSM and PGD attacks again, please put folders inside  **Additional Files** folder into suitable places in the main project folder. Here is a guideline to do this:

For LRP,
* put **Additional Files/LRP/MNIST/Models** into **DeepCustom/LRP/RobustMNIST**
* put **Additional Files/LRP/MNIST/ign_robust_model** into **DeepCustom/LRP/RobustMNIST/Evaluation**
* put **Additional Files/LRP/CIFAR/Models** into **DeepCustom/LRP/RobustCIFAR**
* put **Additional Files/LRP/CIFAR/ign_robust_model** into **DeepCustom/LRP/RobustCIFAR/Evaluation**


For NCE,
* put **Additional Files/NCE/MNIST/Models** into **DeepCustom/OriginalLoss/RobustMNIST**
* put **Additional Files/NCE/MNIST/ign_robust_model** into **DeepCustom/OriginalLoss/RobustMNIST/Evaluation**
* put **Additional Files/NCE/CIFAR/Models** into **DeepCustom/OriginalLoss/RobustCIFAR**
* put **Additional Files/NCE/CIFAR/ign_robust_model** into **DeepCustom/OriginalLoss/RobustCIFAR/Evaluation**


For Suspiciousness,
* put **Additional Files/Suspiciousness/MNIST/Models** into **DeepCustom/Suspiciousness/RobustMNIST**
* put **Additional Files/Suspiciousness/MNIST/ign_robust_model** into **DeepCustom/Suspiciousness/RobustMNIST/Evaluation**
* put **Additional Files/Suspiciousness/CIFAR/Models** into **DeepCustom/Suspiciousness/RobustCIFAR**
* put **Additional Files/Suspiciousness/CIFAR/ign_robust_model** into **DeepCustom/Suspiciousness/RobustCIFAR/Evaluation**


For OtherAttacks,
* put **Additional Files/OtherAttacks/generated_data** into **DeepCustom/OtherAttacks**
* put **Additional Files/OtherAttacks/robust_models** into **DeepCustom/OtherAttacks**



## Example Usage

You can use Anaconda or Pip to create a virtual environments using `requirements.txt` file provided in the project folder.  After cloning this repository, you can move into one of sub-projects (e.g. LRP/RobustMNIST). At this points, you can train new DGN networks or you can evaluate our pre-trained DGN networks which can be found in [Additional Files](https://drive.google.com/file/d/1P4fXZ-g3gIrWuXmMJknrLFg7pkjR49rS/view?usp=sharing).

**Training DGN Networks**

* Run the following command to start training DGN networks:

    `python train.py`


**Generating adversarial data using trained DGN Networks**

* After training DGN networks or download pre-trained DGN networks, you can run the following command to generate adversarial training and adversarial test data:

    `python generate_data.py`

**Applying adversarial training to obtain robust networks using DGN generated adversarial data**

* After generating adversarial training and test data, you can run the following command to apply adversarial training technique offered in [1] to obtain a robust network (DGN-Robust network) which has the same architecture as the target model:

    `python adversarial_training_goodfellow.py`

**Evaluating DGN-Robust, FGSM-Robust and PGD-Robust Networks**

* To evaluate DGN-Robust network, you can run the following command inside **Evalution** folder:

    `python evaluteGoodfellowAT.py`

* To evaluate FGSM and PGD robust networks and target model against FGSM, PGD and DGN generated adversarial test samples, run the following command inside **Evaluation** folder:

    `python evalute2.py`


[1] Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.


## Generating FGSM and PGD generated adversarial samples and robust models

If you want to generate FGSM and PGD generated adversarial samples and FGSM, PGD robust models, instead of using already generated samples and pre-trained models in [Additional Files](https://drive.google.com/file/d/1P4fXZ-g3gIrWuXmMJknrLFg7pkjR49rS/view?usp=sharing), you can use **OtherAttacks** sub-project. The sub-project has a configuration `config.json` file which has the following parameters:

Running Configurations

* `attacks` : a list of attacks that will be used. Valid elements for the list: 'fgsm' and 'pgd'.
* `datasets` : a list of datasets that will be used.  Valid elements for the list: 'mnist' and 'cifar'.
* `adv_training_type` : the adversarial training technique that will be used. Valid parameters are 'gf' or 'default'. When the parameter is 'gf', the adversarial training used in DGNs, is also used in FGSM and PGD adversarial training. When it is 'default', it uses the default adversarial training of IBM adversarial robustness toolbox.
* `mode` : one of 'only_adv_training', 'only_attack' and 'both'. 'only_adv_training' applies only adversarial training, 'only_attack' generates adversarial samples but does not apply adversarial training. 'both' generates adversarial data and then applies adversarial training using generated adversaial samples.

Original Model Configurations:

* `mnist_model_name` : the model name of MNIST target model
* `cifar_model_name` : the model name of CIFAR target model

Attack Configurations

* `epsilon_mnist` : the maximum L_infinity distance between original and generated samples for MNIST
* `epsilon_cifar` : the maximum L_infinity distance between original and generated samples for CIFAR
* `eps_steps_pgd_mnist` : PGD attack step size (input variation) at each iteration for MNIST
* `eps_steps_pgd_cifar` :  PGD attack step size (input variation) at each iteration for CIFAR
* `num_random_init_pgd_mnist` : number of PGD random initialisations within the epsilon ball for MNIST
* `num_random_init_pgd_cifar` : number of PGD random initialisations within the epsilon ball for CIFAR
* `max_iterations_pgd_mnist` : the maximum number of PGD iterations for MNIST
* `max_iterations_pgd_cifar` : the maximum number of PGD iterations for CIFAR

Adversarial Training Configurations

* `adv_ratio_mnist` : a parameter that balances normal and adversarial losses for MNIST adversarial training
* `adv_ratio_cifar` : a parameter that balances normal and adversarial losses for CIFAR adversarial training
* `adv_num_epochs_mnist` : number of epochs for MNIST adversarial training
* `adv_batch_size_mnist` : batch size for MNIST adversarial training
* `adv_num_epochs_cifar_fgsm` : number of epochs for CIFAR FGSM adversarial training
* `adv_num_epochs_cifar_pgd` : number of epochs for CIFAR PGD adversarial training
* `adv_batch_size_cifar` : batch size for CIFAR adversarial training
* `adv_learning_rate_mnist` : learning rate for MNIST adversarial training
* `adv_learning_rate_cifar` : learning rate for CIFAR adversarial training


### Running the code

After setting desired configurations, you can run the code using following command:

`python Main.py`
