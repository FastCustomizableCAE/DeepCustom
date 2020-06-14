# Fast and Customizable Adversarial Data Generation Using Convolutional Autoencoders

This page will continue to be updated. For any questions, please contact via email: deepCustomCAE@gmail.com.




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

It will be available soon.