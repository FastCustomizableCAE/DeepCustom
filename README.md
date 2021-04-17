# Improving Robustness of Deep Learning Systems with Fast and Customizable Adversarial Data Generation

For any questions, please contact via email: deepCustomCAE@gmail.com.


## Overview of Code

The code consists of seven sub-projects. We studied three custom loss functions (LRP, NCE, Suspiciousness) and experiment on MNIST and CIFAR datasets. Since DGN architecture differs based on dataset is used, we decided to split the project into sub-projects to avoid complexity. For each custom loss, we have two sub-projects, namely RobustMNIST and RobustCIFAR. Hence, we have six sub-projects related to CAE based adversarial data generation. One sub-project is for generating FGSM and PGD attack data and FGSM, PGD adversarial training using IBM Robustness Toolbox. 

Each of six CAE based adversarial data generation project, has same Python scripts. 

* `CustomLosses.py` : contains custom loss function that will be used as the loss function of DGN models
* `train.py` :  trains DGN models and save them into automatically generated Models folder.
* `generate_data.py` : generates adversarial training and adversarial test data using DGN models
* `adversarial_train.py` : applies adversarial training using DGN generated adversarial training data 
* `evaluateDGNRobust.py` : evaluates DGN robust model obtained after adversarial training
* `evaluate.py` : evaluates other robust models against eachother and against DGN attacks


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
* put **Additional Files/NCE/MNIST/Models** into **DeepCustom/NCE/RobustMNIST**
* put **Additional Files/NCE/MNIST/ign_robust_model** into **DeepCustom/NCE/RobustMNIST/Evaluation**
* put **Additional Files/NCE/CIFAR/Models** into **DeepCustom/NCE/RobustCIFAR**
* put **Additional Files/NCE/CIFAR/ign_robust_model** into **DeepCustom/NCE/RobustCIFAR/Evaluation**


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

     `python adversarial_train.py`

**Evaluating DGN-Robust, FGSM-Robust and PGD-Robust Networks**

* To evaluate DGN-Robust network, you can run the following command inside **Evalution** folder:

    `python evaluateDGNRobust.py`

* To evaluate FGSM and PGD robust networks and target model against FGSM, PGD and DGN generated adversarial test samples, run the following command inside **Evaluation** folder:

    `python evaluate.py`


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


## Training Configurations


### DGN Training Configurations 

<br/>

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/tables/dgn_training_table.PNG?raw=true" alt="drawing" width="400"/>


The table above shows training configurations of DGN models. All training parameters are determined experimentally to avoid training problems (e.g. underfitting, overfitting). All models use Adam optimizer and learning rate decay.

### Adversarial Training Configurations 

<br/>

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/tables/dgn_adversarial_training_table.PNG?raw=true" alt="drawing" width="400"/>

The table above shows training parameters of DGN adversarial training. Similar to DGN training configurations, training parameters are determined experimentally to overcome training challenges.

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/tables/other_attacks_adversarial_training_table.PNG?raw=true" alt="drawing" width="400"/>

The table above shows training parameters of FGSM and PGD adversarial trainings. The same training parameters are applied to all PGD attacks with different step sizes and random initializations.

## Performance of DGN Models Against Each Other

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/tables/1.png?raw=true" alt="drawing" width="500"/>

Table above shows the performances of DGN models with different custom losses against each other. All DGN models showed better performances against attacks of their own type (i.e. DGN attack with same custom loss) as expected. On MNIST, DGN (LRP) robust network showed better robustness against all DGN attacks. DGN (LRP) attack is successful against other DGN robust networks. The best attack result belongs to DGN (NCE) on DGN (Suspiciousness) robust network with **9.13%** robust test accuracy. On CIFAR, DGN (Suspiciousness) robust network showed better robustness against other DGN attacks. The attack performances of all DGN models against each other are very similar. 

## DGN Architectures

The DGN models used in data generation are convolutional autoencoders. Due to the difference in difficulty of generating data for MNIST and CIFAR datasets, we designed two different DGN architectures for MNIST and CIFAR. Both DGN models use 3x3 filters in their convolutional layers to encode data.  For decoding, transposed convolutional (i.e. deconvolutional) layers are used. Batch normalization is applied after each deconvolutional layer.

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/figures/2.png?raw=true" alt="drawing" width="600"/>

Figure above shows DGN architecture for MNIST dataset. In encoder part, it consists of three convolutional layers with stride 1 and ReLU activations. First two convolutional layers use the same padding (i.e. padding that causes the output as the same as the input) and the third one uses no padding. After each convolutional layer, there is a MaxPool layer which uses 2x2 pool size with stride 2. The size of latent representation is 3x3x128. In decoder part, three deconvolutional layers with 2x2 filters and no padding are used. The first deconvolutional layer uses stride of 3, while others use stride of 2. Finally, the output of the final deconvolutional layer is passed through sigmoid activation function to keep output values between 0-1 which are valid pixel bounds for MNIST. 

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/figures/3.png?raw=true" alt="drawing" width="600"/>

Figure above shows DGN architecture for CIFAR dataset. The architecture is deeper and has much more capacity in comparison to MNIST version. The encoding part has four convolutional layers with no padding where the first and the third have stride size of 2, and others have stride size of 1.  The size of latent representation is 5x5x128. Four deconvolutional layers with different stride and kernel sizes (i.e. filters) are used for decoding. Kernel sizes of 2,1,2,1 and strides of 3,4,4,3 are used, but no padding is used. 


## Target Model Architectures

**MNIST target model.** The model is a fully-connected dense network which consists of five dense layers with ReLU activations where each layer contains 30 neurons. The final output layer has 10 neurons. The model has 160 neurons and 27580 trainable parameters in total. The test accuracy of the model is 96.56%.

**CIFAR target model.** The model is a convolutional neural network, which contains three convolutional and two dense layers. Convolutional layers use 3x3 filters with stride 1 and no padding. There are MaxPool layers with 2x2 pool size after first two convolutional layers. After convolutional layers, there is a dense layer with 64 neurons and a final output layer with 10 neurons. ReLU activations are used in convolutional and dense layers. The model has 122570 trainable parameters in total and achieves 70.75% test set accuracy.

**Target Model Training Configurations.** The Adam optimizer with learning rate 1e-3 is used in the training of both target models. MNIST target model is trained with 20 epochs and batch size 128, while an epoch size of 10 with batch size 32 is used for the training of CIFAR target model.


## Training Configurations

We trained DGN models for data generation. Using adversarial training data generated by DGN models, we also apply adversarial training to target models. Similarly, adversarial training is employed using FGSM and PGD attacks. All training parameters are determined experimentally to avoid training problems (e.g. underfitting, overfitting). All models use Adam optimizer and learning rate decay. Details of parameters used in DGN training and adversarial training, can be found above in **Training Configurations** section.


## Suspiciousness Calculations

Finding suspicious neurons or feature maps is a problem-specific process that is required for suspiciousness custom loss. The first step to find suspicious neurons, is to obtain hit spectrum of each neuron. After finding hit spectrums, suspiciousness measures can be used to identify the most suspicious neurons. 

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/figures/4.png?raw=true" alt="drawing" width="500"/>

A neuron n is **active** when the result of the activation function on a given input t, φ(t,n), is above a predefined threshold (e.g. 0). When the output of the activation function is lower than the threshold, neuron n is **inactive**.

After calculating hit spectrum of each neuron n in a neural network N using Definition 1, suspiciousness measures can be used to identify the most suspicious neurons. In our work, we used Tarantula, which has the following formula;

<img src="https://github.com/FastCustomizableCAE/DeepCustom/blob/master/resources/figures/5.png?raw=true" alt="drawing" width="500"/>

 Top k suspicious neurons can be found after sorting neurons based on their Tarantula scores in descending order and selecting first k of them. 

 We consider working with **the most suspicious feature maps** instead of the most suspicious neurons for convolutional layers. Basically, we take the average Tarantula scores of neurons in a feature map to find its Tarantula score. 