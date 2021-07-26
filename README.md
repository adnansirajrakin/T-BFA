This repository contains the code for Targeted Bit-Flip Attack (T-BFA).


## Description
Traditional Deep Neural Network (DNN) security is mostly related to the well-known adversarial input example attack. Recently, another dimension of adversarial attack, namely, attack on DNN weight parameters, has been shown to be very powerful. As a representative one, the Bit-Flip-based adversarial weight Attack (BFA) injects an extremely small amount of faults into weight parameters to hijack the executing DNN function. Prior works of BFA focus on un-targeted attack that can hack all inputs into a random output class by flipping a very small number of weight bits stored in computer memory. This paper proposes the first work of targeted BFA based (T-BFA) adversarial weight attack on DNNs, which can intentionally mislead selected inputs to a target output class. The objective is achieved by identifying the weight bits that are highly associated with classification of a targeted output through a class-dependent weight bit ranking algorithm. Our proposed T-BFA performance is successfully demonstrated on multiple DNN architectures for image classification tasks. For example, by merely flipping 27 out of 88 million weight bits of ResNet-18, our T-BFA can misclassify all the images from 'Hen' class into 'Goose' class (i.e., 100% attack success rate) in ImageNet dataset, while maintaining 59.35% validation accuracy. Moreover, we successfully demonstrate our T-BFA attack in a real computer prototype system running DNN computation, with Ivy Bridge-based Intel i7 CPU and 8GB DDR3 memory. Link to the paper: https://arxiv.org/pdf/2007.12336.pdf More Details: https://dfan.engineering.asu.edu/ai-security/

## requirement Commands (Anaconda):

conda create -n tbfa python=3.6

source activate tbfa

bash requirement.sh

## Variables

Source Class : --target

Target Class : --toattack

Attack Iterations: --iters

Type of Attack : --type

Attack sample of a given class (usually 500 for CIFAR10): --attacksamp

Test  Sample of a given class (Usually 500 for CIFAR10): --testsamp

Number of layers before the FC layer (9 for vgg11 and 20 for RESNET20) : --fclayer

Model (vgg/res): --model

## Command to run the attack

Type 2 attack on VGG11 model:

python targetedf.py --target 8 --toattack 1 --iters 100 --type 2 --attacksamp 500 --testsamp 500 --fclayer 9 --model vgg

Type 3 attack on ResNet20 model:

python targetedf.py --target 8 --toattack 1 --iters 100 --type 3 --attacksamp 500 --testsamp 500 --fclayer 20 --model res

