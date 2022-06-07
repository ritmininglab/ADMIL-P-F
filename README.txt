# Balancing Bias and Variance for Active Weakly Supervised Learning

This repository contains code for the paper "Balancing Bias and Variance for Active Weakly Supervised Learning"
This repository consists of three types of scripts: (1) Passive Learning Training, (2) Active Learning Training, (3) Query Instances using: (i) admil-p-f, (ii) admil-entropy, and (3) admil-random. The overall flow is first we perform passive learning training, query instances with given batch size, and train active learning models. Once, we train the models, we again query the instances and again perform training. Separate Scripts are written for each 4 datasets: 20 NewsGroup, Cifar10, Cifar100, and Pascal VOC. All the datasets are publicly available.

## Performing Passive Learning Training:
To perform training for Cifar10, Cifar100, and Pascal VOC, execute following command:

python train_dro_passive_+dataset+".py" run gamma


For 20NewsGroup, execute following command:

python train_dro_passive_+dataset+".py" run topic_no gamma

This will run the passive learning models and store the model and evaluation result inside dro_passive

## Performing Query:
To query instances for the first time execute the following command

python query_ins_+query_type+_+dataset+"_first.py" 

To query instances from second time (where we ignore the instances queried in first AL iteration), execute following command

python query_ins_+query_type+_+dataset+"_second.py" 

Above commands query the maximum n instances from positive bags and store inside Dataset folder.
Where, n = 15 for 20 NewsGroup, 150 for Cifar10, Cifarr100, 30 for Pascal VOC

## Performing Active Learning Training:

### For Dataset other than 20NewsGroup 

python train_dro_active_+query_type+_+dataset+".py" run gamma beta

### For 20NewsGroup
python train_dro_active_+query_type+_+dataset+".py" run topic_no gamma beta 


Where,

gamma can be regarded as inverse of lambda in the paper.

dataset one of the: 20newgroup, cifar10, cifar100, pascal_voc.

query_type is one of the: admil_p_f, admil_f_entropy, admil_random

