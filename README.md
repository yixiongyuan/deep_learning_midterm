# deep_learning_midterm

This is our code repository for deep learning midterm project. 
We modified the resnet-18 network to made the amount of parameters below 5M and achieve 93% accuracy on CIFAR-10 dataset.

# Introduction

We have in total 3 .ipynb files which represent 3 steps of our method.

# 1. Baseline Model
First, we replace first two basicblocks in resnet-18 with our own design FusionBlock in which we use 'Separable Convolution' to reduce the number of parameters in kernel and add additional 5*5 convolution layer to offer multi-scale information. Then we have our baseline model, in 'baseline_train.ipynb', we load our initial model, print all layer information and training the network. 



# 2. Model Pruning
In addition to the new design of FusionBlock, we also add new component in classical BasicBlock.
In order to separate redundant channels in network and delete them for simplicity. We add a new item on loss function to train the scale factors in Batch Normalization layer. According to this result, we can eliminate the channel with low factors.
So after training the baseline model, we will use'prune_model.ipynb" file to generate our pruned model.
The details are in code, but always becareful the process of every different layer in network. 



# 3. Fine-tune training
This the last stop, in 'fine_tune_train.ipynb' file, we get away with the learning item in loss function and do a one-time training again with the new pruned model. In our experiment, the modified model has 50% parameters compared to the original model but keep the same performance in accuracy. In reality, we can also try to do the pruning iteratively.
In "eval.ipynb" file, we load both two models and draw their confusion matrix on test dataset respectively as visualization of our evaluation and comparsion
