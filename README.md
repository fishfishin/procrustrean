# Emotion Synthesis through Progressive GAN and Auxiliary Conditional GAN and Multi-Task Learning

**This project is about generation of human emotion evoked by synthesized images through GAN variants, which requires has the same emotion label as the specific given picture with persons or not.**

##  Architecture
This goal is realized by convolutional neuron networks with adaptive weighting on the two losses of image generation and emotion classification. And the detailed weighting method is described in this paper [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115/).
The basic architecture derives from the Progressive GAN [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) and [Conditional Image Synthesis with Auxiliary Classifier GANs](https://www.arxiv-vanity.com/papers/1610.09585/).

## Implementation
The approach is implemented on Python and using Keras.

## Training Dataset
Training dataset is very important for the quality of generated images.

## Imbalanced Classificaiton Model
Emotion recognizition model comes from the paper [Combining Facial Expressions and Electroencephalography to Enhance Emotion Recognition](https://www.mdpi.com/1999-5903/11/5/105). The baseline is a 3=layer Convolutional Network and the more than one output.

## Multiple Tasks Learning

The approach is σ1 and σ2 are the trainable parameters and are regulated by the last term. This counld be the final loss fuction to be minimized. Increasing or decreasing σ1 and σ2 will affect the direction of the optimization.   

![alt text](https://github.com/fishfishin/procrustrean/blob/master/weighted_GAN/formula.png).
