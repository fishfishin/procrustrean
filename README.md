# Emotion Synthesis through Progressive GAN and Auxiliary Conditional GAN and Multi-Task Learning

**This project is about generation of human emotion evoked by synthesized images through GAN variants, which requires has the same emotion label as the specific given picture with persons or not.**

##  Architecture
This goal is realized by convolutional neuron networks with adaptive weighting on the two losses of image generation and emotion classification. And the detailed weighting method is described in this paper [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115/).
The basic architecture derives from the Progressive GAN [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) and [Conditional Image Synthesis with Auxiliary Classifier GANs](https://www.arxiv-vanity.com/papers/1610.09585/).

## Implementation
The approach is implemented on Python and using Keras.

## Training Dataset

## Imbalanced Classificaiton Model


## Multiple Tasks Learning

The approach is 
![alt text](https://github.com/fishfishin/procrustrean/blob/master/weighted_GAN/formula.png).
