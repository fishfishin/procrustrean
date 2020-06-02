# Emotion Synthesis through Progressive GAN and Auxiliary Conditional GAN and Multi-Task Learning

**This project is about generation of human emotion evoked by synthesized images through GAN variants, which requires has the same emotion label as the specific given picture with persons or not.**

##  Architecture
This goal is realized by convolutional neuron networks with adaptive weighting on the two losses of image generation and emotion classification. And the detailed weighting method is described in this paper [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115/).
The basic architecture derives from the Progressive GAN [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) and [Conditional Image Synthesis with Auxiliary Classifier GANs](https://www.arxiv-vanity.com/papers/1610.09585/).

![alt text](https://github.com/fishfishin/procrustrean/blob/master/ProgresiveGAN/progan.png)

![alt text](https://github.com/fishfishin/procrustrean/blob/master/CNN%20_plus_cGAN/acgan.png)


## Implementation
The approach is implemented on Python and using Keras.

## Training Dataset
Training dataset is very important for the quality of generated images.

## Imbalanced Classificaiton Model
Emotion recognizition model comes from the paper [Combining Facial Expressions and Electroencephalography to Enhance Emotion Recognition](https://www.mdpi.com/1999-5903/11/5/105). The baseline is a 3-layer Convolutional Network and the more than one seperate fully connected layer for outputs .

## Multiple Tasks Learning

The approach is σ1 and σ2 are the trainable parameters and are regulated by the last term. This counld be the final loss fuction to be minimized. Increasing or decreasing σ1 and σ2 will affect the direction of the optimization.   

![alt text](https://github.com/fishfishin/procrustrean/blob/master/weighted_GAN/formula.png).


And here is an code example to show us how to implement this method during the training
```ruby
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=4, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,), initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)
    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            else: loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)
    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        return K.concatenate(inputs, -1)
  ```
