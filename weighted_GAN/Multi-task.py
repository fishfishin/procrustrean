from __future__ import print_function, division

from math import sqrt
import cv2
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,Concatenate,Lambda,Add,Layer,AveragePooling2D,add
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, LeakyReLU,ZeroPadding2D,MaxPooling2D,ReLU
#######  adding 'tf.' very essential!!
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.constraints import Constraint
#### donot mix tf.keras and keras !!!!
from keras.losses import binary_crossentropy

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def mse_loss(x):
    y_true, y_pred =x

    return K.sqrt(y_true - y_pred)

def ignoreLoss(true,pred ):  
    return pred

def calLoss(x):
    true,pred = x
    return binary_crossentropy(true,pred)

def softmax(x):
    exps=K.exp(x)  # x-K.max(x)
    return exps/K.sum(exps)

class LossWeighter(Layer):
    def __init__(self, **kwargs): #kwargs can have 'name' and other things
        super(LossWeighter, self).__init__(**kwargs)

    #create the trainable weight here, notice the constraint between 0 and 1
    def build(self, inputShape):
        self.weight = self.add_weight(name='loss_weight', 
                                     shape=(1,),
                                     initializer=Constant(0.5), 
                                     constraint=Between(0,1),
                                     trainable=True)
        super(LossWeighter,self).build(inputShape)

    def call(self,inputs):
        firstLoss, secondLoss = inputs
        return (self.weight * firstLoss) + ((1-self.weight)*secondLoss)

    def compute_output_shape(self,inputShape):
        return inputShape[0]


class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=4, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
             ############### softmax for crossentropy los
            ############### Euclidean distance for continuous value
            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

class Between(Constraint):
    def __init__(self,min_value,max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self,w):
        return K.clip(w,self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.latent_dim = 150

        self.optimizer = Adam(0.0002, 0.5)
        self.loss = ['binary_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = False
        self.discriminator.compile(loss=wasserstein_loss,
            optimizer=self.optimizer)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label

        self.classifier =self.build_classifier()
        self.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model3_weights.h5")

        # For the combined model we will only train the generator
        
        self.classifier.trainable = False
        self.emotion =self.build_emotion()



    def build_classifier(self):

        model = Sequential(name='seq')
        img = Input(shape=self.img_shape)
        # 64 x64 input_shape=self.img_shape,
        model.add(Conv2D(32, kernel_size=3, activation="relu",strides=2,  name ='c1', padding="same"))
        
        model.add(Conv2D(32, kernel_size=3, activation="relu",strides=2, name ='c2', padding="same"))

        model.add(Conv2D(64, kernel_size=3,activation="relu", strides=1, name ='c3',padding="same"))
        
        model.add(Dropout(0.25))
        # 8x8
        model.add(MaxPooling2D((2,2),strides=None, name ='m', padding="same"))
        model.add(Flatten(name ='f'))
        #model.summary()
        # Extract feature representation
        features = model(img)

        # Determine  labels of the image
        a = Dense(64,activation="relu", name='a')(features)
        #a = ReLU()(a)
        v = Dense(64, activation="relu",name='v')(features)
        #v = ReLU()(v)
        d = Dense(64, activation="relu",name='d')(features)
        #d = ReLU()(d)

        a = Dense(1, activation="sigmoid")(a) 
        a = Lambda(lambda x: x * 10)(a)

        v = Dense(1, activation="sigmoid")(v) 
        v = Lambda(lambda x: x * 10)(v)

        d = Dense(1, activation="sigmoid")(d) 
        d = Lambda(lambda x: x * 10)(d)  

        mo = Model(img, [a,v,d]) 

        return mo       


    def build_emotion(self):
        g_model = self.generator ###the last one 64x64 tuned
        d_model= self.discriminator
        self.classifier.trainable = False
        n = Input(shape=(self.latent_dim,)) 
        #image=Input(shape=self.img_shape)
        l1 = Input(shape=(1,))
        l2 = Input(shape=(1,))
        l3 = Input(shape=(1,))
        gen_image = g_model([n,l1,l2,l3])
        a,v,d = self.classifier(gen_image)
        score = d_model(gen_image)
        valid = Input((1,))

        emotion_loss1 = Lambda(mse_loss)([l1, a])
        emotion_loss2 = Lambda(mse_loss)([l2, v])
        emotion_loss3 = Lambda(mse_loss)([l3, d])
        '''
        #####3    add, multiple ([inputs] )
        e_loss = add([emotion_loss1,emotion_loss2, emotion_loss3])       
        
        gen_loss = Lambda(wasserstein_loss)([valid, score])
        #weightedloss = LossWeighter()([gen_loss, e_loss])
        '''
        ###### 4 outputs labels and validation
        weightedloss = CustomMultiLossLayer(nb_outputs=4)([valid,l1,l2,l3,  score, a,v,d])

        model = Model([n,l1,l2,l3,valid], weightedloss  )
        model.compile(optimizer=Adam(0.001, 0.5), loss =ignoreLoss )
        return model

        

    def build_generator(self):

        model = Sequential()

        '''n = Input(shape=(self.latent_dim,))
        noise = Dense(8*8*128)(n)
        noise = LeakyReLU(alpha=0.2)(noise)
        noise = Reshape((8,8,128))(noise)
        l1 = Input(shape=(1, ))
        l2 = Input(shape=(1, ))
        l3 = Input(shape=(1,))        
        label1= Dense(8*8)(Flatten()(Embedding(10, 60)(l1)))
        label2= Dense(8*8)(Flatten()(Embedding(10, 60)(l2)))
        label3= Dense(8*8)(Flatten()(Embedding(10, 60)(l3)))
        label1 = Reshape((8,8,1))(label1)
        label2 = Reshape((8,8,1))(label2)
        label3 = Reshape((8,8,1))(label3)
        '''
        noise = Input(shape=(self.latent_dim,))    
        l1 = Input(shape=(1,))
        l2 = Input(shape=(1,))
        l3 = Input(shape=(1,))        
        label1= Embedding(10, 50)(l1)
        label2= Embedding(10, 50)(l2)
        label3=Embedding(10, 50)(l3)
        label = Concatenate()([label1,label2,label3])

        ### from n to z sapce
        model_input = multiply([noise, label])

        model.add(Dense(256 * dep * dep, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((dep, dep, 256)))

        # 8x8 UpSampling
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        # 16x16 UpSampling
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        # 32x32 UpSampling
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        #64x64 UpSampling
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        img = model(model_input)

        return Model([n,l1,l2,l3], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        data = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/face_image.npy',allow_pickle=True)
        value= np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/face_value.npy',allow_pickle=True)
        X_train = data
        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        arr = np.arange(0,X_train.shape[0])
        samples = X_train[0:4]
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        ######classs a,v,d
        l1 = value[:,0]
        l2 = value[:,1]
        l3 = value[:,2]
        size= int(X_train.shape[0]/batch_size)
        ######
        for epoch in range(epochs):
            for i in range(size) :
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Sample noise as generator input
                in_lat = np.random.normal(0, 1, (batch_size, self.latent_dim))                
                sam = np.random.randint(0,X_train.shape[0], batch_size)
                sam_img = X_train[sam]
                a = l1[sam]
                v = l2[sam]
                d = l3[sam]
                fake_label1 = np.random.randint(1,11, (batch_size, 1))
                fake_label2 = np.random.randint(1,11, (batch_size, 1))
                fake_label3 = np.random.randint(1,11, (batch_size, 1))
                # Generate a half batch of new images
                gen_imgs = self.generator.predict([in_lat,fake_label1,fake_label2,fake_label3])#fake_label1.astype(np.int32),fake_label2.astype(np.int32),fake_label3.astype(np.int32)])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(sam_img ,valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs,fake)
                d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
                # ---------------------
                #  Train Generator
                # ---------------------
                # control the weights of losses
                emotion =self.build_emotion()
               
                g_total_loss = emotion.train_on_batch([in_lat,a,v,d,valid], fake)
                print ("%d [D loss: %.4f] [G loss: %.4f] " % (epoch, d_loss, g_total_loss))
                
            # If at save interval => save generated image samples
                if i % sample_interval == 0:
                    self.save_model()
                    self.sample_images(i,samples)

    def sample_images(self, epoch,samples):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        a ,v,d = self.classifier.predict(samples)
        gen_imgs = self.generator.predict([noise,a.astype(np.int32), v.astype(np.int32) , d.astype(np.int32) ])
        # Rescale images 0 - 1
        gen_imgs = 127.5 * gen_imgs + 127.5 

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs.astype(np.uint8)[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/acgan_model/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/acgan_model/%s.json" % model_name
            weights_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/acgan_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            #json_string = model.to_json()
            #open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    cgan = CGAN()
    e = cgan.emotion
    
    
    #cgan.discriminator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/acgan_model/discriminator_weights.hdf5")
    #cgan.generator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/acgan_model/generator_weights.hdf5")
    
    cgan.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model3_weights.h5")
    cgan.train(epochs=140, batch_size=16, sample_interval=20)
