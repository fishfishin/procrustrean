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
from os import listdir

##################################
#  help function
##################################
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

def l1_distance(y_true, y_pred):
    return tf.norm(y_true-y_pred,ord=1,keepdims=True)
    #return tf.reduce_mean(tf.abs(y_pred - y_true))*10
def bce(x):
    y_true,y_pred =x
    return binary_crossentropy(y_true,y_pred)
def loss(v):
    x,z =v
    return x+ z

###################################
#   construct model
###################################

class DoubleCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        
        self.label_dim = 64
        self.batch_size =1
        self.epochs =100
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoder1 = self.encoder1()
        self.encoder2 = self.encoder2()
        self.encoder3 = self.encoder3()
        self.encoder4 = self.encoder4()
        self.encoder1.summary()
        self.optimizer = Adam(0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
        self.loss = 'binary_crossentropy'
        self.g_a = self.gena()
        self.g_b = self.genb()
        self.g_a.summary()

        self.d_a =self.disa()
        self.d_a.trainable= False

        self.d_b =self.disb()
        self.d_b.trainable= False

        ##### output the gan score
        self.a_2_b =self.build_gana()
        self.b_2_a =self.build_ganb()
        ### output reconstructed images
        self.consta =self.build_consista_2_b()
        self.constb =self.build_consistb_2_a()
        '''
        self.generate=self.build_consist()
        self.generate.summary()
        self.gan = self.build_gan()
        '''
        self.cyclea_b= self.build_cycle_a_b()
        self.cyclea_b.summary()
        self.cycleb_a= self.build_cycle_b_a()
        self.cycleb_a.summary()
        
    def build_cycle_a_b(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)
        
        gen_score1 = self.a_2_b([img1,img2])
        #gen_score2 = self.b_2_a([img2,img1])
        gen_imagea = self.consta([img1,img2])
        #gen_imageb = self.constb([img2,img1])

        total = Model([img1,img2], [gen_score1,gen_imagea])
        total.compile(loss=[self.loss, l1_distance],optimizer=self.optimizer)

        return total

        
    def build_cycle_b_a(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)
        
        #gen_score1 = self.a_2_b([img1,img2])
        gen_score2 = self.b_2_a([img2,img1])
        #gen_imagea = self.consta([img1,img2])
        gen_imageb = self.constb([img2,img1])

        total = Model([img1,img2], [gen_score2,gen_imageb])
        total.compile(loss=[self.loss, l1_distance],optimizer=self.optimizer)

        return total


    def build_consista_2_b(self):
        imga = Input(shape=self.img_shape)
        imgb = Input(shape=self.img_shape)

        label = self.encoder2(imgb)
        latent = self.encoder1(imga)
        gen_img_b = self.g_a([latent,label])

        label = self.encoder4(imga)
        latent = self.encoder3(gen_img_b)

        gen_img_a = self.g_b([latent,label])

        con = Model([imga,imgb], gen_img_a)

        #con.compile(loss=l1_distance, optimizer=self.optimizer)

        return con

    def build_consistb_2_a(self):
        imga = Input(shape=self.img_shape)
        imgb = Input(shape=self.img_shape)

        label = self.encoder4(imga)
        latent = self.encoder3(imgb)
        gen_img_a = self.g_b([latent,label])

        label = self.encoder2(imgb)
        latent = self.encoder1(gen_img_a)

        gen_img_b= self.g_a([latent,label])

        con = Model([imgb,imga], gen_img_b)

        #con.compile(loss=l1_distance, optimizer=self.optimizer)

        return con

    def build_gana(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        label = self.encoder2(img2)
        latent = self.encoder1(img1)
        gen_img = self.g_a([latent,label])

        gen_score = self.d_b([gen_img,label])

        gan = Model([img1,img2], gen_score)

        #gan.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        return gan

    def build_ganb(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        label = self.encoder4(img2)
        latent = self.encoder3(img1)
        gen_img = self.g_b([latent,label])

        gen_score = self.d_a([gen_img,label])

        gan = Model([img1,img2], gen_score)

        #gan.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        return gan
        
    def train(self,data1,data2):

        X_train =data1
        X_train = (X_train.astype(np.float32) - 0.5) / 0.5
        Y_train =data2
        Y_train = (Y_train.astype(np.float32) - 0.5) / 0.5
        samples1 = X_train[0:4]
        samples2 = Y_train[0:4]
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        size = int(X_train.shape[0]/self.batch_size)
        half =int(self.batch_size/2)
        for epoch in range(self.epochs):
            for i in range(size) :
                               
                sam = np.random.randint(0,X_train.shape[0], self.batch_size)
                sam_img1 = X_train[sam]
                sam = np.random.randint(0,Y_train.shape[0], self.batch_size)
                sam_img2 = Y_train[sam]

                ####### share featrues of original domain and target domain
                label_b = self.encoder2.predict(sam_img2)
                gen_img_b = self.g_a.predict([sam_img1,sam_img2])

                d_b_loss1 = self.d_b.train_on_batch([gen_img_b,label_b],fake)               
                d_b_loss2 = self.d_b.train_on_batch([sam_img2,label_b],valid) 
                d_b_loss = 0.5*(d_b_loss2+d_b_loss1)

                gen_img_a = self.g_b.predict([sam_img2,sam_img1])
                label_a = self.encoder4.predict(sam_img1)

                d_a_loss1 = self.d_a.train_on_batch([gen_img_a,label_a],fake)               
                d_a_loss2 = self.d_a.train_on_batch([sam_img1,label_a],valid) 
                d_a_loss = 0.5*(d_a_loss2+d_a_loss1)

                #g_loss = self.gan.train_on_batch([sam_img1,sam_img2], valid)
                #g_rec_loss = self.generate.train_on_batch([sam_img1,sam_img2], sam_img1)

                cycle_loss1 = self.cyclea_b.train_on_batch([sam_img1,sam_img2],[valid,sam_img1])
                cycle_loss2 = self.cycleb_a.train_on_batch([sam_img1,sam_img2],[valid,sam_img2])

                d_loss =d_a_loss+d_b_loss
                cycle_loss = cycle_loss1+cycle_loss2

                print ("%d [D loss: %.4f] [G loss: %.4f] [Recon loss: %.4f] " % (epoch, d_loss,cycle_loss[0], cycle_loss[1]))
                if i % 30 ==0:
                    self.sample_images(epoch,samples1,samples2)
            self.encoder1.save('encoder1.h5')
            self.encoder2.save('encoder2.h5')
            self.g_a.save('g_a.h5')
            self.d_b.save('d_b.h5')
            self.encoder3.save('encoder3.h5')
            self.encoder4.save('encoder4.h5')
            self.g_b.save('g_b.h5')
            self.d_a.save('d_a.h5')
            



    def save_model(self):

        def save(model, model_name):
            model_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/translation/acgan_model/%s.json" % model_name
            weights_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/translation/acgan_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            #json_string = model.to_json()
            #open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generate, "generator")
        save(self.discriminator, "discriminator")



    def encoder1(self):

        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        image = Input(shape=self.img_shape)

        model.add(Conv2D(64, kernel_size=7, strides=1, input_shape=self.img_shape, padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2,  padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const))      
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(self.label_dim))####vector z
        return model


    def encoder2(self):

        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        image = Input(shape=self.img_shape)

        model.add(Conv2D(64, kernel_size=7, strides=1, input_shape=self.img_shape, padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2,  padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const))      
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(self.label_dim))####label l
        return model
           

    def encoder3(self):

        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        image = Input(shape=self.img_shape)

        model.add(Conv2D(64, kernel_size=7, strides=1, input_shape=self.img_shape, padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2,  padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const))      
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(self.label_dim))####vector z
        return model


    def encoder4(self):

        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        image = Input(shape=self.img_shape)

        model.add(Conv2D(64, kernel_size=7, strides=1, input_shape=self.img_shape, padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2,  padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const))      
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(self.label_dim))####label l
        return model
   

    def gena(self):
        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        #########   a to b domain
        imga= Input(shape=self.img_shape)
        imgb= Input(shape=self.img_shape)
        noise1 = self.encoder1(imga)
        noise2 = self.encoder2(imgb)
        noise = Concatenate(axis = 1)([noise1,noise2])
        dep = 16
        noise = Dense(256 * dep * dep, kernel_initializer=init, kernel_constraint=const)(noise)
        noise= Reshape((dep, dep, 256))(noise)

        model.add(Conv2DTranspose(256, kernel_size=3, strides=2,padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(128, kernel_size=3, padding="same",strides=2, kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(64, kernel_size=7, padding="same",strides=1, kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(self.channels, kernel_size=7, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const))

       # model.add(Conv2D(self.channels, kernel_size=3, stride=1,padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("tanh"))

        image = model(noise)

        return Model([imga,imgb],image)

    def genb(self):
        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        #########   b to a domain
        imgb= Input(shape=self.img_shape)
        imga= Input(shape=self.img_shape)
        noise1 = self.encoder3(imgb)
        noise2 = self.encoder4(imga)
        noise = Concatenate(axis = 1)([noise1,noise2])
        dep = 16
        noise = Dense(256 * dep * dep, kernel_initializer=init, kernel_constraint=const)(noise)
        noise= Reshape((dep, dep, 256))(noise)

        model.add(Conv2DTranspose(256, kernel_size=3, strides=2,padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(128, kernel_size=3, padding="same",strides=2, kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(64, kernel_size=7, padding="same",strides=1, kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(self.channels, kernel_size=7, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const))

       # model.add(Conv2D(self.channels, kernel_size=3, stride=1,padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("tanh"))

        image = model(noise)

        return Model([imgb,imga],image)

    def disa(self):
        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        image = Input(shape=self.img_shape)
        label = Input(shape=(self.label_dim,))
        ## 1x1xn to 64x64xn
        ###################################
        row =tf.expand_dims(label,axis=1)
        row = tf.expand_dims(row, axis=1)
        ### tile [ rank same]
        row = tf.tile(row, tf.constant([1,64,64,1]) )

        d= Conv2D(64, kernel_size=7, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const)(image)
        d = LeakyReLU(alpha=0.01)(d)

        d = Concatenate()([d,row])
        
        d=Conv2D(128, kernel_size=7, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const)(d)
        d=LeakyReLU(alpha=0.01)(d)
        

        d=Conv2D(256, kernel_size=7, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const)(d)
        d=LeakyReLU(alpha=0.01)(d)
        

        d=Conv2D(512, kernel_size=7, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const)(d)
        d=LeakyReLU(alpha=0.01)(d)
        d =Flatten()(d)
        d=Dense(1, activation="sigmoid")(d)

        d_model = Model([image,label],d)
        d_model.compile(loss='binary_crossentropy',optimizer=self.optimizer)

        return d_model

    def disb(self):
        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        image = Input(shape=self.img_shape)
        label = Input(shape=(self.label_dim,))
        ## 1x1xn to 64x64xn
        ###################################
        row =tf.expand_dims(label,axis=1)
        row = tf.expand_dims(row, axis=1)
        ### tile [ rank same]
        row = tf.tile(row, tf.constant([1,64,64,1]) )

        d= Conv2D(64, kernel_size=7, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const)(image)
        d = LeakyReLU(alpha=0.01)(d)

        d = Concatenate()([d,row])
        
        d=Conv2D(128, kernel_size=7, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const)(d)
        d=LeakyReLU(alpha=0.01)(d)
        

        d=Conv2D(256, kernel_size=7, strides=2, padding="same", kernel_initializer=init, kernel_constraint=const)(d)
        d=LeakyReLU(alpha=0.01)(d)
        

        d=Conv2D(512, kernel_size=7, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const)(d)
        d=LeakyReLU(alpha=0.01)(d)
        d =Flatten()(d)
        d=Dense(1, activation="sigmoid")(d)

        d_model = Model([image,label],d)
        d_model.compile(loss='binary_crossentropy',optimizer=self.optimizer)

        return d_model


    
    def sample_images(self, epoch,samples1,samples2):
        r, c = 4, 4

        gen_img_b = self.g_a.predict([samples1,samples2])
        gen_img_a = self.g_b.predict([samples2,samples1])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5 

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt <4:
                    axs[i,j].imshow(samples1.astype(np.uint8)[cnt,:,:,:],cmap='gray')
                elif cnt < 8:
                    axs[i,j].imshow(gen_img_b.astype(np.uint8)[cnt-4,:,:,:],cmap='gray')
                elif cnt <12:
                    axs[i,j].imshow(samples2.astype(np.uint8)[cnt-8,:,:,0],cmap='gray')
                else:  
                    axs[i,j].imshow(gen_img_a.astype(np.uint8)[cnt-12,:,:,0],cmap='gray')

                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/translation/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    c = DoubleCGAN()
    '''
    directory = "C:/Users/ZhenjuYin/Documents/Python Scripts/images/train/fear/"
    i =0
    fear=list()
    for filename in listdir(directory):
        image =cv2.imread(directory+filename,cv2.IMREAD_UNCHANGED)
        image =np.expand_dims(image,axis=2)
        image = resize(image,(64,64,1))
        fear.append(image)

    directory = "C:/Users/ZhenjuYin/Documents/Python Scripts/images/train/neutral/"
    neutral=list()
    for filename in listdir(directory):
        image =cv2.imread(directory+filename,cv2.IMREAD_UNCHANGED)
        image =np.expand_dims(image,axis=2)
        image = resize(image,(64,64,1))
        neutral.append(image)
    np.save("neutral.npy",neutral)
    np.save("fear.npy",fear)
    '''
    fear = np.load('fear.npy',allow_pickle=True)
    neutral = np.load('neutral.npy',allow_pickle=True)
    print(len(neutral))
    print(len(fear))
    data1 = neutral
    data2 = fear   
    c.train(data1,data2)
