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
    return tf.reduce_mean(tf.abs(y_pred - y_true))*10
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
        self.encoder1.summary()
        self.optimizer = Adam(0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
        self.loss = 'binary_crossentropy'
        self.g_model = self.gen()
        self.g_model.summary()
        self.d_model =self.dis()
        self.d_model.summary()
        self.d_model.trainable= False
        self.rec_model = self.recon()
        '''
        self.generate=self.build_consist()
        self.generate.summary()
        self.gan = self.build_gan()
        '''
        self.g_total= self.build_total()
        self.g_total.summary()
        
    def build_total(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        label = self.encoder2(img2)
        latent = self.encoder1(img1)
        gen_img = self.g_model([latent,label])

        gen_score = self.d_model([gen_img,label])
        ######   img1 and recon image
        re_img = self.rec_model([gen_img,img1])

        total = Model([img1,img2], [gen_score,re_img])
        total.compile(loss=[self.loss, l1_distance],optimizer=self.optimizer)

        return total

    def build_consist(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        label = self.encoder2(img2)
        latent = self.encoder1(img1)
        gen_img = self.g_model([latent,label])

        #gen_score = self.d_model([gen_img,label])

        re_img = self.rec_model(gen_img)

        #recon_loss= Lambda(l1_distance)([img1,re_img])
        #self.const = Model([img1,re_img])
        #self.const.compile(loss = l1_distance, optimizer=self.optimizer)
        '''
        valid = Input(shape=(1,))
        g_loss=Lambda(bce())([valid, gen_score])

        gloss = Lambda(loss)([g_loss, recon_loss])
        '''
        dec = Model([img1,img2], re_img)

        dec.compile(loss=l1_distance, optimizer=self.optimizer)

        return dec

    def build_gan(self):
        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        label = self.encoder2(img2)
        latent = self.encoder1(img1)
        gen_img = self.g_model([latent,label])

        gen_score = self.d_model([gen_img,label])

        dec = Model([img1,img2], gen_score)

        dec.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        return dec
        
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
                label1 = self.encoder1.predict(sam_img1)
                label2 = self.encoder2.predict(sam_img2)
                gen_img = self.g_model.predict([label1,label2])

                #####
                d_loss1 = self.d_model.train_on_batch([gen_img,label2],fake)
                #label_2 =self.encoder2(sam_img1)
                d_loss2 = self.d_model.train_on_batch([sam_img2,label2],valid) 

                #d_loss2 = 0.5*(self.d_model.train_on_batch([sam_img1,label_2],valid) + self.d_model.train_on_batch([sam_img2,label2],valid))

                d_loss = 0.5*(d_loss2+d_loss1)

                #g_loss = self.gan.train_on_batch([sam_img1,sam_img2], valid)
                #g_rec_loss = self.generate.train_on_batch([sam_img1,sam_img2], sam_img1)

                g_total_loss = self.g_total.train_on_batch([sam_img1,sam_img2],[valid,sam_img1])

                print ("%d [D loss: %.4f] [G loss: %.4f] [Recon loss: %.4f] " % (epoch, d_loss,g_total_loss[0], g_total_loss[1]))
                if i % 30 ==0:
                    self.sample_images(epoch,samples1,samples2)
            self.encoder1.save('encoder1.h5')
            self.encoder2.save('encoder2.h5')
            self.g_model.save('g.h5')
            self.d_model.save('d.h5')
            self.rec.save('rec.h5')



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
           


    def gen(self):
        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        noise1 = Input(shape=(self.label_dim,))
        noise2 = Input(shape=(self.label_dim,))
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

        return Model([noise1,noise2],image)

    def dis(self):
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

    def recon(self):

        model = Sequential()
        const = max_norm(1.0)
        init = RandomNormal(stddev=0.02)
        img1 =Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)
        label =self.encoder2(img2)
        latent= self.encoder1(img1)
        noise = Concatenate(axis = 1)([latent,label])
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
        
        model.add(Conv2DTranspose(self.channels, kernel_size=7, strides=1,padding="same", kernel_initializer=init, kernel_constraint=const))

       # model.add(Conv2D(self.channels, kernel_size=3, stride=1,padding="same", kernel_initializer=init, kernel_constraint=const))
        model.add(Activation("tanh"))

        image = model(noise)

        return Model([img1,img2],image)
    
    def sample_images(self, epoch,samples1,samples2):
        r, c = 2, 2
        label1 = self.encoder1.predict(samples1)
        label2 = self.encoder2.predict(samples2)
        gen_imgs = self.g_model.predict([label1,label2])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5 

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs.astype(np.uint8)[cnt,:,:,0],cmap='gray')
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
