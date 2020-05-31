from __future__ import print_function, division


import cv2
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,Concatenate,Lambda,add
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, LeakyReLU,ZeroPadding2D,MaxPooling2D,ReLU
#######  adding 'tf.' very essential!!
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist


def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)


class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.latent_dim = 100

        self.optimizer = Adam(0.0002, 0.5)
        self.loss = ['binary_crossentropy']

        # Build and compile the discriminator
        #self.classifier = self.build_classifier()
        #self.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model64_weights.h5")

        self.discriminator = self.build_discriminator()

        
        self.discriminator.compile(loss=['binary_crossentropy', tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
            optimizer=self.optimizer)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        in_lat = Input(shape=(self.latent_dim,))
        l1 = Input(shape=(1,))
        l2 = Input(shape=(1,))
        l3 = Input(shape=(1,))

        img = self.generator([in_lat,l1,l2,l3])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        #self.classifier.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid,a,v,d = self.discriminator(img)
        #a,v,d = self.classifier(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.combined1 = Model([in_lat,l1,l2,l3], [valid,a,v,d])
        self.combined1.compile(loss=['binary_crossentropy', tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
            optimizer=opt)  
        '''
        self.combined2 = Model([in_lat, imgs],[a,v,d])
        self.combined2.compile(loss= tf.keras.losses.LogCosh(),
                    optimizer = opt)
        '''

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
        a = Lambda(lambda x: x * 10.)(a)

        v = Dense(1, activation="sigmoid")(v) 
        v = Lambda(lambda x: x * 10.)(v)

        d = Dense(1, activation="sigmoid")(d) 
        d = Lambda(lambda x: x * 10.)(d)  

        mo = Model(img, [a,v,d]) 

        return mo       

    def build_generator(self):

        model = Sequential()
        '''
        image = Input(shape=(self.img_shape))
        l1,l2,l3 = round(Model(inputs=self.classifier.input,outputs=self.classifier.output)(image))
        '''
        n = Input(shape=(self.latent_dim,))
        noise = Dense(8*8*256)(n)
        noise = LeakyReLU(alpha=0.2)(noise)
        noise = Reshape((8,8,256))(noise)
       
        l1 = Input(shape=(1,))
        l2 = Input(shape=(1,))
        l3 = Input(shape=(1,))        
        label1= Dense(8*8)(Flatten()(Embedding(10, 60)(l1)))
        label2= Dense(8*8)(Flatten()(Embedding(10, 60)(l2)))
        label3= Dense(8*8)(Flatten()(Embedding(10, 60)(l3)))
        label1 = Reshape((8,8,1))(label1)
        label2 = Reshape((8,8,1))(label2)
        label3 = Reshape((8,8,1))(label3)

        model_input = Concatenate(axis=3)([noise, label1,label2,label3])
     
        # 16x16 UpSampling
        model.add(UpSampling2D(input_shape=(8,8,256+3)))
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
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)


        #a = Dense(64, activation="relu")(features)
        a = Dense(1, activation="sigmoid")(features)
        a = Lambda(lambda x: x * 9.+1)(a) 

        #v = Dense(64, activation="relu")(features)
        v = Dense(1, activation="sigmoid")(features)
        v = Lambda(lambda x: x * 9.+1)(v) 

        #d = Dense(64, activation="relu")(features)
        d = Dense(1, activation="sigmoid")(features)
        d = Lambda(lambda x: x * 9.+1)(d) 
        

        return Model(img, [validity,a,v,d])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        data = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/train_image_data.npy',allow_pickle=True)
        value= np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/train_label_data.npy',allow_pickle=True)
        X_train = data
        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        arr = np.arange(0,X_train.shape[0])
        samples = X_train[0:4]
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        ######classs a,v,d
        l1 = value[:,0]
        l2 = value[:,1]
        l3 = value[:,2]
        #V = l1*100+l2*10+l3
        #V = V.astype(np.int16) 
        ######

        for epoch in range(epochs):

            size= int(X_train.shape[0]/batch_size)
            random.shuffle(arr)
            
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
                gen_imgs = self.generator.predict([in_lat,fake_label1,fake_label2,fake_label3])

                # Train the discriminator
                

                d_loss_real = self.discriminator.train_on_batch(sam_img ,[valid,a,v,d])
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs,[fake,fake_label1,fake_label2,fake_label3])
                d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
           
                #a,v,d= self.classifier.predict(imgs)
                # control the weights of losses
            
                g_loss1 = self.combined1.train_on_batch([in_lat,fake_label1,fake_label2,fake_label3], [valid,fake_label1,fake_label2,fake_label3])
                print ("%d [D loss: %.4f, label: %.2f] [G loss: %.4f,label: %.2f] " % (epoch, d_loss[0],d_loss[1], g_loss1[0],g_loss1[1]))
                # g_loss1 = self.combined1.train_on_batch([in_lat,imgs], [valid,arousal, valence,dominance])


            # If at save interval => save generated image samples
                if i % sample_interval == 0:
                    self.save_model()
                    self.sample_images(i)




    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        a = np.asarray([5,5,5,5])
        v = np.asarray([2,8,10,5])
        d = np.asarray([4,6,1,3])
        gen_imgs = self.generator.predict([noise,a,v,d])
        # Rescale images 0 - 1
        gen_imgs = 127.5 * gen_imgs + 127.5 

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs.astype(np.uint8)[cnt,:,:,:])
                axs[i,j].axis('off')
                axs[i,j].set_title('a:%d,v:%d,d:%d' %(a[cnt],v[cnt],d[cnt]))
                cnt += 1
        fig.savefig("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/imagesnew/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/%snew.json" % model_name
            weights_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/%snew_weights.h5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    cgan = CGAN()
    
    #cgan.discriminator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/discriminatornew_weights.h5")
    #cgan.generator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/generatornew_weights.h5")
    #cgan.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model64_weights.h5")
    cgan.train(epochs=500, batch_size=16, sample_interval=20)
