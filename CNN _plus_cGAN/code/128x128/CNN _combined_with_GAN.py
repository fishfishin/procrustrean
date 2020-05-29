from __future__ import print_function, division


import cv2
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,Concatenate,Lambda,add
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, LeakyReLU,ZeroPadding2D,MaxPooling2D
#######  adding 'tf.' very essential!!
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import numpy as np

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.latent_dim = 64*3

        self.optimizer = Adam(0.0005, 0.5)
        self.loss = ['binary_crossentropy']

        # Build and compile the discriminator
        self.classifier = self.build_classifier()
        self.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model2_weights.h5")
        
        self.classifier.compile(loss= "mean_squared_error",
                    optimizer = Adam(0.0005, 0.5),
                    metrics=['accuracy'])
        
        self.discriminator = self.build_discriminator()

        self.data = self.loaddata()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=self.optimizer)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        in_lat = Input(shape=(self.latent_dim,))
        imgs = Input(shape=(self.img_shape))
        
        img = self.generator([in_lat, imgs])



        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.classifier.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator(img)
        a,v,d = self.classifier(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined1 = Model([in_lat, imgs], [valid,a,v,d])
        opt = Adam(lr=0.0003, beta_1=0.5)
        self.combined1.compile(loss=['binary_crossentropy',"mean_squared_error","mean_squared_error","mean_squared_error"],
            optimizer=opt)
        '''
        self.combined2 = Model([in_lat, imgs],[a,v,d])
        self.combined2.compile(loss= "mean_squared_error",
                    optimizer = Adam(0.0001, 0.5),
                    metrics=['accuracy'])
        '''

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = Input(shape=self.img_shape)

        l1 = Model(inputs=self.classifier.input,outputs=self.classifier.get_layer('a').output)(img)

        l2 = Model(inputs=self.classifier.input,outputs=self.classifier.get_layer('v').output)(img)

        l3 = Model(inputs=self.classifier.input,outputs=self.classifier.get_layer('d').output)(img)
        
        label =Concatenate()([l1,l2,l3])
        model_input = multiply([noise, label])

        image = model(model_input)

        return Model([noise, img], image)

    def build_classifier(self):

        model = Sequential()
        img = Input(shape=self.img_shape)

        model.add(Conv2D(32, kernel_size=3, activation="relu",strides=2, padding="same"))

        model.add(Conv2D(32, kernel_size=3,activation="relu", strides=2, padding="same"))

        model.add(Conv2D(64, kernel_size=3,activation="relu", strides=1, padding="same"))
        
        model.add(Dropout(0.25))

        model.add(MaxPooling2D((2,2),strides=None, padding="same"))
        model.add(Flatten())

        # Extract feature representation
        features = model(img)

        # Determine  labels of the image
        a = Dense(64, activation="softmax",name='a')(features)
        v = Dense(64, activation="softmax",name='v')(features)
        d = Dense(64, activation="softmax",name='d')(features)

        a = Dense(1, activation="sigmoid")(a) 
        a = Lambda(lambda x: x * 10.)(a)

        v = Dense(1, activation="sigmoid")(v) 
        v = Lambda(lambda x: x * 10.)(v)

        d = Dense(1, activation="sigmoid")(d) 
        d = Lambda(lambda x: x * 10.)(d)  

        mo = Model(img, [a,v,d]) 

        return mo

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))

        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        #model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        #model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        #model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
       

        return Model(img, validity)

    def loaddata(self):

        c = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/test.npy',allow_pickle=True)  
        data =np.zeros(shape=(c.shape[0],128,128,3))
        i=0
        value = np.zeros(shape=(c.shape[0],3))
        for img in c:
            image = cv2.imread(img[3],cv2.IMREAD_UNCHANGED)     
            if len(image.shape)==2: image = np.expand_dims(image, axis=3)
            data[i,:,:,:]= image                
            value[i,:]  = [img[4][0][0][0][0], img[4][0][1][0][0], img[4][0][2][0][0] ]# valence value
            i=i+1

        return data,  value



    def train(self, epochs, batch_size=128, sample_interval=50):

         # Load the dataset
        X_train,value = self.data
        
        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        for epoch in range(epochs):
            
            # ---------------------
            #  Train classifier
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0,  X_train.shape[0], batch_size)
            imgs = X_train[idx]
            arousal = value[idx,0]
            valence = value[idx,1]
            dominance=value[idx,2]            

            c_loss = self.classifier.train_on_batch(imgs, [arousal, valence,dominance])
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise as generator input
            in_lat = np.random.normal(0, 1, (batch_size, self.latent_dim))
            idx = np.random.randint(0,  X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([in_lat,imgs])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs,valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,fake)
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            in_lat = np.random.normal(0, 1, (batch_size, self.latent_dim))
            idx = np.random.randint(0,  X_train.shape[0], batch_size)
            imgs = X_train[idx]
            arousal = value[idx,0]
            valence = value[idx,1]
            dominance=value[idx,2]   

            g_loss1 = self.combined1.train_on_batch([in_lat,imgs], [valid,arousal, valence,dominance])
            #g_loss2 = self.combined2.train_on_batch([in_lat,imgs],[arousal, valence,dominance])

            # Plot the progress
            print ("%d [D loss: %.3f] [G loss: %.3f, g loss: %.3f ]" % (epoch, d_loss, g_loss1[0],  g_loss1[4]))
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch,gen_imgs)

    def sample_images(self, epoch,gen_imgs):
        
        # Rescale images 0 - 1
        gen_imgs = 127.5 * gen_imgs + 127.5

        r, c = 4,4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs.astype(np.uint8)[cnt,:,:,:])
                axs[i,j].axis('off')
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
    cgan.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model2_weights.h5")
    cgan.train(epochs=14000, batch_size=32, sample_interval=30)
