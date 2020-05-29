from __future__ import print_function, division


import cv2
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, LeakyReLU
#######  adding 'tf.' very essential!!
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

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
        self.discriminator = self.build_discriminator()
        self.data = self.loaddata()

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        in_lat = Input(shape=(self.latent_dim,))
        label1 = Input(shape=(64,))
        label2 = Input(shape=(64,))
        label3 = Input(shape=(64,))
        img = self.generator([in_lat, label1,label2,label3])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid  = self.discriminator([img, label1,label2,label3])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([in_lat, label1,label2,label3], [valid])
        opt = Adam(lr=0.001, beta_1=0.5)
        self.combined.compile(loss=self.loss,
            optimizer=opt)

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
        l1 = Input(shape=(64,))
        l2 = Input(shape=(64,))
        l3 = Input(shape=(64,))
        label =Concatenate()([ l1,l2,l3])
        model_input = multiply([noise, label])
        img = model(model_input)

        return Model([noise, l1,l2,l3], img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)
        '''    
        l1 = Input(shape=(64,))
        
        #label1 = Embedding(10, 10 )(l1)
        #label2 = Embedding(10, 10  )(l2)
        n_nodes = 128* 128
        label1 = Dense(n_nodes)(l1)
        label1 = Reshape((128, 128, 1))(label1)
        
        l2 = Input(shape=(64,))
        label2 = Dense(n_nodes)(l2)
        label2 = Reshape((128, 128, 1))(label2)

        l3 = Input(shape=(64,))
        #label3 = Embedding(10, 10 )(l3)
        label3 = Dense(n_nodes)(l3)
        label3 = Reshape((128, 128, 1))(label3)
        
        merge = Concatenate()([img, label1,label2,label3])
        '''
        l1 = Input(shape=(64,))
        l2 = Input(shape=(64,))
        l3 = Input(shape=(64,))
        label =Concatenate()([ l1,l2,l3])
        n_nodes = 128* 128
        label = Dense(n_nodes)(label)
        label = Reshape((128, 128, 1))(label)
        merge = Concatenate()([img, label])
        
        dis = Conv2D(16, kernel_size=3, strides=2, padding="same")(merge)
        dis = LeakyReLU(alpha=0.2)(dis)
        #dis = Dropout(0.25)(dis)

        dis = Conv2D(32, kernel_size=3, strides=2, padding="same")(dis)
        dis = LeakyReLU(alpha=0.2)(dis)
        #dis = Dropout(0.25)(dis)
        #dis = BatchNormalization(momentum=0.8)(dis)

        dis = Conv2D(64, kernel_size=3, strides=2, padding="same")(dis)
        dis = LeakyReLU(alpha=0.2)(dis)
        #dis = Dropout(0.25)(dis)
        #dis = BatchNormalization(momentum=0.8)(dis)

        dis = Conv2D(128, kernel_size=3, strides=2, padding="same")(dis)
        dis = LeakyReLU(alpha=0.2)(dis)
        #dis = Dropout(0.25)(dis)

        dis = Flatten()(dis)

        # Extract feature representation
        features = dis

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        model = Model([img, l1,l2,l3], [validity])

        model.compile(loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        return model


    def loaddata(self):

        c = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/test.npy',allow_pickle=True)  
        data =np.zeros(shape=(c.shape[0],128,128,3))
        a = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/emodb_small/arousal.npy',allow_pickle=True) 
        v = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/emodb_small/valence.npy',allow_pickle=True) 
        d = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/emodb_small/dominance.npy',allow_pickle=True)
        i=0
        value = np.zeros(shape=(c.shape[0],3))
        for img in c:
            image = cv2.imread(img[3],cv2.IMREAD_UNCHANGED)     
            if len(image.shape)==2: image = np.expand_dims(image, axis=3)
            data[i,:,:,:]= image 
            i=i+1

        return data, a,v,d



    def train(self, epochs, batch_size=128, sample_interval=50):

         # Load the dataset
        X_train, a,v,d = self.data
        
        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        for epoch in range(epochs):
           
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0,  X_train.shape[0], batch_size)
            imgs = X_train[idx]
            l1 = a[idx]
            l2 = v[idx]
            l3 = d[idx]

            # Sample noise as generator input
            in_lat = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([in_lat, l1,l2,l3])



            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, l1,l2,l3],[valid])
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, l1,l2,l3],[fake])
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            idx = np.random.randint(0,  X_train.shape[0], batch_size)
            l1 = a[idx]
            l2 = v[idx]
            l3 = d[idx]
            g_loss = self.combined.train_on_batch([in_lat, l1,l2,l3], [valid])

            # Plot the progress
            print ("%d [D loss: %.3f, acc.: %.2f%%,] [G loss: %.3f]" % (epoch, d_loss[0], 100*d_loss[1],  g_loss))

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
        fig.savefig("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/imagesac/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/%sac.json" % model_name
            weights_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/%sac_weights.h5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    cgan = CGAN()
    
    #cgan.discriminator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/discriminatorac_weights.h5")
    #cgan.generator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/generatorac_weights.h5")

    cgan.train(epochs=14000, batch_size=32, sample_interval=30)
