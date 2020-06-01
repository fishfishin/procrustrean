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
from tensorflow.keras.initializers import RandomNormal

def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

class WeightedSum(Add):

    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

class MinibatchStdev(Layer):

    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):

        mean = K.mean(inputs, axis=0, keepdims=True)
        squ_diffs = K.square(inputs - mean)
        mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = K.sqrt(mean_sq_diff)
        mean_pix = K.mean(stdev, keepdims=True)
        shape = K.shape(inputs)
        output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        combined = K.concatenate([inputs, output], axis=-1)
        return combined
 
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)     

class PixelNormalization(Layer):

    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        values = inputs**2.0
        mean_values = K.mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = K.sqrt(mean_values)
        normalized = inputs / l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.latent_dim = 100
        self.loss = ['binary_crossentropy']

        self.discriminator = self.build_discriminator()
        '''
        self.discriminator.compile(loss=['binary_crossentropy', tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
            optimizer=self.optimizer)
        '''
        # Build the generator
        self.generator = self.build_generator()
        self.gan= self.define_gan()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label

        #img = self.generator([in_lat,l1,l2,l3])

        # For the combined model we will only train the generator
       #self.discriminator.trainable = False

       #self.discriminator.compile(loss=['binary_crossentropy', tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
       #     optimizer=self.optimizer)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.opt = Adam(lr=0.0002, beta_1=0.5)
 
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'im_shape': self.img_shape,
            'opt': self.opt,
            'g_model': self.g_model,
            'd_model': self.d_model,
            'loss': self.loss,
            'channels': self.channels,
        })
        return config

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


    def add_d(self,old_model, n_input_layers=3):

        init = RandomNormal(stddev=0.02)
        const = max_norm(1.0)

        in_shape = list(old_model.input.shape)
        input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)

        image = Input(shape=input_shape)
        d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(image)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (3,3), padding='same', kernel_initializer=init,  kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D()(d)
        block_new = d
        # skip the input, 1x1 and activation for the old model
        #print(len(old_model.layers))

        for i in range(n_input_layers, len(old_model.layers)-6):
            #print(i)
            d = old_model.layers[i](d)
        # define straight-through model
        features = d
        validity = Dense(1, activation="sigmoid")(features)
        a = Dense(1, activation="sigmoid")(features)
        a = Lambda(lambda x: x * 9.+1)(a) 
        v = Dense(1, activation="sigmoid")(features)
        v = Lambda(lambda x: x * 9.+1)(v)
        da = Dense(1, activation="sigmoid")(features)
        da = Lambda(lambda x: x * 9.+1)(da)
        out_class = [validity,a,v,da] 

        model1 = Model(image, out_class)

        model1.compile(loss=[wasserstein_loss, tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
                    optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        downsample = AveragePooling2D()(image)
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)

        d = WeightedSum()([block_old, block_new])

        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)-6):
            d = old_model.layers[i](d)
        feature = d
        validity = Dense(1, activation="sigmoid")(feature)
        a = Dense(1, activation="sigmoid")(feature)
        a = Lambda(lambda x: x * 9.+1)(a) 
        v = Dense(1, activation="sigmoid")(feature)
        v = Lambda(lambda x: x * 9.+1)(v)
        da = Dense(1, activation="sigmoid")(feature)
        da = Lambda(lambda x: x * 9.+1)(da)
        out_class = [validity,a,v,da] 

        model2 = Model(image, out_class)

        model2.compile(loss=[wasserstein_loss, tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
                    optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        return [model1, model2]

    def build_discriminator(self,n_blocks=4):

        img = Input(shape=(8,8,3))     
        init = RandomNormal(stddev=0.02)
        const = max_norm(1.0)
        d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(img)       
        d = LeakyReLU(alpha=0.2)(d)
        d = MinibatchStdev()(d)
        d = Conv2D(128, (3,3), padding='same', kernel_initializer=init,name='1', kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Flatten()(d)

        num1 = np.asarray([9.]).astype(np.float32)
        num2 = np.asarray([1.]).astype(np.float32)
        features = d
        validity = Dense(1, activation="sigmoid")(features)
        a = Dense(1, activation="sigmoid")(features)
        a = Lambda(lambda x: x * 9.+1)(a) 
        v = Dense(1, activation="sigmoid")(features)
        v = Lambda(lambda x: x * 9.+1)(v)
        da = Dense(1, activation="sigmoid")(features)
        da = Lambda(lambda x: x * 9.+1)(da)

        out_class = [validity,a,v,da] 
        model = Model(img, out_class)
        model_list=list()
        model.compile(loss=[wasserstein_loss, tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
                    optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            
            models = self.add_d(old_model)
            model_list.append(models)
        return model_list


    def add_g(self,old_model):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)

        block_end = old_model.layers[-2].output

        upsampling = UpSampling2D()(block_end)
        g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

        model1 = Model(old_model.input, out_image)
        out_old = old_model.layers[-1]
        out_image2 = out_old(upsampling)
        merged = WeightedSum()([out_image2, out_image])

        model2 = Model(old_model.input, merged)
        return [model1, model2]


    def build_generator(self,n_blocks=4):
   
        n = Input(shape=(self.latent_dim,))
        noise = Dense(8*8*(256-64))(n)
        noise = LeakyReLU(alpha=0.2)(noise)
        noise = Reshape((8,8,(256-64)))(noise)       
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
     
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)

        model_list = list()

        g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(model_input)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

        model = Model([n,l1,l2,l3], out_image)

        model_list.append([model, model])

        for i in range(1, n_blocks):	
            old_model = model_list[i - 1][0]
            models = self.add_g(old_model)
            model_list.append(models)

        return model_list

    def define_gan(self):
        model_list = list()
        discriminators =self.discriminator
        generators = self.generator
        
        for i in range(len(discriminators)):
            g_models, d_models = generators[i], discriminators[i]
           
            #d_models[0].summary()
            in_lat = Input(shape=(self.latent_dim,))
            l1 = Input(shape=(1,))
            l2 = Input(shape=(1,))
            l3 = Input(shape=(1,)) 

            d_models[0].trainable = False
            img = g_models[0]([in_lat,l1,l2,l3])
            valid,a,v,d = d_models[0](img)
            model1= Model([in_lat,l1,l2,l3], [valid,a,v,d])
            model1.compile(loss=[wasserstein_loss, tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
                             optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            d_models[1].trainable = False
            img = g_models[1]([in_lat,l1,l2,l3])
            valid,a,v,d = d_models[1](img)
            model2= Model([in_lat,l1,l2,l3], [valid,a,v,d])
            model2.compile(loss=[wasserstein_loss, tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh(),tf.keras.losses.LogCosh()],
                             optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            model_list.append([model1, model2])
        return model_list


    def update_model(self, models, step, n_steps):
        alpha = step / float(n_steps - 1)
        # update the alpha for each model      
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    K.set_value(layer.alpha, alpha)

    def train_epoch(self,g_model,d_model,gan_model, data, epochs, batch_size=128, update=False, sample_interval=40):

        # Load the dataset

        value= np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/face_value.npy',allow_pickle=True)
        X_train = data
        X_train  = (X_train .astype(np.float32) - 127.5) / 127.5
        # Configure inputs
        arr = np.arange(0,X_train.shape[0])
        samples = X_train[0:4]     

        bat_per_epo = int(X_train.shape[0] / batch_size)
        n_steps = bat_per_epo * epochs
        half_batch = int(batch_size / 2)

        # Adversarial ground truths
        valid = np.ones((half_batch, 1))
        fake = -np.ones((half_batch, 1))
        valid_g = np.ones((batch_size, 1))
        ######classs a,v,d
        l1 = value[:,0]
        l2 = value[:,1]
        l3 = value[:,2]
        ######

        for i in range(n_steps):
            if update:
                self.update_model( [g_model, d_model, gan_model], i, n_steps)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Sample noise as generator input
            in_lat = np.random.normal(0, 1, (half_batch, self.latent_dim))                
            sam = np.random.randint(0,X_train.shape[0], half_batch)
            sam_img = X_train[sam]
            a = l1[sam]
            v = l2[sam]
            d = l3[sam]
            fake_label1 = np.random.randint(1,11, (half_batch, 1))
            fake_label2 = np.random.randint(1,11, (half_batch, 1))
            fake_label3 = np.random.randint(1,11, (half_batch, 1))
            # Generate a half batch of new images
            gen_imgs = g_model.predict([in_lat,fake_label1,fake_label2,fake_label3])

            # Train the discriminator

            d_loss_real = d_model.train_on_batch(sam_img ,[valid,a,v,d])
            d_loss_fake = d_model.train_on_batch(gen_imgs,[fake,fake_label1,fake_label2,fake_label3])
            d_loss = np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            fake_label1 = np.random.randint(1,11, (batch_size, 1))
            fake_label2 = np.random.randint(1,11, (batch_size, 1))
            fake_label3 = np.random.randint(1,11, (batch_size, 1))
            in_lat = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            g_loss1 = gan_model.train_on_batch([in_lat,fake_label1,fake_label2,fake_label3], [valid_g,fake_label1,fake_label2,fake_label3])
            print ("%d [D loss: %.4f, label: %.4f%%] [G loss: %.4f,label: %.4f%%] " % (i, d_loss[0],100*d_loss[1], g_loss1[0],100*g_loss1[1]))
                # g_loss1 = self.combined1.train_on_batch([in_lat,imgs], [valid,arousal, valence,dominance])

                


    def scale_dataset(self,images, new_shape):
        images_list = list()
        for image in images:
            new_image = resize(image, new_shape, 0)
            images_list.append(new_image)
        return np.asarray(images_list)

    def train(self, e_norm, e_fadein, batch_size):
	
        g_normal, d_normal, gan_normal = self.generator[0][0], self.discriminator[0][0], self.gan[0][0]
        dataset = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/face_image.npy',allow_pickle=True)

        gen_shape = g_normal.output_shape
        scaled_data = self.scale_dataset(dataset, gen_shape[1:])
        print('Scaled Data', scaled_data.shape)

        self.train_epoch(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], batch_size[0])
        for i in range(1, len(self.generator)):
        # retrieve models for this level of growth
            [g_normal, g_fadein] = self.generator[i]
            [d_normal, d_fadein] = self.discriminator[i]
            [gan_normal, gan_fadein] = self.gan[i]
            
            gen_shape = g_normal.output_shape
            scaled_data = self.scale_dataset(dataset, gen_shape[1:])
            print('Scaled Data', scaled_data.shape)
            # train fade-in models for next level of growth

            self.train_epoch(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], batch_size[i], True)
            name = '%03dx%03d-%s_g_model' % (gen_shape[1], gen_shape[2], 'faded')
            self.save_model(g_fadein,name)
            name = '%03dx%03d-%s_d_model' % (gen_shape[1], gen_shape[2], 'faded')
            self.save_model(d_fadein,name)
            
            self.sample_images(name, g_fadein)

            self.train_epoch(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], batch_size[i])
            name = '%03dx%03d-%s_g_model' % (gen_shape[1], gen_shape[2], 'tuned')
            self.save_model(g_normal,name)
            name = '%03dx%03d-%s_d_model' % (gen_shape[1], gen_shape[2], 'tuned')
            self.save_model(d_normal,name)
            self.sample_images(name,g_normal)
		

    def sample_images(self, name,g_model):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        a = np.asarray([5,5,5,5])
        v = np.asarray([2,8,10,5])
        d = np.asarray([4,6,1,3])
        gen_imgs =g_model.predict([noise,a,v,d])
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
        fig.savefig('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/imagespro/'+name+'.png')
        plt.close()

    def save_model(self,model, model_name):

        def save(model, model_name):
            model_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/imagespro/%spro.json" % model_name
            weights_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/imagespro/%spro_weights.h5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            #json_string = model.to_json()
            #open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
        
        save(model, model_name)
        #save(self.discriminator, "discriminator")


if __name__ == '__main__':
    cgan = CGAN()
    batch = [ 16, 16, 8, 4]
    epochs = [8, 8, 10, 10] 

    #cgan.discriminator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/discriminatorpro_weights.h5")
    #cgan.generator.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/cgan/generatorpro_weights.h5")
    #cgan.classifier.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model64_weights.h5")
    cgan.train(epochs,epochs, batch)
