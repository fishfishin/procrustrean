from __future__ import print_function, division

from tensorflow.keras.models import model_from_json

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,Lambda, MaxPooling2D,Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D,LeakyReLU
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np



def mean_loss(y_true, y_pred):

    a_loss = K.mean(K.square(y_true[0] - y_pred[0]))
    v_loss = K.mean(K.square(y_true[1] - y_pred[1]))      
    d_loss = K.mean(K.square(y_true[2] - y_pred[2]))      
    loss = (a_loss+v_loss+d_loss)/3.

    return loss


class classifier():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
       
        optimizer = Adam(0.0002, 0.5)
       
        # Build and compile the model
        self.model = self.build()
        
        self.model.compile(loss= tf.keras.losses.LogCosh(),
            optimizer=optimizer,)
        self.data = self.loaddata()

        self.model.trainable = False



    def build(self):

        model = Sequential()
        img = Input(shape=self.img_shape)

        model.add(Conv2D(32, kernel_size=3, activation="relu",strides=2, padding="same"))

        #model.add(Conv2D(32, kernel_size=3,activation="relu", strides=2, padding="same"))

        model.add(Conv2D(64, kernel_size=3,activation="relu", strides=1, padding="same"))
        
        model.add(Dropout(0.25))

        model.add(MaxPooling2D((2,2),strides=None, padding="same"))
        model.add(Flatten())
        
        # Extract feature representation
        mp  = model(img)
        # Determine valence of the image
       
        v1 = Dense(64, activation="relu",name='valence')(mp)
        valence = Dense(1, activation="sigmoid")(v1) 
        valence= Lambda(lambda x: x * 10.)(valence)

        a1 = Dense(64, activation="relu",name='arousal')(mp)
        arousal = Dense(1, activation="sigmoid")(a1) 
        arousal= Lambda(lambda x: x * 10.)(arousal) 

        d1 = Dense(64, activation="relu",name='dominance')(mp)
        dominance = Dense(1, activation="sigmoid")(d1) 
        dominance= Lambda(lambda x: x * 10.)(dominance) 
              
        #model.summary()


        return Model(img, [arousal,valence,dominance])


    def loaddata(self):

        c = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/test.npy',allow_pickle=True)  
        data =np.zeros(shape=(c.shape[0],64,64,3))
        i=0
        value = np.zeros(shape=(c.shape[0],3))
        for img in c:
            image = cv2.imread(img[3],cv2.IMREAD_UNCHANGED)
            
            if len(image.shape)==2: image = np.expand_dims(image, axis=3)
            data[i,:,:,:]= image 
            value[i,0] = img[4][0][0][0][0]
            value[i,1]  = img[4][0][1][0][0]# valence value
            value[i,2]  = img[4][0][2][0][0]
            i=i+1
        #print(max(value[:,0]))
        #print(min(value[:,0]))
        return data, value


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train, y_train = self.data
        
        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)
        self.model.trainable = True

        for epoch in range(epochs):

            # ---------------------
            #  Train Valence Arousal Dominance
            # ---------------------
            
            #self.valence.trainable = True

            # Select a random batch of images
            idx = np.random.randint(0, len(X_train), batch_size)
            imgs = X_train[idx]
            
            # Image labels. 0-9 
            arousal = y_train[idx,0]
            valence =y_train[idx,1]
            dominance=y_train[idx,2]

            # Train the model  a_loss, v_loss,_,d_loss,_ 
            loss = self.model.train_on_batch(imgs, [arousal, valence,dominance])
            
            value = self.model.predict(imgs[0:1])
            #print((value[0],value[1],value[2]))
            #print(y_train[0:1])
            
            
            # Plot the progress
            
            print ("%d [a loss: %.3f,v loss: %.3f,d loss: %.3f, m_loss: %.3f] " % (epoch, loss[1],loss[2],loss[3],loss[0]))
                                                                                                                            
                                                                                                                         
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()

        self.model.trainable = False
                

    def save_model(self):
        model = self.model
        
        def save(model, model_name):
            model_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model3.json" 
            weights_path = "C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model3_weights.h5" 
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.model, "model")


    def extract(self):
        image,_ = self.data
       
        model=self.model
        # ---------------------
        #  Extract Valence Arousal Dominance
        # ---------------------
        layer_name = 'arousal'
        imagetrain = (image.astype(np.float32) - 127.5) / 127.5
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        #intermediate_layer_model.summary()
        intermediate_output = intermediate_layer_model.predict(imagetrain)
        arousal= intermediate_output

        layer_name = 'valence'
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(imagetrain)
        valence= intermediate_output
        

        layer_name = 'dominance'
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(imagetrain)
        dominance= intermediate_output

        return arousal, valence,dominance


if __name__ == '__main__':
    
    classifier = classifier()
    
    classifier.model.load_weights("C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/class/saved/model3_weights.h5")
    #classifier.train(epochs=20000, batch_size=128, sample_interval=500)
    
    mo = classifier.model
    c = np.load('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/test.npy',allow_pickle=True) 
    image,label = classifier.data
    image = (image.astype(np.float32) - 127.5) / 127.5 # very important!!!
    a,v,d = mo.predict(image[100:101])
    print(label[100])
    print([a,v,d])
    
    arousal, valence,dominance =classifier.extract()
    np.save('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/emodb_small/arousal.npy',arousal) 
    np.save('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/emodb_small/valence.npy',valence) 
    np.save('C:/Users/ZhenjuYin/Documents/Python Scripts/emotic/emodb_small/dominance.npy',dominance) 

