from tensorflow.python.client import device_lib
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy import arange
from numpy.random import randn
from numpy.random import randint
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from skimage import io
from skimage import transform
from matplotlib import pyplot
from tensorflow.keras.models import load_model
dep = 4

def define_discriminator(in_shape=(64,64,3)):
	model = Sequential()
	# normal
	# downsample
	model.add(Conv2D(64, (3,3),  padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
    # downsample
	model.add(Conv2D(512, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 

def define_generator(latent_dim):
	model = Sequential()
	n_nodes = 1024* dep* dep
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((dep, dep, 1024)))
	# upsample to 8x8
	model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x64
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model
 
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0003, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 

def generate_real_samples(path_object,n_samples):

	# choose random instances
	ix = randint(0, path_object.shape[0], n_samples)
	X = path_object[ix]
    
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	X = g_model.predict(x_input)
	#print("a")
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

def save_plot(examples, epoch, n=4):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	# save plot to file
	filename = 'C:/Users/ZhenjuYin/Documents/yolo/gan_64/head/gan_generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

def summarize_performance(epoch, g_model, d_model, path_object,latent_dim, n_samples=64):
	# prepare real samples
	X_real, y_real = generate_real_samples(path_object, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	ix = arange(0, n_samples)
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'C:/Users/ZhenjuYin/Documents/yolo/gan_64/head/gan_generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)
	#filename = 'C:/Users/ZhenjuYin/Documents/yolo/gan_64/lego/gan_discriminator_model_%03d.h5' % (epoch+1)
	#d_model.save(filename)

def train(g_model, d_model, gan_model, path_object, latent_dim, n_epochs=150, n_batch=64):
	bat_per_epo = int(path_object.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	#print("4")
	for i in range(n_epochs):
		
		for j in range(bat_per_epo):
			
			X_real, y_real = generate_real_samples(path_object,half_batch)
			#print(X_real.shape)# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real,  y_real)
			#print("5")
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim,  half_batch)
			# update discriminator model weights
			#print("6")
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			#print("7")
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			#print("8")
            #X_lab = dataset2[0:n_batch-1]
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))         
		if (i+1) % 5 == 0 :
		    summarize_performance(i, g_model, d_model, path_object, latent_dim)


print(device_lib.list_local_devices()) 

latent_dim = 100
path_object = np.load('C:/Users/ZhenjuYin/Documents/yolo/Head64.npy', allow_pickle=True)
#d_model = load_model('C:/Users/ZhenjuYin/Documents/yolo/gan_64/screw/gan_discriminator_model_020.h5')
d_model = define_discriminator()
print("1")
g_model = define_generator(latent_dim)
#g_model = load_model('C:/Users/ZhenjuYin/Documents/yolo/gan_64/lego/gan_generator_model_021.h5')
print("2")
gan_model = define_gan(g_model, d_model)
print("3")
train(g_model, d_model, gan_model, path_object, latent_dim)
