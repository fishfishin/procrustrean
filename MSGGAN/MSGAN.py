from tensorflow.python.client import device_lib
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import stack
from numpy import arange
from numpy.random import randn
from numpy.random import randint
import numpy as np
import math
import pydot_ng as pydot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from skimage import io
from skimage import transform
from tensorflow.keras.layers import add
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer


pydot.find_graphviz()
# mini-batch standard deviation layer
class MinibatchStdev(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		# calculate the mean value for each pixel across channels
		mean = backend.mean(inputs, axis=0, keepdims=True)
		# calculate the squared differences between pixel values and mean
		squ_diffs = backend.square(inputs - mean)
		# calculate the average of the squared differences (variance)
		mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
		# add a small value to avoid a blow-up when we calculate stdev
		mean_sq_diff += 1e-8
		# square root of the variance (stdev)
		stdev = backend.sqrt(mean_sq_diff)
		# calculate the mean standard deviation across each pixel coord
		mean_pix = backend.mean(stdev, keepdims=True)
		# scale this up to be the size of one input feature map for each sample
		shape = backend.shape(inputs)
		output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
		# concatenate with the output
		combined = backend.concatenate([inputs, output], axis=-1)
		return combined
 
	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		# create a copy of the input shape as a list
		input_shape = list(input_shape)
		# add one to the channel dimension (assume channels-last)
		input_shape[-1] += 1
		# convert list to a tuple
		return tuple(input_shape)

        
def define_discriminator(in_shape, n_blocks = 6):
	model_list = list()
	img = []
	in_image = Input(shape=(in_shape[0]),name='input128')
	img.append(in_image)
	# create D0
	d = Conv2D(128, (3,3), padding='same')(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (3,3), padding='same')(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = AveragePooling2D()(d)
	old = d
	# create D1 to D4
	for  i in range(1,n_blocks-1):
		num = pow(2,7-i)
		nam = 'input'+ '%d' % num
		in_image = Input(shape=(in_shape[i]),name=nam)
		img.append(in_image)
		input_= Conv2D(128,(1,1),padding="same")(in_image)
		d = concatenate([input_ ,old],axis=-1)
		d = Conv2D(128, (3,3),padding='same')(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Conv2D(128, (3,3),padding='same')(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = AveragePooling2D()(d)
		old = d
	# create last D5
	in_image = Input(shape=(in_shape[5]),name='input4')
	img.append(in_image)
	input_= Conv2D(128,(1,1),padding="same")(in_image)
	d = concatenate([input_ ,old],axis=-1)
	d = MinibatchStdev()(d)
	d = Conv2D(128, (3,3), padding='same')(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), padding='same')(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Flatten()(d)
	d = Dense(1)(d)
	model = Model(inputs = img, outputs= d)
	model.compile(loss='hinge', optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
	plot_model(model, to_file='Discriminator.png',show_shapes=True, show_layer_names=True)
	return model


def define_generator(latent_dim,n_blocks=6,dep=4):
	
	in_latent = Input(shape=(latent_dim))
	out =[]
	# create G0
	g = Dense(dep*dep*128, input_dim=latent_dim)(in_latent)
	g = LeakyReLU(alpha=0.2)(g)
	g = Reshape((dep, dep, 128))(g)
	g = Conv2DTranspose(128, (4,4), padding='same')(g)
	g = LeakyReLU(alpha=0.2)(g)
	g = Conv2D(128, (3,3), padding='same')(g)
	g = LeakyReLU(alpha=0.2)(g)
	# extract image 4x4
	img = Conv2D(3,(1,1),activation='tanh',padding="same",name='output4')(g)
	out.append(img)
	
	# create G1 to G6 from 8x8 to 128x128
	for i in range(1,n_blocks):
		g = UpSampling2D()(g)
		g = Conv2D(128, (3,3), padding='same')(g)
		g = LeakyReLU(alpha=0.2)(g)
		g = Conv2D(128, (3,3), padding='same')(g)
		g = LeakyReLU(alpha=0.2)(g)
		# extract image
		if i < 5:
			nam = 'output' + '%d' % pow(2, i+2) 
			#print(nam)
			img = Conv2D(3,(1,1),activation='tanh',padding="same",name=nam)(g)
			out.append(img)
			
	# output 128x128
	img = Conv2D(3, (3,3), activation='tanh', padding='same',name='output128')(g)
	out.append(img)
	output=[]

	# sort the outputs in descending order from 128x128 to 4x4
	for image in reversed(out):
		output.append(image)

	model = Model(inputs=in_latent, outputs =output)
	plot_model(model, to_file='Generator.png',show_shapes=True, show_layer_names=True)
	return model

def define_gan(g_model, d_model):
	
	d_model.trainable = False
	output = d_model(g_model.output)
	
	model= Model(inputs=g_model.input, outputs=output)
	# compile model
	model.compile(loss='hinge', optimizer=RMSprop(lr=0.001))
	#print(model.summary())
	plot_model(model, to_file='MSGGAN.png',show_shapes=True, show_layer_names=True)
	return model

def generate_real_samples(n_samples):

	# choose random instances
	ix = randint(0, path_object2.shape[0], n_samples)
	X = [path_object7[ix],path_object6[ix],path_object5[ix],path_object4[ix],path_object3[ix],path_object2[ix]]
	#print(X[0].shape)
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(generator,latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = -ones((n_samples, 1))
	return X, y


def scale_dataset(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

def summarize_performance(ix, g_model,d_model, latent_dim, n_samples=16):
	# devise name
	X_real, y_real = generate_real_samples(n_samples)
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# generate images
	X = (X_fake + 1.0) / 2 *255.0
	# plot real images
	square = int(sqrt(n_samples))
	for i in range(n_blocks):
		for j in range(n_samples):
			pyplot.subplot(square, square, 1 + i)
			pyplot.axis('off')
			pyplot.imshow(X[i][j])
		num = pow(2,7-i)
		filename = 'C:/Users/Documents/yolo/msgan/'+label+'/gan_generated_plot_e%d' % (num) + '_ %d.png' %(ix)
		pyplot.savefig(filename)
		pyplot.close()
	# save the generator model tile file
	filename = 'C:/Users/Documents/yolo/msgan/'+label+'/gan_generator_model_%d.h5' % (ix)
	g_model.save(filename)


def train(g_model, d_model, gan_model,latent_dim, n_epochs=10, n_batch=8):
	bat_per_epo = int(path_object2.shape[0] / n_batch)
	n_steps = bat_per_epo * n_epochs
	half_batch = int(n_batch / 2)
	#print("4")
	for i in range(n_steps):
			X_real, y_real = generate_real_samples(half_batch)
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		    # update discriminator model
			d_loss1 = d_model.train_on_batch(X_real, y_real)
			d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		    # update the generator via the discriminator's error
			z_input= generate_latent_points(latent_dim, n_batch)
			y_real2 = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(z_input, y_real2)			
			print(i+1,  d_loss1, d_loss2, g_loss)
			if (i+1) % 50 == 0 :
				summarize_performance(i, g_model, d_model, latent_dim)        
			
 

print(device_lib.list_local_devices()) 
label ='head'
latent_dim = 256
n_blocks=6

path_object7 = np.load('C:/Users/Documents/yolo/'+label+'128.npy', allow_pickle=True)
path_object6 = np.load('C:/Users/Documents/yolo/'+label+'64.npy', allow_pickle=True)
path_object5 = np.load('C:/Users/Documents/yolo/'+label+'32.npy', allow_pickle=True)
path_object4 = np.load('C:/Users/Documents/yolo/'+label+'16.npy', allow_pickle=True)
path_object3 = np.load('C:/Users/Documents/yolo/'+label+'8.npy', allow_pickle=True)
path_object2 = np.load('C:/Users/Documents/yolo/'+label+'4.npy', allow_pickle=True)

g_model = define_generator(latent_dim )
print("1")
	
image_shape = [(128,128,3),(64,64,3),(32,32,3),(16,16,3),(8,8,3),(4,4,3)]
d_model = define_discriminator(image_shape)
print("2")
gan_model = define_gan(g_model, d_model)
print("3")
"""

n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
"""
train(g_model, d_model, gan_model, latent_dim)
