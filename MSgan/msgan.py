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

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


def add_intra_discriminator(old_model,ix,g_model,latent_dim):
	
	in_shape = list(old_model.input.shape)
	
	input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
	
	in_image = Input(shape=input_shape)
	in_latent = Input(shape = (latent_dim))
    
	d = Conv2D(128, (3,3), padding='same')(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# define new block
	d = Conv2D(128, (3,3), padding='same')(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = AveragePooling2D()(d)
	g = g_model[ix].layers[0](in_latent)
	for i in range(1,len(g_model[ix].layers)):
		g = g_model[ix].layers[i](g)      
    # r of the ix_th output of  generator by 1x1 conv
	out_image = Conv2D(3, (1,1), padding='same')(g)
    # r'  1x1 conv
	out_image  = Conv2D(128, (1,1), padding='same')(out_image)
	
	out_image = backend.stack([d, out_image],axis=0)
	
	d = out_image
	for i in range(0, len(old_model.layers)):
		d = old_model.layers[i](d)
	# define straight-through model
	model2 = Model(in_image, d)
	# compile model
	model2.compile(loss='Hinge', optimizer=RMSprop(lr=0.001))
	return  model2	      


    

def define_discriminator(n_blocks, g_model,latent_dim, input_shape):
   
	model_list = list()
	# base model input
    
	in_image = Input(shape=input_shape)
	
	# conv 3x3 (output block)
	d = MinibatchStdev()(in_image)
	d = Conv2D(128, (3,3), padding='same')(d)
	d = LeakyReLU(alpha=0.2)(d)
	# conv 4x4
	d = Conv2D(128, (4,4), padding='same')(d)
	d = LeakyReLU(alpha=0.2)(d)
	# dense output layer
	d = Flatten()(d)
	out_class = Dense(1)(d)
	# define model
	model = Model(in_image, out_class)
	# compile model
	model.compile(loss='hinge', optimizer=RMSprop(lr=0.001))
	# store model
	model_list.append(model)
	# create submodels
	for i in range(1, n_blocks):
		# get prior model 
		old_model = model_list[i - 1]
		# create new model for next resolution
		models = add_intra_discriminator(old_model,i-1,g_model,latent_dim)
		model_list.append(models)
	return model_list


def add_intra_generator(old_model,latent_dim):
    
	in_latent = Input(shape=(latent_dim))
	g = old_model.layers[0](in_latent)
	for i in range(0,len(old_model.layers)):
		g = old_model.layers[i](g)
	g = UpSampling2D()(g)
	g = Conv2D(128, (3,3), strides=(2,2), padding='same')(g)
	g = LeakyReLU(alpha=0.2)(g)
	g = Conv2D(128, (3,3), strides=(2,2), padding='same')(g)
	g = LeakyReLU(alpha=0.2)(g)
	model = Model(in_latent,g)
	# output layer
	return model

def define_generator(latent_dim,n_blocks=6,in_dim=4):
	model_list = list()
	dep = 4
	in_latent = Input(shape=(latent_dim))
	# linear scale up to activation maps
	g = Dense(dep*dep*128, input_dim=latent_dim)(in_latent)
	g = LeakyReLU(alpha=0.2)(g)
	g = Reshape((dep, dep, 128))(g)
	g = Conv2DTranspose(128, (4,4), strides=(2,2),padding='same')(g)
	g = Conv2D(128, (3,3), strides=(2,2), padding='same')(g)
	g = LeakyReLU(alpha=0.2)(g)
	model = Model(in_latent,g)
	model_list.append(model)

	for i in range(1,n_blocks):
		models = add_intra_generator(model_list[i-1],latent_dim)
		model_list.append(models)
   
	return model_list

def define_gan(generators, discriminators):
	model_list = list()
	# create composite models
	for i in range(len(discriminators)):
		g_models, d_models = generators[i], discriminators[i]
		# straight-through model
		d_models[0].trainable = False
		model1 = Sequential()
		model1.add(g_models)
		model1.add(d_models[1])
		model1.compile(loss='Hinge', optimizer=RMSprop(lr=0.001))
		# store
		model_list.append(model1)
	return model_list
 

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

def summarize_performance(status, g_model, latent_dim, n_samples=25):
	# devise name
	gen_shape = g_model.output_shape
	name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
	# generate images
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# normalize pixel values to the range [0,1]
	X = (X + 1.0) / 2 *255.0
	# plot real images
	square = int(sqrt(n_samples))
	for i in range(n_samples):
		pyplot.subplot(square, square, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X[i])
	filename = 'C:/Users/ZhenjuYin/Documents/yolo/msgan/'+label+'/gan_generated_plot_e%s.png' % (name)
	pyplot.savefig(filename)
	pyplot.close()
	# save the generator model tile file
	filename = 'C:/Users/ZhenjuYin/Documents/yolo/msgan/'+label+'/gan_generator_model_%s.h5' % (name)
	g_model.save(filename)
	#filename = 'C:/Users/ZhenjuYin/Documents/yolo/gan_64/lego/gan_discriminator_model_%03d.h5' % (epoch+1)
	#d_model.save(filename)

def train_epoch(g_model, d_model, gan_model,path_object, latent_dim, n_epochs, n_batch):
	bat_per_epo = int(path_object.shape[0] / n_batch)
	n_steps = bat_per_epo * n_epochs
	half_batch = int(n_batch / 2)
	#print("4")
	for i in range(n_steps):
	
			X_real, y_real = generate_real_samples(dataset, half_batch)
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		    # update discriminator model
			d_loss1 = d_model.train_on_batch(X_real, y_real)
			d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		    # update the generator via the discriminator's error
			z_input= generate_latent_points(latent_dim, n_batch)
			y_real2 = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(z_input, y_real2)			
			print('>%d,  d1=%.3f, d2=%.3f g=%.3f' %
				(i+1,  d_loss1, d_loss2, g_loss))         
			

def train(g_models, d_models, gan_models, size, latent_dim, e_norm,  n_batch):
	# fit the baseline model
    
	for i in range(0, len(g_models)):
		# retrieve models for this level of growth
		g_norma= g_models[i]
		d_normal= d_models[i]
		gan_normal = gan_models[i]
		# scale dataset to appropriate size
		filename = 'C:/Users/ZhenjuYin/Documents/yolo/'+label+'%d.npy' %(size[i])
		scaled_data = np.load(filename, allow_pickle=True)
		print('Scaled Data', scaled_data.shape)
		
		train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
		summarize_performance('tuned', g_normal, latent_dim)
 

print(device_lib.list_local_devices()) 
label ='head'
latent_dim = 128

path_object7 = np.load('C:/Users/ZhenjuYin/Documents/yolo/'+label+'128.npy', allow_pickle=True)
#d_model = load_model('C:/Users/ZhenjuYin/Documents/yolo/gan_64/screw/gan_discriminator_model_020.h5')

g_models = define_generator(latent_dim )
print("1")
d_models = define_discriminator(6,g_models,latent_dim,(4,4,3))

#g_model = load_model('C:/Users/ZhenjuYin/Documents/yolo/gan_64/lego/gan_generator_model_021.h5')
print("2")
gan_model = define_gan(g_models, d_models)

print("3")
datasize = [4,8,16,32,64,128]
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, datasize, latent_dim,  n_epochs, n_batch)
