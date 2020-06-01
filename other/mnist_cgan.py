import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import matplotlib.gridspec as gridspec
import os
from tensorflow.keras.optimizers import Adam
from collections import OrderedDict 

def save(saver, sess, logdir, step): 
    model_name = 'model'  
    checkpoint_path = os.path.join(logdir, model_name)   
    saver.save(sess, checkpoint_path, global_step=step)  
    

def xavier_init(size):
    in_dim = size[0]
    
    xavier_stddev = 1./ tf.sqrt(in_dim/2.)
    return tf.random.normal(shape=size, stddev = xavier_stddev)

#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 128
Z_dim = 100
unrolling =0
##[kernel h , kernel w, in ,out]
D_W1=tf.Variable(tf.random.normal(shape=[784+10,128],stddev=0.02 ))
D_b1=tf.Variable(tf.zeros(shape=[128]))
D_W2=tf.Variable(xavier_init([128,1]))
D_b2=tf.Variable(tf.zeros(shape=[1]))

para_D = [D_W1, D_W2, D_b1, D_b2]
#########################################################
G_W1 = tf.Variable(xavier_init([Z_dim+10, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))
para_G = [G_W1, G_W2, G_b1, G_b2]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])

def sample_Z(m, n):

    return np.random.uniform(-1., 1., size=[m, n])



def generator(z,lab):
    z = tf.concat([z,lab],axis=1)
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)#784
    #G_h1 = tf.concat([G_h1,lab],axis=1)
    G_h2 = tf.matmul(G_h1, G_W2) + G_b2#(img[0]//4)*(img[1]//4)*128
   
    G_log_prob = G_h2
    G_prob = tf.nn.sigmoid(G_log_prob)
    
    return G_prob
    

def discriminator(x,lab, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("discriminator", reuse=reuse):  
        x = tf.concat([x , lab], axis=1)
        D_h1 =  tf.matmul(x, D_W1) + D_b1
        D_h1 = tf.layers.batch_normalization(D_h1)
        D_h1 = tf.nn.leaky_relu(D_h1)
       # D_h1 = tf.concat([D_h1 , lab], axis=1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
  
    return D_prob, D_logit


def plot(samples):

    fig = plt.figure(figsize=(4, 4))

    gs = gridspec.GridSpec(4, 4)

    gs.update(wspace=0.05, hspace=0.05)

    
    
    for i, sample in enumerate(samples):

        ax = plt.subplot(gs[i])

        plt.axis('off')

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        ax.set_aspect('equal')
        
        plt.imshow(sample.reshape([28,28]), cmap='Greys_r')

    return fig


G_sample = generator(Z,y)
#G_sample = tf.reshape(G_sample, shape=[-1,29,29,1])
D_real, D_logit_real = discriminator(X,y)

D_fake, D_logit_fake = discriminator(G_sample,y,reuse=True)



D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))

D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

t = tf.compat.v1.summary.scalar("loss", G_loss+D_loss)

d_loss_sum = tf.compat.v1.summary.scalar("d_loss", D_loss) 

g_loss_sum = tf.compat.v1.summary.scalar("g_loss", G_loss)
summary_writer = tf.compat.v1.summary.FileWriter('snapshots/', graph=tf.compat.v1.get_default_graph()) 


cgan_d = tf.compat.v1.train.AdamOptimizer(0.001, beta1=0.5)
cgan_g =tf.compat.v1.train.AdamOptimizer(0.001, beta1=0.5)


G_solver = cgan_g.minimize(loss =G_loss , var_list=para_G)

D_solver = cgan_g.minimize(loss =D_loss , var_list=para_D)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True) 
sess = tf.compat.v1.Session()
initial= tf.compat.v1.global_variables_initializer()
sess.run(initial)
if not os.path.exists('out/'): 
    os.makedirs('out/')
if not os.path.exists('snapshots/'): 
    os.makedirs('snapshots/')

saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=50)
ii = 0
n = 16

lab_sam  = mnist.train.labels[20:20+n]  
#lab_sam=lab_sam*0.99

for it in range(1):
    with tf.device('/cpu:0'):

        for i in range(int(100000)): 
       
            #X_mb, x_label = databatchset(mb_size)
            
            X_mb, x_label = mnist.train.next_batch(mb_size)
            _, D_loss_curr ,d_loss_sum_value= sess.run([D_solver, D_loss,d_loss_sum], feed_dict={X: X_mb, y:x_label,Z:sample_Z(mb_size, Z_dim)})
            
            _, G_loss_curr,g_loss_sum_value = sess.run([G_solver, G_loss,g_loss_sum], feed_dict={Z: sample_Z(mb_size, Z_dim),y:x_label})
            if i % 500 == 0:
                summary_writer.add_summary(d_loss_sum_value, i)
                summary_writer.add_summary(g_loss_sum_value, i) 
  
                samples = sess.run(G_sample, feed_dict={ Z: sample_Z(n, Z_dim),y:lab_sam})
  
                fig = plot(samples)
           
                plt.savefig('out/{}.png'.format(str(ii).zfill(3)), bbox_inches='tight')
        
                ii += 1

              
                           
                print('Iter: {}'.format(i))
       
                print('D loss: {:.4}'. format(D_loss_curr))

                print('G_loss: {:.4}'.format(G_loss_curr))

                print()
    save(saver, sess, 'snapshots/', i)       
                
       
      
