import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import matplotlib.gridspec as gridspec
#from tensorflow.keras.datasets import mnist
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

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 128
Z_dim = 100
unrolling =6
##[kernel h , kernel w, in ,out]
with tf.variable_scope("discriminator"):
    k1_d = tf.Variable(tf.random.normal(shape=[5, 5, 11, 64 ],stddev=0.02 ))
    k2_d = tf.Variable(tf.random.normal(shape=[5, 5, 74 ,128],stddev=0.02 ))
'''
D_W1=tf.Variable(tf.random.normal(shape=[6282,1024],stddev=0.02 ),"discriminator")
D_b1=tf.Variable(tf.zeros(shape=[1024]),"discriminator")
D_W2=tf.Variable(xavier_init([1034,1]),"discriminator")
D_b2=tf.Variable(tf.zeros(shape=[1]),"discriminator")
'''
#para_D = [D_W1, D_W2, D_b1, D_b2,k1_d,k2_d]
#########################################################
G_W1 = tf.Variable(xavier_init([Z_dim, 1024]))
G_b1 = tf.Variable(tf.zeros(shape=[1024]))

G_W2 = tf.Variable(xavier_init([1034, 128*49]))
G_b2 = tf.Variable(tf.zeros(shape=[128*49]))

#[kernel h, kernel w, out chanels, in chanels]
k1_g = tf.Variable(tf.random.normal(shape=[5, 5, 64, 138 ],stddev=0.02))
k2_g = tf.Variable(tf.random.normal(shape=[5, 5, 1, 74 ],stddev=0.02 ))

para_G = [G_W1, G_W2,  G_b1, G_b2,k1_g,k2_g]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])

def sample_Z(m, n):

    return np.random.uniform(-1., 1., size=[m, n])



def generator(z,lab):
   
    size = mb_size
    G_h1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(z, G_W1) + G_b1))#[size, 1024]
    G_h1 = tf.concat([G_h1,lab],1)
    G_h2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(G_h1, G_W2) + G_b2))#[1034, 49*128]
   
    G_h = tf.reshape(G_h2, shape=[-1, 7, 7, 128])
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, 7, 7, 1])
    f = tf.cast(f, tf.float32) #######################
    G_h = tf.concat([G_h , f],axis=3) 
    ### input  [ size , 7,7,128+10]
    
    G_h3 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d_transpose(G_h,filter = k1_g,
             output_shape=[size, 14,14,64], padding='SAME', strides = [1, 2,2, 1])))####bias1
    
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, 14, 14, 1])
    f = tf.cast(f, tf.float32)
    G_h3 = tf.concat([G_h3 , f],axis=3)
    # input [ size , 14,14, 64+10]
    
    G_h4 = tf.nn.conv2d_transpose(G_h3,output_shape=[size, 28,28,1],
            filter = k2_g, padding='SAME', strides = [1, 2,2, 1]) #+bias2
    G_prob = tf.nn.sigmoid(G_h4)
    G_prob = tf.reshape(G_prob,shape=[-1,28*28])

    return G_prob
    



def discriminator(x,lab, reuse=tf.AUTO_REUSE):

    x = tf.reshape(x, shape=[-1,28,28,1])
    
    ##input [mb size, 28,28,1]
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, 28, 28, 1])
    f = tf.cast(f, tf.float32)
    x = tf.concat([x , f], axis=3)
    f1 = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, 14, 14, 1])
    f1 = tf.cast(f1, tf.float32)
    with tf.variable_scope("discriminator", reuse=reuse):   
    ##x input [mb size, 28,28, 1+ 10]
        D_h1=tf.nn.conv2d(x, padding='SAME', filter = k1_d, strides = [1, 2,2, 1])
        D_h1=tf.layers.batch_normalization(D_h1)
        D_h1 = tf.nn.leaky_relu(D_h1)
    ##[mb size, 14, 14, 64]
        D_h1 = tf.concat([D_h1 , f1], axis=3)
    ### D_h1 [mb size, 14, 14, 64+10]
        D_h1=tf.nn.conv2d(D_h1, padding='SAME', filter = k2_d, strides = [1, 2,2, 1])
        D_h1=tf.layers.batch_normalization(D_h1)
        D_h1 = tf.nn.leaky_relu(D_h1)
    ##D_h2 [mb size, 7, 7, 128]
        D_h1= tf.reshape(D_h1, shape=[mb_size,-1])
        D_h1 = tf.concat( [D_h1,lab],1)
    ##D_h2 [mb size, 49*128+10]
        D_h1.set_shape([None, 49*128+10])
   
        D_h1 =  tf.layers.dense(inputs=D_h1, units=1024,trainable =True, name='layer3')
        D_h1 = tf.layers.batch_normalization(D_h1)
        D_h1 = tf.nn.leaky_relu(D_h1)
        D_h1 = tf.concat( [D_h1,lab],1)
    ##D_h2 [mb size, 1024+10]
        D_h1.set_shape([None, 1034])
    
        D_logit = tf.layers.dense(inputs=D_h1, units=1,trainable =True,
                     kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='layer4') 
        D_prob = tf.nn.sigmoid(D_logit)
    
    return D_prob, D_logit


def plot(samples):

    fig = plt.figure(figsize=(4, 4))

    gs = gridspec.GridSpec(4, 4)

    gs.update(wspace=0.05, hspace=0.05)

    sam = samples [0:16,:]
    
    for i, sample in enumerate(sam):

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
#d_loss_sum_fake= tf.compat.v1.summary.scalar("d_loss_fake", D_loss_fake) 
#d_loss_sum_real = tf.compat.v1.summary.scalar("d_loss_real", D_loss_real)
g_loss_sum = tf.compat.v1.summary.scalar("g_loss", G_loss)
summary_writer = tf.compat.v1.summary.FileWriter('snapshots/', graph=tf.compat.v1.get_default_graph()) 

_graph_replace = tf.contrib.graph_editor.graph_replace
def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None
        
def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)

def extract_update_dict(update_ops):
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        #####  tyoe is "AssignVariableOp"  raise error ><,./:"|\}_\||--"
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update_op.op.type)
        
    return updates

cgan_d = Adam(lr=1e-4, beta_1=0.5, epsilon=1e-8)
cgan_g =tf.compat.v1.train.AdamOptimizer(0.001, beta1=0.5)
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
updates = cgan_d.get_updates(D_loss,disc_vars )

d_train_op = tf.group(*updates, name="d_train_op")
update_dict = extract_update_dict(updates)

for i in range(unrolling- 1):
    cur_update_dict = update_dict
    cur_update_dict = tf.contrib.graph_editor.graph_replace(update_dict, cur_update_dict)

unrolled_loss = graph_replace(D_loss, cur_update_dict)
G_solver = cgan_g.minimize(loss =-unrolled_loss, var_list=para_G)
unrolled_loss_sum = tf.compat.v1.summary.scalar("unrolled_loss", unrolled_loss) 



sess = tf.compat.v1.Session()
initial= tf.compat.v1.global_variables_initializer()
sess.run(initial)
if not os.path.exists('out/'): 
    os.makedirs('out/')
if not os.path.exists('snapshots/'): 
    os.makedirs('snapshots/')
saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=50)

ii = 0
n = mb_size

for it in range(1):
    with tf.device('/cpu:0'):

        for i in range(int(1000)): 
       
            #X_mb, x_label = databatchset(mb_size)
            
            X_mb, x_label = mnist.train.next_batch(mb_size)
            d_loss_curr,unrolled_loss_curr, unrolled_loss_sum_value,d_loss_sum_value, _ = sess.run([D_loss,
                             unrolled_loss, unrolled_loss_sum ,d_loss_sum,  d_train_op],
                              feed_dict={X: X_mb, y:x_label,Z:sample_Z(mb_size, Z_dim)})
            _, G_loss_curr,g_loss_sum_value = sess.run([G_solver, G_loss,g_loss_sum], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim),y:x_label})
            if i % 50 == 0:
                summary_writer.add_summary(d_loss_sum_value, i)
                #summary_writer.add_summary(d_loss_real_value, i)
                #summary_writer.add_summary(d_loss_fake_value, i)
                summary_writer.add_summary(g_loss_sum_value, i) 
                summary_writer.add_summary(unrolled_loss_sum_value, i) 
                save(saver, sess, 'snapshots/', it)  
                 
                _,lab_sam  = mnist.train.next_batch(n)  
                samples = sess.run(G_sample, feed_dict={ Z: sample_Z(n, Z_dim),y:lab_sam})
  
                fig = plot(samples)
           
                plt.savefig('out/{}.png'.format(str(ii).zfill(3)), bbox_inches='tight')
        
                ii += 1

              
                           
                print('Iter: {}'.format(i))
       
                print('D loss: {:.4}'. format(d_loss_curr))

                print('G_loss: {:.4}'.format(G_loss_curr))

                print()
                 
                
       
        
