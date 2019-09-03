from loaddata_conv import *
import matplotlib.pyplot as plt
import tensorflow as tf 
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

type = "illumination"
h=200
w=200
mb_size = 32
Z_dim = 100
unrolling =0
lr=0.001
classes = 4
rgb =3 
##[kernel h , kernel w, in ,out]
k1_d = tf.Variable(tf.random.normal(shape=[5, 5, rgb+classes, 64 ],stddev=0.02 ))
k2_d = tf.Variable(tf.random.normal(shape=[5, 5, 64+classes ,128],stddev=0.02 ))

D_W1=tf.Variable(tf.random.normal(shape=[h*w*8+classes, 1024],stddev=0.02 ))
D_b1=tf.Variable(tf.zeros(shape=[1024]))
D_W2=tf.Variable(xavier_init([1024+classes,1]))
D_b2=tf.Variable(tf.zeros(shape=[1]))

para_D = [D_W1, D_W2, D_b1, D_b2,k1_d,k2_d]
#########################################################

G_W1 = tf.Variable(xavier_init([Z_dim+classes, 1024]))
G_b1 = tf.Variable(tf.zeros(shape=[1024]))

G_W2 = tf.Variable(xavier_init([1024+classes, 8*h*w]))
G_b2 = tf.Variable(tf.zeros(shape=[8*h*w]))

#[kernel h, kernel w, out chanels, in chanels]
k1_g = tf.Variable(tf.random.normal(shape=[5, 5, 64,  128+classes ],stddev=0.02))
k2_g = tf.Variable(tf.random.normal(shape=[5, 5, rgb,  64+classes ],stddev=0.02 ))

para_G = [G_W1, G_W2,  G_b1, G_b2,k1_g,k2_g]


X = tf.compat.v1.placeholder(tf.float32, shape=[None, h*w,rgb])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])

def sample_Z(m, n):

    return np.random.uniform(-1., 1., size=[m, n])



def generator(z,lab):
    size = mb_size
    z= tf.concat([z,lab],1)
    G_h1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(z, G_W1) + G_b1))#[size, 1024]
    G_h1 = tf.concat([G_h1,lab],1)

    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h = tf.reshape(G_h2, shape=[-1, int(h//4), int(w//4), 128])
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1,int(h//4), int(w//4),  1])
    f = tf.cast(f, tf.float32) #######################
    G_h = tf.concat([G_h , f],axis=3) 
    lh=int(h//4)
    lw=int(w//4)
    G_h3 = tf.nn.relu(tf.nn.conv2d_transpose(G_h,filter = k1_g,
             output_shape=[size,(lh-1)*2+2,(lw-1)*2+2,64], padding='SAME', strides = [1, 2,2, 1]))####bias1
    
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, (lh-1)*2+2,(lw-1)*2+2, 1])
    f = tf.cast(f, tf.float32)
    G_h3 = tf.concat([G_h3 , f],axis=3)
    lh=(lh-1)*2+2
    lw=(lw-1)*2+2
    
    G_h4 = tf.nn.conv2d_transpose(G_h3,output_shape=[size,(lh-1)*2+2,(lw-1)*2+2,rgb],
            filter = k2_g, padding='SAME', strides = [1, 2,2, 1]) #+bias2
    G_prob = tf.nn.sigmoid(G_h4)
    G_prob = tf.reshape(G_prob,shape=[-1, h*w, rgb])
    
    return G_prob
    

def discriminator(x,lab):
    x = tf.reshape(x ,shape=[-1, h,w, rgb])
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, h,w, 1])
    f = tf.cast(f, tf.float32)
    x = tf.concat([x , f], axis=3)  
    
    D_h1=tf.nn.conv2d(x, padding='SAME', filter = k1_d, strides = [1, 2,2, 1])
    D_h1=tf.layers.batch_normalization(D_h1)
    D_h1 = tf.nn.leaky_relu(D_h1)
    lh = D_h1.shape[1]
    lw = D_h1.shape[2]
    f = tf.tile(lab[:, tf.newaxis, tf.newaxis, :], multiples=[1, lh,lw, 1])
    f = tf.cast(f, tf.float32)
    D_h1 = tf.concat([D_h1 , f], axis=3)
    
    D_h2=tf.nn.conv2d(D_h1, padding='SAME', filter = k2_d, strides = [1, 2,2, 1])
    D_h2=tf.layers.batch_normalization(D_h2)
    D_h2 = tf.nn.leaky_relu(D_h2)
    
    D_h2= tf.reshape(D_h2, shape=[mb_size,-1])
    D_h2 = tf.concat( [D_h2,lab],1)
   
    D_h3 =  tf.matmul(D_h2, D_W1)+D_b1
    D_h3 = tf.layers.batch_normalization(D_h3)
    D_h3 = tf.nn.leaky_relu(D_h3)
    D_h3 = tf.concat( [D_h3,lab],1)
    D_logit = tf.matmul(D_h3, D_W2)+D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    
    return D_prob, D_logit
  

G_sample = generator(Z,y)

D_real, D_logit_real = discriminator(X,y)

D_fake, D_logit_fake = discriminator(G_sample,y)



D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))

D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

t = tf.compat.v1.summary.scalar("loss", G_loss+D_loss)

d_loss_sum = tf.compat.v1.summary.scalar("d_loss", D_loss) 

g_loss_sum = tf.compat.v1.summary.scalar("g_loss", G_loss)
summary_writer = tf.compat.v1.summary.FileWriter('snapshots/', graph=tf.compat.v1.get_default_graph()) 


cgan_d = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
cgan_g =tf.compat.v1.train.AdamOptimizer(lr, beta1=0.5)


G_solver = cgan_g.minimize(loss =G_loss , var_list=para_G)

D_solver = cgan_g.minimize(loss =D_loss , var_list=para_D)


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
image, label = getdata(type)
#image = tf.compat.v1.convert_to_tensor(image)
lab_sam  = label[120:120+n]
### happy expression
index = train(label,mb_size)

for it in range(1000):
    with tf.device('/cpu:0'):
        for i in range(int(label.shape[0]//mb_size)): 
       
            X_mb, x_label =  next_batch(index,i,image, label,mb_size)
            
            
            _, D_loss_curr ,d_loss_sum_value= sess.run([D_solver, D_loss,d_loss_sum], feed_dict={X: X_mb, y:x_label,Z:sample_Z(mb_size, Z_dim)})

            _, G_loss_curr,g_loss_sum_value = sess.run([G_solver, G_loss,g_loss_sum], feed_dict={Z: sample_Z(mb_size, Z_dim),y:x_label})
            if i % 10 == 0:
                summary_writer.add_summary(d_loss_sum_value, i)
                summary_writer.add_summary(g_loss_sum_value, i) 
                samples = sess.run(G_sample, feed_dict={ Z: sample_Z(n, Z_dim),y:lab_sam})         
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
            if i % 20 ==0 :
                a = samples.reshape([-1,h,w,rgb])
                b = np.array(int(a[1]*255))
                plt.imshow(b.reshape(size=[h,w,rgb]))
           
                plt.savefig('out/{}.jpg'.format(str(ii).zfill(3)), bbox_inches='tight')
        
                ii += 1
        index = train(label,mb_size)
    save(saver, sess, 'snapshots/', i)    
