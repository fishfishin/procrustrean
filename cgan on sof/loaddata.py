import cv2
import os
import glob
import numpy as np
from PIL import Image
import random
import scipy.io
import matplotlib.pyplot as plt
'''
original images size is 480 x 640 x 3,and I downsize the images for accelerate the learning to h x w x 1.
there are two kinds of partition:  illumination and in/outdoor
you can choose images with good illumination(1025 images) or outdoor(829 images) for better face synthesis
total amount of images is 2662
'''
h = 200
w = 200
loaddir =".\originalimages\originalimages"
img_dir =  loaddir # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
mat = scipy.io.loadmat('metadata.mat')
f2 = mat['metadata'][0]

def labelload():
    mat = scipy.io.loadmat('metadata.mat')
    labels = []
    f2 = mat['metadata'][0]
    data = []
    for f1 in f2 :
        labels.append(f1[7][0][0])
    for f1 in files :
            img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(h ,w), cv2.INTER_CUBIC)   
            img =  np.reshape(img,(h* w))
            data.append(img)
    '''
    normal, happy, sad/angry/disgusted, and surprised/fearful
    label:1,  2,  3,  4
    '''
    data = np.array(data)
    label = np.array(labels)
    label_matrix = np.zeros(shape=[label.shape[0], label.max()])
    for i, lab in enumerate(label):
        label_matrix[i, lab-1] = 1
    
    return data,label_matrix

image_whole, lable_data=labelload()

def get_whole():
    return image_whole, lable_data

def illumination(f2,labeldata):
    label =[]
    data = []
    i=0
    for f1 in f2:
        if f1[16][0][0]==1:
          
            img = cv2.imread(files[i],cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(h ,w), cv2.INTER_CUBIC)   
            img =  np.reshape(img,(h* w))
            data.append(img)
            label.append(labeldata[i,:])
        i+=1
    data = np.array(data)
    label = np.array(label)
    #print(data.shape)
    return data,label

def outdoor(f2,labeldata):
    label =[]
    data = []
    i=0
    for f1 in f2:
        l = f1[9][0]
        if l == '1' : ###indoor 289 images with lighting
      
            img = cv2.imread(files[i],cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(h ,w), cv2.INTER_CUBIC)   
            img =  np.reshape(img,(h*w))
            data.append(img)
            label.append(labeldata[i,:])
        i+=1
    data = np.array(data)
    label = np.array(label)
    #print(data.shape)
    return data,label


def getdata(type):
    if type == "illumination":
        image1, label1= illumination(f2,lable_data)
    elif type == "indoor" or "outdoor" :
        image1, label1= outdoor(f2,lable_data)
    return image1,label1

def train(image1,mb_size):

    train_index = np.arange(start=0, stop=image1.shape[0]-mb_size, step=mb_size, dtype=int)
    return train_index

    
def fetch(data,label,mb_size):
    idx =random.sample(range(data.shape[0]), mb_size)
    batch_data=[]
    batch_label=[]
    for i in idx: 
        batch_data.append(data[i])
        batch_label.append(label[i])
    
    batch_data = np.array(batch_data)
    batch_label = np.array(batch_label)
    return batch_data,batch_label


def next_batch(train_index,i,image1,label1,mb_size) :
    idx = train_index [i]
    batch_image = image1[idx:idx+mb_size,:]
    batch_label = label1[idx:idx+mb_size,:]
    return batch_image, batch_label
  
