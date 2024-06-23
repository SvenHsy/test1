# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 23:24:24 2020

@author: Administrator
"""

import os
import h5py
import numpy as np
import tensorflow as tf
import time
import scipy.misc
import scipy.ndimage

try:
  xrange
except:
  xrange = range

  



def pnsr(img1,img2):
    (x,y)=np.shape(label)
    img1=img1[3:(x-3),3:(y-3)] #减去边缘，matlab中即减去边缘的三个像素点
    img2=img2[3:(x-3),3:(y-3)] 
    diff = np.abs(img1*255.0- img2*255.0)
    mse = np.square(diff).mean() #mse表示当前图像与原有图像的均方误差
    psnr = 20 * np.log10(255 / np.sqrt(mse)) #评价指标即峰值信噪比
    return psnr

def imsave(image, path):
  return scipy.misc.imsave(path, image) 


data_dir = os.path.join(os.getcwd(), 'h5/test.h5')  #h5文件的路径
padding="SAME"                                      #因为是预测，所以需要填充
trainable = tf.Variable(False, dtype=tf.bool)       
path=data_dir                                         #路径

with h5py.File(path, 'r') as hf:         #读取h5文件
    train_data = np.array(hf.get('data'))
    train_label = np.array(hf.get('label'))

    
#待喂数据  
c_dim=1    
images = tf.placeholder(tf.float32, [None, None, None, c_dim], name='images')
labels = tf.placeholder(tf.float32, [None, None, None, c_dim], name='labels')


#网络
weights = {
      'w1': tf.Variable(tf.zeros([9, 9, 1, 64]),trainable=trainable, name='w1'),
      'w2': tf.Variable(tf.zeros([1, 1, 64, 32]),trainable=trainable, name='w2'),
      'w3': tf.Variable(tf.zeros([5, 5, 32, 1]), trainable=trainable,name='w3')
    }
biases = {
      'b1': tf.Variable(tf.zeros([64]),trainable=trainable ,name='b1'),
      'b2': tf.Variable(tf.zeros([32]),trainable=trainable, name='b2'),
      'b3': tf.Variable(tf.zeros([1]),trainable=trainable, name='b3')
    }
conv1 = tf.nn.relu(tf.nn.conv2d(images, weights['w1'], strides=[1,1,1,1], padding=padding) + biases['b1'])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding=padding) + biases['b2'])
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding=padding) + biases['b3']


saver=tf.train.Saver()

pred=conv3
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("checkpoint")
    if ckpt and ckpt.model_checkpoint_path:  # 加载保存的模型
       saver.restore(sess, ckpt.model_checkpoint_path)           
       #img1 = (weights['w1'].eval())  #查看卷积核是否在变化                     
       result = pred.eval({images: train_data, labels: train_label}) #得到训练后的结果
       result1 = result.squeeze()                                    #降维
       result2 =np.around(result1 ,decimals=4)                       #取小数点的后四位
       image_path = os.path.join(os.getcwd(),'sample')                #保存预测的图片到sample文件夹中
       image_path = os.path.join(image_path, "test_image.png")
       imsave(result2, image_path)                       
       label=train_label.squeeze()                                    #label数据降维
       print(pnsr(label,result2))                                     #计算并打印pnsr值
   