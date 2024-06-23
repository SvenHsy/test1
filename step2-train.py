# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 23:28:53 2020

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


epoch=150000 
batch_size=128
c_dim=1
#读取h5文件
is_train=True   #训练时为True,测试时改为False，因为预测和测试分开了，因此就一致True就行啦

if is_train:     
    data_dir = os.path.join(os.getcwd(), 'h5/train.h5')
    padding="VALID"
    trainable = tf.Variable(True, dtype=tf.bool)


path=data_dir #获取训练数据集的路径
with h5py.File(path, 'r') as hf:         #读取h5文件
    train_data = np.array(hf.get('data'))
    train_label = np.array(hf.get('label'))


images = tf.placeholder(tf.float32, [None, None, None, c_dim], name='images')
labels = tf.placeholder(tf.float32, [None, None, None, c_dim], name='labels')

weights = {
      'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3),trainable=trainable, name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3),trainable=trainable, name='w2'),
      'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), trainable=trainable,name='w3')
    }
biases = {
      'b1': tf.Variable(tf.zeros([64]),trainable=trainable ,name='b1'),
      'b2': tf.Variable(tf.zeros([32]),trainable=trainable, name='b2'),
      'b3': tf.Variable(tf.zeros([1]),trainable=trainable, name='b3')
    }
conv1 = tf.nn.relu(tf.nn.conv2d(images, weights['w1'], strides=[1,1,1,1], padding=padding) + biases['b1'])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding=padding) + biases['b2'])
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding=padding) + biases['b3']


pred=conv3 
loss = tf.reduce_mean(tf.square(labels - pred)) #loss函数为mse值

if is_train:
     
     var_list1=[weights['w1'],biases['b1'],weights['w2'],biases['b2']]
     var_list2=[weights['w3'],biases['b3']]
     
     opt1 = tf.train.GradientDescentOptimizer(1e-4)  #前两层参数学习率为1e-4
     opt2 = tf.train.GradientDescentOptimizer(1e-5)  #第三层参数学习率为1e-5
     grads = tf.gradients(loss, var_list1 + var_list2)
     grads1 = grads[:len(var_list1)]
     grads2 = grads[len(var_list1):]
     train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
     train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
     train_op = tf.group(train_op1, train_op2)
            
     #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

counter = 0
start_time = time.time() #记录开始时间
saver=tf.train.Saver(max_to_keep=5)  #只保留最近的五个模型的参数值


with tf.Session() as sess:

    if is_train:
      print("Training...")
      sess.run(tf.initialize_all_variables())
      
      ckpt = tf.train.get_checkpoint_state("checkpoint")      
      if ckpt and ckpt.model_checkpoint_path:  # 加载上次训练保存的模型继续训练
       print("Continuing ")
       saver.restore(sess, ckpt.model_checkpoint_path)    
       
      for ep in xrange(epoch):
        # Run by batch images
        batch_idxs = len(train_data) // batch_size  
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*batch_size : (idx+1)*batch_size]
          batch_labels = train_label[idx*batch_size : (idx+1)*batch_size]

          counter +=1
          _, err = sess.run([train_op, loss], feed_dict={images: batch_images, labels: batch_labels})

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            saver.save(sess,os.path.join('checkpoint', 'SRCNN'),global_step=counter,write_meta_graph=False) 
            
            #img1=(weights['w1'].eval())
            #img2=(weights['w2'].eval())
            #img3=(weights['w3'].eval())
            