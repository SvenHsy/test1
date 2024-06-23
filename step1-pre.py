# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 22:31:24 2020

@author: Administrator
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from scipy.misc import imread
from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import scipy.misc as smi
import tensorflow as tf

try:
  xrange
except:
  xrange = range


def rgb2ycbcr(img, only_y=True):  #自己重新写的rgb2ycbcr函数以求对应到matlab的rgb2ycbcr函数
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


  
is_train=False  #预处理训练数据时时为True,预处理测试数据时改为False
scale=3;       #插值规模

#定义一个保存图片的函数
def imsave(image, path):
  return scipy.misc.imsave(path, image)   

#将数据读入进来
if  is_train:
        dataset="Train"
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
else:
        dataset="Test"
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))

   
    
if is_train:
        image_size=33  #训练输入的图片的大小33*33
        stride=14      #
        label_size=21  #训练输入图片经过训练变成21*21大小
        padding = abs(image_size - label_size) / 2 # 6像素点的边缘
        sub_input_sequence = []
        sub_label_sequence = []
        for i in xrange(len(data)):   #xrange产生随机数且优于range函数
        
          image=imread(data[i])
          image=rgb2ycbcr(image) #自己写rgb2ycbcr函数可取
          #image=scipy.misc.imread(data[i], flatten=True, mode='YCbCr').astype(np.float) #提取Y通道
          
          if len(image.shape) == 3:
            h, w, _ = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            label_ = image[0:h, 0:w, :] #长宽进行裁剪，第三维原样，但这里并没有变成33*33
          else:
            h, w = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            label_ = image[0:h, 0:w]
          image = image / 255.
          label_ = label_ / 255.
          
          #进行两次插值构造低分辨率图片
          label_1=Image.fromarray(label_)
          input_= label_1.resize(( w // scale,h // scale),Image.BICUBIC)
          input_= input_.resize((w,h), Image.BICUBIC)
          input_=np.float64(input_)
          
          #保存四位小数
          label_=np.around(label_, decimals=4)
          input_=np.around(input_,decimals=4)
          
          #下面这个插值函数与matlab中不一致，因此舍弃
          #input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)#进行双三次插值变为低分辨率图片
          #input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)#再次进行双三次插值变为与高分辨率图片一样大小
          
          if len(input_.shape) == 3:  #取train图片的长宽，因为这里是在is_train=True下的if语句
            h, w, _ = input_.shape
          else:
            h, w = input_.shape
          for x in range(0, h-image_size+1, stride):  #以stride为步长进行取子图片操作
            for y in range(0, w-image_size+1, stride):
              sub_input = input_[x:x+image_size, y:y+image_size] # [33 x 33]
              sub_label = label_[x+int(padding):x+int(padding)+label_size, y+int(padding):y+int(padding)+label_size] # [21 x 21]
    
              # Make channel value
              sub_input = sub_input.reshape([image_size, image_size, 1])  
              sub_label = sub_label.reshape([label_size, label_size, 1])
    
              sub_input_sequence.append(sub_input) #append为在列表末尾添加新的对象
              sub_label_sequence.append(sub_label)
    
    
if not is_train:
        image_size=33
        stride=14
        label_size=21
        padding = abs(image_size - label_size) / 2 # 6
        sub_input_sequence = []
        sub_label_sequence = []
        for i in xrange(len(data)):   #xrange产生随机数且优于range函数
        
          image=imread(data[i])
          image=rgb2ycbcr(image) #自己写rgb2ycbcr函数可取
          #image=scipy.misc.imread(data[i], flatten=True, mode='YCbCr').astype(np.float)#读取方法不可取
          
          if len(image.shape) == 3:
            h, w, _ = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            label_ = image[0:h, 0:w, :] #长宽进行裁剪，第三维原样，但这里并没有变成33*33
          else:
            h, w = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            label_ = image[0:h, 0:w]
          image = image / 255.
          label_ = label_ / 255.      #此时label_为真图，后续进行psnr计算时与预测图片进行对比
          
          #方法一最好，基本达到与matlab相同水平
          label_1=Image.fromarray(label_)
          input_= label_1.resize((w//scale,h//scale),Image.BICUBIC)
          input_= input_.resize((w,h), Image.BICUBIC)
          input_=np.float64(input_)
          
          #方法二优于方法三
          #input_ =smi.imresize(label_, (85,85),interp='bicubic')
          #input_ =smi.imresize(input_, (255,255),interp='bicubic')/255.
          
          #方法三不太行
          #input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
          #input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False) 
          if len(input_.shape) == 3:  #取输入图片的长宽
            h, w, _ = input_.shape
          else:
            h, w = input_.shape
            
          #保留四位小数
          label_=np.around(label_, decimals=4)
          input_=np.around(input_,decimals=4)
          
          image_path = os.path.join(os.getcwd(),'sample')
          image_path = os.path.join(image_path, "label_image.png")
          imsave(label_, image_path)   #保存真图
          image_path = os.path.join(os.getcwd(),'sample')
          image_path = os.path.join(image_path, "input_image.png")
          imsave(input_, image_path)  #保存输入图片
          
          sub_input = input_.reshape([h, w, 1])  
          sub_label = label_.reshape([h, w, 1])
    
          sub_input_sequence.append(sub_input) #append为在列表末尾添加新的对象
          sub_label_sequence.append(sub_label)
      
arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

if is_train:
    savepath = os.path.join(os.getcwd(), 'h5/train.h5')  #os.getcwd()为获取当前工作目录
else:
    savepath = os.path.join(os.getcwd(), 'h5/test.h5')

with h5py.File(savepath, 'w') as hf:   #数据集的制作,图片大小不一样，不能转成h5，这里无效，可以在test时直接读取图片
    hf.create_dataset('data', data=arrdata)
    hf.create_dataset('label', data=arrlabel)