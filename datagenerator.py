#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:25:37 2019

@author: tonee
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization , Concatenate,Conv2D
from tensorflow.python.keras.layers.core import Dropout, Lambda
from tensorflow.python.keras.layers.convolutional import  Conv2DTranspose

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.optimizers import RMSprop, Adam

from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
import cv2

batchsize=5
batchsize_val=12



img_rows=512
img_cols=512

 



    
    
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)




    
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss



def unet_512(input_shape=(512, 512, 3),
                 num_classes=1, dropout = 0.25):
    inputs = Input(shape=input_shape)
#    s = Lambda(lambda x: x / 255) (inputs) 
#       
#     512
    
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    p2 = Dropout(dropout)(down0_pool)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(p2)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    p3 = Dropout(dropout)(down1_pool)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(p3)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    p4 = Dropout(dropout)(down2_pool)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(p4)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    p5 = Dropout(dropout)(down3_pool)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(p5)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    p6 = Dropout(dropout)(down4_pool)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(p6)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    p7 = Dropout(dropout)(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(p7)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    p8 = Dropout(dropout)(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(p8)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    p9 = Dropout(dropout)(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(p9)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    p10 = Dropout(dropout)(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(p10)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    p11 = Dropout(dropout)(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(p11)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    

    # 512
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=[inputs], outputs=[classify])
    
    model.load_weights('/home/tonee/.config/spyder-py3/deeplearning practise/line_crop_best_9000.h5')


#    model.compile(optimizer = RMSprop(lr=0.001), loss = [jacard_coef_loss], metrics = [jacard_coef])
#    model.compile(optimizer = Adam(lr = 1e-4), loss = [jacard_coef_loss], metrics = [jacard_coef])
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff, jacard_coef])
#    model.summary()
    
    return model


model=unet_512()


tensorboard = TensorBoard(log_dir='/home/tonee/segmentation/mapilary_train/logs', histogram_freq=0,
                          write_graph=True, write_images=False)
earlystopper = EarlyStopping(monitor='val_loss',patience=8, verbose=1)
checkpointer = ModelCheckpoint('line_crop_best_9000_roma.h5', verbose=1, save_best_only=True)
reduce=ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4)


#data_gen_args = dict(rescale =1./255
#                     
#                     )

def preprocess(qwe):
    zero_arr = np.zeros((1,img_rows,img_cols,1), dtype=np.uint8)
    
    if (qwe.size == 262144):
#        new_qwe = qwe/qwe
#        zero_arr[0]=new_qwe
#        return zero_arr[0]
        return qwe/255.0
    else:
        return qwe/255.0
    







data_gen_args = dict(zoom_range=[0.8,1.2],
                     brightness_range=[0.7,1.3],
                     horizontal_flip=True,
                     rotation_range=7,
                     shear_range=0.05,
                     preprocessing_function = preprocess)





#
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)







seed = 69

image_generator = image_datagen.flow_from_directory('/home/tonee/segmentation/roma_v2/image', target_size=(img_rows,img_cols),
                                                        class_mode=None, seed=seed, batch_size=batchsize, color_mode='rgb')


mask_generator = mask_datagen.flow_from_directory('/home/tonee/segmentation/roma_v2/mask', target_size=(img_rows,img_cols),
                                                       class_mode=None, seed=seed, batch_size=batchsize, color_mode='grayscale')


    
plt.imshow(image_generator[2][0])
plt.imshow(np.squeeze(mask_generator[2][0]), alpha = 0.4)
plt.show()
    
    


train_generator = zip(image_generator, mask_generator)
num_train = len(image_generator)


val_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen_m = ImageDataGenerator(rescale = 1./255)



val_image_generator = val_datagen.flow_from_directory('/home/tonee/segmentation/roma_v2/val_image', target_size=(img_rows,img_cols),
                                                        class_mode=None, seed=seed, batch_size=batchsize_val, color_mode='rgb')
##
#
val_mask_generator = val_datagen_m.flow_from_directory('/home/tonee/segmentation/roma_v2/val_mask', target_size=(img_rows,img_cols),
                                                       class_mode=None, seed=seed, batch_size=batchsize_val, color_mode='grayscale')








val_generator = zip(val_image_generator, val_mask_generator)
num_val = len(val_image_generator)
#
model.fit_generator(
    train_generator,
    steps_per_epoch=num_train,
    epochs=100,
    callbacks=[earlystopper, checkpointer,reduce,tensorboard],
    validation_data=val_generator,
    validation_steps=num_val)





print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("map_512_line2.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("map_line_512_crop_68_roma.h5")
print("Сохранение сети завершено")
