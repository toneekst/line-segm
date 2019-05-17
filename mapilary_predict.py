#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 12:34:55 2019

@author: tonee
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:23:21 2018

@author: tonee
"""
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from tensorflow.python.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras.models import  model_from_json
import os
import cv2
from tensorflow.python.keras.preprocessing import image
import math
from tensorflow.python.keras.optimizers import RMSprop,Adam


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


num=6

#json_file = open("road_mark_v5_Adam.json", "r")
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("road_mark_v5_Adam.h5")
#loaded_model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
#
#model = loaded_model



###
#json_file = open("map_512_line.json", "r")
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
##loaded_model.load_weights("lane_mapilary_68batch_withfloatmask.h5")
#loaded_model.load_weights("line_crop_best_9000_roma.h5")
#loaded_model.compile(optimizer=RMSprop(lr=0.0001), loss=[bce_dice_loss], metrics=[dice_coeff])
#
#model = loaded_model
#
img = imread('/home/tonee/segmentation/mapilary/testing/images/_8Vr5ty8fzJwigfnnZdIEg.jpg')
img = resize(img, (512, 512))

squ = imread('/home/tonee/segmentation/test3.png')
squ = resize(squ, (500, 500))
plt.imshow(squ)
#
#plt.show()

##
##
#
##
#
#x = np.expand_dims(img, axis=0)
##
#predict=model.predict(x,verbose=1)
#plt.imshow(img)
#plt.show()
##
#plt.imshow(np.squeeze(predict))
#plt.show()
##
#pred = (predict > 0.25).astype(np.uint8)
#squ=(np.squeeze(pred))*255
#plt.imshow(img)
#plt.imshow(squ, alpha=0.7)
#plt.show()
#plt.imshow(img) # Show results
#plt.show()
#
#save_img=imsave('/home/tonee/segmentation/test3.png',squ)


img = resize(img, (500, 500))

def perspective(images,mask_images,rshag):
    plt.imshow(images) # Show results
    plt.show()
    
    
    IMAGE_H = rshag
    IMAGE_W = 500
        
    src = np.float32([[0, 500-rshag], [500, 500-rshag], [0, 500], [500, 500]])
        
    dst = np.float32([[0, 0], [500, 0], [rshag+75, rshag], [500-rshag-75, rshag]])
        
    print (src)
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        
    warped_img = cv2.warpPerspective(images, M, (IMAGE_W, IMAGE_H)) # Image warping
    warped_img_mask = cv2.warpPerspective(mask_images, M, (IMAGE_W, IMAGE_H))   
    
    plt.imshow(warped_img) # Show results
    plt.show()
    plt.imshow(warped_img_mask) # Show results
    plt.show()
    
    
    
    hist = []
    column_width = 10 
    
    for x in range(2,(int(500/column_width))-2):
    
    
        
        
        su =np.sum(warped_img_mask[0:rshag,x*column_width:column_width+x*column_width])
        
       
        
        su_plus=np.sum(warped_img_mask[0:rshag,(x+1)*column_width:column_width+(x+1)*column_width])
        su_plus2=np.sum(warped_img_mask[0:rshag,(x+2)*column_width:column_width+(x+2)*column_width])
    
        su_minus = np.sum(warped_img_mask[0:rshag,(x-1)*column_width:column_width+(x-1)*column_width])
        su_minus2 = np.sum(warped_img_mask[0:rshag,(x-2)*column_width:column_width+(x-2)*column_width])
    
        if (su!=0 and su>su_plus and su>su_plus2 and su>su_minus and su>su_minus2 and  su>50):
        
            hist.append(x*column_width)
            print(su,x*column_width)
    
    
    
    
    
    
    
    
    
    
    
    
    return  warped_img,warped_img_mask , hist  







#IMAGE_H = 200
#IMAGE_W = 500
#
#src = np.float32([[6, 300], [506, 300], [6, 512], [506, 512]])
#dst = np.float32([[0, 0], [500, 0], [200, 200], [300, 200]])
#M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
#Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
#
##squ = squ[380:512, 0:512]
##plt.imshow(squ) # Show results
##plt.show()
#warped_imgs = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
#
#
#warped_img = cv2.warpPerspective(squ, M, (IMAGE_W, IMAGE_H)) # Image warping
#plt.imshow(warped_img) # Show results
#plt.show()
#
#hist = []
#column_width = 10   # this allows you to speed up the result,
           
        # at the expense of horizontal resolution. (higher is faster)
        
        
#def find_line_nums(warped_img):
#    hist = []
#    column_width = 10 
#    
#    for x in range(2,(int(500/column_width))-2):
#    
#    
#    
#        su =np.sum(warped_img[0:170,x*column_width:column_width+x*column_width])/255
#        
#        su_plus=np.sum(warped_img[0:170,(x+1)*column_width:column_width+(x+1)*column_width])/255
#        su_plus2=np.sum(warped_img[0:170,(x+2)*column_width:column_width+(x+2)*column_width])/255
#    
#        su_minus = np.sum(warped_img[0:170,(x-1)*column_width:column_width+(x-1)*column_width])/255
#        su_minus2 = np.sum(warped_img[0:170,(x-2)*column_width:column_width+(x-2)*column_width])/255
#    
#        if (su!=0 and su>su_plus and su>su_plus2 and su>su_minus and su>su_minus2 and  su>50):
#        
#            hist.append(x*column_width)
#            print(su,x*column_width)
#            
#    return hist   


 
#for x in range(2,(int(500/column_width))-2):
#    
#    
#    
#    su =np.sum(warped_img[0:170,x*column_width:column_width+x*column_width])/255
#    su_plus=np.sum(warped_img[0:170,(x+1)*column_width:column_width+(x+1)*column_width])/255
#    su_plus2=np.sum(warped_img[0:170,(x+2)*column_width:column_width+(x+2)*column_width])/255
#    
#    su_minus = np.sum(warped_img[0:170,(x-1)*column_width:column_width+(x-1)*column_width])/255
#    su_minus2 = np.sum(warped_img[0:170,(x-2)*column_width:column_width+(x-2)*column_width])/255
#    
#    if (su!=0 and su>su_plus and su>su_plus2 and su>su_minus and su>su_minus2 and  su>50):
#        
#        hist.append(x*column_width)
#        print(su,x*column_width)
        
    








def find_lines(imag_mask ,imags,hist,kol):
    nums=len(hist)
    
    if (nums!=None):
        
        
        
        
        
        mid_dot=[]
        for dot in hist:
            xverhh=[]
            xnizz=[]
            for num in range(0,kol):
                
                y = np.shape(imag_mask)[0]
                
                dy = int(y/kol)
                
                print (np.shape(imags))
                
                mask = np.zeros((y,500), dtype = "uint8")
                
                maskimg = np.zeros(np.shape(imags), dtype = "float64")
                
                cv2.rectangle(mask, (dot-30, 0+(dy*num)), (dot+30, 0+(dy*(num+1))), (255, 255, 255), -1)
                cv2.rectangle(maskimg, (dot-30, 0+(dy*num)), (dot+30, 0+(dy*(num+1))), (255, 255, 255), -1)
                
                
                masked = cv2.bitwise_and(imag_mask, mask)
                
                masked_img = cv2.bitwise_and(imags, maskimg)
                
                sliced_image = imag_mask[0+(dy*num):0+(dy*(num+1)),dot-30:dot+30]
            
                sliced_image2 = imags[0+(dy*num):0+(dy*(num+1)),dot-30:dot+30]
            
                blur = cv2.blur(masked,(7,7))
                
                
                
            
                canny = cv2.Canny(blur, 10, 250, None, 3)
            
                lines = cv2.HoughLinesP(canny,rho = 1,theta = 1*np.pi/100,threshold = 20,minLineLength = 15,maxLineGap = 50)
            
                if (np.any(lines) != None):
                
                    for x1,y1,x2,y2 in lines[0]:
                        cv2.line(imags,(x1,y1),(x2,y2),(0,255,0),3)
                    
#                    print(lines[0][0])        
                    plt.imshow(imags) 
                    plt.show()
            
                    x1,y1,x2,y2 =lines[0][0][0],lines[0][0][1],lines[0][0][2],lines[0][0][3]
                
                    if ((x2-x1)==0):
                    
                        tangens = (y2-y1)
                    else:
                    
                        tangens = ((y2-y1)/(x2-x1))
                
                    
#                    print(tangens)
#                    print(math.degrees(math.atan(tangens)))
                    print (lines[0][0][0],lines[0][0][1],lines[0][0][2],lines[0][0][3] )
                    
                    if (len(xnizz)==0):
                        xv = (tangens*lines[0][0][0]+(num*dy)-lines[0][0][1])/tangens
                        
                    else:
                        print (num)
                        print (xnizz)
                        xv  = xnizz[-1]
                        
           
#                    xverh=(0+num*dy-y/2+tangens*lines[0][0][0])/tangens
#                    print(xverh)
                    
                    
                    print ('xv = ' + str(xv))
                   
#                    xniz=(0+(num+1)*dy-y/2+tangens*lines[0][0][2])/tangens
#                    print(xniz)
                    
                    xn= (tangens*lines[0][0][2]+((num+1)*dy)-lines[0][0][3])/tangens
                    
                    print('xn = '+str(xn))
                    
                    
                    xverhh.append(xv)
                    xnizz.append(xn)
            
#                    print(xverh,xniz)
            
            print('line ' + str(dot))
            
            
            
            for num in range(0,len(xverhh)):
                
                x1,y1,x2,y2 =int(xverhh[num]),0+(dy*num),int(xnizz[num]),0+(dy*(num+1))
                
#                print (x1,y1,x2,y2)
                cv2.line(imags,(x1,y1),(x2,y2),(255,0,0),2)
            
            x_val = []
            if (175<dot<325):
                for num in range(0,len(xverhh)):
                    x1,y1,x2,y2 =int(xverhh[num]),0+(dy*num),int(xnizz[num]),0+(dy*(num+1))
                    x_val.append(x1)
                    x_val.append(x2)
                    
                x_mid = sum(x_val)/len(x_val)
                mid_dot.append(x_mid)
                
                
#            print(xverhh[num],0)
#            print(xnizz[num],200)
                
                
                
        print  (mid_dot)  
        if (len(mid_dot)>0):
            if (len(mid_dot)==2):
                x_center =sum(mid_dot)/2
            elif (len(mid_dot)==1):
                if (mid_dot[0]<250):
                    x_center = mid_dot[0]+45
            else:
                x_center = mid_dot[0]-45
                 
            cv2.circle(imags,(int(x_center),50), 2, (0,0,255), -1)    
        else:
            print('image witout lines')
        
        cv2.circle(imags,(250,50), 2, (255,0,255), -1)
        plt.imshow(imags) 
        plt.show()



imagees,mask,hist = perspective(img,squ,rshag=140)                
mask  = mask*255
mask = mask.astype(np.uint8)


find_lines(mask,imagees,hist,kol=2)


#def find_lines(imag_mask ,imags,nums= len(hist)):
#    if (nums!=None):
#        
#        
#        
#        
#        
#        xverhh=[]
#        xnizz=[]
#        for dot in hist:
#            
#            sliced_image = imag_mask[0:200,dot-30:dot+30]
#            
#            sliced_image2 = imags[0:200,dot-30:dot+30]
#            
#            blur = cv2.blur(sliced_image,(7,7))
#            
#            plt.imshow(blur) 
#            plt.show()
#            
#            canny = cv2.Canny(blur, 10, 250, None, 3)
#            
#            lines = cv2.HoughLinesP(canny,rho = 1,theta = 1*np.pi/180,threshold = 20,minLineLength = 20,maxLineGap = 50)
#            
#            if (np.any(lines) != None):
#                
#                for x1,y1,x2,y2 in lines[0]:
#                    cv2.line(sliced_image2,(x1,y1),(x2,y2),(0,255,0),3)
#                    
#                print(lines[0][0])        
#                plt.imshow(sliced_image2) 
#                plt.show()
#            
#                x1,y1,x2,y2 =lines[0][0][0],lines[0][0][1],lines[0][0][2],lines[0][0][3]
#                
#                if ((x2-x1)==0):
#                    
#                    tangens = (y2-y1)
#                else:
#                    
#                    tangens = ((y2-y1)/(x2-x1))
#                
#                    
#                print(tangens)
#                print(math.degrees(math.atan(tangens)))
#            
#                xverh=(0-50+tangens*dot)/tangens
#                xniz=(200-50+tangens*dot)/tangens
#            
#                xverhh.append(xverh)
#                xnizz.append(xniz)
#            
#                print(xverh,xniz)
#            
#            
#        for num in range(0,len(xverhh)):
#            x1,y1,x2,y2 =int(xverhh[num]),0,int(xnizz[num]),200
##            print(xverhh[num],0)
##            print(xnizz[num],200)
#            cv2.line(imags,(x1,y1),(x2,y2),(255,0,0),2)
#        plt.imshow(imags) 
#        plt.show()















#median = cv2.medianBlur(squ,33)
#plt.imshow(median)
#plt.show()
#save_img=imsave('/home/tonee/segmentation/image_mask/' +'mask'+ str(num) +'.png',squ)
#plt.imshow(predict)
#plt.show()





#image_path_train = '/home/tonee/segmentation/bdd100k/seg/images/test/'
#mask_path_train='/home/tonee/segmentation/roma_v2/mask/0/'
#
#
#train_ids = next(os.walk(image_path_train))[2]


#for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#    path = image_path_train + id_
#    img = imread(path)
#    img = resize(img, (256, 256))
#    x = np.expand_dims(img, axis=0)
#    predict=model.predict(x,verbose=1)
#    pred = (predict > 0.9).astype(np.uint8)
##    path_m=mask_path_train + id_
##    mask = imread(path_m)
#    
#    
#    
#    
#    plt.imshow(img)
#    plt.show()
##    plt.imshow(mask)
##    plt.show()
#    plt.imshow(np.squeeze(pred))
#    plt.show()
    
    