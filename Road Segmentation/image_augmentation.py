# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:54:19 2017

@author: Andreas
"""

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

# get image data:
sateliteImages = np.zeros((100,400,400,3))
sateliteImages = sateliteImages.astype(np.uint8)

for i in range (1, 101):
    zeros = ''
    if(i < 10):
        zeros = '00'
    if(i < 100 and i >=10):
        zeros = '0'
    path = "./training/images/satImage_"+zeros+ str(i) +".png"
    print(path)
    image = misc.imread(path)
    sateliteImages[i-1] = image
    sateliteImages[i-1].astype(np.uint8)
print(sateliteImages.shape)

#%%

#   rotating/mirroring images

mirroredPic = np.zeros((800,400,400,3))
mirroredPic = mirroredPic.astype(np.uint8)
for i in range (0, 100):
    mirroredPic[i] = np.flip(sateliteImages[i,:,:,:],0)

for i in range (100, 200):
    mirroredPic[i] = np.flip(sateliteImages[i-100,:,:,:],1)

for i in range (200, 300):
    for d in range (0, 3):
        mirroredPic[i,:,:,d] = np.transpose(sateliteImages[i-200,:,:,d])

for i in range (300, 400):
    mirroredPic[i] = np.flip(mirroredPic[i-100,:,:,:],0)

for i in range (400, 500):
    mirroredPic[i] = np.flip(mirroredPic[i-200,:,:,:],1)
    
for i in range (500, 600):
    mirroredPic[i] = np.flip(mirroredPic[i-100,:,:,:],0)
    
for i in range (600, 700):
    mirroredPic[i] = np.flip(mirroredPic[i-600,:,:,:],1)
    
mirroredPic[700:] = sateliteImages[0:]
    

#%%


#im = sateliteImages[99,:,:,:]
#plt.imshow(im)
#plt.show()

#plt.imshow(mirroredPic[99,:,:,:])
#plt.show()

#plt.imshow(mirroredPic[199,:,:,:])
#plt.show()

#plt.imshow(mirroredPic[299,:,:,:])
#plt.show()

#plt.imshow(mirroredPic[399,:,:,:])
#plt.show()

#plt.imshow(mirroredPic[499,:,:,:])
#plt.show()

#plt.imshow(mirroredPic[599,:,:,:])
#plt.show()

#plt.imshow(mirroredPic[699,:,:,:])
#plt.show()


#%%

# adding noise to images

mirroredPicsNoisy = np.zeros((800,400,400,3))
mirroredPicsNoisy = mirroredPicsNoisy.astype(np.uint8)

for i in range(0, mirroredPic.shape[0]):
    noise = np.random.rand(400,400)*30
    temp = np.zeros((400,400,3))
    for t in range(0,3):
        temp[:,:,t] = noise
    
    noise = temp - 15
    noise = noise.astype(np.uint8)
    
    mirroredPicsNoisy[i,:,:,:] = mirroredPic[i,:,:,:] + noise





plt.imshow(mirroredPicsNoisy[-1,:,:,:])
plt.show()
print(mirroredPicsNoisy[699])

plt.imshow(mirroredPic[-1,:,:,:])
plt.show()
print(mirroredPic[699])

#%%

augmentedImages = np.zeros((4800,400,400,3))
augmentedImages = augmentedImages.astype(np.uint8)

#augmentedImages[0:100,:,:,:] = sateliteImages[0:,:,:,:]
augmentedImages[0:800,:,:,:] = mirroredPic[0:,:,:,:]
augmentedImages[800:1600,:,:,:] = mirroredPicsNoisy[0:,:,:,:]


#%%

# take image parts, and scale them up

# per image, take 2 random parts, sized 200x200

counter = 1600

for i in range(0, 1600):
    for j in range(0,2):
        cropStartX = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
        cropStartY = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
        
        im = augmentedImages[i,cropStartX:cropStartX+200,cropStartY:cropStartY+200,:]
        im = np.repeat(np.repeat(im,2, axis=0), 2, axis=1)
        augmentedImages[counter] = im
        counter = counter + 1

