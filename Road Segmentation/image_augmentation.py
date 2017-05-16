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

GTImages = np.zeros((100,400,400))
GTImages = GTImages.astype(np.uint8)

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
    
    path = "./training/groundtruth/satImage_"+zeros+ str(i) +".png"
    print(path)
    image = misc.imread(path)
    GTImages[i-1] = image
    GTImages[i-1].astype(np.uint8)
    
    
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

def getRandomImagePartAsNewImage(inputImage):
    cropStartX = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
    cropStartY = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
    if(len(GTImages[-1,:,:].shape) > 2):
        im = inputImage[cropStartX:cropStartX+200,cropStartY:cropStartY+200,:]
    else:
        im = inputImage[cropStartX:cropStartX+200,cropStartY:cropStartY+200]        
    im = np.repeat(np.repeat(im,2, axis=0), 2, axis=1)
    
    return im

#%%


counter = 1600

for i in range(0, 1600):
    for j in range(0,2):
        
        augmentedImages[counter] = getRandomImagePartAsNewImage(augmentedImages[i,:,:,:])
        counter = counter + 1


#%%

# returns tupels with coordinates where the road segment starts, and ends. additional there are 2 numbers indicating which side of the image the road is.
# 0 , 0 is the top of the image
# 0 , 1 is the bottom of the image
# 1 , 0 is the left side of the image
# 1 , 1 is the right side of the image


def findStreets(image):
    foundRoad = False
    
    roads = set()
    
    for i in range(0, image.shape[0]):
        if(image[0, i] >= 10 and not foundRoad):
            roadStartCoord = i
            foundRoad = True
        else:
            if(foundRoad and image[0, i] < 10):
                foundRoad = False
                roadEndCoord = i
                roads.add((roadStartCoord, roadEndCoord, 0 , 0))
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        roadEndCoord = i
        print('adding option 2')
        roads.add((roadStartCoord, roadEndCoord, 0 , 0))

    for i in range(0, image.shape[0]):
        if(image[image.shape[0]-1, i] >= 10 and not foundRoad):
            roadStartCoord = i
            foundRoad = True
        else:
            if(foundRoad and image[image.shape[0]-1, i] < 10):
                foundRoad = False
                roadEndCoord = i
                roads.add((roadStartCoord, roadEndCoord, 0 , 1))
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        roadEndCoord = i
        roads.add((roadStartCoord, roadEndCoord, 0 , 1))
        
    
    for i in range(0, image.shape[1]):
        if(image[i, 0] >= 10 and not foundRoad):
            roadStartCoord = i
            foundRoad = True
        else:
            if(foundRoad and image[i, 0] < 10):
                foundRoad = False
                roadEndCoord = i
                roads.add((roadStartCoord, roadEndCoord, 1 , 0))
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        roadEndCoord = i
        roads.add((roadStartCoord, roadEndCoord, 1 , 0))

    for i in range(0, image.shape[0]):
        if(image[i, image.shape[1]-1] >= 10 and not foundRoad):
            roadStartCoord = i
            foundRoad = True
        else:
            if(foundRoad and image[i, image.shape[1]-1] < 10):
                foundRoad = False
                roadEndCoord = i
                roads.add((roadStartCoord, roadEndCoord, 1 , 1))
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        roadEndCoord = i
        roads.add((roadStartCoord, roadEndCoord, 1 , 1))
        
    return roads



#%%
test = getRandomImagePartAsNewImage(GTImages[-1,:,:])


plt.imshow(test)
plt.show()

s = findStreets(test)
print(s)

#%%

print(test[0,132:210])