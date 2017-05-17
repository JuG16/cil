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

GTmirroredPic = np.zeros((800,400,400))
GTmirroredPic = GTmirroredPic.astype(np.uint8)


for i in range (0, 100):
    mirroredPic[i] = np.flip(sateliteImages[i,:,:,:],0)
    GTmirroredPic[i] = np.flip(GTImages[i,:,:],0)

for i in range (100, 200):
    mirroredPic[i] = np.flip(sateliteImages[i-100,:,:,:],1)
    GTmirroredPic[i] = np.flip(GTImages[i-100,:,:],1)

for i in range (200, 300):
    GTmirroredPic[i,:,:] = np.transpose(GTImages[i-200,:,:])
    for d in range (0, 3):
        mirroredPic[i,:,:,d] = np.transpose(sateliteImages[i-200,:,:,d])
    

for i in range (300, 400):
    mirroredPic[i] = np.flip(mirroredPic[i-100,:,:,:],0)
    GTmirroredPic[i] = np.flip(GTmirroredPic[i-100,:,:],0)

for i in range (400, 500):
    mirroredPic[i] = np.flip(mirroredPic[i-200,:,:,:],1)
    GTmirroredPic[i] = np.flip(GTmirroredPic[i-200,:,:],1)
    
for i in range (500, 600):
    mirroredPic[i] = np.flip(mirroredPic[i-200,:,:,:],1)
    GTmirroredPic[i] = np.flip(GTmirroredPic[i-200,:,:],1)
    
for i in range (600, 700):
    mirroredPic[i] = np.flip(mirroredPic[i-600,:,:,:],1)
    GTmirroredPic[i] = np.flip(GTmirroredPic[i-600,:,:],1)
    
mirroredPic[700:] = sateliteImages[0:]
GTmirroredPic[700:] = GTImages[0:]
    

#%%

def showImageConcGT(image, GT):
    
    if(len(GTImages[-1,:,:].shape) < 3): #we have BW image
        temp = np.zeros((GT.shape[0],GT.shape[1],3))
        temp = temp.astype(np.uint8)
        temp[:,:,0] = GT
        GT= temp
        
    plt.imshow( np.concatenate((image[:,:,:], GT[:,:,:]), axis=1) )
    plt.show()

#%%



showImageConcGT(mirroredPic[-1,:,:,:], GTmirroredPic[-1,:,:])
showImageConcGT(mirroredPic[99,:,:,:], GTmirroredPic[99,:,:])
showImageConcGT(mirroredPic[199,:,:,:], GTmirroredPic[199,:,:])
showImageConcGT(mirroredPic[299,:,:,:], GTmirroredPic[299,:,:])
showImageConcGT(mirroredPic[399,:,:,:], GTmirroredPic[399,:,:])
showImageConcGT(mirroredPic[499,:,:,:], GTmirroredPic[499,:,:])
showImageConcGT(mirroredPic[599,:,:,:], GTmirroredPic[599,:,:])
showImageConcGT(mirroredPic[699,:,:,:], GTmirroredPic[699,:,:])


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

def getRandomImagePartAsNewImage(inputImage, upScale):
    cropStartX = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
    cropStartY = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
    if(len(GTImages[-1,:,:].shape) > 2):
        im = inputImage[cropStartX:cropStartX+200,cropStartY:cropStartY+200,:]
    else:
        im = inputImage[cropStartX:cropStartX+200,cropStartY:cropStartY+200]        
    
    if(upScale):
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


# TODO: find way to add mutable data constructs (np array) to a datastructure like a set, and then use that


def findStreets(image):
    foundRoad = False
    
    roads = set()
    temp = np.zeros((4))
    temp = temp.astype(np.uint8)
    
    for i in range(0, image.shape[0]):
        if(image[0, i] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[0, i] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 0
                temp[3] = 0
                roads.add(temp)
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 0
        temp[3] = 0
        roads.add(temp)

    for i in range(0, image.shape[0]):
        if(image[image.shape[0]-1, i] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[image.shape[0]-1, i] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 0
                temp[3] = 1
                roads.add(temp)
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 0
        temp[3] = 1
        roads.add(temp)
        
    
    for i in range(0, image.shape[1]):
        if(image[i, 0] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[i, 0] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 1
                temp[3] = 0
                roads.add(temp)
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 1
        temp[3] = 0
        roads.add(temp)

    for i in range(0, image.shape[0]):
        if(image[i, image.shape[1]-1] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[i, image.shape[1]-1] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 1
                temp[3] = 1
                roads.add(temp)
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 1
        temp[3] = 1
        roads.add(temp)
        
    return roads



#%%

def findFittingPart( image, coords, axis):
    
    originalDistance = coords[1] - coords[0]
    
    ret = [-1,-1]
    
    if (axis == 0): #horizontal
        for i in range(0, image.shape[1]):
            foundRoad = False
            roadStartCoord = 0
            for j in range(0, image.shape[0]):
                if ( image[i, j] >= 10 and not foundRoad):
                    roadStartCoord = j
                    foundRoad = True
                else:
                    if(foundRoad and image[i,j] < 10):
                        foundRoad = False
                        distance = i - roadStartCoord
                        if(abs(distance - originalDistance) <=3 ):
                            # found fitting road part
                            ret[0]= i
                            ret[1] = j-coords[1]
                            return ret
                        else:
                            foundRoad = False
                            roadStartCoord = 0
    
    return ret


#%%
test = getRandomImagePartAsNewImage(GTImages[-1,:,:], False)


plt.imshow(test)
plt.show()

s = findStreets(test)
print(s)

t=np.zeros((3))
b=np.zeros((3))
l=np.zeros((3))
r=np.zeros((3))

while(len(s) > 0):
    item = s.pop
    print(np.dtype(item))
    if(item[2] == 0 and item[3] == 1): #bottom edge
        b[0] = b[0]+1
        b[1] = item[0]
        b[2] = item[1]

if(b[0] == 1): ## we're good, just look for one street matching now
    y = b[2]
    x = b[1]
    print('extracted coords: ' + (x,y))

#%%
rslt = findFittingPart( GTImages[-1,:,:], (y, x), 0)
print(rslt)


plt.imshow(GTImages[-1,rslt[0]:rslt[0]+200 , rslt[1]:rslt[1]+200 ])
plt.show()

temp = np.concatenate((test, GTImages[-1,rslt[0]:rslt[0]+200 , rslt[1]:rslt[1]+200 ]), axis = 0)

plt.imshow(temp)
plt.show()
