
"""
Created on Wed May 10 10:54:19 2017

@author: Andreas
"""

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import winsound


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
    
    if(len(GT.shape) < 3): #we have BW image
        temp = np.zeros((GT.shape[0],GT.shape[1],3))
        temp = temp.astype(np.uint8)
        temp[:,:,0] = GT
        GT= temp
        
    plt.imshow( np.concatenate((image[:,:,:], GT[:,:,:]), axis=1) )
    plt.show()

#%%



#showImageConcGT(mirroredPic[-1,:,:,:], GTmirroredPic[-1,:,:])
#showImageConcGT(mirroredPic[99,:,:,:], GTmirroredPic[99,:,:])
#showImageConcGT(mirroredPic[199,:,:,:], GTmirroredPic[199,:,:])
#showImageConcGT(mirroredPic[299,:,:,:], GTmirroredPic[299,:,:])
#showImageConcGT(mirroredPic[399,:,:,:], GTmirroredPic[399,:,:])
#showImageConcGT(mirroredPic[499,:,:,:], GTmirroredPic[499,:,:])
#showImageConcGT(mirroredPic[599,:,:,:], GTmirroredPic[599,:,:])
#showImageConcGT(mirroredPic[699,:,:,:], GTmirroredPic[699,:,:])


#%%

# adding noise to images

#mirroredPicsNoisy = np.zeros((800,400,400,3))
#mirroredPicsNoisy = mirroredPicsNoisy.astype(np.uint8)

#for i in range(0, mirroredPic.shape[0]):
#    noise = np.random.rand(400,400)*30
#    temp = np.zeros((400,400,3))
#    for t in range(0,3):
#        temp[:,:,t] = noise
#    
#    noise = temp - 15
#    noise = noise.astype(np.uint8)
#    
#    mirroredPicsNoisy[i,:,:,:] = mirroredPic[i,:,:,:] + noise



#%%

#augmentedImages = np.zeros((4800,400,400,3))
#augmentedImages = augmentedImages.astype(np.uint8)

#augmentedImages[0:100,:,:,:] = sateliteImages[0:,:,:,:]
#augmentedImages[0:800,:,:,:] = mirroredPic[0:,:,:,:]
#augmentedImages[800:1600,:,:,:] = mirroredPicsNoisy[0:,:,:,:]

allImages = np.zeros((2,mirroredPic.shape[0],mirroredPic.shape[1],mirroredPic.shape[2], mirroredPic.shape[3] ))

allImages[0,:,:,:,:] = mirroredPic
allImages[1,:,:,:,0] = GTmirroredPic
allImages[1,:,:,:,1] = GTmirroredPic
allImages[1,:,:,:,2] = GTmirroredPic

allImages =  allImages.astype(np.uint8)

print(allImages.shape)

#%%

# take image parts, and scale them up

# per image, take 2 random parts, sized 200x200

def getRandomImagePartAsNewImage(inputImage, upScale):
    
    if(inputImage.shape[0] == 2):
        cropStartX = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
        cropStartY = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
        
        return inputImage[:,cropStartX:cropStartX+200,cropStartY:cropStartY+200,:] 
        
    
    cropStartX = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
    cropStartY = np.floor(200*np.random.rand(1)[0]).astype(np.uint8)
    if(len(inputImage[-1,:,:].shape) > 2):
        im = inputImage[cropStartX:cropStartX+200,cropStartY:cropStartY+200,:]
    else:
        im = inputImage[cropStartX:cropStartX+200,cropStartY:cropStartY+200]        
    
    if(upScale):
        im = np.repeat(np.repeat(im,2, axis=0), 2, axis=1)
    
    return im

#%%


#counter = 1600

#for i in range(0, 1600):
#    for j in range(0,2):
#        
#        augmentedImages[counter] = getRandomImagePartAsNewImage(augmentedImages[i,:,:,:])
#        counter = counter + 1


#%%

# returns tupels with coordinates where the road segment starts, and ends. additional there are 2 numbers indicating which side of the image the road is.
# 0 , 0 is the top of the image
# 0 , 1 is the bottom of the image
# 1 , 0 is the left side of the image
# 1 , 1 is the right side of the image


# TODO: find way to add mutable data constructs (np array) to a datastructure like a set, and then use that


def findStreets(image):
    if(len(image.shape) > 2):
        image = image[:,:,0]
    foundRoad = False
    
    roads = []
    temp = [0,0,0,0]
    
    for i in range(0, image.shape[1]):
        if(image[0, i] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[0, i] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 0
                temp[3] = 0
                
                roads.append(temp[:])
                
    if(foundRoad and i == image.shape[1]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 0
        temp[3] = 0
        roads.append(temp[:])

    for i in range(0, image.shape[1]):
        if(image[image.shape[0]-1, i] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[image.shape[0]-1, i] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 0
                temp[3] = 1
                roads.append(temp[:])
                
    if(foundRoad and i == image.shape[1]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 0
        temp[3] = 1
        roads.append(temp[:])
        
    
    for i in range(0, image.shape[0]):
        if(image[i, 0] >= 10 and not foundRoad):
            temp[0] = i
            foundRoad = True
        else:
            if(foundRoad and image[i, 0] < 10):
                foundRoad = False
                temp[1] = i
                temp[2] = 1
                temp[3] = 0
                roads.append(temp[:])
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 1
        temp[3] = 0
        roads.append(temp[:])

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
                roads.append(temp[:])
                
    if(foundRoad and i == image.shape[0]-1):
        foundRoad = False
        temp[1] = i
        temp[2] = 1
        temp[3] = 1
        roads.append(temp[:])
        
    
    return roads[:]


#%%

def findFittingPart( image, coords, axis):
    
    if(len(image.shape) > 2):
        image = image[:,:,0]
    
    originalDistance = abs(coords[1] - coords[0])
    print('original distance:', originalDistance)
    
    
    ret = [-1,-1]
    
    if (axis == 0): #horizontal
        for i in range(0, int(image.shape[0]/2)):
            foundRoad = False
            roadStartCoord = 0
            for j in range(coords[0], image.shape[1]):
                if ( image[i, j] >= 10 and not foundRoad and j-coords[1] >= 0):
                    roadStartCoord = j
                    foundRoad = True
                else:
                    if(foundRoad and image[i,j] < 10):
                        foundRoad = False
                        distance = j - roadStartCoord
                        if(abs(distance - originalDistance) <=3 ):
                            # found fitting road part
                            ret[0]= i
                            ret[1] = j-coords[1]-1
                            return ret
                        else:
                            foundRoad = False
                            roadStartCoord = 0
    
    return ret

#%%



def findFittingPart3( image, coords, edge, debug):
    
    if(len(image.shape) > 2):
        image = image[:,:,0]
    
    if(edge != [1,1]): 
        originalDistance = abs(coords[1] - coords[0])
    else: 
        originalDistance = [ coords[1][1] - coords[1][0] , coords[0][1] - coords[0][0] ]
    if(debug):
        print('original distance:', originalDistance, 'coords', coords)
    
    tolerance = 3
    threshold = 100
    
    ret = [-1,-1]
    i=0
    j=0
    
    while(i < int(image.shape[0]/2)):
        roadTooBig = False
        foundRet = False
        if(debug):
            print('now at line', i, 'coords', coords)
        if(edge == [0,0] or edge == [1,1]): # look on the bottom side
            
            if(edge == [1,1]):
                origDist = originalDistance[0]
            else:
                origDist = originalDistance
            if(edge == [0,0]):
                bottomCoords = coords
            else: 
                bottomCoords = coords[1]
            line = image[i,:]
            roadFound = False
            j=0
            sinceLastRoad = 0
            over = 0
            while( j < line.shape[0]):
                if( foundRet and line[j] >= threshold ):
                    over += 1
                    
                if( over >= tolerance * 2 ):
                    foundRet = False
                    ret = [-1,-1]
                    roadTooBig = True
                    
                if(line[j] >= threshold and not roadFound):
                    if(debug):
                        print('found road at coord' , j )
                    roadFound = True
                    start = j
                    if(sinceLastRoad < bottomCoords[0]):
                        if(debug):
                            print('sadly, coordinates are too far to the left, we can not use them')
                        
                        roadFound = False
                else:
                    if(line[j] < threshold and not roadFound):
                        sinceLastRoad += 1
                    if(line[j] < threshold and roadFound):
                        if(debug):
                            print('road segment ends at', j,'distance', j-start)
                        if(start - bottomCoords[0]  > int(image.shape[1])/2 ):
                            if(debug):
                                print('sadly, too far right', )
                            break
                        roadFound = False
                        sinceLastRoad = 0
                        if(abs((j-start) - origDist) <= tolerance):
                            foundRet = True
                            ret = [i, start-bottomCoords[0]]
                j+=1
        






        if(edge == [1,0] or edge == [1,1]): # look on the right side
            if(edge == [1,1]):
                origDist = originalDistance[1]
            else:
                origDist = originalDistance
            if(edge == [1,1]):
                rightCoords = coords[0]
            else: 
                rightCoords = coords
            col = image[:,i]
            roadFound = False
            j=0
            sinceLastRoad = 0
            over = 0
            while( j < col.shape[0]):
                if( foundRet and col[j] >= threshold ):
                    over += 1
                    
                if( over >= tolerance * 2 ):
                    foundRet = False
                    ret = [-1,-1]
                    roadTooBig = True
                    
                if(col[j] >= threshold and not roadFound):
                    if(debug):
                        print('found road at coord' , j )
                    roadFound = True
                    start = j
                    if(sinceLastRoad < rightCoords[0]):
                        if(debug):
                            print('sadly, coordinates are too far to the left, we can not use them')
                        
                        roadFound = False
                else:
                    if(col[j] < threshold and not roadFound):
                        sinceLastRoad += 1
                    if(col[j] < threshold and roadFound):
                        if(debug):
                            print('road segment ends at', j,'distance', j-start)
                        if(start - rightCoords[0]  > int(image.shape[0])/2 ):
                            if(debug):
                                print('sadly, too far right', )
                            break
                        roadFound = False
                        sinceLastRoad = 0
                        if(abs((j-start) - origDist) <= tolerance):
                            foundRet = True
                            ret = [ start-rightCoords[0], i ]
                j+=1
        
        if(ret != [-1,-1] and not roadTooBig and edge != [1,1]):
            streetSet = findStreets(image[ret[0]:ret[0]+200, ret[1]:ret[1]+200])
            
            u=np.zeros((3))
            b=np.zeros((3))
            l=np.zeros((3))
            r=np.zeros((3))
            
            while(len(streetSet) > 0):
                item = streetSet[-1]
                streetSet = streetSet[:-1]
                print(item)
                
                if(item[2] == 0 and item[3] == 1): #bottom edge
                    b[0] = b[0]+1
                    b[1] = item[0]
                    b[2] = item[1]
                    
                if(item[2] == 0 and item[3] == 0): #upper edge
                    u[0] = u[0]+1
                    u[1] = item[0]
                    u[2] = item[1]
                    
                if(item[2] == 1 and item[3] == 0): #left edge
                    l[0] = l[0]+1
                    l[1] = item[0]
                    l[2] = item[1]
                    
                if(item[2] == 1 and item[3] == 1): #right edge
                    r[0] = r[0]+1
                    r[1] = item[0]
                    r[2] = item[1]
            
            
            nrCrossingRoads = 1
            if( edge == [0 , 0] ):
                nrCrossingRoads = r[0]
            if( edge == [1 , 0] ):
                nrCrossingRoads = b[0]
            
            if( nrCrossingRoads == 1 ):
                return ret
            else: 
                return [-1,-1]
        i += 1
    
    return ret



#%%


def findFittingPart4( image, coords, edge, debug):
    
    if(len(image.shape) > 2):
        image = image[:,:,0]
    
    if(edge != [1,1]): 
        originalDistance = abs(coords[1] - coords[0])
    else: 
        originalDistance = [ coords[1][1] - coords[1][0] , coords[0][1] - coords[0][0] ] # coords = [horizontal, vertical]
    if(debug):
        print('original distance:', originalDistance, 'coords', coords)
    
    tolerance = 6
    threshold = 100
    
    ret = [-1,-1]
    i=0
    j=0
    
    if(edge != [1,1]):
        while(i < int(image.shape[0]/2)):
            roadTooBig = False
            foundRet = False
            if(debug):
                print('now at line', i, 'coords', coords)
            if(edge == [0,0]): # look on the bottom side
                
                origDist = originalDistance
                bottomCoords = coords
                
                line = image[i,:]
                roadFound = False
                j=0
                sinceLastRoad = 0
                over = 0
                while( j < line.shape[0]):
                    if( foundRet and line[j] >= threshold ):
                        over += 1
                        
                    if( over >= tolerance * 2 ):
                        foundRet = False
                        ret = [-1,-1]
                        roadTooBig = True
                        
                    if(line[j] >= threshold and not roadFound):
                        if(debug):
                            print('found road at coord' , j )
                        roadFound = True
                        start = j
                        if(sinceLastRoad < bottomCoords[0]):
                            if(debug):
                                print('sadly, coordinates are too far to the left, we can not use them')
                            
                            roadFound = False
                    else:
                        if(line[j] < threshold and not roadFound):
                            sinceLastRoad += 1
                        if(line[j] < threshold and roadFound):
                            if(debug):
                                print('road segment ends at', j,'distance', j-start)
                            if(start - bottomCoords[0]  > int(image.shape[1])/2 ):
                                if(debug):
                                    print('sadly, too far right', )
                                break
                            roadFound = False
                            sinceLastRoad = 0
                            if(abs((j-start) - origDist) <= tolerance):
                                foundRet = True
                                ret = [i, start-bottomCoords[0]]
                    j+=1
            
    
    
    
    
    
    
            if(edge == [1,0]): # look on the right side
                origDist = originalDistance
                rightCoords = coords
                
                col = image[:,i]
                roadFound = False
                j=0
                sinceLastRoad = 0
                over = 0
                while( j < col.shape[0]):
                    if( foundRet and col[j] >= threshold ):
                        over += 1
                        
                    if( over >= tolerance * 2 ):
                        foundRet = False
                        ret = [-1,-1]
                        roadTooBig = True
                        
                    if(col[j] >= threshold and not roadFound):
                        if(debug):
                            print('found road at coord' , j )
                        roadFound = True
                        start = j
                        if(sinceLastRoad < rightCoords[0]):
                            if(debug):
                                print('sadly, coordinates are too far to the left, we can not use them')
                            
                            roadFound = False
                    else:
                        if(col[j] < threshold and not roadFound):
                            sinceLastRoad += 1
                        if(col[j] < threshold and roadFound):
                            if(debug):
                                print('road segment ends at', j,'distance', j-start)
                            if(start - rightCoords[0]  > int(image.shape[0])/2 ):
                                if(debug):
                                    print('sadly, too far right', )
                                break
                            roadFound = False
                            sinceLastRoad = 0
                            if(abs((j-start) - origDist) <= tolerance):
                                foundRet = True
                                ret = [ start-rightCoords[0], i ]
                    j+=1
            
            if(ret != [-1,-1] and not roadTooBig and edge != [1,1]):
                streetSet = findStreets(image[ret[0]:ret[0]+200, ret[1]:ret[1]+200])
                
                u=np.zeros((3))
                b=np.zeros((3))
                l=np.zeros((3))
                r=np.zeros((3))
                
                while(len(streetSet) > 0):
                    item = streetSet[-1]
                    streetSet = streetSet[:-1]
                    print(item)
                    
                    if(item[2] == 0 and item[3] == 1): #bottom edge
                        b[0] = b[0]+1
                        b[1] = item[0]
                        b[2] = item[1]
                        
                    if(item[2] == 0 and item[3] == 0): #upper edge
                        u[0] = u[0]+1
                        u[1] = item[0]
                        u[2] = item[1]
                        
                    if(item[2] == 1 and item[3] == 0): #left edge
                        l[0] = l[0]+1
                        l[1] = item[0]
                        l[2] = item[1]
                        
                    if(item[2] == 1 and item[3] == 1): #right edge
                        r[0] = r[0]+1
                        r[1] = item[0]
                        r[2] = item[1]
                
                
                nrCrossingRoads = 1
                if( edge == [0 , 0] ):
                    nrCrossingRoads = r[0]
                if( edge == [1 , 0] ):
                    nrCrossingRoads = b[0]
                
                if( nrCrossingRoads == 1 ):
                    return ret
                else: 
                    return [-1,-1]
            i += 1
        
        return ret
    else: # both must be fitting. the left, and the above edge
        
        toFitLeft = np.zeros((200))
        toFitAbove = np.zeros((200))
        
        
        
        for w in range(coords[0][0], coords[0][1]):
            toFitLeft[w] = 1
        for w in range(coords[1][0], coords[1][1]):
            toFitAbove[w] = 1
            
        
            
        while(i < int(image.shape[0]/2)):
            j=0
            while(j < int(image.shape[1]/2)):
                tooManyRoads = False
                
                above = np.zeros((200))
                above = above.astype(np.uint8)
                
                left = np.zeros((200))
                left = left.astype(np.uint8)
                
                above[:] = image [ i, j : j + int(image.shape[1]/2)]
                left[:] = image [ i: i + int(image.shape[0]/2), j ]               
                
                for w in range(0, above.shape[0]):
                    
                    if(above[w] > threshold):
                        above[w] = 1
                    else:
                        above[w] = 0
                    if(left[w] > threshold):
                        left[w] = 1
                    else:
                        left[w] = 0
                
                
                
                nrOfRoads = 0
                foundRoad = False
                for w in range(0, above.shape[0]):
                    if(above[w] == 1 and not foundRoad):
                        nrOfRoads = nrOfRoads + 1
                        foundRoad = True
                    if(above[w] == 0 and foundRoad):
                        foundRoad = False
                
                if(nrOfRoads != 1):
                    if(debug):
                        print('nr of roads is not one 1', nrOfRoads)
                    tooManyRoads = True
                
                
                nrOfRoads = 0
                foundRoad = False
                for w in range(0, left.shape[0]):
                    if(left[w] == 1 and not foundRoad):
                        nrOfRoads = nrOfRoads + 1
                        foundRoad = True
                    if(left[w] == 0 and foundRoad):
                        foundRoad = False
                
                if(nrOfRoads != 1):
                    if(debug):
                        print('nr of roads is not one 2', nrOfRoads)
                    tooManyRoads = True
                
                if(not tooManyRoads):
                    aboveStart = 0
                    leftStart = 0
                    toFitLeftStart = 0
                    toFitAboveStart = 0
                    
                    for w in range(0, above.shape[0]):
                        if(above[w] == 1):
                            aboveStart = w
                            break
                    
                    for w in range(0, left.shape[0]):
                        if(left[w] == 1):
                            leftStart = w
                            break
                        
                    
                    for w in range(0, toFitLeft.shape[0]):
                        if(toFitLeft[w] == 1):
                            toFitLeftStart = w
                            break
                        
                    
                    for w in range(0, toFitAbove.shape[0]):
                        if(toFitAbove[w] == 1):
                            toFitAboveStart = w
                            break
                        
                    
                        
                    aboveEnd = above.shape[0]-1
                    leftEnd = left.shape[0]-1
                    toFitLeftEnd = toFitLeft.shape[0]-1
                    toFitAboveEnd = toFitAbove.shape[0]-1
                    
                    for w in range(above.shape[0]-1,-1,-1):
                        if(above[w] == 1):
                            aboveEnd = w
                            break
                    
                    for w in range(left.shape[0]-1, -1,-1):
                        if(left[w] == 1):
                            leftEnd = w
                            break
                        
                    
                    for w in range(toFitLeft.shape[0]-1, -1,-1):
                        if(toFitLeft[w] == 1):
                            toFitLeftEnd = w
                            break
                        
                    
                    for w in range(toFitAbove.shape[0]-1, -1,-1):
                        if(toFitAbove[w] == 1):
                            toFitAboveEnd = w
                            break
                    
                    if(debug):
                        print('aboveStart, leftStart, toFitAboveStart, toFitLeftStart ',aboveStart, leftStart, toFitAboveStart, toFitLeftStart)
                        print('aboveEnd, leftEnd, tofitAvobeEnd, toFitLeftEnd', aboveEnd, leftEnd, toFitAboveEnd, toFitLeftEnd)
                    
                    if( abs(aboveStart - toFitAboveStart) + abs(aboveEnd - toFitAboveEnd) <= tolerance and abs(leftStart - toFitLeftStart) + abs(leftEnd - toFitLeftEnd) <= tolerance):
                        print('aboveStart, leftStart, toFitAboveStart, toFitLeftStart ',aboveStart, leftStart, toFitAboveStart, toFitLeftStart)
                        print('aboveEnd, leftEnd, tofitAvobeEnd, toFitLeftEnd', aboveEnd, leftEnd, toFitAboveEnd, toFitLeftEnd)
                        return [i, j]
                
                
                j = j + 1          
            
            i = i + 1
    
    #print('reached end, nothing found')
    return [-1,-1]





#%%

def findFittingPart2( image, coords, edge, debug):
    
    if(len(image.shape) > 2):
        image = image[:,:,0]
    
    originalDistance = abs(coords[1] - coords[0])
    if(debug):
        print('original distance:', originalDistance, 'coords', coords)
    
    tolerance = 3
    threshold = 100
    
    ret = [-1,-1]
    i=0
    j=0
    
    while(i < int(image.shape[0]/2)):
        roadTooBig = False
        foundRet = False
        if(debug):
            print('now at line', i, 'coords', coords)
        if(edge == [0,0]):
            line = image[i,:]
            roadFound = False
            j=0
            sinceLastRoad = 0
            over = 0
            while( j < line.shape[0]):
                if( foundRet and line[j] >= threshold ):
                    over += 1
                    
                if( over >= tolerance * 2 ):
                    foundRet = False
                    ret = [-1,-1]
                    roadTooBig = True
                    
                if(line[j] >= threshold and not roadFound):
                    if(debug):
                        print('found road at coord' , j )
                    roadFound = True
                    start = j
                    if(sinceLastRoad < coords[0]):
                        if(debug):
                            print('sadly, coordinates are too far to the left, we can not use them')
                        
                        roadFound = False
                else:
                    if(line[j] < threshold and not roadFound):
                        sinceLastRoad += 1
                    if(line[j] < threshold and roadFound):
                        if(debug):
                            print('road segment ends at', j,'distance', j-start)
                        if(start - coords[0]  > int(image.shape[1])/2 ):
                            if(debug):
                                print('sadly, too far right', )
                            break
                        roadFound = False
                        sinceLastRoad = 0
                        if(abs((j-start) - originalDistance) <= tolerance):
                            foundRet = True
                            ret = [i, start-coords[0]]
                j+=1
        






        if(edge == [1,0]):
            col = image[:,i]
            roadFound = False
            j=0
            sinceLastRoad = 0
            over = 0
            while( j < col.shape[0]):
                if( foundRet and col[j] >= threshold ):
                    over += 1
                    
                if( over >= tolerance * 2 ):
                    foundRet = False
                    ret = [-1,-1]
                    roadTooBig = True
                    
                if(col[j] >= threshold and not roadFound):
                    if(debug):
                        print('found road at coord' , j )
                    roadFound = True
                    start = j
                    if(sinceLastRoad < coords[0]):
                        if(debug):
                            print('sadly, coordinates are too far to the left, we can not use them')
                        
                        roadFound = False
                else:
                    if(col[j] < threshold and not roadFound):
                        sinceLastRoad += 1
                    if(col[j] < threshold and roadFound):
                        if(debug):
                            print('road segment ends at', j,'distance', j-start)
                        if(start - coords[0]  > int(image.shape[0])/2 ):
                            if(debug):
                                print('sadly, too far right', )
                            break
                        roadFound = False
                        sinceLastRoad = 0
                        if(abs((j-start) - originalDistance) <= tolerance):
                            foundRet = True
                            ret = [ start-coords[0], i ]
                j+=1
                
        if(ret != [-1,-1] and not roadTooBig and edge != [1,1]):
            streetSet = findStreets(image[ret[0]:ret[0]+200, ret[1]:ret[1]+200])
            
            u=np.zeros((3))
            b=np.zeros((3))
            l=np.zeros((3))
            r=np.zeros((3))
            
            while(len(streetSet) > 0):
                item = streetSet[-1]
                streetSet = streetSet[:-1]
                print(item)
                
                if(item[2] == 0 and item[3] == 1): #bottom edge
                    b[0] = b[0]+1
                    b[1] = item[0]
                    b[2] = item[1]
                    
                if(item[2] == 0 and item[3] == 0): #upper edge
                    u[0] = u[0]+1
                    u[1] = item[0]
                    u[2] = item[1]
                    
                if(item[2] == 1 and item[3] == 0): #left edge
                    l[0] = l[0]+1
                    l[1] = item[0]
                    l[2] = item[1]
                    
                if(item[2] == 1 and item[3] == 1): #right edge
                    r[0] = r[0]+1
                    r[1] = item[0]
                    r[2] = item[1]
            
            
            nrCrossingRoads = 1
            if( edge == [0 , 0] ):
                nrCrossingRoads = r[0]
            if( edge == [1 , 0] ):
                nrCrossingRoads = b[0]
            
            if( nrCrossingRoads == 1 ):
                return ret
            else: 
                return [-1,-1]
        i += 1
    
    return ret

#%%

order = np.arange(allImages.shape[1])
np.random.shuffle(order)


u=np.ones((3))*10
b=np.ones((3))*10
l=np.ones((3))*10
r=np.ones((3))*10
count = 0

while( not (b[0] == 1 and r[0] == 1)):
    
    count =( count + 1 )% allImages.shape[1]
    
    
    
    
    u=np.zeros((3))
    b=np.zeros((3))
    l=np.zeros((3))
    r=np.zeros((3))
    
    test = getRandomImagePartAsNewImage(allImages[:,order[count],:,:,:], False)
    testGT = test[1,:,:,:]
    testSat = test[0,:,:,:]
    
    showImageConcGT(testSat,testGT)
    
    s = findStreets(testGT)
    sb = s
    print(s)
    
    
    while(len(s) > 0):
        item = s[-1]
        s = s[:-1]
        print(item)
        
        if(item[2] == 0 and item[3] == 1): #bottom edge
            b[0] = b[0]+1
            b[1] = item[0]
            b[2] = item[1]
            
        if(item[2] == 0 and item[3] == 0): #upper edge
            u[0] = u[0]+1
            u[1] = item[0]
            u[2] = item[1]
            
        if(item[2] == 1 and item[3] == 0): #left edge
            l[0] = l[0]+1
            l[1] = item[0]
            l[2] = item[1]
            
        if(item[2] == 1 and item[3] == 1): #right edge
            r[0] = r[0]+1
            r[1] = item[0]
            r[2] = item[1]
    
    print(b[0] , r[0])
    
    if(b[0] <= 1 and r[0] <= 1): ## we're good, just look for at most one street matching now
        x2 = int(b[2])
        x1 = int(b[1])
        print('extracted coords bottom: ' , [x1,x2])

        y2 = int(r[2])
        y1 = int(r[1])
        print('extracted coords right: ' , [y1,y2])
        

imagenr=0
index = order[imagenr]

plt.imshow(allImages[1,index,: , :, 0])
plt.show()
rsltRight = findFittingPart4( allImages[1,index,:,:,0], [y1, y2], [1,0], False)
#print('nr',  index, 'res',rsltRight,'looking for match',[y1, y2])

order = np.arange(allImages.shape[1])
np.random.shuffle(order)


while(imagenr < allImages.shape[1]-1 and rsltRight == [-1,-1]):
    imagenr =(imagenr + 1 )% allImages.shape[1]
    index = order[imagenr]
    #plt.imshow(allImages[1,index,: , :, 0])
    #plt.show()
    rsltRight = findFittingPart4( allImages[1,index,:,:,0], [y1, y2], [1,0], False)
    #print('nr',  index, 'res',rsltRight ,'looking for match',[y1, y2])

rightIndex = index

imagenr=0
index = order[imagenr]

plt.imshow(allImages[1,index,: , :, 0])
plt.show()
rslt = findFittingPart4( allImages[1,index,:,:,0], [x1, x2], [0,0], False)
#print('nr',  index, 'res',rslt,'looking for match',[x1, x2])

order = np.arange(allImages.shape[1])
np.random.shuffle(order)


while(imagenr < allImages.shape[1]-1 and rslt == [-1,-1]):
    imagenr =(imagenr + 1 )% allImages.shape[1]
    index = order[imagenr]
    #plt.imshow(allImages[1,index,: , :, 0])
    #plt.show()
    rslt = findFittingPart4( allImages[1,index,:,:,0], [x1, x2], [0,0], False)
    #print('nr',  index, 'res',rslt,'looking for match',[x1, x2])


    
plt.imshow(allImages[1,index,: , :,0])
plt.show()



extractedGTRight = allImages[1, rightIndex,rsltRight[0]:rsltRight[0]+200 , rsltRight[1]:rsltRight[1]+200 ,0]
extractedSatRight = allImages[0,rightIndex,rsltRight[0]:rsltRight[0]+200 , rsltRight[1]:rsltRight[1]+200 ,:]

extractedGT = allImages[1,index,rslt[0]:rslt[0]+200 , rslt[1]:rslt[1]+200 ,0]
extractedSat = allImages[0,index,rslt[0]:rslt[0]+200 , rslt[1]:rslt[1]+200 ,:]

showImageConcGT(extractedSat, extractedGT)
showImageConcGT(extractedSatRight, extractedGTRight)

bottomLeftSat = np.zeros((200,200,3))
bottomLeftSat = bottomLeftSat.astype(np.uint8)
bottomLeftGT = np.zeros((200,200))
bottomLeftGT = bottomLeftGT.astype(np.uint8)


tempSat = np.concatenate((testSat, extractedSat), axis = 0)
rightSat = np.concatenate((extractedSatRight, bottomLeftSat), axis = 0)

print(testGT.shape)
print(extractedGT.shape)
tempGT = np.concatenate((testGT[:,:,0], extractedGT), axis = 0)
rightGT = np.concatenate((extractedGTRight, bottomLeftGT), axis = 0)

showImageConcGT(tempSat, tempGT)
showImageConcGT(rightSat, rightGT)

finalPicSat = np.concatenate((tempSat, rightSat), axis = 1)
finalPicGT = np.concatenate((tempGT, rightGT), axis = 1)

showImageConcGT(finalPicSat, finalPicGT)



print('new pictures streets', findStreets(tempGT))
winsound.Beep(500,1100)

#%%
showImageConcGT(finalPicSat, finalPicGT)
bottomSet = findStreets(extractedGT)
rightSet = findStreets(extractedGTRight)

print(bottomSet)
plt.imshow(extractedGT)
plt.show
print(rightSet)
plt.imshow(extractedGTRight)
plt.show

while(len(bottomSet) > 0):
        item = bottomSet[-1]
        bottomSet = bottomSet[:-1]
        if(item[2] == 1 and item[3] == 1): #right edge
            horizontalCoords = [item[0], item[1]]
            
print(horizontalCoords)

while(len(rightSet) > 0):
        item = rightSet[-1]
        rightSet = rightSet[:-1]        
        if(item[2] == 0 and item[3] == 1): #bottom edge
            verticalCoords = [item[0], item[1]]
            
print(verticalCoords)
print([horizontalCoords, verticalCoords][0])
bottomLeftRslt= [-1,-1]
times = 0
while(bottomLeftRslt == [-1,-1]):
    bottomLeftRslt = findFittingPart4( allImages[1,times,:,:,0], [horizontalCoords, verticalCoords], [1,1], False)
    print(times, bottomLeftRslt)
    if(times%10 == 0):
        winsound.Beep(500,400)
    times = times + 1
    if(times == 800):
        print('could not find an image at all...')
        break

if(bottomLeftRslt != [-1,-1]):
        
    showImageConcGT(allImages[0,times-1,:,:,:], allImages[1,times-1,:,:,:])
    
    bottomLeftGT = allImages[1,times-1,bottomLeftRslt[0]:bottomLeftRslt[0]+200 , bottomLeftRslt[1]:bottomLeftRslt[1]+200 ,0]
    bottomLeftSat = allImages[0,times-1,bottomLeftRslt[0]:bottomLeftRslt[0]+200 , bottomLeftRslt[1]:bottomLeftRslt[1]+200 ,:]
    
    #plt.imshow(bottomLeftGT)
    #plt.show
    #plt.imshow(bottomLeftSat)
    #plt.show
    
    finalPicSat[200:,200:,:] = bottomLeftSat
    finalPicGT[200:,200:] = bottomLeftGT
    
    showImageConcGT(finalPicSat, finalPicGT)

winsound.Beep(700,1100)


#%%

goodPictureCounter = np.load('counter.npy')
goodPictures = np.load('pictures.npy')


goodPictures[0,int(goodPictureCounter[0]),:,:,:] = finalPicSat
goodPictures[1,int(goodPictureCounter[0]),:,:,0] = finalPicGT

goodPictureCounter[0] = goodPictureCounter[0] + 1

np.save('counter.npy', goodPictureCounter)
np.save('pictures.npy', goodPictures)


#%%

#%%


functionTest = np.ones((400,400))
functionTest = functionTest.astype(np.uint8)


functionTest[150:170, 50:] = 240
functionTest[50:, 130:190] = 240



testCoords = [[100,121],[80, 140]]

testrslt = findFittingPart4(functionTest, testCoords,[1,1], False)
print(testrslt)

