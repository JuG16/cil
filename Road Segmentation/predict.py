
from train import IMG_PATCH_SIZE
from train import img_crop
from train import data_iterator_samples
from train import batch_size
import matplotlib.image as mpimg
import os
import PIL
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.misc
import time
import re
import math
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter


foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def smooth_labels(labels):
    #we have squared images
    w = int(math.sqrt(len(labels)))
    h = w

    #transform it into a minipicture array
    labelpicture = np.zeros((w,h))
    idx = 0
    for i in range(0,h):
        for j in range(0,w):
            labelpicture
            if labels[idx] > 0.5:
                l = 1
            else:
                l = 0
            labelpicture[j, i] = l
            idx += 1
    #check if not on the border of the picture      
    for i in range(1,h-1):
        for j in range(1,w-1):
            #put all neighbors in a list from top left to bottom right
            neighbors = [labelpicture[i-1, j-1], labelpicture[i-1, j], labelpicture[i-1, j+1], labelpicture[i, j-1], labelpicture[i, j+1],  labelpicture[i+1, j-1], labelpicture[i+1, j], labelpicture[i+1, j+1]]
            
            #delete if it has less than three neighbors
            if sum(neighbors) < 2:
                labelpicture[i,j] = 0

            #if all neighbors are streets, you are a street
            if sum(neighbors) >= len(neighbors) -1 :
                labelpicture[i,j] = 1
    
    for dim in range(4):

        #CHECK all 4 borders
        i = 0
        for j in range(1,w-1):

            neighbors = [labelpicture[i, j-1], labelpicture[i, j+1],  labelpicture[i+1, j-1], labelpicture[i+1, j], labelpicture[i+1, j+1]]
            if sum(neighbors) < 2:
                labelpicture[i,j] = 0
            #if all neighbors are streets, you are a street
            if sum(neighbors) == len(neighbors):
                labelpicture[i,j] = 1

        #check if not on the border of the picture      
        for i in range(1,h-1):
            for j in range(1,w-1):
                #enlist all neighbors on top and on the side (not bottom)
                neighbors = [labelpicture[i-1, j-1], labelpicture[i-1, j], labelpicture[i-1, j+1], labelpicture[i, j-1], labelpicture[i, j+1]]
                if (sum(neighbors) == 0):
                    labelpicture[i,j] = 0
        labelpicture = np.rot90(labelpicture)

    
    #does not seem to improve
    #street detection
    streetpicture = np.zeros((w,h))
    for dim in range(4):
        for i in range(0, h):
            street_streak = 0
            background_streak = 0
            for j in range(0,w):
                if labelpicture[i,j] == 1:
                    streetpicture[i,j] = 1
                    street_streak += 1
                    background_streak = 0
                else:
                    background_streak += 1
                    if street_streak > 3:
                        street_streak +=1
                        streetpicture[i,j] = 1

                if background_streak > 2:
                    for b in range(background_streak):
                        streetpicture[i,j-b] = 0
                    street_streak = 0


        labelpicture = np.rot90(labelpicture)
        streetpicture = np.rot90(streetpicture)

    #uncomment if you want to enable street detection
    #labelpicture = streetpicture


    smoothed_labels = []
    for i in range(0,h):
        for j in range(0,w):
            smoothed_labels.append(labelpicture[j,i])
    return smoothed_labels
    



def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels



# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    oimg = make_img_overlay(img, img_prediction)

    return oimg

def extract_test_data(parent_dir):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1,51):
        filepath = parent_dir + "test_" + str(i) + "/test_" + str(i) + ".png"
        print ('Loading ' + filepath)
        img = mpimg.imread(filepath)
        imgs.append(img)


    num_images = len(imgs)
    imgs = np.asarray(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]

    print("Images shape: " + str(imgs.shape))
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    print(np.asarray(img_patches).shape)
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return np.asarray(data)

def main(unused_argv):

    sess = tf.Session()

    saver = tf.train.import_meta_graph(modelPath + '.meta')
    # Restore variables.
    saver.restore(sess,modelPath)
    # Restore ops.
    predictions = tf.get_collection('predictions')[0]
    input_samples_op = tf.get_collection('input_samples_op')[0]
    mode = tf.get_collection('mode')[0]

    def do_prediction(sess, samples):
        batches = data_iterator_samples(samples, batch_size)
        test_predictions = []
        for batch_samples in batches:
            feed_dict = {input_samples_op: batch_samples,
                         mode: False}
            test_predictions.extend(sess.run(predictions, feed_dict=feed_dict))
        return np.asarray(test_predictions)

    y_val = do_prediction(sess, X_val)
    imgs = np.split(y_val, 50)
    print(len(imgs))
    i = 1

    submission_filename = 'submission-'+(time.strftime('%Y-%m-%d-%a-%Hh%Mmin')) + "0.83.csv"
    image_filenames = []
    for predicted_labels in imgs:

        predicted_labels = smooth_labels(predicted_labels)
        img = label_to_img(608, 608, IMG_PATCH_SIZE, IMG_PATCH_SIZE, predicted_labels)
        #Image.fromarray(img).save(open('./predict_output/' + "mask_" + str(i) + ".png", 'w'))
        image_filename = './predict_output/' + "mask_" + str(i) + ".png"
        print(image_filename)
        scipy.misc.imsave(image_filename, img)
        image_filenames.append(image_filename)
        i += 1
    masks_to_submission(submission_filename, *image_filenames)




if __name__ == '__main__':

    modelPath = './tmp/1498306546/model-138800'#131200 #135600 #138800 #148000

    X_val = extract_test_data('./test_set_images/')
    print("X_val shape: " + str(X_val.shape))

    tf.app.run()










