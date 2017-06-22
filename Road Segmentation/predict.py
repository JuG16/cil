
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
        img.resize((400,400,3))
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
    for predicted_labels in imgs:
        img = label_to_img(400, 400, IMG_PATCH_SIZE, IMG_PATCH_SIZE, predicted_labels)
        #Image.fromarray(img).save(open('./predict_output/' + "mask_" + str(i) + ".png", 'w'))

        scipy.misc.imsave('./predict_output/' + "mask_" + str(i) + ".png", img)
        i += 1




if __name__ == '__main__':

    modelPath = './tmp/1498134747/model-35800'

    X_val = extract_test_data('./test_set_images/')
    print("X_val shape: " + str(X_val.shape))

    tf.app.run()










