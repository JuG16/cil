



import numpy as np
import time
#import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.contrib import learn
import tensorflow as tf


import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf




# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


#needs to be called main because tf.app.run() runs this
def main(unused_argv):
    
        best_f1 = 0.84 #score to beat, gets printed if better

        # Get input dimensionality.
        IMAGE_HEIGHT = X_train.shape[1]
        IMAGE_WIDTH = X_train.shape[2]
        NUM_CHANNELS = X_train.shape[3]


        input_samples_op = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS], name="input_samples")
        input_label_op = tf.placeholder(tf.int32, shape=[None], name="input_labels")
        # Some layers/functions have different behaviours during training and evaluation.
        # If model is in the training mode, then pass True.
        mode = tf.placeholder(tf.bool, name="mode")
        # loss_avg and accuracy_avg will be used to update summaries externally.
        # Since we do evaluation by using batches, we may want average value.
        # (1) Keep counting number of correct predictions over batches.
        # (2) Calculate the average value, evaluate the corresponding summaries
        # by using loss_avg and accuracy_avg placeholders.
        loss_avg = tf.placeholder(tf.float32, name="loss_avg")
        accuracy_avg = tf.placeholder(tf.float32, name="accuracy_avg")

        # Call the function that builds the network. You should pass all the
        # parameters that controls external inputs.
        # It returns "logits" layer, i.e., the top-most layer of the network.
        #logits = krizshevsky_model(input_samples_op, dropout_rate, mode)
        logits = model(input_samples_op, mode)

        #logits = small_cnn_model(input_samples_op, dropout_rate, mode)

        # Optional:
        # Tensorflow provides a very simple and useful API (summary) for
        # monitoring the training via tensorboard
        # (https://www.tensorflow.org/get_started/summaries_and_tensorboard)
        # However, it is not trivial to visualize average accuracy over whole
        # dataset. Create two tensorflow variables in order to count number of
        # samples fed and correct predictions made. They are attached to
        # a summary op (see below).
        counter_correct_prediction = tf.Variable(0, name='counter_correct_prediction', trainable=False)
        counter_samples_fed = tf.Variable(0, name='counter_samples_fed', trainable=False)

        # Loss calculations: cross-entropy
        with tf.name_scope("cross_entropy_loss"):
            # Takes predictions of the network (logits) and ground-truth labels
            # (input_label_op), and calculates the cross-entropy loss.

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_label_op))

        # Accuracy calculations.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            predictions = tf.argmax(logits, 1, name="predictions")
            # Return a bool tensor with shape [batch_size] that is true for the
            predicted = tf.cast(predictions, tf.int32)
            actual = input_label_op
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # correct predictions.
            correct_predictions = tf.nn.in_top_k(logits, input_label_op, 1)
            # Calculate the accuracy per minibatch.
            batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            # Number of correct predictions in order to calculate average accuracy afterwards.
            num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))


        def do_evaluation(sess, samples, labels):
            '''
            Evaluation function.
            @param sess: tensorflow session object.
            @param samples: input data (numpy tensor)
            @param labels: ground-truth labels (numpy array)
            '''
            batches = data_iterator(samples, labels, batch_size)
            # Keep track of this run.
            counter_accuracy = 0.0
            counter_loss = 0.0
            counter_batches = 0
            counter_f1 = 0.0
            for batch_samples, batch_labels in batches:
                counter_batches += 1
                feed_dict = {input_samples_op: batch_samples,
                             input_label_op: batch_labels,
                             mode: False}
                results = sess.run([loss, num_correct_predictions, f1], feed_dict=feed_dict)
                counter_loss += results[0]
                counter_accuracy += results[1]
                counter_f1 += results[2]
            return (counter_loss/counter_batches, counter_accuracy/(counter_batches*batch_size), counter_f1/counter_batches)

        # Create summary ops for monitoring the training.
        # Each summary op annotates a node in the computational graph and collects
        # data data from it.
        summary_trian_loss = tf.summary.scalar('loss', loss)
        summary_train_acc = tf.summary.scalar('accuracy_training', batch_accuracy)
        summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg)
        summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg)

        # Group summaries.
        summaries_training = tf.summary.merge([summary_trian_loss, summary_train_acc])
        summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

        # Generate a variable to contain a counter for the global training step.
        # Note that it is useful if you save/restore your network.
        global_step = tf.Variable(1, name='global_step', trainable=False)

        # Create optimization op.
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            
            train_op = optimizer.minimize(loss, global_step=global_step)

        # For saving/restoring the model.
        # Save important ops (which can be required later!) by adding them into
        # the collection. We will use them in order to evaluate our model on the test
        # data after training.
        # See tf.get_collection for details.
        tf.add_to_collection('predictions', predictions)
        tf.add_to_collection('logits', logits)
        tf.add_to_collection('input_samples_op', input_samples_op)
        tf.add_to_collection('mode', mode)

        # Create session object
        sess = tf.Session()
        # Add the ops to initialize variables.
        init_op = tf.global_variables_initializer()
        # Actually intialize the variables
        sess.run(init_op)

        # Register summary ops.
        train_summary_dir = os.path.join(model_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        valid_summary_dir = os.path.join(model_dir, "summary", "validation")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=20)
        
        import matplotlib.pyplot as plt
        plt.figure()
        accurracy_list = []
        accurracy_loss = []
        validation_list = []
        validation_loss = []

        #index that counts which data is currently in memory
        data_index = 0

        # Define counters in order to accumulate measurements.
        counter_correct_predictions_training = 0.0
        counter_loss_training = 0.0
        for epoch in range(1, num_epochs+1):
            # Generate training batches
            training_batches = data_iterator(X_train, y_train, batch_size, 1)
            # Training loop.
            for batch_samples, batch_labels in training_batches:
                step = tf.train.global_step(sess, global_step)
                if (step%checkpoint_every_step) == 0 and not train:

                    ckpt_save_path = saver.save(sess, os.path.join(model_dir, 'model'), global_step)
                    np.save("latest_save_path", ckpt_save_path)
                    print("Model saved in file: %s" % ckpt_save_path)

                # This dictionary maps the batch data (as a numpy array) to the
                # placeholder variables in the graph.
                feed_dict = {input_samples_op: batch_samples,
                             input_label_op: batch_labels,
                             mode: True}

                # Run the optimizer to update weights.
                # Note that "train_op" is responsible from updating network weights.
                # Only the operations that are fed are evaluated.
                # Run the optimizer to update weights.
                train_summary, correct_predictions_training, loss_training, _ = sess.run([summaries_training, num_correct_predictions, loss, train_op], feed_dict=feed_dict)
                # Update counters.
                counter_correct_predictions_training += correct_predictions_training
                counter_loss_training += loss_training
                # Write summary data.
                train_summary_writer.add_summary(train_summary, step)

                # Occasionally print status messages.
                if (step%print_every_step) == 0:
                    # Calculate average training accuracy.
                    accuracy_avg_value_training = counter_correct_predictions_training/(print_every_step*batch_size)
                    loss_avg_value_training = counter_loss_training/(print_every_step)
                    # [Epoch/Iteration]
                    
                    counter_correct_predictions_training = 0.0
                    counter_loss_training = 0.0
                    # Report
                    # Note that accuracy_avg and loss_avg placeholders are defined
                    # just to feed average results to summaries.
                    summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_training, loss_avg:loss_avg_value_training})
                    train_summary_writer.add_summary(summary_report, step)
                    
                    accurracy_list.append(accuracy_avg_value_training)
                    accurracy_loss.append(loss_avg_value_training)

                    print("[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f" % (epoch, step, accuracy_avg_value_training, loss_avg_value_training))
                    

                if (step%evaluate_every_step) == 0 and train:
                    validation_data = X_test
                    validation_labels = y_test
                    # Calculate average validation accuracy.
                    (loss_avg_value_validation, accuracy_avg_value_validation, f1_report) = do_evaluation(sess, validation_data, validation_labels)
                    # Report
                    summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_validation, loss_avg:loss_avg_value_validation})
                    valid_summary_writer.add_summary(summary_report, step)
                    validation_list.append(accuracy_avg_value_validation)
                    validation_loss.append(loss_avg_value_validation)
                    if f1_report > best_f1:
                        best_f1 = f1_report
                        print("Found better F1 score!")

                    print("[%d/%d] [Validation] Accuracy: %.3f, F1: %.3f, Loss: %.3f" % (epoch, step, accuracy_avg_value_validation, f1_report, loss_avg_value_validation))

#utils, helper functions

#######################################################################
## Helper functions.
#######################################################################
def data_iterator(data, labels, batch_size, num_epochs=1, shuffle=True):
    """
    A simple data iterator for samples and labels.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param labels: Numpy array.
    @param batch_size:
    @param num_epochs:
    @param shuffle: Boolean to shuffle data before partitioning the data into batches.
    """
    data_size = data.shape[0]
    
    # shuffle labels and features
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_samples = data[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]
    else:
        shuffled_samples = data
        shuffled_labels = labels
    for batch_idx in range(0, data_size-batch_size, batch_size):
        batch_samples = shuffled_samples[batch_idx:batch_idx + batch_size]
        batch_labels = shuffled_labels[batch_idx:batch_idx + batch_size]
        yield batch_samples, batch_labels

def data_iterator_samples(data, batch_size):
    """
    A simple data iterator for samples.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param batch_size:
    @param num_epochs:
    """
    data_size = data.shape[0]
    for batch_idx in range(0, data_size, batch_size):
        batch_samples = data[batch_idx:batch_idx + batch_size]
        yield batch_samples

#models

def model(input_layer, mode):

    with tf.name_scope("network"):

    #input img_patch_size*img_patch_size
        filter_size = 512 #192
        #init_bias2d = None     
        init_bias2d = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32) #none
        #init_kernel2d = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32) #xavier
        init_kernel2d = None
        #init_bias = None
        init_bias = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32) #none
        #init_kernel = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32) #xavier
        init_kernel = None
        with tf.name_scope("cnn1"):net = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[2, 2],padding="same",activation=tf.nn.relu, bias_initializer= init_bias2d, kernel_initializer=init_kernel2d)
       
        with tf.name_scope("cnn2"):net = tf.layers.conv2d( inputs=net,filters=128, kernel_size=[1, 1],padding="same",activation=tf.nn.relu, bias_initializer= init_bias2d, kernel_initializer=init_kernel2d)
        with tf.name_scope("pooling1"): net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        with tf.name_scope("cnn3"):net = tf.layers.conv2d( inputs=net,filters=256, kernel_size=[1, 1],padding="same",activation=tf.nn.relu, bias_initializer= init_bias2d, kernel_initializer=init_kernel2d)
        with tf.name_scope("cnn4"):net = tf.layers.conv2d( inputs=net,filters=256, kernel_size=[1, 1],padding="same",activation=tf.nn.relu, bias_initializer= init_bias2d, kernel_initializer=init_kernel2d)
        with tf.name_scope("pooling2"): net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    
        #with tf.name_scope("pooling3"): net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        with tf.name_scope("cnn6"):net = tf.layers.conv2d( inputs=net,filters=128, kernel_size=[1, 1],padding="same",activation=tf.nn.relu, bias_initializer= init_bias2d, kernel_initializer=init_kernel2d)
      
        with tf.name_scope("flatten"): net = tf.reshape(net, [-1, int(IMG_PATCH_SIZE/4  * IMG_PATCH_SIZE/4 * 128)]) #16 * 16 * 256 oom
       
        with tf.name_scope("dropout1"): net = tf.layers.dropout(inputs=net, rate=dropout_rate, training=mode)
        with tf.name_scope("dense1"): net = tf.layers.dense(inputs=net, units=8, activation=tf.nn.relu, bias_initializer= init_bias, kernel_initializer=init_kernel)
        with tf.name_scope("logits"): net = tf.layers.dense(inputs=net, units=NUM_LABELS)
        
        return net

train = True

learning_rate = 0.0001 #0.0001
epsilon=1e-08 #1e-08
beta1=0.9 #0.9
beta2=0.999 #0.999
batch_size = 80 #80... 64, 120 are worse
num_epochs = 1000
print_every_step = 400
evaluate_every_step = 400
checkpoint_every_step = 400
log_dir = './tmp/'
dropout_rate = 0.9 #0.9


PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16


if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS


    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    print (train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]


    train_size = train_labels.shape[0]
    print("Train size: " + str(train_size))

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))


    train_labels_number_not_array = []
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            train_labels_number_not_array.append(0)
        else:
            train_labels_number_not_array.append(1)
    train_labels = np.asarray(train_labels_number_not_array)
    if train:
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels)
        print("Xtrain shape: " + str(X_train.shape))
        print("Starting training")

        #turn gpu off:
        #CUDA_VISIBLE_DEVICES=""

        #create a new saving directory
        timestamp = str(int(time.time()))
        model_dir = os.path.abspath(os.path.join(log_dir, timestamp))

        #run tensorflow
        tf.app.run()

    else:
        X_train = train_data
        y_train = train_labels

        print("Xtrain shape: " + str(X_train.shape))
        print("Starting training")


        #turn gpu off:
        #CUDA_VISIBLE_DEVICES=""

        #create a new saving directory
        timestamp = str(int(time.time()))
        model_dir = os.path.abspath(os.path.join(log_dir, timestamp))

        #run tensorflow
        tf.app.run()