import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


# cnn layer 1
filter_size1 = 5
num_filters1 = 16

# cnn layer 2
filter_size2 = 5
num_filters2 = 36

# fully connected layer
fc_size = 128

data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28
img_size_flat = img_size ** 2
img_shape = (img_size, img_size)

num_channels = 1
num_classes = 10


def plot_images(images, cls_true, cls_pred=None):

    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true)
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def new_weights(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):

    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


if __name__ == '__main__':

    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    layer_conv1, weights_conv1 = new_conv_layer(x_image, num_channels, filter_size1, num_filters1, use_pooling=True)
    layer_conv2, weights_conv2 = new_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, use_pooling=True)
    layer_flat, num_features = flatten_layer(layer_conv2)

    layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, use_relu=True)
    layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # the function calculates the softmax internally so we must use the output of layer_fc2 directly rather than y_pred
    # which has already had the softmax applied
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    train_batch_size = 64
    test_batch_size = 256

    total_iterations = 0


    def optimize(num_iterations):

        global total_iterations

        start_time = time.time()

        for i in range(total_iterations, total_iterations + num_iterations):
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
            if i % 100 == 0:
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = 'Optimization iterations: {0:>6}, Training accuracy: {1:>6.1%}'
                print(msg.format(i + 1, acc))
        total_iterations += num_iterations
        end_time = time.time()
        time_dif = end_time - start_time
        print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))))


    def plot_example_errors(cls_pred, correct):

        incorrect = (correct == False)
        images = data.test.images[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = data.test.cls[incorrect]
        plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


    def plot_confusion_matrix(cls_pred):

        cls_true = data.test.cls
        cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        print cm
        plt.matshow(cm)
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


    def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):

        # Number of images in the test-set.
        num_test = len(data.test.images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = data.test.images[i:j, :]

            # Get the associated labels.
            labels = data.test.labels[i:j, :]

            # Create a feed-dict with these images and labels.
            feed_dict = {x: images,
                         y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Convenience variable for the true class-numbers of the test-set.
        cls_true = data.test.cls

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)


    optimize(1000)
    print_test_accuracy()
    session.close()










