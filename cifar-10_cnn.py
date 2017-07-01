import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'Tensorflow-Tutorials/'))
import cifar10
from cifar10 import img_size, num_channels, num_classes


cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

img_size_cropped = 24


def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)

        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


def pre_process_image(image, training):
    """
    For training, the input images are randomly cropped, randomly flipped horizontally, 
    and the hue, contrast and saturation is adjusted with random values. This artificially 
    inflates the size of the training-set by creating random variations of the original 
    input images. Examples of distorted images are shown further below.
    For testing, the input images are cropped around the centre and nothing else is 
    adjusted. 
    """
    if training:
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.maximum(image, 1.0)
        image = tf.minimum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image


def pre_process(images, training):

    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images


def main_network(images, training):

    x_pretty = pt.wrap(images)

    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss


def create_network(training):

    with tf.variable_scope('network', reuse=not training):
        images = x
        images = pre_process(images, training)
        y_pred, loss = main_network(images, training)

    return y_pred, loss


if __name__ == '__main__':

    distorted_images = pre_process(images=x, training=True)
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    _, loss = create_network(training=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
    y_pred, _ = create_network(training=False)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_predictions = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    saver = tf.train.Saver()

    session = tf.Session()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'cifar10_cnn')

    try:
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        saver.restore(session, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        session.run(tf.initialize_all_variables())

    train_batch_size = 64


    def random_batch():

        num_images = len(images_train)
        idx = np.random.choice(num_images, size=train_batch_size, replace=False)
        x_batch = images_train[idx, :, :, :]
        y_batch = labels_train[idx, :]

        return x_batch, y_batch


    def optimize(num_iterations):

        start_time = time.time()

        for i in range(num_iterations):
            x_batch, y_true_batch = random_batch()
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)
            if (i_global % 100) == 0 or i == (num_iterations - 1):
                batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = 'Global step: {0:>6}, Train Batch Accuracy: {1:>6.1%}'
                print(msg.format(i_global, batch_acc))
            if (i_global % 1000) == 0 or i == (num_iterations - 1):
                saver.save(sess=session, save_path=save_path, global_step=global_step)
                print('Saved checkpoint.')
        end_time = time.time()
        time_dif = end_time - start_time
        print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))))


    batch_size = 256


    def predict_cls(images, labels, cls_true):

        num_images = len(images)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        i = 0
        while i < num_images:
            j = min(1 + batch_size, num_images)
            feed_dict = {x: images[i:j, :], y_true: labels[i:j, :]}
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
            i = j
        correct = (cls_true == cls_pred)
        return correct, cls_pred


    def predict_cls_test():

        return predict_cls(images_test, labels_test, cls_test)


    def classification_accuracy(correct):

        return correct.mean(), correct.sum()


    def print_test_accuracy():

        correct, cls_pred = predict_cls_test()
        acc, num_correct = classification_accuracy(correct)
        num_images = len(correct)
        msg = 'Accuracy on Test-Set: {0:.1%} ({1} / {2})'
        print(msg.format(acc, num_correct, num_images))


    optimize(10000)
    print_test_accuracy()