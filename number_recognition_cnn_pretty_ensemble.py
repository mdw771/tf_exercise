import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import prettytensor as pt
import os


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
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# combined_images.shape = (60000, 784)
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

combined_size = len(combined_images)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size

img_size = 28
img_size_flat = img_size ** 2
img_shape = (img_size, img_size)

num_channels = 1
num_classes = 10


def random_training_set():

    idx = np.random.permutation(combined_size)
    idx_train = idx[:train_size]
    idx_validation = idx[train_size:]

    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    x_validation = combined_images[idx_validation, :]
    y_validation = combined_images[idx_validation, :]

    return x_train, y_train, x_validation, y_validation


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

    # layer_conv1, weights_conv1 = new_conv_layer(x_image, num_channels, filter_size1, num_filters1, use_pooling=True)
    # layer_conv2, weights_conv2 = new_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, use_pooling=True)
    # layer_flat, num_features = flatten_layer(layer_conv2)
    #
    # layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, use_relu=True)
    # layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, use_relu=False)
    #
    # y_pred = tf.nn.softmax(layer_fc2)
    # y_pred_cls = tf.argmax(y_pred, dimension=1)
    #
    # # the function calculates the softmax internally so we must use the output of layer_fc2 directly rather than y_pred
    # # which has already had the softmax applied
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    # cost = tf.reduce_mean(cross_entropy)

    x_pretty = pt.wrap(x_image)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=16, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=36, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=128, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    def get_weights_variable(layer_name):

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable('weights')

        return variable


    batch_size = 256


    def predict_labels(images):

        # Number of images.
        num_images = len(images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        pred_labels = np.zeros(shape=(num_images, num_classes), dtype=np.float)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: images[i:j, :]}

            # Calculate the predicted class using TensorFlow.
            pred_labels[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        return pred_labels


    def correct_prediction(images, labels, cls_true):
        # Calculate the predicted labels.
        pred_labels = predict_labels(images=images)

        # Calculate the predicted class-number for each image.
        cls_pred = np.argmax(pred_labels, axis=1)

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        return correct


    def test_correct():

        return correct_prediction(data.test.images, data.test.labels, data.test.cls)


    def validation_correct():

        return correct_prediction(data.validation.images, data.validation.labels, data.validation.cls)


    def classification_accuracy(correct):

        return correct.mean()


    def test_accuracy():

        correct = test_correct()
        return classification_accuracy(correct)


    def validation_accuracy():

        correct = validation_correct()
        return classification_accuracy(correct)


    weights_conv1 = get_weights_variable(layer_name='layer_conv1')
    weights_conv2 = get_weights_variable(layer_name='layer_conv2')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    train_batch_size = 64
    test_batch_size = 256


    def random_batch(x_train, y_train):

        num_images = len(x_train)
        idx = np.random.choice(num_images, size=train_batch_size, replace=False)
        x_batch = x_train[idx, :]
        y_batch = y_train[idx, :]

        return x_batch, y_batch


    best_validation_accuracy = 0.0
    last_improvement = 0
    require_improvement = 1000

    total_iterations = 0


    def optimize(num_iterations, x_train, y_train):

        global total_iterations
        global best_validation_accuracy
        global last_improvement

        start_time = time.time()

        for i in range(num_iterations):
            total_iterations += 1
            x_batch, y_true_batch = random_batch(x_train, y_train)
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
            if (i % 100 == 0) or (i == (num_iterations - 1)):
                acc_train = session.run(accuracy, feed_dict=feed_dict_train)
                acc_validation, _ = validation_accuracy()
                if acc_validation > best_validation_accuracy:
                    best_validation_accuracy = acc_validation
                    last_improvement = total_iterations
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''
                msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
                print(msg.format(i + 1, acc_train, acc_validation, improved_str))
            if total_iterations - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")
                break
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


    def get_save_path(net_number):
        return save_dir + 'network' + str(net_number)


    num_networks = 5
    num_iterations = 10000

    for i in range(num_networks):
        print ('Neural network: {0}'.format(i))
        x_train, y_train, _, _ = random_training_set()
        session.run(tf.initialize_all_variables())
        optimize(num_iterations, x_train, y_train)
        saver.save(sess=session, save_path=get_save_path(i))
        print('\n')


    def ensemble_predictions():

        pred_labels = []
        test_accuracies = []
        val_accuracies = []

        for i in range(num_networks):
            saver.restore(sess=session, save_path=get_save_path(i))
            test_acc = test_accuracy()
            test_accuracies.append(test_acc)
            val_acc = validation_accuracy()
            val_accuracies.append(val_acc)
            msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
            print(msg.format(i, val_acc, test_acc))
            pred = predict_labels(data.test.images)
            pred_labels.append(pred)

        return np.array(pred_labels), np.array(test_accuracies), np.array(val_accuracies)


    pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

    ensemble_pred_labels = np.mean(pred_labels, axis=0)
    ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
    ensemble_correct = (ensemble_cls_pred == data.test.cls)
    np.sum(ensemble_correct)






