import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'Tensorflow-Tutorials'))
import inception
import prettytensor as pt
import cifar10
from cifar10 import num_classes
from inception import transfer_values_cache
from sklearn.decomposition import PCA


def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()


cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

inception.maybe_download()
model = inception.Inception()

file_path_cache_train = os.path.join('data', 'CIFAR-10', 'inception_cifar10_train.pk1')
file_path_cache_test = os.path.join('Tensorflow-Tutorials', cifar10.data_path, 'inception_cifar10_test.pk1')

images_scaled = images_train * 255.0
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, images=images_scaled, model=model)
images_scaled = images_test * 255.0
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, images=images_scaled, model=model)

pca = PCA(n_components=2)
transfer_values = transfer_values_train[0:3000]
cls = cls_train[:3000]
transfer_values_reduced = pca.fit_transform(transfer_values)

plot_scatter(transfer_values_reduced, cls)
