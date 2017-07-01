import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'Tensorflow-Tutorials'))
import inception


inception.maybe_download()
model = inception.Inception()
print inception.data_dir

def classify(image_path):

    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)


image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)
