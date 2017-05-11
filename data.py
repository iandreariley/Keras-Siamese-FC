import numpy as np
from keras.preprocessing import image
from math import ceil
from util import euclidean_distance, inclusive_range

# Create labels for a score map of size dim x dim, where the label of a score is positive for any cell within
# radius of the center, and negative otherwise.
def make_label(dim, radius):
    label = np.full((dim, dim), -1)
    center = int(dim / 2.0)
    start = center - ceil(radius)
    end = center + ceil(radius)
    for i in inclusive_range(start, end):
        for j in inclusive_range(start, end):
            if euclidean_distance(i, j, center, center) <= radius:
                label[i,j] = 1
    return label

def n_copy_array(source, n):
    copies = np.empty((n,) + source.shape)
    copies[:] = source
    return copies

def make_label_weight_mask(dim, radius, n_imgs):
    label = make_label(dim, radius)
    values, counts = np.unique(labels, return_counts=True)
    
    weights = np.empty(label.shape)
    for i in range(len(values)):
        weights[np.where(label == values[i])] = 0.5 / counts[i] * 100
    
    weight_labels = n_copy_array(weights, n_imgs)
    return weight_labels

# Creates a matrix of weights where the sum of the weights for each label in {-1, 1} is 0.5. Used to account for
# the fact that, given a search image we may have more negative than positive examples or vice versa.
def make_label_weights(labels):
    values, counts = np.unique(labels, return_counts=True)
    class_weights = {}
    for i in range(len(values)):
        class_weights[values[i]] = 0.5 / counts[i]
    return class_weights

def normalize(images):
    color_means = np.mean(images, axis=(0,1,2), keepdims=True)
    images -= color_means
    color_std_dev = np.std(images, axis=(0,1,2), keepdims=True)
    images /= color_std_dev
    return images

def load_images(directory, dimension, n_images, suffix, normalize_images=False):
    img_array = np.empty((n_images, dimension, dimension, 3))
    for i in range(1, n_images + 1):
        img = image.load_img(directory + str(i) + suffix, target_size=(dimension, dimension))
        img_array[i - 1] = image.img_to_array(img)
    return normalize(img_array) if normalize_images else img_array