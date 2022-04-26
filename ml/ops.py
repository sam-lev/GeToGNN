from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from .layers import Layer
import tensorflow as tf

from ml.utils import pout

flags = tf.app.flags
FLAGS = flags.FLAGS


def _map_slice( input):
    sigma, geto_dims, n_dimensions = input
    sigma = tf.slice(sigma, [0, 0], [geto_dims, n_dimensions])
    return sigma


def euclideanMeanDistance( x, y):

    dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))
    return tf.reduce_mean(dist)


def euclideanDistance( x, y):
    #dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))
    dist = tf.sqrt(tf.reduce_sum(tf.square(
        tf.subtract(x, y, name='subtract1')),axis=1), name='dist_embedd')
    return dist


def reduce_pca( X, geto_dims = None):
    keep_info = 0
    x_shape = tf.shape(X)
    X_reshaped = tf.reshape(X, (2, x_shape[0] / 2))  # (8 , x_shape[0] / 8))
    singular_values, u, v = tf.linalg.svd(X_reshaped)
    sigma = tf.linalg.tensor_diag(singular_values)

    tf.print(singular_values, output_stream=sys.stdout)
    tf.print(x_shape, output_stream=sys.stdout)

    singular_values_sum = tf.reduce_sum(singular_values, axis=0)  # tf.map_fn(sum,singular_values)
    normalized_singular_values = singular_values / singular_values_sum
    # Create the aggregated ladder of kept information per dimension
    ladder = np.cumsum(normalized_singular_values)
    # Get the first index which is above the given information threshold
    # index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
    n_dimensions = 2  # index
    # with self.graph.as_default():
    # Cut out the relevant part from sigma
    slice_input = (sigma, geto_dims, n_dimensions)
    # sigma = tf.map_fn(self._map_slice, slice_input)
    slice_dim = tf.shape(X_reshaped)[0]
    sigma = tf.slice(sigma, [0, 0], [slice_dim, n_dimensions])
    # PCA
    pca = tf.matmul(u, sigma)

    return pca