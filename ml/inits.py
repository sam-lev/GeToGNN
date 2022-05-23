import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32, seed=801)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init. (uniform xavier)"""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32, seed=801)
    return tf.Variable(initial, name=name)

def normxav(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init. (normal dist Xavier)"""
    init_range = np.sqrt(2.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32, seed=801)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32, seed=801)
    return tf.Variable(initial, name=name)

def normxav_lrhype(shape, maxval=1,minval=-1,name=None):
    """Positive uniform init."""
    dist = np.sqrt(6.0/(shape[0]+shape[1]),dtype=np.float32)
    initial = tf.multiply(dist,
                          tf.random_uniform(shape, minval=minval, maxval=maxval,
                                            dtype=tf.float32, seed=801))
    return tf.Variable(initial, name=name)