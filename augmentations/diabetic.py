from tensorflow.python.keras.preprocessing.image import random_zoom
import tensorflow as tf


def resize(img, width, height):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return tf.image.resize(img, [width, height])


def augment(img):
    def rand_zoom(x):
        return random_zoom(x, zoom_range=(0.85, 1.15), channel_axis=2, row_axis=0, col_axis=1,
                           fill_mode='constant', cval=0.0)

    img = tf.numpy_function(rand_zoom, [img], tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img
