import tensorflow as tf


def split_batch_and_concat(num_transform=2, axis=-1):
    def transform(features, label):
        features_list = tf.split(features, num_or_size_splits=num_transform, axis=axis)
        features = tf.concat(features_list, 0)
        return features, label

    return transform


def get_preprocess_fn(name):
    if name == 'densenet':
        preprocess_fn = tf.keras.applications.densenet.preprocess_input
    elif name == 'resnet':
        preprocess_fn = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError()

    def preprocess(x, y):
        x = tf.numpy_function(preprocess_fn, [tf.cast(x * 255, dtype=tf.uint8)], tf.float32)
        return x, y

    return preprocess
