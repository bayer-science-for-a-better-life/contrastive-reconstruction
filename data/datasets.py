import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from data.subsets import TFDataSubset
from data.aptos2019 import load_aptos_2019
from augmentations import contrastive
from augmentations import diabetic

DATASETS = ['numpy', 'stanford_dogs', 'oxford_flowers102', 'aptos2019', 'image-folder']


def load_dataset(identifier, data_path, threads=4, test_split='test', train_split='train',
                 cache=False, shuffle_train=True, shuffle_test=False, docker_down=False):
    test = None
    if identifier == 'numpy':
        data = np.load(data_path)

        def convert_to_float(images):
            if images.dtype in [np.uint8, np.int32, np.int64]:
                return images.astype(np.float32) / 255.0
            return images

        x_train, y_train = shuffle(convert_to_float(data['x_train']), data['y_train'], random_state=0)
        x_test, y_test = shuffle(convert_to_float(data['x_test']), data['y_test'], random_state=0)
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        num_examples = len(x_train)
    elif identifier == 'image-folder':
        builder = tfds.ImageFolder(data_path)
        train = builder.as_dataset(split=train_split, shuffle_files=True, as_supervised=True)
        num_examples = builder.info.splits[train_split].num_examples
        if test_split is not None:
            test = builder.as_dataset(split=test_split, shuffle_files=True, as_supervised=True)
    elif identifier == 'aptos2019':
        train, num_examples = load_aptos_2019(base_path=data_path, split_file=train_split, threads=threads)
        if test_split is not None:
            test, _ = load_aptos_2019(base_path=data_path, split_file=test_split, threads=threads)
    else:
        if data_path is None:
            builder = tfds.builder(identifier)
            if docker_down:
                builder = tfds.builder(identifier, data_dir="/home/tensorflow_datasets")
            builder.download_and_prepare()
            train = builder.as_dataset(split=train_split, shuffle_files=shuffle_train, as_supervised=True)
            if test_split is not None:
                test = builder.as_dataset(split=test_split, shuffle_files=shuffle_test, as_supervised=True)
            num_examples = builder.info.splits[train_split].num_examples
        else:
            subset = TFDataSubset(identifier, file=data_path, train_split=train_split,
                                  test_split=test_split, shuffle_train=shuffle_train, shuffle_test=shuffle_test)
            train = subset.train()
            if test_split is not None:
                test = subset.test()
            num_examples = subset.num_train_examples()
    if cache:
        train = train.cache()
        if test is not None:
            test = test.cache()

    return train, test, num_examples


def preprocess_and_batch(train, test, height, width, preprocess_implementation='none', batch_size=8):
    def preprocess(training=False):
        def preprocess_func(img, y):
            if preprocess_implementation == 'simclr':
                img = contrastive.preprocess_image(img, height, width,
                                                   is_training=training,
                                                   color_distort=False, test_crop=True)
            elif preprocess_implementation == 'simclr-no-aug':
                img = contrastive.preprocess_image(img, height, width,
                                                   is_training=False,
                                                   color_distort=False, test_crop=True)
            elif preprocess_implementation == '1-channel-aug':
                img = tf.image.convert_image_dtype(img, dtype=tf.float32)
                if training:
                    img = contrastive.random_crop_with_resize(img, height, width)
                    img = tf.image.random_flip_left_right(img)

                img = tf.reshape(img, [height, width, 1])
                img = tf.clip_by_value(img, 0., 1.)
            elif preprocess_implementation == 'diabetic':
                img = tf.image.convert_image_dtype(img, dtype=tf.float32)
                if training:
                    img = diabetic.augment(img)
            elif preprocess_implementation == 'greyscale':
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            elif preprocess_implementation == 'none':
                img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            else:
                raise ValueError('Unknown preprocessing')
            return img, y

        return preprocess_func

    train = train.map(preprocess(training=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.map(preprocess(training=False), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    if batch_size is not None:
        train, test = [ds.batch(batch_size, drop_remainder=False) for ds in [train, test]]
    return train, test


def setup_dataset(dataset, data_path, height, width, batch_size, test_split='test',
                  train_split='train', threads=8, preprocess_implementation='none', train_shuffle=True, *args, **kwargs):
    train, test, num_examples = load_dataset(dataset, data_path, test_split=test_split, train_split=train_split,
                                             shuffle_train=train_shuffle, shuffle_test=False,
                                             threads=threads)
    if train_shuffle:
        train = train.shuffle(batch_size * 20)
    train, test = preprocess_and_batch(train=train, test=test, height=height, width=width, batch_size=batch_size,
                                       preprocess_implementation=preprocess_implementation)
    return train, test, num_examples
