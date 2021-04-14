from os.path import join
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
import random


def _construct_dataset(base_path, id_codes, diagnosis, shuffle=False, threads=8):
    dataset = tf.data.Dataset.from_tensor_slices((id_codes, diagnosis))
    if shuffle:
        dataset = dataset.shuffle(len(id_codes))

    def generator(file_ids, labels):
        for filename, label in zip(file_ids, labels):
            filename = filename.decode("utf-8")
            img_path = join(base_path, 'images', filename + '.png')
            im_frame = Image.open(img_path)
            img = np.asarray(im_frame, dtype="float32")
            img /= 255.0
            if label == 0:
                label = [1, 0, 0, 0, 0]
            elif label == 1:
                label = [1, 1, 0, 0, 0]
            elif label == 2:
                label = [1, 1, 1, 0, 0]
            elif label == 3:
                label = [1, 1, 1, 1, 0]
            elif label == 4:
                label = [1, 1, 1, 1, 1]
            transformed_label = np.array(label)
            yield img, transformed_label

    dataset = dataset.batch(len(id_codes) // threads + 1)
    dataset = dataset.interleave(lambda file_ids, labels: tf.data.Dataset.from_generator(generator,
                                                                                         output_types=(
                                                                                         tf.float32, tf.int32),
                                                                                         args=(file_ids, labels)),
                                 cycle_length=threads, num_parallel_calls=threads)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_aptos_2019(base_path, split_file, shuffle=True, threads=4):
    csv = pd.read_csv(join(base_path, 'splits', split_file))
    num_examples = len(csv)
    dataset = _construct_dataset(base_path, csv.id_code.values, csv.diagnosis.values,
                                 shuffle=shuffle, threads=threads)
    return dataset, num_examples


def cross_validation(base_path, csv_file, k=5, shuffle=True, threads=4, label_percentage=1.0):
    csv = pd.read_csv(join(base_path, 'splits', csv_file))
    X = csv.id_code.values
    Y = csv.diagnosis.values

    kf = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(X, Y):
        random.shuffle(train_index)
        train_index = train_index[:int(len(train_index) * label_percentage)]
        train = _construct_dataset(base_path, X[train_index], csv.diagnosis.values[train_index],
                                   shuffle=shuffle, threads=threads)
        test = _construct_dataset(base_path, X[test_index], csv.diagnosis.values[test_index],
                                  shuffle=shuffle, threads=threads)
        yield train, test, len(train_index), train_index, test_index
