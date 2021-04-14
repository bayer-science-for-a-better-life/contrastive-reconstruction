import tensorflow_datasets as tfds
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import numpy as np

IDENTIFIERS = {
    'oxford_flowers102': 'file_name'
}


def unique_key_identifier(tf_dataset_identifier):
    if tf_dataset_identifier in IDENTIFIERS.keys():
        return IDENTIFIERS[tf_dataset_identifier]
    return 'image/filename'


class TFSubsetGenerator:
    def __init__(self, tf_dataset_identifier, train_split='train', test_split='test', label_key='label'):
        builder = tfds.builder(tf_dataset_identifier)
        builder.download_and_prepare()
        self.train = builder.as_dataset(split=train_split, shuffle_files=False)
        self.test = builder.as_dataset(split=test_split, shuffle_files=False)
        self.unique_identifier_key = unique_key_identifier(tf_dataset_identifier)
        self.label_key = label_key
        self.train_split = train_split
        self.test_split = test_split

    def generate_subset(self, classes, filename, num_train=None, num_test=None):
        train_samples = []
        train_counter = defaultdict(int)
        for features in tqdm(self.train):
            label = features[self.label_key].numpy().item()
            if label in classes and (num_train is None or train_counter[label] < num_train):
                train_counter[label] += 1
                train_samples.append((features[self.unique_identifier_key].numpy().decode("utf-8"), label))
        test_samples = []
        test_counter = defaultdict(int)
        for features in tqdm(self.test):
            label = features[self.label_key].numpy().item()
            if label in classes and (num_test is None or test_counter[label] < num_test):
                test_counter[label] += 1
                test_samples.append((features[self.unique_identifier_key].numpy().decode("utf-8"), label))
        result = []
        for samples, split in ((train_samples, self.train_split), (test_samples, self.test_split)):
            for i, label in samples:
                result.append({
                    'split': split,
                    'id': i,
                    'label': label
                })
        pd.DataFrame(result).to_csv(filename, index=False)


class TFDataSubset:

    def __init__(self, tf_dataset_identifier, file, train_split='train', test_split='test', shuffle_train=False,
                 shuffle_test=False, convert_labels=True):
        builder = tfds.builder(tf_dataset_identifier)
        builder.download_and_prepare()
        self._train = builder.as_dataset(split=train_split, shuffle_files=shuffle_train)
        self._test = builder.as_dataset(split=test_split, shuffle_files=shuffle_test)
        self.subset = pd.read_csv(file)
        self.train_split = train_split
        self.test_split = test_split
        self.unique_identifier_key = unique_key_identifier(tf_dataset_identifier)
        self.convert_labels = convert_labels
        self.labels = {j: i for i, j in enumerate(list(sorted(np.unique(self.subset.label.values).tolist())))}

    @property
    def _feature_key_identifier(self):
        return 'image'

    @property
    def _label_key_identifier(self):
        return 'label'

    def _filter_ds(self, ds, split):
        def filter_func(x):
            x = x.decode("utf-8")
            return x in self.subset[self.subset['split'] == split].id.values.tolist()

        def map_func(x):
            return x[self._feature_key_identifier], x[self._label_key_identifier]

        ds = ds.filter(
            lambda x: tf.numpy_function(filter_func, inp=[x[self.unique_identifier_key]], Tout=tf.bool)) \
            .map(map_func)

        if self.convert_labels:
            def convert(x, y):
                return x, self.labels[y]

            ds = ds.map(lambda x, y: tf.numpy_function(convert, inp=[x, y], Tout=(tf.uint8, tf.int64)))
        return ds

    def train(self):
        return self._filter_ds(self._train, self.train_split)

    def test(self):
        return self._filter_ds(self._test, self.test_split)

    def num_train_examples(self):
        return len(self.subset[self.subset['split'] == self.train_split])
