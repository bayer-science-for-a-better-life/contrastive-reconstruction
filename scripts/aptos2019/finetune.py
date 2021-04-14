import sys

sys.path.insert(0, '.')

import argparse
import numpy as np
from collections import defaultdict
import pandas as pd
import warnings
from tqdm import tqdm
import tensorflow as tf
from utils.metrics import score_cat_acc_kaggle, score_kappa_kaggle
import models.model_utils as model_utils
from tensorflow.keras.layers import Activation
from data.aptos2019 import cross_validation
import json
from data.utils import get_preprocess_fn
from utils import train_utils
import os
from data.datasets import preprocess_and_batch


def train_and_evaluate(embedding_model, train, test, optimizer, epochs=30):
    classifier = model_utils.add_linear_head(embedding_model, 5)

    classifier = tf.keras.Sequential([classifier, Activation('sigmoid')])

    def cat_acc(y1, y2):
        return tf.numpy_function(score_cat_acc_kaggle, [y1, y2], tf.float32)

    def score_kappa(y1, y2):
        return tf.numpy_function(score_kappa_kaggle, [y1, y2], tf.float32)

    metrics = [tf.keras.metrics.BinaryAccuracy(),
               cat_acc,
               score_kappa]
    loss = tf.keras.losses.BinaryCrossentropy()

    classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    classifier.fit(train, epochs=epochs, verbose=0)
    return classifier.evaluate(test, return_dict=True)


warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
train_utils.add_train_args(parser)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--models', required=True)
parser.add_argument('--splits')
parser.add_argument('--data-path', default='resources/aptos2019')
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--output', required=True)
parser.add_argument('--gpu', default='0', help="The id of the gpu device")
parser.add_argument('--preprocess', default='none')
parser.add_argument('--linear', action='store_true')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

with open(args.models, 'r') as f:
    models = json.load(f)

eval_results = {model_config['name']: defaultdict(list) for model_config in models}

indices = []
for r in range(args.repetitions):
    repetition_indices = []
    for train, test, num_examples, train_index, test_index in cross_validation(args.data_path, 'all.csv', k=5,
                                                                               shuffle=True, threads=4,
                                                                               label_percentage=1.0):
        repetition_indices.append({
            'train': train_index.tolist(),
            'test': test_index.tolist()
        })
        for model_config in tqdm(models):
            name = model_config['name']
            path = model_config['path']
            preprocess_method = model_config['preprocess']
            out_layer = model_config['output_layer']

            model = tf.keras.models.load_model(path, compile=False)
            if out_layer:
                model = model_utils.remove_layers(model, out_layer)

            transformed_train, transformed_test = preprocess_and_batch(train=train, test=test,
                                                                       height=224, width=224,
                                                                       preprocess_implementation=args.preprocess,
                                                                       batch_size=None)

            if preprocess_method is not None:
                preprocess = get_preprocess_fn(preprocess_method)
                transformed_train = train.map(preprocess)
                transformed_test = test.map(preprocess)

            schedule = train_utils.learning_rate_schedule(args, num_examples)
            optimizer = train_utils.get_optimizer(args, schedule)

            if args.linear:
                # Freeze model
                model.trainable = False

            result = train_and_evaluate(embedding_model=model, train=transformed_train.batch(args.batch_size),
                                        test=transformed_test.batch(args.batch_size), optimizer=optimizer,
                                        epochs=args.epochs)
            print(result)
            for metric, value in result.items():
                eval_results[name][metric].append(value)
    indices.append(repetition_indices)

    data = []
    for name, results in eval_results.items():
        model_data = dict(name=name, repetitions=r)
        for metric, values in results.items():
            model_data[metric] = np.mean(values)
        data.append(model_data)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)

    if args.splits:
        with open(args.splits, 'w') as f:
            json.dump(indices, f)
