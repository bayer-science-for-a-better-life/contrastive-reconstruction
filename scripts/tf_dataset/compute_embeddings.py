import sys

sys.path.insert(0, '.')

import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import models.model_utils as model_utils
import argparse
from data.datasets import setup_dataset
import utils.evaluation as evaluation
import json
from data.utils import get_preprocess_fn

parser = argparse.ArgumentParser()
parser.add_argument('--models', required=True)
parser.add_argument('--dataset', required=True, default='cars196')
parser.add_argument('--out-dir', default='./resources/cars196/embeddings/')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--width', type=int, default=224)
parser.add_argument('--height', type=int, default=224)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--train-split', default='train')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

os.makedirs(args.out_dir, exist_ok=True)

train, test, _ = setup_dataset(args.dataset, None, args.width, args.height, batch_size=None, test_split='test',
                               train_split=args.train_split, threads=8,
                               preprocess_implementation='simclr-no-aug',
                               train_shuffle=False)

with open(args.models, 'r') as f:
    models = json.load(f)

for model_config in tqdm(models):
    name = model_config['name']
    path = model_config['path']
    preprocess_method = model_config['preprocess']
    out_layer = model_config['output_layer']

    model = tf.keras.models.load_model(path, compile=False)
    if out_layer:
        model = model_utils.remove_layers(model, out_layer)

    transformed_train = train
    transformed_test = test

    if preprocess_method is not None:
        preprocess = get_preprocess_fn(preprocess_method)
        transformed_train = train.map(preprocess)
        transformed_test = test.map(preprocess)

    x_train, y_train = evaluation.compute_embeddings(embedding_model=model,
                                                     dataset=transformed_train.batch(args.batch_size))
    x_test, y_test = evaluation.compute_embeddings(embedding_model=model,
                                                   dataset=transformed_test.batch(args.batch_size))
    np.savez_compressed(os.path.join(args.out_dir, f'{name}.npz'), x_train=x_train, y_train=y_train, x_test=x_test,
                        y_test=y_test, model_config=model_config)
