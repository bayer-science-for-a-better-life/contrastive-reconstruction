import sys

sys.path.insert(0, '.')

from data.datasets import load_aptos_2019
import tensorflow as tf
from utils.evaluation import compute_embeddings
import os
from tqdm import tqdm
import numpy as np
import models.model_utils as model_utils
import argparse
import json
from data.utils import get_preprocess_fn


parser = argparse.ArgumentParser()
parser.add_argument('--models', required=True)
parser.add_argument('--data-path', default='resources/aptos2019')
parser.add_argument('--out-dir', default='./resources/aptos2019/embeddings/')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--gpu', default='0', help="The id of the gpu device")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset, _ = load_aptos_2019(args.data_path, 'all.csv', shuffle=False, threads=4)
os.makedirs(args.out_dir, exist_ok=True)

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

    transformed_ds = dataset

    if preprocess_method is not None:
        preprocess = get_preprocess_fn(preprocess_method)
        transformed_ds = dataset.map(preprocess)

    x, y = compute_embeddings(model, transformed_ds.batch(args.batch_size))
    np.savez_compressed(os.path.join(args.out_dir, f'{name}.npz'), x=x, y=y, model_config=model_config)
