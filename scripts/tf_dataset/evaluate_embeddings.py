import sys
import os

sys.path.insert(0, '.')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import random
import numpy as np
import utils.evaluation as evaluation
from collections import defaultdict
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class ModelEmbedding:

    def __init__(self, name, x_train, y_train, x_test, y_test, model_path=None):
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_path = model_path

    @staticmethod
    def from_file(name, path):
        contents = np.load(path)
        return ModelEmbedding(name=name, x_train=contents['x_train'], y_train=contents['y_train'],
                              x_test=contents['x_test'],
                              y_test=contents['y_test'])


parser = argparse.ArgumentParser()
parser.add_argument('--embeddings-dir', default='resources/aptos2019/embeddings')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('--label-percentages', nargs='+', type=float, default=[1.0])
parser.add_argument('--C', type=float, default=1.0)
args = parser.parse_args()

y_train = None
y_test = None
embeddings = []
for file in os.listdir(args.embeddings_dir):
    if not file.endswith('.npz'):
        print('Skipping file: ', file)
    embedding = ModelEmbedding.from_file(name=file.split('.npz')[0], path=os.path.join(args.embeddings_dir, file))
    # sanity check that all embeddings are in same order
    assert y_train is None or (y_train == embedding.y_train).all()
    assert y_test is None or (y_test == embedding.y_test).all()
    y_train = embedding.y_train
    y_test = embedding.y_test
    embeddings.append(embedding)

num_examples = len(y_train)

eval_results = {label_percentage: {embedding.name: defaultdict(list) for embedding in embeddings}
                for label_percentage in args.label_percentages}
for label_percentage in args.label_percentages:
    for r in tqdm(range(args.repetitions)):
        train_index = random.sample(list(range(num_examples)), k=int(num_examples * label_percentage))
        random.shuffle(train_index)
        for embedding in tqdm(embeddings):
            x_train = embedding.x_train[train_index]
            x_test = embedding.x_test
            y_train = embedding.y_train[train_index]
            y_test = embedding.y_test
            result = evaluation.linear_evaluation_with_embeddings(x_train=x_train,
                                                                  y_train=y_train,
                                                                  x_test=x_test,
                                                                  y_test=y_test,
                                                                  logistic_params={'C': args.C},
                                                                  type='categorical')
            for metric, value in result.items():
                eval_results[label_percentage][embedding.name][metric].append(value)

data = []
for label_percentage, model_results in eval_results.items():
    for name, results in model_results.items():
        model_data = dict(name=name, label_percentage=label_percentage)
        for metric, values in results.items():
            model_data[metric] = np.mean(values)
        data.append(model_data)

df = pd.DataFrame(data)
print(df)
df.to_csv(args.output, index=False)
