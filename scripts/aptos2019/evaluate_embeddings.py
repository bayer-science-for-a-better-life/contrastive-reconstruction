import sys

sys.path.insert(0, '.')

import argparse
import random
from sklearn.model_selection import KFold
import numpy as np
import os
import utils.evaluation as evaluation
from collections import defaultdict
import pandas as pd
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings("ignore", category=UserWarning)


class ModelEmbedding:

    def __init__(self, name, x, y, model_path):
        self.name = name
        self.x = x
        self.y = y
        self.model_path = model_path

    @staticmethod
    def from_file(name, path):
        contents = np.load(path)
        return ModelEmbedding(name=name, x=contents['x'], y=contents['y'], model_path=None)


parser = argparse.ArgumentParser()
parser.add_argument('--embeddings-dir', default='resources/aptos2019/embeddings')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('--split-file', default=None)
parser.add_argument('--label-percentages', nargs='+', type=float, default=[1.0])
args = parser.parse_args()

y = None
embeddings = []
for file in os.listdir(args.embeddings_dir):
    if not file.endswith('.npz'):
        print('Skipping file: ', file)
    embedding = ModelEmbedding.from_file(name=file.split('.npz')[0], path=os.path.join(args.embeddings_dir, file))
    # sanity check that all embeddings are in same order
    assert y is None or (y == embedding.y).all()
    y = embedding.y
    embeddings.append(embedding)

num_examples = len(y)

eval_results = {label_percentage: {embedding.name: defaultdict(list) for embedding in embeddings}
                for label_percentage in args.label_percentages}
for label_percentage in args.label_percentages:
    for r in tqdm(range(args.repetitions)):
        kf = KFold(n_splits=args.folds, shuffle=True)

        if args.split_file is None:
            iterator = kf.split(list(range(num_examples)))
        else:
            with open(args.split_file, 'r') as f:
                splits = json.load(f)


            def build_iterator():
                for split in splits[r]:
                    yield split['train'], split['test']


            iterator = build_iterator()

        for train_index, test_index in tqdm(iterator):
            print(len(train_index), len(test_index))
            for idx in test_index:
                assert idx not in train_index
            random.shuffle(train_index)
            train_index = train_index[:int(len(train_index) * label_percentage)]
            for embedding in tqdm(embeddings):
                x_train = embedding.x[train_index]
                x_test = embedding.x[test_index]
                y_train = embedding.y[train_index]
                y_test = embedding.y[test_index]
                result = evaluation.linear_evaluation_with_embeddings(x_train=x_train,
                                                                      y_train=y_train,
                                                                      x_test=x_test,
                                                                      y_test=y_test,
                                                                      type='diabetic')
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
