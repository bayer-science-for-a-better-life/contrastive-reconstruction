import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import numpy as np

METRIC_NAMES = {
    'accuracy': 'Accuracy',
    'kappa_kaggle': 'Avg QW Kappa'
}

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='resources/aptos2019/results.csv')
parser.add_argument('--metric', default='accuracy')
parser.add_argument('--legend', default='lower right')
parser.add_argument('--output', required=True)
args = parser.parse_args()

results = pd.read_csv(args.input)

sns.set_context("paper")
sns.set_style("whitegrid")
marker = ['o'] * len(np.unique(results.name.values))
g = sns.lineplot(x="label_percentage", y=args.metric, hue='name', style='name',
                 data=results.sort_values(by=['name']), dashes=False, markers=marker, legend='brief')
plt.legend(loc=args.legend)

percentages = np.unique(results.label_percentage.values)
labels = [str(int(percentage * 100)) + '%' for percentage in percentages]
plt.xticks(percentages, labels)
plt.xlabel('Percentage of labelled Images')

plt.ylabel(METRIC_NAMES[args.metric] if args.metric in METRIC_NAMES.keys() else args.metric)
sns.set(rc={'figure.figsize': (10, 5)})
plt.savefig(args.output)
plt.show()
