import sys

sys.path.insert(0, '.')

from data.artificial.rectangle_triangle import RectangleTriangleDatasetGenerator
from data.artificial.circle_square import CircleSquareDatasetGenerator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--output', default='rectangle-triangles.npz')
parser.add_argument('--num-train', type=int, default=1000)
parser.add_argument('--num-test', type=int, default=500)
parser.add_argument('--type', choices=['circle-square', 'rectangle-triangle'], default='rectangle-triangle')
args = parser.parse_args()

if args.type == 'circle-square':
    generator = CircleSquareDatasetGenerator()
elif type == 'rectangle-triangle':
    generator = RectangleTriangleDatasetGenerator()
else:
    raise ValueError("Unknown synthetic dataset type")

generator.generate_dataset(filename=args.output, num_train=args.num_train, num_test=args.num_test)
