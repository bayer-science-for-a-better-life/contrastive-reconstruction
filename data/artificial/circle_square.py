from data.artificial.generator import ArtificialDatasetGenerator
from PIL import ImageDraw, Image
import random
import math
import numpy as np


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class NonOverlappingObjectDrawer:

    def __init__(self, draw, width, height, fill=(255, 255, 255)):
        self._draw = draw
        self._width = width
        self._height = height
        self._objects = []
        self._fill = fill

    def _overlaps(self, center, radius):
        for p, p_radius in self._objects:
            if dist((center[0], center[1]), p) < radius + p_radius + 10:
                return True
        return False

    def _random_x(self, radius):
        return random.randint(radius, self._width - radius)

    def _random_y(self, radius):
        return random.randint(radius, self._height - radius)

    def _find_center_position(self, radius):
        center = (self._random_x(radius), self._random_y(radius))
        while self._overlaps(center, radius):
            center = (self._random_x(radius), self._random_y(radius))
        return center

    def _square_like(self, func, min_length=10, max_length=15):
        length = random.randint(min_length, max_length)
        center = self._find_center_position(length // 2)
        radius = length // 2
        func((center[0] - radius, center[1] - radius,
              center[0] + radius, center[1] + radius), fill=self._fill)
        self._objects.append((center, radius))

    def random_square(self, **kwargs):
        self._square_like(self._draw.rectangle, **kwargs)

    def random_circle(self, **kwargs):
        self._square_like(self._draw.ellipse, **kwargs)


class CircleSquareDatasetGenerator(ArtificialDatasetGenerator):

    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height

    @property
    def num_classes(self):
        return 3

    def generate_sample(self, cls):
        im = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(im)
        drawer = NonOverlappingObjectDrawer(draw=draw, height=self.height, width=self.width)

        min_length = self.width // 8
        max_length = self.width // 3

        for condition in [cls in [0, 1], cls == 0]:
            if condition:
                drawer.random_circle(min_length=min_length, max_length=max_length)
            else:
                drawer.random_square(min_length=min_length, max_length=max_length)

        angle = random.randint(-90, 90)
        im = im.rotate(angle)
        return np.expand_dims(np.asarray(im)[:, :, 0], -1)
