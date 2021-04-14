import random
import numpy as np
import math
from data.artificial.generator import ArtificialDatasetGenerator

from PIL import Image, ImageDraw


def rounded_rectangle(self: ImageDraw, xy, corner_radius, corners=(True, True, True, True), fill=None, outline=None):
    upper_left_point = xy[0]
    bottom_right_point = xy[1]
    self.rectangle(
        [
            (upper_left_point[0], upper_left_point[1] + corner_radius),
            (bottom_right_point[0], bottom_right_point[1] - corner_radius)
        ],
        fill=fill,
        outline=outline
    )
    self.rectangle(
        [
            (upper_left_point[0] + corner_radius, upper_left_point[1]),
            (bottom_right_point[0] - corner_radius, bottom_right_point[1])
        ],
        fill=fill,
        outline=outline
    )
    if corners[0]:
        self.pieslice(
            [upper_left_point, (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)],
            180,
            270,
            fill=fill,
            outline=outline
        )
    else:
        self.rectangle(
            [upper_left_point, (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)],
            fill=fill,
            outline=outline)

    if corners[3]:
        self.pieslice([(bottom_right_point[0] - corner_radius * 2, bottom_right_point[1] - corner_radius * 2),
                       bottom_right_point],
                      0,
                      90,
                      fill=fill,
                      outline=outline
                      )
    else:
        self.rectangle([(bottom_right_point[0] - corner_radius * 2, bottom_right_point[1] - corner_radius * 2),
                        bottom_right_point],
                       fill=fill,
                       outline=outline
                       )

    if corners[2]:
        self.pieslice([(upper_left_point[0], bottom_right_point[1] - corner_radius * 2),
                       (upper_left_point[0] + corner_radius * 2, bottom_right_point[1])],
                      90,
                      180,
                      fill=fill,
                      outline=outline
                      )
    else:
        self.rectangle([(upper_left_point[0], bottom_right_point[1] - corner_radius * 2),
                        (upper_left_point[0] + corner_radius * 2, bottom_right_point[1])],
                       fill=fill,
                       outline=outline
                       )

    if corners[1]:
        self.pieslice([(bottom_right_point[0] - corner_radius * 2, upper_left_point[1]),
                       (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)],
                      270,
                      360,
                      fill=fill,
                      outline=outline
                      )
    else:
        self.rectangle([(bottom_right_point[0] - corner_radius * 2, upper_left_point[1]),
                        (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)],
                       fill=fill,
                       outline=outline
                       )


class RectangleTriangleDatasetGenerator(ArtificialDatasetGenerator):

    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height

    @property
    def num_classes(self):
        return 4

    def generate_sample(self, cls):
        assert cls in list(range(self.num_classes))

        im = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(im)

        min_length = self.width // 2
        padding = 20
        max_length = self.width - 2 * padding
        lengthx = random.randint(min_length, max_length)
        lengthy = random.randint(min_length, max_length)
        startx = random.randint(padding, self.width - lengthx - padding)
        starty = random.randint(padding, self.height - lengthy - padding)

        rectangle_points = [(startx, starty), (startx + lengthx, starty + lengthy)]

        corners = [cls in [0, 1]] * 4
        rounded_rectangle(draw, rectangle_points, 20, fill=(255, 255, 255), corners=corners)
        padding = 4

        def equiliteral_triangle():
            start = (random.randint(startx + padding, startx + lengthx // 2),
                     random.randint(starty + padding, starty + lengthy // 2))
            length = random.randint(self.width // 8, min(lengthx - (start[0] - startx), lengthy - (start[1] - starty)))
            second = (start[0] + length, start[1])
            third = (start[0] + length // 2, start[1] + length * math.sqrt(3) / 2)
            draw.polygon([start, second, third], fill=(0, 0, 0))

        def right_triangle():
            start = (random.randint(startx + padding, startx + lengthx // 2),
                     random.randint(starty + padding, starty + lengthy // 2))
            x_len = random.randint(self.width // 8, lengthx - (start[0] - startx))
            y_len = random.randint(self.height // 8, lengthy - (start[1] - starty))
            second = (start[0] + x_len, start[1])
            third = (start[0], start[1] + y_len)
            draw.polygon([start, second, third], fill=(0, 0, 0))

        if cls % 2 == 0:
            equiliteral_triangle()
        else:
            right_triangle()

        del draw
        angle = random.randint(-90, 90)
        im = im.rotate(angle)
        return np.expand_dims(np.asarray(im)[:, :, 0], -1)
