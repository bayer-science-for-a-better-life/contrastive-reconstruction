from scipy.special import comb
import numpy as np
import random
import tensorflow as tf


def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals


def non_linear_transformation(x):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x.astype(np.float32)


def generate_indices_grid(start_x, start_y, dist_x, dist_y):
    x_range = tf.range(start_x, start_x + dist_x)
    y_range = tf.range(start_y, start_y + dist_y)
    pair_indices = tf.stack(tf.meshgrid(*[y_range, x_range]))
    pair_indices = tf.transpose(pair_indices, perm=[2, 1, 0])
    return tf.reshape(pair_indices, (-1, 2))


def local_pixel_shuffling(x, width, height, channels=1, max_block_factor=0.2, num_block=15, stop_prob=0.95):
    shuffled_image = tf.identity(x)
    orig_image = tf.identity(x)
    count = num_block
    while count > 0 and rand_float() < stop_prob:
        block_noise_size_x = randint(1, width * max_block_factor)
        block_noise_size_y = randint(1, height * max_block_factor)
        noise_x = randint(0, width - block_noise_size_x)
        noise_y = randint(0, height - block_noise_size_y)
        window = orig_image[noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x]
        window = tf.reshape(window, (-1, channels))
        window = tf.random.shuffle(window)
        indices = generate_indices_grid(noise_x, noise_y, block_noise_size_x, block_noise_size_y)
        shuffled_image = tf.tensor_scatter_nd_update(shuffled_image, indices=indices, updates=window)
        count -= 1
    return shuffled_image


def image_in_painting(x, width, height, channels=1, count=5, min_block_factor=0.14, max_block_factor=0.2,
                      stop_prob=0.95):
    while count > 0 and rand_float() < stop_prob:
        block_noise_size_x = randint(width * min_block_factor, width * max_block_factor)
        block_noise_size_y = randint(height * min_block_factor, height * max_block_factor)
        noise_x = randint(3, width - block_noise_size_x - 3)
        noise_y = randint(3, height - block_noise_size_y - 3)
        indices = generate_indices_grid(noise_x, noise_y, block_noise_size_x, block_noise_size_y)
        random_color = tf.random.uniform(shape=[1, channels])
        updates = tf.repeat(random_color, repeats=block_noise_size_x * block_noise_size_y, axis=0)
        x = tf.tensor_scatter_nd_update(x, indices=indices, updates=updates)
        count -= 1
    return x


def image_out_painting(x, width, height, channels=1):
    random_color = tf.random.uniform(shape=[1, channels])
    cnt = 5
    random_color_image = tf.reshape(tf.repeat(random_color, repeats=width * height, axis=0),
                                    (height, width, channels))
    out = random_color_image
    prob = 1.0
    while cnt > 0 and rand_float() < prob:
        prob = 0.95
        block_noise_size_x = width - randint(2 * width // 7, 3 * width // 7)
        block_noise_size_y = height - randint(2 * height // 7, 3 * height // 7)
        noise_x = randint(3, width - block_noise_size_x - 3)
        noise_y = randint(3, height - block_noise_size_y - 3)
        updates = tf.reshape(x[noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x],
                             (-1, channels))
        indices = generate_indices_grid(noise_x, noise_y, block_noise_size_x, block_noise_size_y)
        out = tf.tensor_scatter_nd_update(out, indices=indices, updates=updates)
        cnt -= 1
    return out


def randint(min, max):
    min = tf.cast(min, dtype=tf.int32)
    max = tf.cast(max, dtype=tf.int32)
    return tf.random.uniform([], minval=min, maxval=max, dtype=tf.dtypes.int32)


def rand_float():
    return tf.random.uniform(shape=[])


def transform_image(img, width, height, channels, mode='all'):
    r = rand_float()

    if channels == 1:
        if mode == 'all':
            if r <= 0.25:
                result = local_pixel_shuffling(img, channels=channels, width=width, height=height, stop_prob=1.0)
            elif 0.25 < r and r <= 0.5:
                result = tf.numpy_function(non_linear_transformation, [img], tf.float32)
            elif 0.5 < r and r <= 0.75:
                result = image_in_painting(img, channels=channels, width=width, height=height)
            else:
                result = image_out_painting(img, channels=channels, width=width, height=height)
        else:
            raise ValueError()

    else:
        if mode == 'all':
            if r <= 0.33:
                result = local_pixel_shuffling(img, channels=channels, width=width, height=height, stop_prob=0.99)
            elif r <= 0.66:
                result = image_in_painting(img, channels=channels, width=width, height=height)
            else:
                result = image_out_painting(img, channels=channels, width=width, height=height)
        else:
            raise ValueError()
    return result
