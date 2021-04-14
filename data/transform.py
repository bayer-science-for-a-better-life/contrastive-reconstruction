from augmentations import contrastive as augmentations2d
import tensorflow as tf
import augmentations.masking as masking


def preprocess_image(tf_dataset, img_height=224, img_width=224, test_crop=True, is_training=False):
    def transform(image, label):
        image = augmentations2d.preprocess_image(image,
                                                 height=img_height,
                                                 width=img_width,
                                                 is_training=is_training,
                                                 color_distort=False,
                                                 test_crop=test_crop)
        return image, label

    return tf_dataset.map(transform)


def conrec_dataset(dataset, batch_size, height=224, width=224, channels=3, num_transform=2,
                   implementation='conrec', buffer_multiplier=10, do_shuffle=False,
                   color_jitter_strength=1.0, postprocess_fn=None, use_blur=False):
    def map_fn(image, _):
        xs = []
        ys = []
        for _ in range(num_transform):
            if implementation == 'conrec':
                if channels == 1:
                    aug_image = augment_one_channel(image, height=height, width=width, strength=color_jitter_strength)
                    y = tf.identity(aug_image)
                    x = masking.transform_image(aug_image, width=width, height=height, channels=channels)
                else:
                    aug_image = augmentations2d.preprocess_image(image, height=height, width=width,
                                                                 is_training=True,
                                                                 color_distort=True,
                                                                 color_jitter_strength=color_jitter_strength,
                                                                 test_crop=False)
                    if use_blur:
                        if tf.random.uniform(shape=[]) < 0.5:
                            aug_image = augmentations2d.random_blur(aug_image, height, width, p=1.)
                    y = tf.identity(aug_image)
                    x = masking.transform_image(aug_image, width=width, height=height, channels=channels)
            elif implementation == 'simclr':
                if channels == 1:
                    aug_image = augment_one_channel(image, height=height, width=width, strength=color_jitter_strength)
                    y = tf.identity(aug_image)
                    x = aug_image
                else:
                    aug_image = augmentations2d.preprocess_image(image, height=height, width=width,
                                                                 is_training=True,
                                                                 color_distort=True,
                                                                 color_jitter_strength=color_jitter_strength,
                                                                 test_crop=False)
                    if use_blur:
                        if tf.random.uniform(shape=[]) < 0.5:
                            aug_image = augmentations2d.random_blur(aug_image, height, width, p=1.)
                    y = tf.identity(aug_image)
                    x = aug_image
            elif implementation == 'none':
                y = tf.identity(image)
                x = image
            else:
                raise ValueError()

            if postprocess_fn is not None:
                x = postprocess_fn(x)
                y = postprocess_fn(y)
            xs.append(x)
            ys.append(y)
        return tf.concat(xs, -1), tf.concat(ys, -1)

    if do_shuffle:
        dataset = dataset.shuffle(batch_size * buffer_multiplier)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def transform_fn(features1, features2):
        result = []
        all_features = [features1, features2]
        for i in range(len(all_features)):
            features_list = tf.split(all_features[i], num_or_size_splits=2, axis=-1)
            features = tf.concat(features_list, 0)
            result.append(features)
        return tuple(result)

    return dataset.map(transform_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def augment_one_channel(image, height, width, strength=1.0):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = augmentations2d.random_crop_with_resize(image, height, width)
    image = tf.image.random_flip_left_right(image)

    # Color Jitter
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    image = tf.image.random_contrast(image, lower=1 - contrast, upper=1 + contrast)
    image = augmentations2d.random_brightness(image, max_delta=brightness, impl='simclrv2')

    image = tf.reshape(image, [height, width, 1])
    image = tf.clip_by_value(image, 0., 1.)
    return image
