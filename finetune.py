import tensorflow as tf
import models.model_utils as models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation
import argparse
from data.datasets import DATASETS, setup_dataset
from os.path import join
from utils.metrics import score_cat_acc_kaggle, score_kappa_kaggle
from models import model_factory
from utils import train_utils
import os
import json
from data.utils import get_preprocess_fn
from datetime import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    model_factory.add_model_args(parser)
    train_utils.add_image_dimension_args(parser)
    train_utils.add_train_args(parser)

    parser.add_argument('-d', '--dataset', default='covid', choices=DATASETS)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--test-split', default='test')

    parser.add_argument('-p', '--pretrained')
    parser.add_argument('--freeze-until', default=None)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--validation-freq', type=int, default=1)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--classes', type=int, default=1)

    parser.add_argument('--multilabel', action='store_true')
    parser.add_argument('--prune-layer', default='encoder_output')
    parser.add_argument('--prune', choices=['true', 'false'], default='true')
    parser.add_argument('--save-metric', default='val_sparse_categorical_accuracy')
    parser.add_argument('--preprocess', choices=['none', 'simclr', 'simclr-no-aug', 'diabetic', 'greyscale', '1-channel-aug'], default='none')
    parser.add_argument('--transform', default=None)
    parser.add_argument('--gpu', default='0', help="The id of the gpu device")
    parser.add_argument('--flatten', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    input_shape = (args.width, args.height, args.channels)
    skip_connections = [False] * args.depth

    logdir = join(args.logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir)

    params = vars(args)
    with open(join(logdir, 'params.json'), 'w+') as f:
        json.dump(params, f, indent=4)

    if args.pretrained:
        model = tf.keras.models.load_model(args.pretrained, compile=False)
    else:
        model = model_factory.construct_model_from_args(args)

    if args.prune == 'true':
        embedding_model = models.remove_layers(model, args.prune_layer)
    else:
        embedding_model = model

    if args.freeze_until is not None:
        for layer in embedding_model.layers:
            layer.trainable = False
            if layer.name == args.freeze_until:
                break

    embedding_model.summary()

    file_writer = tf.summary.create_file_writer(join(logdir, 'tensor', 'metrics'))
    file_writer.set_as_default()

    train, test, num_examples = setup_dataset(dataset=args.dataset, data_path=args.data_path,
                                              height=args.height, width=args.width,
                                              batch_size=args.batch_size,
                                              test_split=args.test_split, train_split=args.train_split,
                                              threads=args.threads,
                                              preprocess_implementation=args.preprocess)

    learning_rate_schedule = train_utils.learning_rate_schedule(args, num_examples=num_examples)
    optimizer = train_utils.get_optimizer(args, learning_rate_schedule)

    with open(join(logdir, 'model.txt'), 'w+') as fh:
        embedding_model.summary(print_fn=lambda x: fh.write(x + '\n'))

    tensorboard_dir = join(logdir, 'tensor')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    callbacks = [tensorboard_callback]

    if args.save_model:
        checkpoint = ModelCheckpoint(
            join(logdir, f'model_{args.save_metric}.hdf5'),
            monitor=args.save_metric,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max")
        callbacks.append(checkpoint)

    classifier = models.add_linear_head(embedding_model, args.classes, flatten=args.flatten, name='supervised_head')

    if args.classes == 1 or args.multilabel:
        classifier = tf.keras.Sequential([classifier, Activation('sigmoid')])


        def cat_acc(y1, y2):
            return tf.numpy_function(score_cat_acc_kaggle, [y1, y2], tf.float32)


        def score_kappa(y1, y2):
            return tf.numpy_function(score_kappa_kaggle, [y1, y2], tf.float32)


        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=args.classes > 1),
                   cat_acc, score_kappa]
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classifier.summary()
    classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if args.transform is not None and args.transform != 'none':
        preprocess = get_preprocess_fn(args.transform)
        train = train.map(preprocess)
        test = test.map(preprocess)

    classifier.fit(train, epochs=args.epochs, validation_data=test, verbose=1, validation_freq=args.validation_freq,
                   callbacks=callbacks)
