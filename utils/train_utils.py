import tensorflow as tf
from os.path import join
import json
from tensorflow.keras.callbacks import (
    CSVLogger
)
from utils.callbacks import EpochModelSaver
from tensorflow.python.framework import ops
from tensorflow.keras.experimental import CosineDecay
import math
from utils.lars_optimizer import LARSOptimizer
from tensorflow_addons.optimizers import ExponentialCyclicalLearningRate
from tensorboard.plugins.hparams import api as hp
import copy
import re

EXCLUDED_WEIGHT_DECAY_PARAMS = ['batch_normalization', 'bias', 'bn', 'supervised_head']


class WarmupCosineDecay(CosineDecay):

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 warmup_steps=0,
                 alpha=0.0,
                 name=None):
        assert warmup_steps <= decay_steps
        super(WarmupCosineDecay, self).__init__(initial_learning_rate=initial_learning_rate,
                                                decay_steps=decay_steps - warmup_steps, alpha=alpha, name=name)
        self.warmup_steps = warmup_steps
        self.current_lr = initial_learning_rate

    def __call__(self, step):
        warmup_steps = ops.convert_to_tensor_v2(self.warmup_steps, dtype=tf.float32)
        initial_learning_rate = ops.convert_to_tensor_v2(self.initial_learning_rate, name="initial_learning_rate")
        lr = tf.cond(step <= warmup_steps, lambda: (step / warmup_steps) * initial_learning_rate,
                     lambda: super(WarmupCosineDecay, self).__call__(step))
        self.current_lr = lr
        return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
            "name": self.name
        }


def add_image_dimension_args(parser):
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--channels', type=int, default=3)


def add_train_args(parser):
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.075)
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-bs', '--batch-size', type=int, default=16)
    parser.add_argument('-t', '--temperature', type=float, default=0.5)
    parser.add_argument('-we', '--warmup-epochs', type=int, default=0)
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'adam', 'lars'], default='lars')
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', choices=['static', 'warmup-cosine-decay', 'cyclic'], default='static')
    parser.add_argument('--lr-scaling', choices=['none', 'linear', 'sqrt'], default='none')
    parser.add_argument('--logdir', default='logs')


def learning_rate_schedule(args, num_examples):
    if args.lr_scaling == 'none':
        lr = args.learning_rate
    elif args.lr_scaling == 'linear':
        lr = args.learning_rate * args.batch_size / 256.
    elif args.lr_scaling == 'sqrt':
        lr = args.learning_rate * math.sqrt(args.batch_size)
    else:
        raise ValueError('Unknown lr scaling')
    if args.lr_schedule == 'lr-finder':
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-10, decay_rate=1.3, decay_steps=200)
    elif args.lr_schedule == 'static':
        schedule = lr
    elif args.lr_schedule == 'cyclic':
        step_size = num_examples * 10 // args.batch_size + 1
        schedule = ExponentialCyclicalLearningRate(initial_learning_rate=lr * 0.01, maximal_learning_rate=lr * 1,
                                                   step_size=step_size, gamma=1.0)
    elif args.lr_schedule == 'warmup-cosine-decay':
        total_steps = num_examples * args.epochs // args.batch_size + 1
        warmup_steps = int(round(args.warmup_epochs * num_examples // args.batch_size))
        schedule = WarmupCosineDecay(initial_learning_rate=lr, decay_steps=total_steps,
                                     warmup_steps=warmup_steps)
    else:
        raise ValueError('Unknown lr schedule')
    return schedule


def get_optimizer(args, learning_rate_schedule):
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    elif args.optimizer == 'lars':
        optimizer = LARSOptimizer(learning_rate=learning_rate_schedule,
                                  momentum=0.9,
                                  weight_decay=args.weight_decay,
                                  exclude_from_weight_decay=EXCLUDED_WEIGHT_DECAY_PARAMS)
    else:
        raise ValueError("Unknow optimizer")
    return optimizer


def save_train_params(args, logdir, file_name='params.json', tensorboard_dir='tensorboard'):
    params = vars(args)
    with open(join(logdir, file_name), 'w+') as f:
        json.dump(params, f, indent=4)
    params = copy.deepcopy(params)
    del params['save_epochs']
    params = {key: value if value is not None else 'None' for key, value in params.items()}
    with tf.summary.create_file_writer(join(logdir, tensorboard_dir, 'params')).as_default():
        hp.hparams(params)


def setup_logging_callbacks(logdir, model_prefix='model', save_model=True, save_epochs=[]):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=join(logdir, 'tensorboard'))
    csv_logger = CSVLogger(join(logdir, 'log.csv'), append=True)
    callbacks = [csv_logger, tensorboard_callback]
    if save_model:
        epoch_saver = EpochModelSaver(directory=logdir, base_name=model_prefix, epochs=save_epochs)
        callbacks.extend([epoch_saver])
    return callbacks


def allow_gpu_memory_growth():
    # Stop TF from claiming all GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def add_weight_decay(model, weight_decay, excluded_params=EXCLUDED_WEIGHT_DECAY_PARAMS):
    if (weight_decay is None) or (weight_decay == 0.0):
        return

    def should_regularize(param_name):
        if excluded_params is not None:
            for r in excluded_params:
                if re.search(r, param_name) is not None:
                    return False
        return True

    # recursion inside the model
    def add_decay_loss(m, factor):
        if isinstance(m, tf.keras.Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                if should_regularize(param.name):
                    print(param.name)
                    with tf.keras.backend.name_scope('weight_regularizer'):
                        regularizer = lambda: tf.keras.regularizers.l2(factor)(param)
                        m.add_loss(regularizer)

    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay / 2.0)
    return
