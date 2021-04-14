import tensorflow as tf
from os.path import join

from sklearn.model_selection import KFold

from models import model_utils
from utils.evaluation import linear_evaluation_with_embeddings, compute_embeddings
from models.conrec import ENCODER_OUTPUT_NAME
from threading import Thread
import os
import numpy as np
from collections import defaultdict


class LearningRateLogger(tf.keras.callbacks.Callback):

    def __init__(self, schedule, num_examples, batch_size):
        super().__init__()
        self.schedule = schedule
        self.num_examples = num_examples
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if callable(self.schedule):
            lr = self.schedule(self.num_examples * epoch // self.batch_size + 1)
        else:
            lr = self.schedule
        logs.update({'lr': lr})
        tf.summary.scalar('lr', data=lr, step=epoch)


class EpochModelSaver(tf.keras.callbacks.Callback):

    def __init__(self, directory, base_name, epochs, save_newest=True):
        super().__init__()
        self.epochs = epochs
        self.directory = directory
        self.base_name = base_name
        self.save_newest = save_newest

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs:
            tf.keras.models.save_model(self.model, join(self.directory, f'{self.base_name}_{epoch}.hdf5'))
        if self.save_newest:
            tf.keras.models.save_model(self.model, join(self.directory, f'{self.base_name}_newest.hdf5'))


class LinearFinetuneCallback(tf.keras.callbacks.Callback):

    def __init__(self, train, test, logdir, batch_size=256, interval=10, num_train=None, num_test=None,
                 output_layer_name=ENCODER_OUTPUT_NAME, linear_type='categorical', async_eval=False):
        super().__init__()
        self.train = train
        self.test = test
        self.interval = interval
        self.num_train = num_train
        self.num_test = num_test
        self.batch_size = batch_size
        self.output_layer_name = output_layer_name
        self.best_roc_auc = 0
        self.linear_type = linear_type
        self.best_score = 0
        self.file_writer = tf.summary.create_file_writer(join(logdir, 'tensorboard', 'metrics'))
        self.model_path = join(logdir, 'best_score.hdf5')
        self.tmp_path = join(logdir, 'best_score_tmp.hdf5')
        self.async_eval = async_eval

    def generate_model(self):
        embedding_model = model_utils.remove_layers(self.model, self.output_layer_name)
        return embedding_model

    def logistic_regression(self, x_train, y_train, x_test, y_test, epoch):
        if self.linear_type == 'diabetic':
            kf = KFold(n_splits=5, shuffle=True)
            x = x_train
            y = y_train
            means = defaultdict(list)
            for train_index, test_index in kf.split(list(range(len(x)))):
                fold_train_x = x[train_index]
                fold_test_x = x[test_index]
                fold_train_y = y[train_index]
                fold_test_y = y[test_index]
                result = linear_evaluation_with_embeddings(x_train=fold_train_x,
                                                           y_train=fold_train_y,
                                                           x_test=fold_test_x,
                                                           y_test=fold_test_y,
                                                           type='diabetic')
                for metric, value in result.items():
                    means[metric].append(value)
            result = dict()
            for key, values in means.items():
                result[key] = np.mean(values)
        else:
            result = linear_evaluation_with_embeddings(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                       type=self.linear_type)
        if self.linear_type == 'categorical':
            score = result['accuracy']
        elif self.linear_type == 'diabetic':
            score = result['kappa_kaggle']
        elif self.linear_type == 'multilabel':
            score = result['roc_auc']
        else:
            raise ValueError('Unknown linear type')

        if score >= self.best_score:
            print('Improved Score, saving model to ', self.model_path)
            if self.async_eval:
                os.rename(self.tmp_path, self.model_path)
            else:
                tf.keras.models.save_model(self.model, self.model_path)
            self.best_score = score
        else:
            if self.async_eval:
                os.remove(self.tmp_path)

        for name, val in result.items():
            print(name, val)
            with self.file_writer.as_default():
                tf.summary.scalar(name, data=val, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        embedding_model = self.generate_model()
        train = self.train.take(self.num_train // self.batch_size + 1) if self.num_train is not None else self.train
        test = self.test.take(self.num_test // self.batch_size + 1) if self.num_test is not None else self.test
        if epoch % self.interval == 0:
            print('Updating Scores')
            x_train, y_train = compute_embeddings(embedding_model, train)
            x_test, y_test = compute_embeddings(embedding_model, test)
            if self.async_eval:
                tf.keras.models.save_model(self.model, self.tmp_path)
                thread = Thread(target=self.logistic_regression, args=(x_train, y_train, x_test, y_test, epoch))
                thread.start()
            else:
                self.logistic_regression(x_train, y_train, x_test, y_test, epoch)


class ImageLoggerCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset, logdir, log_interval=8, num_images=5, postprocess_fn=None, log_predictions=True):
        super(ImageLoggerCallback, self).__init__()
        self.dataset = dataset
        self.num_images = num_images
        self.file_writer = tf.summary.create_file_writer(join(logdir, 'tensorboard', 'images'))
        self.log_interval = log_interval
        self.postprocess_fn = postprocess_fn
        self.log_predictions = log_predictions

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            x, y = next(iter(self.dataset.take(1)))
            if self.log_predictions:
                predictions = self.model.predict(x)[0]
                result = tf.stack([x, predictions, y], axis=1)
            else:
                result = tf.stack([x, y], axis=1)
            with self.file_writer.as_default():
                for i, img in enumerate(result[:self.num_images]):
                    if self.postprocess_fn is not None:
                        img = self.postprocess_fn(img)
                    tf.summary.image(f"{i}", img, step=epoch, max_outputs=self.num_images)
