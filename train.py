from data import transform
from utils.contrastive import NTXentLoss, ContrastiveAccuracy, ContrastiveEntropy
from datetime import datetime
from os.path import join
import os
import argparse
from utils.callbacks import LearningRateLogger, LinearFinetuneCallback
from models.conrec import ENCODER_OUTPUT_NAME, \
    RECONSTRUCTION_OUTPUT, CONTRASTIVE_OUTPUT
from utils import train_utils
import tensorflow as tf
from utils.callbacks import ImageLoggerCallback
from data.datasets import load_dataset, DATASETS
import models.model_utils as model_utils
from models import model_factory

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # add options for lr, optimizer, batch size etc
    train_utils.add_train_args(parser)

    # add model constructing options
    model_factory.add_model_args(parser)

    # height, width, channels
    train_utils.add_image_dimension_args(parser)

    # Dataset options
    parser.add_argument('-d', '--dataset', default='oxford_flowers102', choices=DATASETS,
                        help="Dataset identifier that either is the name of a tf dataset or dataset type e.g. numpy")
    parser.add_argument('--data-path', default=None,
                        help="Optional supporting dataset path that is required for some datasets e.g. aptos2019")
    parser.add_argument('-ed', '--eval-dataset', default=None, choices=DATASETS)
    parser.add_argument('--eval-data-path', default=None)
    parser.add_argument('--test-split', default='test')
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--eval-test-split', default='test')
    parser.add_argument('--eval-train-split', default='train')
    parser.add_argument('--eval-center-crop', action='store_true', help="Makes a center crop for linear evaluation")
    parser.add_argument('--no-test-data', action='store_true')

    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--linear-interval', type=int, default=20,
                        help='How often the linear evaluation hook is executed')
    parser.add_argument('--validation-freq', type=int, default=50,
                        help='Specifies how often the validation hook is executed')

    parser.add_argument('--log-images', action='store_true',
                        help='Enables Logging of images to tensorboard during training')
    parser.add_argument('--image-log-interval', type=int, default=50,
                        help='The interval in which images should be logged in tensorboard')

    parser.add_argument('-p', '--pretrained',
                        help='Path to a pretrained model that should be used as initialization')

    parser.add_argument("--lambda-rec", type=float, default=100.0, help='Loss factor of the reconstruction loss')
    parser.add_argument("--lambda-con", type=float, default=1.0, help='Loss factor of the contrastive loss')
    parser.add_argument('--reconst-loss', choices=['mse', 'ssim', 'huber'], default='mse',
                        help='The reconstruction loss type')

    # Augmentation Policy
    parser.add_argument('--aug-impl', default='conrec')
    parser.add_argument('--use-blur', action='store_true', help='Enables blurring of images after augmentation')
    parser.add_argument('--color-jitter-strength', type=float, default=1.0,
                        help='Specifies how much color jitter should be used while augmenting the images')

    parser.add_argument('--no-save', action='store_true', help='turns off saving of the models')
    parser.add_argument('--threads', type=int, default=8,
                        help='How many threads should be used when loading the dataset')
    parser.add_argument('--linear-type', default='categorical')
    parser.add_argument('--save-epochs', type=int, nargs='*', default=[], help='Specifies ')
    parser.add_argument('--no-cache', action='store_true', help='Prevents the dataset from being cached')
    parser.add_argument('--simclr', action='store_true')

    parser.add_argument('--shuffle-buffer-multiplier', type=int, default=10,
                        help='The shuffle buffer has size batch_size x shuffle-buffer-multiplier')

    parser.add_argument('--async-eval', action='store_true', help='If set, the linear embedding evaluation '
                                                                  'will be done in a separate thread')

    parser.add_argument('--docker-download', action='store_true',
                        help='If set, if code is run inside docker and datasets download path needs to be adopted')

    parser.add_argument('--gpu', default='0', help="The id of the gpu device")


    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.pretrained:
        model = tf.keras.models.load_model(args.pretrained, compile=False)
    else:
        # Choose encoder model
        model = model_factory.construct_model_from_args(args)

        if args.simclr:
            model = model_utils.remove_layers(model, CONTRASTIVE_OUTPUT)


    # Conrec Dataset (unsupervised)
    train, test, num_examples = load_dataset(args.dataset, args.data_path,
                                             test_split=args.test_split if not args.no_test_data else None,
                                             train_split=args.train_split,
                                             cache=not args.no_cache,
                                             threads=args.threads, docker_down=args.docker_download)

    conrec_train, conrec_test = (transform.conrec_dataset(ds, args.batch_size, height=args.height,
                                                          width=args.width, implementation=args.aug_impl,
                                                          channels=args.channels,
                                                          color_jitter_strength=args.color_jitter_strength,
                                                          do_shuffle=s,
                                                          buffer_multiplier=args.shuffle_buffer_multiplier,
                                                          use_blur=args.use_blur) if ds is not None else None
                                 for ds, s in [(train, True), (test, False)])

    # loss, learning rate, optimizer
    contrastive_loss = NTXentLoss(temperature=args.temperature)
    learning_rate_schedule = train_utils.learning_rate_schedule(args, num_examples=num_examples)
    optimizer = train_utils.get_optimizer(args, learning_rate_schedule)

    # make dir for logging results
    logdir = join(args.logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir)
    train_utils.save_train_params(args, logdir)

    # Callbacks
    callbacks = train_utils.setup_logging_callbacks(logdir, save_model=not args.no_save, save_epochs=args.save_epochs)
    callbacks.append(LearningRateLogger(learning_rate_schedule, num_examples, batch_size=args.batch_size))
    model.summary()

    if args.linear_type != 'none':
        if args.eval_dataset is None:
            assert test is not None, "Linear Evaluation with the option --no-test-data does not work"
            train_eval, test_eval = train, test
        else:
            train_eval, test_eval, _ = load_dataset(args.eval_dataset, args.eval_data_path,
                                                    test_split=args.eval_test_split,
                                                    train_split=args.eval_train_split,
                                                    cache=True,
                                                    threads=args.threads)

        # Resize with center crop if desired
        if args.eval_center_crop:
            train_eval, test_eval = (transform.preprocess_image(ds, img_width=args.width,
                                                                img_height=args.height,
                                                                is_training=False,
                                                                test_crop=True) for ds in [train_eval, test_eval])

        train_eval, test_eval = train_eval.batch(args.batch_size), test_eval.batch(args.batch_size)

        lft = LinearFinetuneCallback(train=train_eval, test=test_eval, batch_size=args.batch_size,
                                     interval=args.linear_interval, output_layer_name=ENCODER_OUTPUT_NAME,
                                     async_eval=args.async_eval,
                                     linear_type=args.linear_type, logdir=logdir)
        callbacks.append(lft)

    if args.log_images:
        img_logger = ImageLoggerCallback(conrec_train, logdir=logdir, log_interval=args.image_log_interval,
                                         postprocess_fn=None, log_predictions=not args.simclr)
        img_logger.model = model
        callbacks.append(img_logger)

    if args.reconst_loss == 'ssim':
        def reconstruction_loss_fn(y_true, y_pred):
            return - (tf.image.ssim(y_true, y_pred, max_val=1.0) - 1)
    elif args.reconst_loss == 'huber':
        reconstruction_loss_fn = tf.keras.losses.Huber()
    elif args.reconst_loss == 'mse':
        reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError('Unknown loss')

    # only include reconstruction loss when not in simclr mode
    if not args.simclr:
        loss = {RECONSTRUCTION_OUTPUT: reconstruction_loss_fn, CONTRASTIVE_OUTPUT: contrastive_loss}
        loss_weights = {RECONSTRUCTION_OUTPUT: args.lambda_rec, CONTRASTIVE_OUTPUT: args.lambda_con}
    else:
        loss = {CONTRASTIVE_OUTPUT: contrastive_loss}
        loss_weights = {CONTRASTIVE_OUTPUT: args.lambda_con}

    # Lars optimizer handles weight decay already
    if args.optimizer not in ['lars']:
        train_utils.add_weight_decay(model, weight_decay=args.weight_decay)


    def compile_model():
        metrics = {CONTRASTIVE_OUTPUT: [ContrastiveAccuracy(temperature=args.temperature),
                                        ContrastiveEntropy(temperature=args.temperature)]}
        if not args.simclr:
            metrics[RECONSTRUCTION_OUTPUT] = ['mse', 'mae'],

        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)


    # If we vary the steps per epoch, repeat the dataset infinitely
    conrec_train = conrec_train.prefetch(tf.data.experimental.AUTOTUNE)
    if args.steps_per_epoch is not None:
        conrec_train = conrec_train.repeat()

    fit_args = dict(x=conrec_train, verbose=1, validation_data=conrec_test, callbacks=callbacks,
                    steps_per_epoch=args.steps_per_epoch, validation_freq=args.validation_freq)

    # Compile and fit
    compile_model()
    model.fit(epochs=args.epochs, **fit_args)
