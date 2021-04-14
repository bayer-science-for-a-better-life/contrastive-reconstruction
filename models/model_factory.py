from models.conrec import conrec_model, EncoderReduction, DecoderType


def add_model_args(parser):
    parser.add_argument('-m', '--model', choices=['unet', 'backbone', 'resnet', 'std-resnet50', 'densenet-custom'],
                        help='Model identifier', default='unet')

    # Unet options
    parser.add_argument('-f', '--filters', type=int, default=64, help="How many base filters should be used")
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('-sc', '--skip-connections', type=int, default=None)
    parser.add_argument('--sc-strength', type=float, default=1.0, help='Strength factor of the skip connections')

    #
    parser.add_argument('-pd', '--projection-dim', type=int, default=128,
                        help='Dimension at the output of the projection head')
    parser.add_argument('-pl', '--projection-layers', type=int, default=3, help='Number of projection layers')
    parser.add_argument('--encoder-reduction', choices=EncoderReduction.values(), default='ga_pooling',
                        help='Determines the method which should be used to reduce the dimensionality output. '
                             'ga_pooling for GlobalAveragePooling and ga_attention for the attention mechanism')
    parser.add_argument('--decoder', default='upsampling', choices=DecoderType.values(),
                        help="which method should be used to enlarge the dimension in the decoder")
    return parser


def construct_model_from_args(args):
    input_shape = (args.height, args.width, args.channels)

    proj_parameters = dict(projection_dim=args.projection_dim, projection_head_layers=args.projection_layers)

    model_parameters = dict(input_shape=input_shape, decoder_type=DecoderType(args.decoder),
                            encoder_reduction=EncoderReduction(args.encoder_reduction), **proj_parameters)

    if args.model == 'unet':
        model = conrec_model(depth=args.depth,
                             basemap=args.filters, skip_connections=args.skip_connections,
                             batch_normalization=True, p_dropout=None, **model_parameters,
                             sc_strength=args.sc_strength)
    else:
        raise ValueError('unknown model')

    return model
