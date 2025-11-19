import argparse
from pathlib import Path


def get_args():

    parser = argparse.ArgumentParser(description="MARAN arguments")
    parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--dataset',
                        default='Gowalla',
                        type=str,
                        help='Gowalla/Foursquare/Yelp')
    parser.add_argument('--model_path',
                        default='./Model/model_gowalla.pkl',
                        type=str,
                        help='model path')
    parser.add_argument('--model_dir', default='./Model', type=str, help='Model dir.')
    parser.add_argument('--database',
                        default='./Datasets',
                        type=str,
                        help='Database dir.')

    parser.add_argument('--max_sequence_length',
                        default=20,
                        type=int,
                        help='Max sequence length.')
    parser.add_argument('--long_sequence_length',
                        default=200,
                        type=int,
                        help='Long sequence length.')
    parser.add_argument('--min_neighborhood_num',
                        default=1500,
                        type=int,
                        help='Min neighborhood num.')
    parser.add_argument('--mask', default=True, type=bool, help='Mask.')

    parser.add_argument('--batch_size', default=64, help='Batch size.')
    parser.add_argument('--hidden_size', default=64, type=int, help='Hidden size.')
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=50, type=int, help='Train epochs.')
    parser.add_argument('--alpha', default=0.1, type=float, help='.')
    parser.add_argument('--beta', default=0.1, type=float, help='.')
    # Phase-0 参数化：多粒度与 warm-up
    parser.add_argument('--geohash_precisions',
                        nargs='+',
                        type=int,
                        default=[5],
                        help='Enabled geohash granularities, e.g., 4 5 or 4 5 6')
    parser.add_argument('--lambda_regions',
                        nargs='+',
                        type=float,
                        default=[0.2],
                        help='Region loss weights aligned with geohash_precisions; single value will broadcast')
    parser.add_argument('--use_warmup',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Enable warm-up (0/1)')
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=2,
                        help='Warm-up epochs (placeholder when use_warmup=0)')
    parser.add_argument('--use_fusion_short',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Enable LiteFusion on short-term branch (0/1)')
    parser.add_argument('--use_fusion_long',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Enable LiteFusion on long-term branch (0/1)')
    parser.add_argument('--share_gcn_weights',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Share traj/geohash GCN weights in LocalCenterEncoder (0/1)')
    parser.add_argument('--fusion_init_bias',
                        type=str,
                        default='g5-dominant',
                        help='LiteFusion weight init bias strategy')
    parser.add_argument('--fusion_lr_scale',
                        type=float,
                        default=0.3,
                        help='LR scale for Fusion parameter group during warm-up')
    parser.add_argument('--fusion_dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate inside LiteFusion')
    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    args.model_dir = Path(args.model_dir)
    args.database = Path(args.database)
    args.dataset_train_path = args.database / f'dataset_{args.dataset}_train.pkl'
    args.dataset_test_path = args.database / f'dataset_{args.dataset}_test.pkl'

    if args.dataset == 'Gowalla':
        args.dataset_file = './Datasets/checkins-gowalla.txt'
    elif args.dataset == 'Foursquare':
        args.dataset_file = './Datasets/checkins-4sq.txt'
    elif args.dataset == 'Yelp':
        args.dataset_file = './Datasets/checkins-yelp.txt'
    try:
        assert 5 in args.geohash_precisions
    except AssertionError:
        raise SystemExit('Assertion failed: geohash_precisions must include 5 (G5 as NNS anchor).')
    try:
        assert len(args.lambda_regions) == len(args.geohash_precisions)
    except AssertionError:
        raise SystemExit('Assertion failed: lambda_regions length must equal geohash_precisions length.')
    try:
        levels_str = ','.join([f'G{p}' for p in args.geohash_precisions])
        lambda_map_str = ', '.join([f'λ(G{p})={args.lambda_regions[i]:.2f}' for i, p in enumerate(args.geohash_precisions)])
        print(f'[Args] geohash_precisions: {levels_str}')
        print(f'[Args] lambda_regions: {lambda_map_str}')
        pairs = [(int(p), float(args.lambda_regions[i])) for i, p in enumerate(args.geohash_precisions)]
        print(f'[Args] precision-lambda pairs: {pairs}')
        print(f'[Args] use_fusion_short: {int(args.use_fusion_short)}')
        print(f'[Args] use_fusion_long: {int(args.use_fusion_long)}')
        print(f'[Args] share_gcn_weights: {int(args.share_gcn_weights)}')
        print(f'[Args] fusion_init_bias: {str(args.fusion_init_bias)}')
        print(f'[Args] fusion_lr_scale: {float(args.fusion_lr_scale)}')
        print(f'[Args] fusion_dropout: {float(args.fusion_dropout)}')
        print(f'[Args] use_warmup: {int(args.use_warmup)}')
        print(f'[Args] warmup_epochs: {int(args.warmup_epochs)}')
    except Exception:
        pass
    return args