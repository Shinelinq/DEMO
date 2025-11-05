import argparse
from pathlib import Path


def get_args():

    parser = argparse.ArgumentParser(description="MARAN arguments")
    parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
    parser.add_argument('--mode', default='test', type=str, help='train/test')
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
                        required=True,
                        help='Region loss weights aligned with geohash_precisions; single value will broadcast')
    parser.add_argument('--use_warmup',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Enable warm-up (0/1)')
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=10,
                        help='Warm-up epochs (placeholder when use_warmup=0)')
    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    args.model_dir = Path(args.model_dir)
    args.database = Path(args.database)
    args.dataset_train_path = args.database / f'dataset_{args.dataset}_train.pkl'
    args.dataset_test_path = args.database / f'dataset_{args.dataset}_test.pkl'

    if args.dataset == 'Gowalla':
        args.loss_rate = 0.8
        args.dataset_file = './Datasets/checkins-gowalla.txt'
    elif args.dataset == 'Foursquare':
        args.loss_rate = 0.2
        args.dataset_file = './Datasets/checkins-4sq.txt'
    elif args.dataset == 'Yelp':
        args.loss_rate = 0.0
        args.dataset_file = './Datasets/checkins-yelp.txt'
    return args