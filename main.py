import utils
import sys
from args import get_args
from dataloader import Poidataloader
from model import *
from trainer import Trainer

# load config file
args = get_args()

# init logger
logger = utils.init_logger()

# init seed
utils.init_seed(3407)


def main():
    # Phase-0 参数校验与人类可读打印（入口最小改动）
    # 规范化：geohash_precisions 去重并升序
    args.geohash_precisions = sorted(set(args.geohash_precisions))

    # Phase-0 fail-fast：仅允许 {4,5}；如出现 6 或其他粒度，立即退出并打印示例命令
    invalid = [p for p in args.geohash_precisions if p not in (4, 5)]
    if len(invalid) > 0:
        logger.error('检测到不支持的地理粒度: %s', invalid)
        logger.error('G6 属于 Phase-1 的功能（局部 Softmax 与掩码），Phase-0 不支持。')
        logger.error('Phase-0 示例：--geohash_precisions 4 5 --lambda_regions 0.10 0.20')
        logger.error('Phase-1 示例：--geohash_precisions 4 5 6 --lambda_regions 0.10 0.25 0.05')
        raise SystemExit(1)

    # 对齐：若 lambda_regions 长度为 1 且粒度数 > 1，则广播；否则长度必须相等
    if len(args.lambda_regions) == 1 and len(args.geohash_precisions) > 1:
        args.lambda_regions = [args.lambda_regions[0]] * len(args.geohash_precisions)
    elif len(args.lambda_regions) != len(args.geohash_precisions):
        logger.error(
            'lambda_regions 数量(%d)与 geohash_precisions 数量(%d)不一致，且无法广播。',
            len(args.lambda_regions), len(args.geohash_precisions))
        raise SystemExit(1)

    # 合法：所有 λ 非负；sum(lambda_regions) > 1.0 打印警告（不中断）
    if any(l < 0 for l in args.lambda_regions):
        logger.error('检测到负的区域损失权重 λ：%s', args.lambda_regions)
        raise SystemExit(1)
    lambda_sum = float(sum(args.lambda_regions))
    if lambda_sum > 1.0:
        logger.warning('区域权重之和大于 1.0：%.2f（不中断，仅提示）', lambda_sum)

    # 人类可读打印（启动摘要）
    levels_str = ','.join([f'G{p}' for p in args.geohash_precisions])
    lambda_map_str = ', '.join([f'λ(G{p})={args.lambda_regions[i]:.2f}'
                                for i, p in enumerate(args.geohash_precisions)])
    warmup_state = 'ON' if args.use_warmup == 1 else 'OFF'
    logger.info('Geohash levels: %s', levels_str)
    logger.info('Lambda by level: %s (sum=%.2f)', lambda_map_str, lambda_sum)
    logger.info('Warm-up: %s (epochs=%d)', warmup_state, int(args.warmup_epochs))
    # 仅打印已有关键开关（不改逻辑）
    if hasattr(args, 'mask'):
        logger.info('Mask: %s', str(args.mask))

    logger.info('start loading checkin data...')
    poi_loader = Poidataloader(args)
    # Gowalla / Foursquare
    if not args.dataset_test_path.exists():
        checkins = poi_loader.load(args.dataset, args.dataset_file)
    logger.info('loading checkin data done!')

    logger.info('start creating dataset...')
    poi_loader.create_dataset(args.mode, args.dataset)
    config = poi_loader.config
    logger.info('creating dataset done!')

    logger.info('start loading model...')
    model = PoiModel(config)
    logger.info('loading model done!')

    trainer = Trainer(config=config, logger=logger, gpu=config.gpu)
    if config.mode == 'train':
        trainer.train(model=model, dataloader=poi_loader)
    else:
        trainer.test(model, dataloader=poi_loader, model_path=config.model_path)


if __name__ == '__main__':
    main()

# train
# python -u main.py --mode=train --dataset=Gowalla --gpu=1
# nohup python -u main.py --mode=train --dataset=Gowalla --gpu=1 > ./go.log 2>&1 &
# nohup python -u main.py --mode=train --dataset=Yelp --gpu=2 --mask=False > ./yelp.log 2>&1 &

# test
# python main.py --mode=test --dataset=Gowalla --model_path='./Model/model_gowalla.pkl' --gpu=1
