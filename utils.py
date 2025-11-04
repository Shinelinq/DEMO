import logging
import os
import random

import numpy as np
import pandas as pd
import pygeohash as gh
import torch


def init_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger():
    """
    初始化日志系统：
    - 保留原有终端输出；
    - 将 logging、print、tqdm 等同时写入项目根目录下 Logs/run-YYYYmmdd-HHMMSS.log；
    - 使用行缓冲（buffering=1）确保实时落盘；
    - 仅在本函数内完成修改。
    """
    import sys
    from pathlib import Path
    from datetime import datetime
    import atexit

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 防重：若已初始化则直接返回
    if getattr(logger, "_file_logging_initialized", False):
        return logger

    # 日志目录与文件
    root_dir = Path(__file__).resolve().parent
    logs_dir = root_dir / 'Logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_path = logs_dir / log_filename

    # 行缓冲模式打开日志文件
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)

    # Tee 到终端与文件，确保 print/tqdm 也写入文件
    class TeeStream:
        def __init__(self, orig_stream, file_stream):
            self._orig = orig_stream
            self._file = file_stream

        def write(self, data):
            # 同时写入原始终端与日志文件
            try:
                self._orig.write(data)
            except Exception:
                pass
            try:
                self._file.write(data)
            except Exception:
                pass

        def flush(self):
            try:
                self._orig.flush()
            except Exception:
                pass
            try:
                self._file.flush()
            except Exception:
                pass

        def isatty(self):
            try:
                return self._orig.isatty()
            except Exception:
                return False

        @property
        def encoding(self):
            return getattr(self._orig, "encoding", "utf-8")

        def fileno(self):
            fn = getattr(self._orig, "fileno", None)
            return fn() if callable(fn) else None

        def writable(self):
            return True

    # 先替换 sys.stdout/sys.stderr，再创建控制台 handler，使其写入到 Tee（从而被同时记录到文件）
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    # 控制台输出（保留原有行为与格式）
    consol_handler = logging.StreamHandler()  # 默认写入 sys.stderr（已被 Tee 包装）
    consol_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s line:%(lineno)d process:%(process)d] %(levelname)s: %(message)s'
    )
    consol_handler.setFormatter(formatter)

    # 为避免重复添加 handler，仅在不存在时添加
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(consol_handler)

    # 退出时关闭日志文件
    atexit.register(log_file.close)

    # 记录初始化状态
    logger._file_logging_initialized = True
    logger._file_logging_fp = log_file

    return logger


def geohash_encode(checkins, precision=5):
    '''Encode the latitude and longitude use geohash
    '''
    checkins['geohash'] = checkins.apply(
        lambda x: gh.encode(x['latitude'], x['longitude'], precision=precision), axis=1)
    return checkins


def geohash_neighbors(geohash):
    '''Get the neighboring areas around the area, including itself for a total of 9
    '''
    neighbors = []
    lat_range, lon_range = 180, 360
    x, y = gh.decode(geohash)
    num = len(geohash) * 5
    dx = lat_range / (2**(num // 2))
    dy = lon_range / (2**(num - num // 2))
    for i in range(1, -2, -1):
        for j in range(-1, 2):
            neighbors.append(gh.encode(x + i * dx, y + j * dy, num // 5))
    return neighbors


def split_user_train_test(checkins, train_size):
    def split_train_test(df):
        n_train = int(len(df) * train_size)
        checkins_train.append(df.iloc[:n_train])
        checkins_test.append(df.iloc[n_train - 1:])

    checkins_train, checkins_test = [], []
    _ = checkins.groupby('user_id').apply(split_train_test)
    checkins_train = pd.concat(checkins_train).reset_index(drop=True)
    checkins_test = pd.concat(checkins_test).reset_index(drop=True)
    return checkins_train, checkins_test


def calculate_acc(pred, labels):
    '''calculate acc
    
    `result[0, 1, 2, 3]` represent `recall@1`, `recall@5`, `recall@10`, `MAP` respectively
    '''
    # pred shape: (batch_size, max_sequence_length, max_loc_num)
    # labels shape: (batch_size, max_sequence_length)

    # pred shape: (batch_size * max_sequence_length, max_loc_num)
    pred = pred.view(-1, pred.shape[2])
    # labels shape: (1, batch_size * max_sequence_length)
    labels = labels.view(-1).unsqueeze(0)

    result = torch.zeros(5, labels.shape[1], device=pred.device)
    # calculate recall@k (k=1, 5, 10, 20)
    # get topk predict pois
    pred_val, pred_poi = pred.topk(20, dim=1, sorted=True)
    recall = torch.stack([labels == pred_poi[:, i] for i in range(20)])
    result[0] = recall[:1].sum(dim=0)
    result[1] = recall[:5].sum(dim=0)
    result[2] = recall[:10].sum(dim=0)
    result[3] = recall[:20].sum(dim=0)

    # calculate MRR
    # find the score of the label corresponding to the POI
    score = pred.gather(dim=1, index=labels.T)
    result[4] = 1 / (1 + (pred > score).sum(dim=1))

    return result


def cal_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    R = 6371  # Radius of the earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    distance = 2 * np.arcsin(np.sqrt(a)) * R
    return distance
