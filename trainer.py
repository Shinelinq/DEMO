from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from utils import *


class Trainer:
    '''
    
    Args:
    -------
    config: Config
        configuration file
    logger: logging = None
        logging object
    gpu: int = -1
        Specify the GPU device. if `gpu=-1` then use CPU.

    Example:
    -------
    >>> model = TestModel()
    >>> dataloader = TestDataloader()
    >>> trainer = Trainer(config=config, gpu=0)
    >>> trainer.train(model=model, dataloader=dataloader)
    '''

    def __init__(self, config, logger=None, gpu=-1):
        self.config = config
        self.logger = logger if logger is not None else init_logger()

        if gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(self.config.model_dir)
        if not self.model_dir.exists():
            self.model_dir.mkdir()

    def train(self, model, dataloader):
        assert 5 in self.config.geohash_precisions
        assert len(self.config.lambda_regions) == len(self.config.geohash_precisions)
        train_dl = dataloader.train_dataloader()
        val_dl = dataloader.val_dataloader()
        model = model.to(self.device)

        with torch.no_grad():
            try:
                dl = next(iter(train_dl))
                if isinstance(dl, dict):
                    user = dl["user"].to(self.device)
                    traj = dl["traj"].to(self.device)
                    geo = dl["geo"].to(self.device)
                    center_traj = dl["center_traj"].to(self.device)
                    long_traj = dl["long_traj"].to(self.device)
                    dt = dl["dt"].to(self.device)
                    traj_graph = dl["traj_graph"].to(self.device)
                    geo_graph = dl["geo_graph"].to(self.device)
                else:
                    user, traj, geo, center_traj, long_traj, dt, label_traj, \
                        label_geo, negihbors_mask, traj_graph, geo_graph = dl
                    user = user.to(self.device)
                    traj = traj.to(self.device)
                    geo = geo.to(self.device)
                    center_traj = center_traj.to(self.device)
                    long_traj = long_traj.to(self.device)
                    dt = dt.to(self.device)
                    traj_graph = traj_graph.to(self.device)
                    geo_graph = geo_graph.to(self.device)
                geo_seqs = dl.get("geo_seqs", None) if isinstance(dl, dict) else None
                if isinstance(dl, dict) and "geo_graphs" in dl:
                    setattr(geo_graph, "graphs_p", dl["geo_graphs"])
                _ = model(user, traj, geo, center_traj, long_traj, dt, traj_graph, geo_graph, geo_seqs=geo_seqs)
            except Exception:
                pass

        base_lr = float(self.config.learning_rate)
        fusion_params = []
        lcl = getattr(model, 'LocalCenterEncoder', None)
        ste = getattr(model, 'ShortTermEncoder', None)
        if lcl is not None:
            lf = getattr(lcl, 'lite_fusion_long', None)
            if lf is not None:
                fusion_params += list(lf.parameters())
        if ste is not None:
            lf = getattr(ste, 'lite_fusion_short', None)
            if lf is not None:
                fusion_params += list(lf.parameters())
        fusion_ids = {id(p) for p in fusion_params}
        backbone_params = [p for p in model.parameters() if id(p) not in fusion_ids]
        if len(fusion_params) == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
            self._fusion_group_index = None
        else:
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': base_lr},
                {'params': fusion_params, 'lr': base_lr}
            ])
            self._fusion_group_index = 1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        self._warmup_log_start = False
        self._warmup_log_end = False

        self.logger.info('start training...')
        for epoch in range(self.config.epochs):
            cond = (epoch < int(self.config.warmup_epochs)) and \
                   (int(getattr(self.config, 'use_fusion_long', 0)) == 1 or int(getattr(self.config, 'use_fusion_short', 0)) == 1)
            if self._fusion_group_index is not None:
                if cond:
                    new_lr = base_lr * float(getattr(self.config, 'fusion_lr_scale', 0.3))
                    optimizer.param_groups[self._fusion_group_index]['lr'] = new_lr
                    if not self._warmup_log_start:
                        self.logger.info('[fusion-warmup] lr <- base_lr * fusion_lr_scale')
                        self._warmup_log_start = True
                else:
                    optimizer.param_groups[self._fusion_group_index]['lr'] = base_lr
                    if self._warmup_log_start and not self._warmup_log_end:
                        self.logger.info('[fusion-warmup] lr restored to base_lr')
                        self._warmup_log_end = True

            self._train_epoch(epoch, model, train_dl, optimizer, scheduler, criterion)
            if (epoch + 1) % 1 == 0:
                self._val_epoch(model, val_dl, criterion)
            model_path = self.model_dir / f"model_{epoch+1}.pkl"
            torch.save(model.state_dict(), model_path)
        self.logger.info('training done!')

    def _train_epoch(self, epoch, model, train_dl, optimizer, scheduler, criterion):
        model.train()
        train_loss = []
        pbar = tqdm(train_dl, total=len(train_dl))
        for idx, dl in enumerate(pbar):
            # 兼容 batch 为 dict 或 tuple，两种路径均支持
            if isinstance(dl, dict):
                user = dl["user"].to(self.device)
                traj = dl["traj"].to(self.device)
                geo = dl["geo"].to(self.device)
                center_traj = dl["center_traj"].to(self.device)
                long_traj = dl["long_traj"].to(self.device)
                dt = dl["dt"].to(self.device)
                label_traj = dl["label_traj"].to(self.device)
                label_geo_5 = dl["label_geo"].to(self.device)
                label_geo_4 = dl.get("label_geo_4", None)
                if label_geo_4 is not None:
                    label_geo_4 = label_geo_4.to(self.device)
                negihbors_mask = dl["negihbors_mask"].to(self.device)
                traj_graph = dl["traj_graph"].to(self.device)
                geo_graph = dl["geo_graph"].to(self.device)
            else:
                user, traj, geo, center_traj, long_traj, dt, label_traj, \
                    label_geo, negihbors_mask, traj_graph, geo_graph = dl
                user = user.to(self.device)
                traj = traj.to(self.device)
                geo = geo.to(self.device)
                center_traj = center_traj.to(self.device)
                long_traj = long_traj.to(self.device)
                dt = dt.to(self.device)
                label_traj = label_traj.to(self.device)
                label_geo_5 = label_geo.to(self.device)
                label_geo_4 = None
                negihbors_mask = negihbors_mask.to(self.device)
                traj_graph = traj_graph.to(self.device)
                geo_graph = geo_graph.to(self.device)

            optimizer.zero_grad()
            geo_seqs = dl.get("geo_seqs", None) if isinstance(dl, dict) else None
            if isinstance(dl, dict) and "geo_graphs" in dl:
                setattr(geo_graph, "graphs_p", dl["geo_graphs"])
            outputs = model(user, traj, geo, center_traj, long_traj, dt, traj_graph,
                            geo_graph, geo_seqs=geo_seqs)
            if isinstance(outputs, tuple):
                pred_traj = outputs[0]
                pred_geo_5 = outputs[1]
                pred_regions = outputs[2] if len(outputs) >= 3 and isinstance(outputs[2], dict) else getattr(model, "pred_regions", None)
            else:
                pred_traj = outputs
                pred_geo_5 = None
                pred_regions = getattr(model, "pred_regions", None)

            if hasattr(self.config, 'mask') and self.config.mask:
                negihbors_mask = negihbors_mask.unsqueeze(1).repeat(
                    1, self.config.max_sequence_length, 1)
                pred_traj.masked_fill_(negihbors_mask, -1000)

            # 计算损失（全局 softmax + CE），保持 ignore_index 与现有 criterion 一致
            loss_poi = criterion(pred_traj.permute(0, 2, 1), label_traj)
            total_loss = loss_poi
            loss_geo_g5 = None
            loss_geo_g4 = None
            if isinstance(pred_regions, dict):
                labels_map = {5: label_geo_5}
                if label_geo_4 is not None:
                    labels_map[4] = label_geo_4
                for i, p in enumerate(self.config.geohash_precisions):
                    lam = float(self.config.lambda_regions[i])
                    if lam <= 0:
                        continue
                    logits_p = pred_regions.get(f"G{int(p)}", None)
                    label_p = labels_map.get(int(p), None)
                    if logits_p is None or label_p is None:
                        continue
                    lp = criterion(logits_p.permute(0, 2, 1), label_p)
                    if int(p) == 5:
                        loss_geo_g5 = lp
                    if int(p) == 4:
                        loss_geo_g4 = lp
                    total_loss = total_loss + lam * lp

            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

            # update pbar（记录各项损失）
            pbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            postfix = {
                'loss': float(np.mean(train_loss)),
                'poi': float(loss_poi.item()),
                'g5': float(loss_geo_g5.item()) if loss_geo_g5 is not None else None,
                'g4': float(loss_geo_g4.item()) if loss_geo_g4 is not None else None,
                'lr': scheduler.get_last_lr()[0]
            }
            pbar.set_postfix(**{k: v for k, v in postfix.items() if v is not None})

        scheduler.step()

    @torch.no_grad()
    def _val_epoch(self, model, val_dl, criterion):
        model.eval()
        val_loss, val_acc = [], []
        vbar = tqdm(val_dl, desc='valid', total=len(val_dl))
        for idx, dl in enumerate(vbar):
            # 兼容 batch 为 dict 或 tuple
            if isinstance(dl, dict):
                user = dl["user"].to(self.device)
                traj = dl["traj"].to(self.device)
                geo = dl["geo"].to(self.device)
                center_traj = dl["center_traj"].to(self.device)
                long_traj = dl["long_traj"].to(self.device)
                dt = dl["dt"].to(self.device)
                label_traj = dl["label_traj"].to(self.device)
                label_geo_5 = dl["label_geo"].to(self.device)
                label_geo_4 = dl.get("label_geo_4", None)
                if label_geo_4 is not None:
                    label_geo_4 = label_geo_4.to(self.device)
                negihbors_mask = dl["negihbors_mask"].to(self.device)
                traj_graph = dl["traj_graph"].to(self.device)
                geo_graph = dl["geo_graph"].to(self.device)
            else:
                user, traj, geo, center_traj, long_traj, dt, label_traj, \
                    label_geo, negihbors_mask, traj_graph, geo_graph = dl
                user = user.to(self.device)
                traj = traj.to(self.device)
                geo = geo.to(self.device)
                center_traj = center_traj.to(self.device)
                long_traj = long_traj.to(self.device)
                dt = dt.to(self.device)
                label_traj = label_traj.to(self.device)
                label_geo_5 = label_geo.to(self.device)
                label_geo_4 = None
                negihbors_mask = negihbors_mask.to(self.device)
                traj_graph = traj_graph.to(self.device)
                geo_graph = geo_graph.to(self.device)

            geo_seqs = dl.get("geo_seqs", None) if isinstance(dl, dict) else None
            if isinstance(dl, dict) and "geo_graphs" in dl:
                setattr(geo_graph, "graphs_p", dl["geo_graphs"])
            outputs = model(user, traj, geo, center_traj, long_traj, dt, traj_graph,
                            geo_graph, geo_seqs=geo_seqs)
            if isinstance(outputs, tuple):
                pred_traj = outputs[0]
                pred_geo_5 = outputs[1]
                pred_regions = outputs[2] if len(outputs) >= 3 and isinstance(outputs[2], dict) else getattr(model, "pred_regions", None)
            else:
                pred_traj = outputs
                pred_geo_5 = None
                pred_regions = getattr(model, "pred_regions", None)

            if hasattr(self.config, 'mask') and self.config.mask:
                negihbors_mask = negihbors_mask.unsqueeze(1).repeat(
                    1, self.config.max_sequence_length, 1)
                pred_traj.masked_fill_(negihbors_mask, -1000)

            loss_poi = criterion(pred_traj.permute(0, 2, 1), label_traj)
            total_loss = loss_poi
            loss_geo_g5 = None
            loss_geo_g4 = None
            if isinstance(pred_regions, dict):
                labels_map = {5: label_geo_5}
                if label_geo_4 is not None:
                    labels_map[4] = label_geo_4
                for i, p in enumerate(self.config.geohash_precisions):
                    lam = float(self.config.lambda_regions[i])
                    if lam <= 0:
                        continue
                    logits_p = pred_regions.get(f"G{int(p)}", None)
                    label_p = labels_map.get(int(p), None)
                    if logits_p is None or label_p is None:
                        continue
                    lp = criterion(logits_p.permute(0, 2, 1), label_p)
                    if int(p) == 5:
                        loss_geo_g5 = lp
                    if int(p) == 4:
                        loss_geo_g4 = lp
                    total_loss = total_loss + lam * lp

            val_acc.append(calculate_acc(pred_traj, label_traj))
            val_loss.append(total_loss.item())

            # update pbar
            mean_acc = torch.concat(val_acc, dim=1).mean(dim=1).cpu().tolist()
            mean_acc = [round(acc, 4) for acc in mean_acc]
            postfix = {
                'val_loss': f'{np.mean(val_loss):.4f}',
                'poi': float(loss_poi.item()),
                'g5': float(loss_geo_g5.item()) if loss_geo_g5 is not None else None,
                'g4': float(loss_geo_g4.item()) if loss_geo_g4 is not None else None,
                'acc': mean_acc
            }
            vbar.set_postfix(**{k: v for k, v in postfix.items() if v is not None})

    @torch.no_grad()
    def test(self, model, dataloader, model_path):
        # prepare dataloader
        test_dl = dataloader.test_dataloader()

        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_acc = []
        tbar = tqdm(test_dl, desc='test', total=len(test_dl))
        self.logger.info('start testing...')
        for idx, dl in enumerate(tbar):
            # 兼容 batch 为 dict 或 tuple（测试保持主指标不变）
            if isinstance(dl, dict):
                user = dl["user"].to(self.device)
                traj = dl["traj"].to(self.device)
                geo = dl["geo"].to(self.device)
                center_traj = dl["center_traj"].to(self.device)
                long_traj = dl["long_traj"].to(self.device)
                dt = dl["dt"].to(self.device)
                label_traj = dl["label_traj"].to(self.device)
                negihbors_mask = dl["negihbors_mask"].to(self.device)
                traj_graph = dl["traj_graph"].to(self.device)
                geo_graph = dl["geo_graph"].to(self.device)
            else:
                user, traj, geo, center_traj, long_traj, dt, label_traj, \
                    label_geo, negihbors_mask, traj_graph, geo_graph = dl
                user = user.to(self.device)
                traj = traj.to(self.device)
                geo = geo.to(self.device)
                center_traj = center_traj.to(self.device)
                long_traj = long_traj.to(self.device)
                dt = dt.to(self.device)
                label_traj = label_traj.to(self.device)
                negihbors_mask = negihbors_mask.to(self.device)
                traj_graph = traj_graph.to(self.device)
                geo_graph = geo_graph.to(self.device)

            geo_seqs = dl.get("geo_seqs", None) if isinstance(dl, dict) else None
            if isinstance(dl, dict) and "geo_graphs" in dl:
                setattr(geo_graph, "graphs_p", dl["geo_graphs"])
            outputs = model(user, traj, geo, center_traj, long_traj, dt, traj_graph,
                            geo_graph, geo_seqs=geo_seqs)
            if isinstance(outputs, tuple):
                pred_traj = outputs[0]
            else:
                pred_traj = outputs
            if hasattr(self.config, 'mask') and self.config.mask:
                negihbors_mask = negihbors_mask.unsqueeze(1).repeat(
                    1, self.config.max_sequence_length, 1)
                pred_traj.masked_fill_(negihbors_mask, -1000)

            test_acc.append(calculate_acc(pred_traj, label_traj))
            # update pbar
            mean_acc = torch.concat(test_acc, dim=1).mean(dim=1).cpu().tolist()
            mean_acc = [round(acc, 4) for acc in mean_acc]
            tbar.set_postfix(acc=mean_acc)

        self.logger.info('testing done.')
        self.logger.info('-------------------------------------')
        self.logger.info('test result:')
        self.logger.info(f'Acc@1: {mean_acc[0]}')
        self.logger.info(f'Acc@5: {mean_acc[1]}')
        self.logger.info(f'Acc@10: {mean_acc[2]}')
        self.logger.info(f'Acc@20: {mean_acc[3]}')
        self.logger.info(f'MRR: {mean_acc[4]}')
