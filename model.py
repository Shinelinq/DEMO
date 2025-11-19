import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import logging


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None, other=None):
        # Q shape: (batch_size, n_heads, len_q, d_k)
        # K shape: (batch_size, n_heads, len_k, d_k)
        # V shape: (batch_size, n_heads, len_v(=len_k), d_v)
        # attn_mask shape: (batch_size, n_heads, seq_len, seq_len)
        # other shape: (batch_size, n_heads, seq_len, seq_len)

        # scores shape: (batch_size, n_heads, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(3))
        if other is not None:
            scores = scores + other
        if attn_mask is not None:
            # Fills elements of self tensor with value where mask is True.
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        # context shape: (batch_size, n_heads, len_q, d_v)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads=1, dropout=0.5, d_v=64, d_k=64):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask=None, other=None):
        # input_Q shape: (batch_size, len_q, d_model)
        # input_K shape: (batch_size, len_k, d_model)
        # input_V sahpe: (batch_size, len_v(=len_k), d_model)
        # attn_mask shape: (batch_size, seq_len, seq_len)
        # other shape: (batch_size, seq_len, seq_len)

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q shape: (batch_size, n_heads, len_q, d_k)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K shape: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V shape: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            # attn_mask shape: (batch_size, n_heads, seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        if other is not None:
            other = other.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context shape: (batch_size, n_heads, len_q, d_v)
        # attn shape: (batch_size, n_heads, len_q, len_k)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, other)
        # context shape: (batch_size, len_q, n_heads * d_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # output shape: (batch_size, len_q, d_model)
        output = self.fc(context)
        output = self.dropout(output)
        return output, attn


class GCN(torch.nn.Module):

    def __init__(self, d_model, n_layers=3):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [GCNConv(d_model, d_model) for _ in range(n_layers)])

    def forward(self, x, edge_index):
        for conv in self.conv_list:
            x = conv(x, edge_index)
            x = F.dropout(x, training=self.training)
        return x


class LiteFusion(nn.Module):

    def __init__(self, num_inputs, d_model, dropout, init_bias='g5-dominant'):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_inputs)])
        self.dropout = nn.Dropout(dropout)
        if isinstance(init_bias, str) and init_bias == 'g5-dominant':
            if num_inputs >= 1:
                with torch.no_grad():
                    self.weights.fill_(0.1)
                    idx = 1 if num_inputs >= 2 else 0
                    self.weights[idx] = 2.0

    def forward(self, x_list):
        assert len(x_list) == len(self.norms)
        xs = []
        for i, x in enumerate(x_list):
            x = self.norms[i](x)
            x = self.dropout(x)
            xs.append(x)
        pi = torch.softmax(self.weights, dim=0)
        out = None
        for i, x in enumerate(xs):
            w = pi[i]
            xw = x * w
            out = xw if out is None else out + xw
        return out


class EmbeddingLayer(nn.Module):

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.config = config

        # define embedding layer
        self.userEmbLayer = nn.Embedding(config.max_user_num, config.hidden_size, 0)
        self.locEmbLayer = nn.Embedding(config.max_loc_num, config.hidden_size, 0)
        self.geoEmbLayer = nn.Embedding(config.max_geo_num, config.hidden_size, 0)

        # init embedding layer
        nn.init.normal_(self.userEmbLayer.weight, std=0.1)
        nn.init.normal_(self.locEmbLayer.weight, std=0.1)
        nn.init.normal_(self.geoEmbLayer.weight, std=0.1)
        self.geo_embs = nn.ModuleDict()
        p_list = getattr(config, 'geohash_precisions', [5])
        vocab_map = getattr(config, 'geo_vocab_size', {})
        for p in p_list:
            ip = int(p)
            if ip == 5:
                self.geo_embs[str(ip)] = self.geoEmbLayer
                continue
            v = vocab_map.get(ip, None)
            if v is None:
                v = getattr(config, f'max_geo_num_{ip}', None)
            if v is None:
                v = getattr(config, 'max_geo_num', None)
            emb = nn.Embedding(int(v), config.hidden_size, 0)
            nn.init.normal_(emb.weight, std=0.1)
            self.geo_embs[str(ip)] = emb

    def forward(self, user, traj, geo, long_traj, traj_graph, geo_graph, geo_seqs=None):
        #! Embedding user, traj, geohash
        # emb shape: (batch_size, max_sequence_length, hidden_size)
        user_emb = self.userEmbLayer(user)
        traj_emb = self.locEmbLayer(traj)
        geo_emb = self.geoEmbLayer(geo)

        long_traj_emb = self.locEmbLayer(long_traj)

        traj_graph.x = self.locEmbLayer(traj_graph.x)
        geo_graph.x = self.geoEmbLayer(geo_graph.x)
        if hasattr(geo_graph, 'graphs_p') and isinstance(getattr(geo_graph, 'graphs_p'), dict):
            keys_embedded = []
            dev = getattr(geo_graph.x, 'device', None)
            for p, g in geo_graph.graphs_p.items():
                key = str(int(p))
                if key not in self.geo_embs:
                    logging.getLogger().error('[embed] missing embedding table for G%s', key)
                    raise KeyError(f'missing embedding for precision {key}')
                emb_layer = self.geo_embs[key]
                g.x = emb_layer(g.x.to(dtype=torch.long, device=dev))
                keys_embedded.append(int(p))
            if not hasattr(self, '_embed_log_done') or not self._embed_log_done:
                try:
                    logging.getLogger().info('[embed] graphs_p embedded for keys: %s', str(keys_embedded))
                except Exception:
                    pass
                self._embed_log_done = True
        if geo_seqs is not None:
            geo_emb_dict = {}
            for p, seq in geo_seqs.items():
                emb_layer = self.geo_embs.get(str(int(p)), self.geoEmbLayer)
                geo_emb_dict[int(p)] = emb_layer(seq)
            self.geo_emb_dict = geo_emb_dict

        return user_emb, traj_emb, geo_emb, long_traj_emb, traj_graph, geo_graph


class LocalCenterEncoder(nn.Module):

    def __init__(self, d_model, n_heads=4, dropout=0.5):
        super(LocalCenterEncoder, self).__init__()
        self.d_model = d_model
        self.traj_conv = GCN(d_model, 3)
        self.geo_conv = GCN(d_model, 3)
        self.gcn_modules = nn.ModuleDict()
        self.gcn_modules['5'] = self.geo_conv
        self.attn = nn.MultiheadAttention(d_model * 2, n_heads, dropout, batch_first=True)
        self.linear = nn.Linear(d_model * 2, d_model)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self._init_done = False
        self._log_once = False
        self.lite_fusion_long = None

    def forward(self, center_traj, traj_graph, geo_graph):
        # center_traj shape: (batch_size, long_sequence_length)

        # traj_conv_out/geo_conv_out shape: (batch_node_num, d_model)
        traj_conv_out = self.traj_conv(traj_graph.x, traj_graph.edge_index)
        geo_conv_out = self.geo_conv(geo_graph.x, geo_graph.edge_index)
        traj_graph.x = traj_conv_out
        geo_graph.x = geo_conv_out

        # center_traj_emb shape: (batch_size, long_sequence_length, d_model)
        center_traj = center_traj + traj_graph.ptr[:-1].unsqueeze(1)
        center_traj_emb = traj_conv_out[center_traj]

        sub_traj_graph = traj_graph.subgraph(traj_graph.freq >= traj_graph.thr)
        sub_geo_graph = geo_graph.subgraph(geo_graph.freq >= geo_graph.thr)
        # traj_personal/geo_personal shape: (batch_size, d_model)
        traj_personal = global_mean_pool(sub_traj_graph.x, sub_traj_graph.batch)
        precisions = sorted(set([int(x) for x in getattr(self, 'geohash_precisions', [5])]))
        if not self._init_done:
            share = int(getattr(self, 'share_gcn_weights', 0))
            first = None
            for i, p in enumerate(precisions):
                key = str(p)
                if p == 5:
                    self.gcn_modules[key] = self.geo_conv
                    first = self.geo_conv
                else:
                    if share == 1 and first is not None:
                        self.gcn_modules[key] = first
                    else:
                        self.gcn_modules[key] = GCN(self.d_model, 3)
            num_inputs = len(precisions)
            drop = float(getattr(self, 'fusion_dropout', 0.1))
            bias = getattr(self, 'fusion_init_bias', 'g5-dominant')
            self.lite_fusion_long = LiteFusion(num_inputs, self.d_model, drop, bias)
            self._init_done = True

        graphs_p = getattr(geo_graph, 'graphs_p', None)
        vec_list = []
        for p in precisions:
            if p == 5:
                g = geo_graph
            else:
                g = graphs_p.get(p) if isinstance(graphs_p, dict) else None
            if g is None:
                vec_list.append(global_mean_pool(sub_geo_graph.x, sub_geo_graph.batch))
            else:
                gc = self.gcn_modules.get(str(p), self.geo_conv)
                gout = gc(g.x, g.edge_index)
                g.x = gout
                sg = g.subgraph(g.freq >= g.thr)
                vec_list.append(global_mean_pool(sg.x, sg.batch))

        use_fusion_long = int(getattr(self, 'use_fusion_long', 0))
        if not self._log_once:
            assert 5 in precisions
            if use_fusion_long == 0:
                logging.getLogger().info('[long] fallback=G5')
            else:
                pi = torch.softmax(self.lite_fusion_long.weights, dim=0).detach().cpu().tolist()
                logging.getLogger().info('[long] fusion_pi=%s', str(pi))
            self._log_once = True

        if use_fusion_long == 1 and self.lite_fusion_long is not None:
            geo_vec = self.lite_fusion_long(vec_list)
        else:
            try:
                idx = precisions.index(5)
            except Exception:
                idx = 0
            geo_vec = vec_list[idx] if len(vec_list) > idx else global_mean_pool(sub_geo_graph.x, sub_geo_graph.batch)
        geo_personal = geo_vec

        # traj_personal/geo_personal shape: (batch_size, 1, d_model)
        traj_personal = traj_personal.unsqueeze(1)
        geo_personal = geo_personal.unsqueeze(1)
        # personal shape: (batch_size, 1, d_model * 2)
        personal = torch.concat([traj_personal, geo_personal], dim=-1)

        # personal shape: (batch_size, 1, d_model)
        user_perfence, _ = self.attn(personal, personal, personal)
        user_perfence = self.linear(user_perfence)
        center_traj_emb = self.dropout1(center_traj_emb)
        user_perfence = self.dropout2(user_perfence)
        return center_traj_emb, user_perfence


class ShortTermEncoder(nn.Module):

    def __init__(self, d_model) -> None:
        super(ShortTermEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=d_model * 4,
                            hidden_size=d_model * 2,
                            batch_first=True)
        self.attn = MultiHeadAttention(d_model * 2, n_heads=4, dropout=0.5)
        self.w = nn.Parameter(torch.ones(2))
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.lite_fusion_short = None
        self._init_short = False
        self._log_once_short = False

    def forward(self, user_emb, traj_emb, geo_emb, center_traj_emb, long_traj_emb,
                user_perfence, dt):
        # traj_emb/geo_emb shape: (batch_size, max_sequence_length, d_model)
        # center_traj_emb shape: (batch_size, long_sequence_length, d_model)
        # long_traj_emb shape: (batch_size, long_sequence_length, d_model)
        # user_perfence shape: (batch_size, 1, d_model * 2)
        # dt shape: (batch_size, max_sequence_length, max_sequence_length)

        precisions = sorted(set([int(x) for x in getattr(self, 'geohash_precisions', [5])]))
        if not self._init_short:
            num_inputs = len(precisions)
            drop = float(getattr(self, 'fusion_dropout', 0.1))
            bias = getattr(self, 'fusion_init_bias', 'g5-dominant')
            self.lite_fusion_short = LiteFusion(num_inputs, int(traj_emb.size(-1)), drop, bias)
            self._init_short = True
        use_fusion_short = int(getattr(self, 'use_fusion_short', 0))
        if not self._log_once_short:
            assert 5 in precisions
            if use_fusion_short == 0:
                logging.getLogger().info('[short] fallback=G5')
            else:
                pi = torch.softmax(self.lite_fusion_short.weights, dim=0).detach().cpu().tolist()
                logging.getLogger().info('[short] fusion_pi=%s', str(pi))
            self._log_once_short = True

        fused_geo = None
        if isinstance(geo_emb, dict):
            emb_list = []
            for p in precisions:
                t = geo_emb.get(int(p), None)
                if t is None and p == 5 and not fused_geo is not None:
                    t = geo_emb.get(5, None)
                if t is None:
                    t = geo_emb.get(5, None)
                emb_list.append(t)
            if use_fusion_short == 1 and self.lite_fusion_short is not None:
                fused_geo = self.lite_fusion_short(emb_list)
            else:
                try:
                    fused_geo = emb_list[precisions.index(5)]
                except Exception:
                    fused_geo = emb_list[0]
        else:
            fused_geo = geo_emb

        user_perfence = user_perfence.repeat(1, traj_emb.size(1), 1)
        user_perfence = torch.concat([user_emb, user_perfence], dim=-1)

        # input shape: (batch_size, max_sequence_length, d_model * 4)
        input = torch.concat([traj_emb, fused_geo, user_perfence], dim=-1)

        # lstm_output shape: (batch_size, max_sequence_length, hidden_size)
        lstm_output, (hidden_state, cell_state) = self.lstm(input)
        lstm_output = self.dropout1(lstm_output)

        # center_input shape: (batch_size, long_sequence_length, d_model)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        center_input = F.relu(w1 * long_traj_emb + w2 * center_traj_emb)

        center_input = torch.concat([center_input, \
            user_emb.repeat(1, long_traj_emb.size(1) // traj_emb.size(1), 1)], dim=-1)
        # center_out shape: (batch_size, max_sequence_length, d_model)
        center_out, _ = self.attn(lstm_output,
                                  center_input,
                                  center_input,
                                  other=(1 / (1 + dt)))

        # out shape: (batch_size, max_sequence_length, d_model)
        out = lstm_output * torch.exp(-center_out)
        out = self.dropout2(out)
        return out


class PoiModel(nn.Module):

    def __init__(self, config):
        super(PoiModel, self).__init__()
        self.config = config
        self.EmbeddingLayer = EmbeddingLayer(config)
        self.LocalCenterEncoder = LocalCenterEncoder(config.hidden_size)
        self.ShortTermEncoder = ShortTermEncoder(config.hidden_size)
        for m in (self.LocalCenterEncoder, self.ShortTermEncoder):
            m.geohash_precisions = config.geohash_precisions
            m.use_fusion_long = getattr(config, "use_fusion_long", 0)
            m.use_fusion_short = getattr(config, "use_fusion_short", 0)
            m.fusion_init_bias = getattr(config, "fusion_init_bias", "g5-dominant")
            m.fusion_dropout = getattr(config, "fusion_dropout", 0.1)
            m.share_gcn_weights = getattr(config, "share_gcn_weights", 0)
        self.fc_traj = nn.Linear(config.hidden_size * 2, config.max_loc_num)
        self.geo_heads = nn.ModuleDict()
        p_list = sorted(set([int(p) for p in getattr(config, 'geohash_precisions', [5])]))
        vocab_map = getattr(config, 'geo_vocab_size', {})
        for p in p_list:
            key = f"G{p}"
            if p == 5:
                out_features = int(getattr(config, 'max_geo_num', 0))
            else:
                out_features = int(vocab_map.get(p, getattr(config, f'max_geo_num_{p}', getattr(config, 'max_geo_num', 0))))
            self.geo_heads[key] = nn.Linear(config.hidden_size * 2, out_features)
        self._heads_logged = False

    def forward(self, user, traj, geo, center_traj, long_traj, dt, traj_graph, geo_graph, geo_seqs=None):
        # user/traj/geo shape: (batch_size, max_sequence_length)
        # center_traj/long_traj shape: (batch_size, long_sequence_length)
        # dt shape: (batch_size, max_sequence_length, max_sequence_length)

        # user_emb/traj_emb/geo_emb shape: (batch_size, max_sequence_length, hidden_size)
        user_emb, traj_emb, geo_emb, long_traj_emb, traj_graph, geo_graph = self.EmbeddingLayer(
            user, traj, geo, long_traj, traj_graph, geo_graph, geo_seqs=geo_seqs)

        # user_perfence shape: (batch_size, 1, hidden_size)
        # center_traj_emb shape: (batch_size, long_sequence_length, hidden_size)
        center_traj_emb, user_perfence = self.LocalCenterEncoder(
            center_traj, traj_graph, geo_graph)

        # short_enc_out shape: (batch_size, max_sequence_length, hidden_size)
        geo_for_short = getattr(self.EmbeddingLayer, "geo_emb_dict", None) or geo_emb
        short_enc_out = self.ShortTermEncoder(user_emb, traj_emb, geo_for_short,
                                              center_traj_emb, long_traj_emb,
                                              user_perfence, dt)

        # pred_traj shape: (batch_size, max_sequence_length, max_loc_num)
        pred_traj = self.fc_traj(short_enc_out)
        # 多粒度区域 head 映射
        pred_regions = {}
        for key, head in self.geo_heads.items():
            pred_regions[key] = head(short_enc_out)
        # 兼容：返回 G5 作为第二项；同时侧带字典用于训练器读取
        pred_geo_5 = pred_regions.get('G5', None)
        self.pred_regions = pred_regions
        if not self._heads_logged:
            try:
                info = {k: getattr(v, 'out_features', None) for k, v in self.geo_heads.items()}
                logging.getLogger().info('geo_heads: %s', str(info))
            except Exception:
                pass
            self._heads_logged = True
        return pred_traj, pred_geo_5, pred_regions
