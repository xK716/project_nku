# import torch
# import torch.nn as nn
# from enum import Enum
# import time
# import numpy as np
# from utils import *
# import scipy.sparse as sp
# from scipy.sparse import coo_matrix
# import math
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from math import pi
# import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# class Rnn(Enum):
#     """ The available RNN units """
#
#     RNN = 0
#     GRU = 1
#     LSTM = 2
#
#     @staticmethod
#     def from_string(name):
#         if name == 'rnn':
#             return Rnn.RNN
#         if name == 'gru':
#             return Rnn.GRU
#         if name == 'lstm':
#             return Rnn.LSTM
#         raise ValueError('{} not supported in --rnn'.format(name))
#
#
# class RnnFactory():
#     """ Creates the desired RNN unit. """
#
#     def __init__(self, rnn_type_str):
#         self.rnn_type = Rnn.from_string(rnn_type_str)
#
#     def __str__(self):
#         if self.rnn_type == Rnn.RNN:
#             return 'Use pytorch RNN implementation.'
#         if self.rnn_type == Rnn.GRU:
#             return 'Use pytorch GRU implementation.'
#         if self.rnn_type == Rnn.LSTM:
#             return 'Use pytorch LSTM implementation.'
#
#     def is_lstm(self):
#         return self.rnn_type in [Rnn.LSTM]
#
#     def create(self, hidden_size):
#         if self.rnn_type == Rnn.RNN:
#             return nn.RNN(hidden_size, hidden_size)
#         if self.rnn_type == Rnn.GRU:
#             return nn.GRU(hidden_size, hidden_size)
#         if self.rnn_type == Rnn.LSTM:
#             return nn.LSTM(hidden_size, hidden_size)
#
# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#
#     def __init__(self, d_model, dropout, max_len=20):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(1)
#         self.register_buffer("pe", pe)
#
#     def forward(self, x):
#         x = x + self.pe
#         return self.dropout(x)
#
# class TransformerModel(nn.Module):
#     def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.1):
#         super(TransformerModel, self).__init__()
#         self.embed_size = embed_size
#         self.pos_encoder = PositionalEncoding(embed_size, dropout)
#         encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#
#     def generate_square_subsequent_mask(self, sz, device):
#         mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def forward(self, src, src_mask=None):
#         if src_mask is None:
#             src_mask = self.generate_square_subsequent_mask(src.size(0), src.device)
#         src = src * math.sqrt(self.embed_size)
#         src = self.pos_encoder(src)
#         src = self.transformer_encoder(src, src_mask)
#
#         return src
#
#
#
# class Flashback(nn.Module):
#     """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
#     of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
#     """
#
#     def __init__(self, input_size, user_count, region_count, slot_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight, graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, device, region_graph, region_dis, cl_decay_steps, lat_min, lat_max, lng_min, lng_max, dropout, dataset_name):
#         super().__init__()
#         self.input_size = input_size
#         self.user_count = user_count
#         self.region_count = region_count
#         self.slot_count = slot_count
#         self.hidden_size = hidden_size
#         self.f_t = f_t
#         self.f_s = f_s
#         self.device = device
#         self.use_curriculum_learning = True
#         self.cl_decay_steps = cl_decay_steps
#         self.lat_min, self.lat_max, self.lng_min, self.lng_max = lat_min, lat_max, lng_min, lng_max
#         self.dr_ratio = dropout
#         self.dataset_name = dataset_name
#
#         self.lambda_loc = lambda_loc
#         self.lambda_user = lambda_user
#         self.use_weight = use_weight
#         self.use_graph_user = use_graph_user
#         self.use_spatial_graph = use_spatial_graph
#         self.region_graph = torch.from_numpy(region_graph).to(self.device) if region_graph is not None else None
#         self.region_dis = torch.from_numpy(region_dis).to(self.device) if region_dis is not None else None
#         self.I_rg = identity(region_graph.shape[0], format='coo')
#         if self.region_graph is not None:
#             self.region_graph_coo = coo_matrix(region_graph)
#             self.region_graph_coo = sparse_matrix_to_tensor(calculate_random_walk_matrix((self.region_graph_coo * self.lambda_loc + self.I_rg).astype(np.float32)))
#         if graph is not None:
#             self.graph = sparse_matrix_to_tensor(calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))
#         else:
#             self.graph = None
#
#         self.spatial_graph = spatial_graph
#         if interact_graph is not None:
#             self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
#                 interact_graph))
#         else:
#             self.interact_graph = None
#
#
#         self.poi_dropout = nn.Dropout(self.dr_ratio)
#         self.region_dropout = nn.Dropout(0.2)
#         self.dropout = nn.Dropout(0.2)
#         self.encoder = nn.Embedding(input_size, hidden_size)
#         self.time_encoder = nn.Embedding(slot_count, hidden_size//2)
#         self.region_encoder = nn.Embedding(region_count, hidden_size)
#         self.user_encoder = nn.Embedding(user_count, hidden_size)
#         self.up_encoder = nn.Embedding(user_count, hidden_size)
#         self.ur_encoder = nn.Embedding(user_count, hidden_size)
#
#         self.rnn_size = hidden_size * 2
#         self.rnn = rnn_factory.create(self.rnn_size)
#         self.rnn_r = rnn_factory.create(self.rnn_size)
#         self.fc_size = self.rnn_size + hidden_size * 2 + hidden_size // 2
#         self.fc = nn.Linear(self.fc_size, input_size)
#         self.fc_r = nn.Linear(self.rnn_size * 2, hidden_size)
#
#
#
#     def forward(self, x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h, active_user, batches_seen=None):
#         seq_len, user_len = x.size()
#         x_emb = self.encoder(x)
#         rg_emb = self.region_encoder(r)
#         rg_emb_nx = self.region_encoder(y_r)
#         t_emb = self.time_encoder(t_slot)
#         t_emb_nx = self.time_encoder(y_t_slot)
#
#         p_u = self.user_encoder(active_user).repeat(seq_len, 1, 1)
#         up_pref = self.up_encoder(active_user).repeat(seq_len, 1, 1)
#         ur_pref = self.ur_encoder(active_user).repeat(seq_len, 1, 1)
#         up_pref = rotate_batch(up_pref, t_emb, self.hidden_size)
#         ur_pref = rotate_batch(ur_pref, t_emb, self.hidden_size)
#
#         new_x_emb = torch.cat([x_emb, up_pref], dim=-1)
#         new_rg_emb = torch.cat([rg_emb, ur_pref], dim=-1)
#         out, h_ = self.rnn(new_x_emb, h)
#         out_r, h_r = self.rnn_r(new_rg_emb, h)
#         out = self.poi_dropout(out)
#         out_r = self.region_dropout(out_r)
#
#         user_loc_similarity = torch.exp(-(torch.norm(up_pref - x_emb, p=2, dim=-1))).to(x.device)
#         user_region_similarity = torch.exp(-(torch.norm(ur_pref - rg_emb, p=2, dim=-1))).to(x.device)
#
#         out_w = torch.zeros(seq_len, user_len, self.rnn_size, device=x.device)
#         out_wr = torch.zeros(seq_len, user_len, self.rnn_size, device=x.device)
#
#         for i in range(seq_len):
#             dist_t = t[i].unsqueeze(0) - t[:i+1]
#             dist_s = torch.norm(s[i].unsqueeze(0) - s[:i+1], dim=-1)
#             a_j = self.f_t(dist_t, user_len).unsqueeze(-1)
#             b_j = self.f_s(dist_s, user_len).unsqueeze(-1)
#             # Compute the weights
#             w_j = a_j * b_j
#             w_jp = w_j * user_loc_similarity[:i+1].unsqueeze(-1) + 1e-10
#             w_jr = w_j * user_region_similarity[:i+1].unsqueeze(-1) + 1e-10
#             sum_wp = w_jp.sum(dim=0)
#             sum_wr = w_jr.sum(dim=0)
#             out_w[i] = (w_jp * out[:i+1]).sum(dim=0) / sum_wp
#             out_wr[i] = (w_jr * out_r[:i+1]).sum(dim=0) / sum_wr
#
#         out_pu_r = self.fc_r(torch.cat([out_wr, p_u, ur_pref], dim=-1))
#         y_linear_r = out_pu_r.matmul(self.region_encoder.weight.transpose(1, 0))
#         out_pu = torch.cat([out_w+out_wr, p_u, up_pref+ur_pref, t_emb_nx], dim=-1)
#         out_pu = self.dropout(out_pu)
#         y_linear = self.fc(out_pu)
#
#         return y_linear, y_linear_r, h
#
#
#
# '''
# ~~~ h_0 strategies ~~~
# Initialize RNNs hidden states
# '''
#
#
# def create_h0_strategy(hidden_size, is_lstm):
#     if is_lstm:
#         return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
#     else:
#         return FixNoiseStrategy(hidden_size)
#
#
# class H0Strategy():
#
#     def __init__(self, hidden_size):
#         self.hidden_size = hidden_size
#
#     def on_init(self, user_len, device):
#         pass
#
#     def on_reset(self, user):
#         pass
#
#     def on_reset_test(self, user, device):
#         return self.on_reset(user)
#
#
# class FixNoiseStrategy(H0Strategy):
#     """ use fixed normal noise as initialization """
#
#     def __init__(self, hidden_size):
#         super().__init__(hidden_size)
#         mu = 0
#         sd = 1 / self.hidden_size
#         self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
#
#     def on_init(self, user_len, device):
#         hs = []
#         for i in range(user_len):
#             hs.append(self.h0)
#         # (1, 200, 10)
#         return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
#
#     def on_reset(self, user):
#         return self.h0
#
#
# class LstmStrategy(H0Strategy):
#     """ creates h0 and c0 using the inner strategy """
#
#     def __init__(self, hidden_size, h_strategy, c_strategy):
#         super(LstmStrategy, self).__init__(hidden_size)
#         self.h_strategy = h_strategy
#         self.c_strategy = c_strategy
#
#     def on_init(self, user_len, device):
#         h = self.h_strategy.on_init(user_len, device)
#         c = self.c_strategy.on_init(user_len, device)
#         return h, c
#
#     def on_reset(self, user):
#         h = self.h_strategy.on_reset(user)
#         c = self.c_strategy.on_reset(user)
#         return h, c






import torch
import torch.nn as nn
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from math import pi
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(0), src.device)
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask)

        return src



class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, region_count, slot_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight, graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, device, region_graph, region_dis, cl_decay_steps, lat_min, lat_max, lng_min, lng_max, dropout, dataset_name, semantic_count=1, poi_semantic=None, poi_region=None, use_sem_branch=False, semantic_fusion_weight=0.5):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.region_count = region_count
        self.slot_count = slot_count
        self.hidden_size = hidden_size
        self.f_t = f_t
        self.f_s = f_s
        self.device = device
        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.lat_min, self.lat_max, self.lng_min, self.lng_max = lat_min, lat_max, lng_min, lng_max
        self.dr_ratio = dropout
        self.dataset_name = dataset_name
        # Sem-DPRL: semantic branch configuration.
        self.semantic_count = int(semantic_count) if semantic_count is not None else 1
        self.use_sem_branch = bool(use_sem_branch and self.semantic_count > 1)
        self.semantic_fusion_weight = float(semantic_fusion_weight)
        if poi_semantic is None:
            poi_semantic = torch.zeros(input_size, dtype=torch.long)
        else:
            poi_semantic = torch.as_tensor(poi_semantic, dtype=torch.long)
        if poi_region is None:
            poi_region = torch.zeros(input_size, dtype=torch.long)
        else:
            poi_region = torch.as_tensor(poi_region, dtype=torch.long)
        self.register_buffer('poi_semantic', poi_semantic.clamp(min=0, max=max(self.semantic_count - 1, 0)))
        self.register_buffer('poi_region', poi_region.clamp(min=0, max=max(region_count - 1, 0)))


        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.region_graph = torch.from_numpy(region_graph).to(self.device) if region_graph is not None else None
        self.region_dis = torch.from_numpy(region_dis).to(self.device) if region_dis is not None else None
        self.I_rg = identity(region_graph.shape[0], format='coo')
        if self.region_graph is not None:
            self.region_graph_coo = coo_matrix(region_graph)
            self.region_graph_coo = sparse_matrix_to_tensor(calculate_random_walk_matrix((self.region_graph_coo * self.lambda_loc + self.I_rg).astype(np.float32)))
        if graph is not None:
            self.graph = sparse_matrix_to_tensor(calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))
        else:
            self.graph = None

        self.spatial_graph = spatial_graph
        if interact_graph is not None:
            self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
                interact_graph))
        else:
            self.interact_graph = None


        self.poi_dropout = nn.Dropout(self.dr_ratio)
        self.region_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.time_encoder = nn.Embedding(slot_count, hidden_size//2)
        self.region_encoder = nn.Embedding(region_count, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.up_encoder = nn.Embedding(user_count, hidden_size)
        self.ur_encoder = nn.Embedding(user_count, hidden_size)
        # Sem-DPRL: user-semantic preference and semantic embedding.
        self.semantic_encoder = nn.Embedding(self.semantic_count, hidden_size)
        self.us_encoder = nn.Embedding(user_count, hidden_size)


        self.rnn_size = hidden_size * 2
        self.rnn = rnn_factory.create(self.rnn_size)
        self.rnn_r = rnn_factory.create(self.rnn_size)
        self.rnn_s = rnn_factory.create(self.rnn_size)
        self.fc_size = self.rnn_size + hidden_size * 2 + hidden_size // 2
        self.fc = nn.Linear(self.fc_size, input_size)
        self.fc_r = nn.Linear(self.rnn_size * 2, hidden_size)
        # Semantic auxiliary decoder: predict next semantic class / intent.
        self.fc_s = nn.Linear(self.rnn_size * 2, self.semantic_count)

        # Sem-DPRL v2: gated semantic residual fusion.
        # Instead of adding the semantic branch with a fixed weight, a learnable gate
        # decides how much semantic information should enter the POI branch.
        self.sem_residual = nn.Linear(self.rnn_size, self.rnn_size)
        self.sem_pref_residual = nn.Linear(hidden_size, hidden_size)
        self.sem_gate = nn.Linear(self.rnn_size * 3 + hidden_size, 1)
        nn.init.constant_(self.sem_gate.bias, 0.0)



    def forward(self, x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h, active_user, batches_seen=None):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        rg_emb = self.region_encoder(r)
        t_emb = self.time_encoder(t_slot)
        t_emb_nx = self.time_encoder(y_t_slot)

        p_u = self.user_encoder(active_user).repeat(seq_len, 1, 1)
        up_pref = self.up_encoder(active_user).repeat(seq_len, 1, 1)
        ur_pref = self.ur_encoder(active_user).repeat(seq_len, 1, 1)
        up_pref = rotate_batch(up_pref, t_emb, self.hidden_size)
        ur_pref = rotate_batch(ur_pref, t_emb, self.hidden_size)

        # 1) POI branch: concrete-location transition modeling.
        # 2) Region branch: spatial/region transition modeling.
        new_x_emb = torch.cat([x_emb, up_pref], dim=-1)
        new_rg_emb = torch.cat([rg_emb, ur_pref], dim=-1)
        out, h_ = self.rnn(new_x_emb, h)
        out_r, h_r = self.rnn_r(new_rg_emb, h)
        out = self.poi_dropout(out)
        out_r = self.region_dropout(out_r)

        # User preference factors for personalized flashback aggregation.
        user_loc_similarity = torch.exp(-(torch.norm(up_pref - x_emb, p=2, dim=-1))).to(x.device)
        user_region_similarity = torch.exp(-(torch.norm(ur_pref - rg_emb, p=2, dim=-1))).to(x.device)

        # Sem-DPRL semantic branch: semantic sequence + user-semantic preference.
        if self.use_sem_branch:
            sem_x = self.poi_semantic[x].long()
            sem_emb = self.semantic_encoder(sem_x)
            us_pref = self.us_encoder(active_user).repeat(seq_len, 1, 1)
            us_pref = rotate_batch(us_pref, t_emb, self.hidden_size)
            new_sem_emb = torch.cat([sem_emb, us_pref], dim=-1)
            out_s, h_s = self.rnn_s(new_sem_emb, h)
            out_s = self.dropout(out_s)
            user_sem_similarity = torch.exp(-(torch.norm(us_pref - sem_emb, p=2, dim=-1))).to(x.device)
        else:
            us_pref = torch.zeros_like(up_pref)
            out_s = torch.zeros_like(out)
            user_sem_similarity = torch.ones(seq_len, user_len, device=x.device)

        out_w = torch.zeros(seq_len, user_len, self.rnn_size, device=x.device)
        out_wr = torch.zeros(seq_len, user_len, self.rnn_size, device=x.device)
        out_ws = torch.zeros(seq_len, user_len, self.rnn_size, device=x.device)

        for i in range(seq_len):
            dist_t = t[i].unsqueeze(0) - t[:i+1]
            dist_s = torch.norm(s[i].unsqueeze(0) - s[:i+1], dim=-1)
            a_j = self.f_t(dist_t, user_len).unsqueeze(-1)
            b_j = self.f_s(dist_s, user_len).unsqueeze(-1)
            # Base spatial-temporal correlation w(DeltaT, DeltaD).
            w_j = a_j * b_j
            # Personalized correlation: POI / region / semantic preference factors.
            w_jp = w_j * user_loc_similarity[:i+1].unsqueeze(-1) + 1e-10
            w_jr = w_j * user_region_similarity[:i+1].unsqueeze(-1) + 1e-10
            w_js = w_j * user_sem_similarity[:i+1].unsqueeze(-1) + 1e-10
            sum_wp = w_jp.sum(dim=0)
            sum_wr = w_jr.sum(dim=0)
            sum_ws = w_js.sum(dim=0)
            out_w[i] = (w_jp * out[:i+1]).sum(dim=0) / sum_wp
            out_wr[i] = (w_jr * out_r[:i+1]).sum(dim=0) / sum_wr
            out_ws[i] = (w_js * out_s[:i+1]).sum(dim=0) / sum_ws

        # Region auxiliary prediction, following DPRL's multi-task design.
        out_pu_r = self.fc_r(torch.cat([out_wr, p_u, ur_pref], dim=-1))
        y_linear_r = out_pu_r.matmul(self.region_encoder.weight.transpose(1, 0))

        # Semantic auxiliary prediction: next semantic / intent class.
        if self.use_sem_branch:
            y_linear_s = self.fc_s(torch.cat([out_ws, p_u, us_pref], dim=-1))

            # Sem-DPRL v2: gated semantic residual fusion.
            # This makes semantic information directly influence POI prediction,
            # but the gate limits noisy pseudo-semantics from overwhelming POI/region signals.
            sem_gate = torch.sigmoid(self.sem_gate(torch.cat([out_w, out_wr, out_ws, p_u], dim=-1)))
            sem_traj = self.sem_residual(out_ws)
            sem_pref = self.sem_pref_residual(us_pref)
            fused_traj = out_w + out_wr + self.semantic_fusion_weight * sem_gate * sem_traj
            fused_pref = up_pref + ur_pref + self.semantic_fusion_weight * sem_gate * sem_pref
        else:
            y_linear_s = torch.zeros(seq_len, user_len, self.semantic_count, device=x.device)
            fused_traj = out_w + out_wr
            fused_pref = up_pref + ur_pref

        # Query-enhanced POI prediction with next time slot as query.
        out_pu = torch.cat([fused_traj, p_u, fused_pref, t_emb_nx], dim=-1)
        out_pu = self.dropout(out_pu)
        y_linear = self.fc(out_pu)

        return y_linear, y_linear_r, y_linear_s, h




    # def apply_semantic_candidate_filter(self, poi_logits, region_logits=None, semantic_logits=None,
    #                                     top_m=2000, semantic_weight=0.6, region_weight=0.4,
    #                                     candidate_mode='mask'):
    #     """Soft semantic-aware candidate filtering for evaluation/inference.
    #
    #     Args:
    #         poi_logits: Tensor [N, #POI]
    #         region_logits: Tensor [N, #Region]
    #         semantic_logits: Tensor [N, #Semantic]
    #     Returns:
    #         Tensor [N, #POI]. In candidate_mode='score', only soft semantic/region reranking is applied.
    #         In candidate_mode='mask', POIs outside top_m candidates are masked.
    #     """
    #     if (not self.use_sem_branch) or semantic_logits is None:
    #         return poi_logits
    #
    #     # Semantic score for every candidate POI via its semantic id.
    #     sem_log_prob = F.log_softmax(semantic_logits, dim=-1)
    #     candidate_score = poi_logits + semantic_weight * sem_log_prob[:, self.poi_semantic]
    #
    #     # Region branch offers a spatially-aware soft constraint.
    #     if region_logits is not None:
    #         reg_log_prob = F.log_softmax(region_logits, dim=-1)
    #         candidate_score = candidate_score + region_weight * reg_log_prob[:, self.poi_region]
    #
    #     if candidate_mode == 'score':
    #         return candidate_score
    #
    #     if top_m is not None and 0 < int(top_m) < self.input_size:
    #         top_idx = torch.topk(candidate_score, k=int(top_m), dim=-1).indices
    #         masked_score = torch.full_like(candidate_score, -1e9)
    #         candidate_score = masked_score.scatter(1, top_idx, candidate_score.gather(1, top_idx))
    #     return candidate_score


    def apply_semantic_candidate_filter(self, poi_logits, region_logits=None, semantic_logits=None,
                                        top_m=2000, semantic_weight=0.6, region_weight=0.4,
                                        candidate_mode='mask'):
        if (not self.use_sem_branch) or semantic_logits is None:
            return poi_logits

        def align_to_poi_scale(aux_score, ref_score):
            """
            把 semantic / region score 调整到和 POI logits 相近的尺度。
            这样不会让语义分数太弱，也不会让语义分数直接压倒 POI logits。
            """
            aux_mean = aux_score.mean(dim=1, keepdim=True)
            aux_std = aux_score.std(dim=1, keepdim=True).clamp_min(1e-6)

            ref_std = ref_score.detach().std(dim=1, keepdim=True).clamp_min(1e-6)

            aux_score = (aux_score - aux_mean) / aux_std
            aux_score = aux_score * ref_std

            return aux_score

        # 1. POI logits 仍然作为主分数
        candidate_score = poi_logits

        # 2. semantic score：每个候选 POI 取其所属 semantic 类别的预测分数
        # sem_log_prob = F.log_softmax(semantic_logits, dim=-1)
        # sem_score = sem_log_prob[:, self.poi_semantic]
        # 2. semantic score：每个候选 POI 取其所属 semantic 类别的预测分数
        # temperature < 1 会让 semantic 分布更尖锐，增强高置信类别的区分度
        sem_temp = 0.7
        sem_prob = F.softmax(semantic_logits / sem_temp, dim=-1)
        sem_log_prob = torch.log(sem_prob + 1e-8)

        sem_score = sem_log_prob[:, self.poi_semantic]

        # 关键：对齐语义分数和 POI logits 的尺度
        sem_score = align_to_poi_scale(sem_score, poi_logits)

        candidate_score = candidate_score + semantic_weight * sem_score

        # 3. region score：每个候选 POI 取其所属 region 的预测分数
        if region_logits is not None and region_weight != 0:
            reg_log_prob = F.log_softmax(region_logits, dim=-1)
            reg_score = reg_log_prob[:, self.poi_region]

            # 关键：对齐 region 分数和 POI logits 的尺度
            reg_score = align_to_poi_scale(reg_score, poi_logits)

            candidate_score = candidate_score + region_weight * reg_score

        # 4. score 模式：只做软重排序，不硬过滤
        if candidate_mode == 'score':
            return candidate_score

        # 5. mask 模式：软重排序后只保留 top_m 个候选
        if top_m is not None and 0 < int(top_m) < self.input_size:
            top_idx = torch.topk(candidate_score, k=int(top_m), dim=-1).indices
            masked_score = torch.full_like(candidate_score, -1e9)
            candidate_score = masked_score.scatter(
                1,
                top_idx,
                candidate_score.gather(1, top_idx)
            )

        return candidate_score




'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        # (1, 200, 10)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c



