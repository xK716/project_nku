# import time
#
# import torch
# import torch.nn as nn
# import numpy as np
# from utils import *
# from network import Flashback
# from scipy.sparse import csr_matrix
#
#
# class FlashbackTrainer():
#     """ Instantiates Flashback module with spatial and temporal weight functions.
#     Performs loss computation and prediction.
#     """
#
#     def __init__(self, dataset_name, lambda_t, lambda_s, dropout, lambda_loc, lambda_user, use_weight, transition_graph,
#                  spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, log):
#         """ The hyper parameters to control spatial and temporal decay.
#         """
#         self.lambda_t = lambda_t
#         self.lambda_s = lambda_s
#         self.dropout = dropout
#         self.dataset_name = dataset_name
#
#         self.lambda_loc = lambda_loc
#         self.lambda_user = lambda_user
#         self.use_weight = use_weight
#         self.use_graph_user = use_graph_user
#         self.use_spatial_graph = use_spatial_graph
#         self.graph = transition_graph
#         self.spatial_graph = spatial_graph
#         self.friend_graph = friend_graph
#         self.interact_graph = interact_graph
#         self._log = log
#
#     def __str__(self):
#         return 'Use flashback training.'
#
#     def count_parameters(self):
#         param_count = 0
#         log_string(self._log, 'Trainable parameter list:')
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 log_string(self._log, name + ',   ' + str(param.numel()))
#                 param_count += param.numel()
#
#         return param_count
#
#     def Loss_l2(self):
#         base_params = dict(self.model.named_parameters())
#         loss_l2 = 0.
#         count = 0
#         for key, value in base_params.items():
#             if 'bias' not in key and 'pre_model' not in key:
#                 loss_l2 += torch.sum(value ** 2)
#                 count += value.nelement()
#         return loss_l2
#
#     def parameters(self):
#         return self.model.parameters()
#
#     def prepare(self, loc_count, user_count, region_count, slot_count, hidden_size, gru_factory, device,
#                 region_graph, region_dis, cl_decay_steps, pid_lat_lon):
#         def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
#             -(delta_t / 86400 * self.lambda_t))
#
#         def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))
#
#         self.loc_count = loc_count
#         self.region_count = region_count
#         self.slot_count = slot_count
#         self.pid_lat_lon = pid_lat_lon
#         self.lat_min, self.lat_max = self.pid_lat_lon[:, 0].min(), self.pid_lat_lon[:, 0].max()
#         self.lng_min, self.lng_max = self.pid_lat_lon[:, 1].min(), self.pid_lat_lon[:, 1].max()
#         self.cross_entropy_loss = nn.CrossEntropyLoss()
#         self.model = Flashback(loc_count, user_count, region_count, slot_count, hidden_size, f_t, f_s, gru_factory,
#                                self.lambda_loc,
#                                self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
#                                self.use_graph_user, self.use_spatial_graph, self.interact_graph, device, region_graph,
#                                region_dis, cl_decay_steps,
#                                self.lat_min, self.lat_max, self.lng_min, self.lng_max, self.dropout,
#                                self.dataset_name).to(device)
#
#     def evaluate(self, x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h, active_users):
#         """ takes a batch (users x location sequence)
#         then does the prediction and returns a list of user x sequence x location
#         describing the probabilities for each location at each position in the sequence.
#         t, s are temporal and spatial data related to the location sequence x
#         y_t, y_s are temporal and spatial data related to the target sequence y.
#         Flashback does not access y_t and y_s for prediction!
#         """
#
#         self.model.eval()
#         out, out_r, h = self.model(x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h,
#                                    active_users)
#
#         out = out.transpose(0, 1)
#         out_r = out_r.transpose(0, 1)
#         return out, out_r, h
#
#     def loss(self, x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h, active_users,
#              batches_seen, logits, logits_reg):
#         """ takes a batch (users x location sequence)
#         and corresponding targets in order to compute the training loss """
#
#         self.model.train()
#         out, out_r, h = self.model(x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h,
#                                    active_users, batches_seen)
#         out = out.view(-1, self.loc_count)
#         out_r = out_r.view(-1, self.region_count)
#
#         y = y.view(-1)
#         y_r = y_r.view(-1)
#         l = self.cross_entropy_loss(out, y)
#         l_r = self.cross_entropy_loss(out_r, y_r)
#
#         l_r, l_mar = focal_loss(out_r, y_r)
#
#         return l, l_r
#
#
#
#
#
#
#
#
#
#
# # --lambda-sem
# # 0.3
# # --sem-dim
# # 64
# # --use-candidate-filter
# # True
# # --candidate-topk
# # 200






import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, dataset_name, lambda_t, lambda_s, dropout, lambda_loc, lambda_user, use_weight, transition_graph,
                 spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, log):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.dropout = dropout
        self.dataset_name = dataset_name

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph
        self._log = log

    def __str__(self):
        return 'Use flashback training.'

    def count_parameters(self):
        param_count = 0
        log_string(self._log, 'Trainable parameter list:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                log_string(self._log, name + ',   ' + str(param.numel()))
                param_count += param.numel()

        return param_count

    def Loss_l2(self):
        base_params = dict(self.model.named_parameters())
        loss_l2 = 0.
        count = 0
        for key, value in base_params.items():
            if 'bias' not in key and 'pre_model' not in key:
                loss_l2 += torch.sum(value ** 2)
                count += value.nelement()
        return loss_l2

    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, region_count, slot_count, hidden_size, gru_factory, device,
                region_graph, region_dis, cl_decay_steps, pid_lat_lon,
                semantic_count=1, poi_semantic=None, poi_region=None, use_sem_branch=False,
                filter_candidate=False, topm_candidate=2000,
                candidate_sem_weight=0.6, candidate_region_weight=0.4,
                candidate_mode='mask', semantic_fusion_weight=0.5, sem_label_smoothing=0.0):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))

        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))

        self.loc_count = loc_count
        self.region_count = region_count
        self.slot_count = slot_count
        self.pid_lat_lon = pid_lat_lon
        self.semantic_count = int(semantic_count) if semantic_count is not None else 1
        self.use_sem_branch = bool(use_sem_branch and self.semantic_count > 1)
        self.filter_candidate = bool(filter_candidate)
        self.topm_candidate = int(topm_candidate)
        self.candidate_sem_weight = float(candidate_sem_weight)
        self.candidate_region_weight = float(candidate_region_weight)
        self.candidate_mode = str(candidate_mode)
        self.semantic_fusion_weight = float(semantic_fusion_weight)
        self.sem_label_smoothing = float(sem_label_smoothing)
        self.lat_min, self.lat_max = self.pid_lat_lon[:, 0].min(), self.pid_lat_lon[:, 0].max()
        self.lng_min, self.lng_max = self.pid_lat_lon[:, 1].min(), self.pid_lat_lon[:, 1].max()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, region_count, slot_count, hidden_size, f_t, f_s, gru_factory,
                               self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph, device, region_graph,
                               region_dis, cl_decay_steps,
                               self.lat_min, self.lat_max, self.lng_min, self.lng_max, self.dropout,
                               self.dataset_name, semantic_count=self.semantic_count,
                               poi_semantic=poi_semantic, poi_region=poi_region,
                               use_sem_branch=self.use_sem_branch,
                               semantic_fusion_weight=self.semantic_fusion_weight).to(device)


    def evaluate(self, x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h, active_users):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        """

        self.model.eval()
        out, out_r, out_s, h = self.model(x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h,
                                          active_users)

        out = out.transpose(0, 1)
        out_r = out_r.transpose(0, 1)
        out_s = out_s.transpose(0, 1) if out_s is not None else None
        return out, out_r, out_s, h

    def loss(self, x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h, active_users,
             batches_seen, logits, logits_reg):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model.train()
        out, out_r, out_s, h = self.model(x, t, t_slot, s, r, r_nv, r_s, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h,
                                          active_users, batches_seen)
        out = out.view(-1, self.loc_count)
        out_r = out_r.view(-1, self.region_count)
        out_s = out_s.view(-1, self.semantic_count) if out_s is not None else None

        y = y.view(-1)
        y_r = y_r.view(-1)
        l = self.cross_entropy_loss(out, y)

        # Keep DPRL's region focal loss: region transitions are imbalanced and repeated.
        l_r, l_mar = focal_loss(out_r, y_r)

        if self.use_sem_branch and out_s is not None:
            y_sem = self.model.poi_semantic[y].long()
            if self.sem_label_smoothing > 0:
                l_s = F.cross_entropy(out_s, y_sem, label_smoothing=self.sem_label_smoothing)
            else:
                l_s = self.cross_entropy_loss(out_s, y_sem)
        else:
            l_s = torch.zeros((), device=out.device)

        return l, l_r, l_s

    def filter_candidates(self, out, out_r=None, out_s=None):
        if not self.filter_candidate:
            return out
        return self.model.apply_semantic_candidate_filter(
            out, out_r, out_s,
            top_m=self.topm_candidate,
            semantic_weight=self.candidate_sem_weight,
            region_weight=self.candidate_region_weight,
            candidate_mode=self.candidate_mode,
        )

