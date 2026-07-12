# import random
# from enum import Enum
# import torch
# import pickle
# from torch.utils.data import Dataset
# import numpy as np
# import os
#
# class Split(Enum):
#     """ Defines whether to split for train or test.
#     """
#     TRAIN = 0
#     TEST = 1
#
#
# class Usage(Enum):
#     """
#     Each user has a different amount of sequences. The usage defines
#     how many sequences are used:
#
#     MAX: each sequence of any user is used (default)
#     MIN: only as many as the minimal user has
#     CUSTOM: up to a fixed amount if available.
#
#     The unused sequences are discarded. This setting applies after the train/test split.
#     """
#
#     MIN_SEQ_LENGTH = 0
#     MAX_SEQ_LENGTH = 1
#     CUSTOM = 2
#
#
# class PoiDataset(Dataset):
#     """
#     Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
#     "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
#     In order to fill the batch all the time we wrap around the available users: if an active user
#     runs out of locations we replace him with a new one. When there are no unused users available
#     we reuse already processed ones. This happens if a single user was way more active than the average user.
#     The batch guarantees that each sequence of each user was processed at least once.
#
#     This data management has the implication that some sequences might be processed twice (or more) per epoch.
#     During training you should call PoiDataset::shuffle_users before the start of a new epoch. This
#     leads to more stochastic as different sequences will be processed twice.
#     During testing you *have to* keep track of the already processed users.
#
#     Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user. We
#     work with a 80/20 train/test spilt, where test check-ins are strictly after training checkins. To obtain at least
#     one test sequence with label we require any user to have at least (5*<sequence-length>+1) checkins in total.
#     """
#
#     def reset(self):
#         # reset training state:
#         self.next_user_idx = 0  # current user index to add
#         self.active_users = []  # current active users
#         self.active_user_seq = []  # current active users sequences
#         self.user_permutation = []  # shuffle users during training
#
#         # set active users:
#         for i in range(self.batch_size):
#             self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
#             self.active_users.append(i)
#             self.active_user_seq.append(0)
#
#         # use 1:1 permutation:
#         for i in range(len(self.users)):
#             self.user_permutation.append(i)
#
#     def shuffle_users(self):
#         random.shuffle(self.user_permutation)
#         # reset active users:
#         self.next_user_idx = 0
#         self.active_users = []
#         self.active_user_seq = []
#         for i in range(self.batch_size):
#             self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
#             self.active_users.append(self.user_permutation[i])
#             self.active_user_seq.append(0)
#
#     def get_region_graph(self):
#         return self.region_graph
#
#     def get_user_region_graph(self):
#         return self.user_region_graph
#
#     def get_user_poi_graph(self):
#         return self.user_poi_graph
#
#     def __init__(self, users, times, time_slots, coords, locs, regions, regions_nv, regions_gps, sequence_length, batch_size, split, usage, loc_count, custom_seq_count, cluster, file_dir, dataset_name):
#         self.users = users
#         self.locs = locs
#         self.times = times
#         self.time_slots = time_slots
#         self.coords = coords
#         self.regions = regions
#         self.regions_nv = regions_nv
#         self.regions_gps = regions_gps
#         self.cluster = cluster
#
#         self.labels = []
#         self.lbl_times = []
#         self.lbl_time_slots = []
#         self.lbl_coords = []
#         self.lbl_regions = []
#         self.lbl_regions_nv = []
#         self.lbl_regions_gps = []
#
#         self.sequences = []
#         self.sequences_times = []
#         self.sequences_time_slots = []
#         self.sequences_coords = []
#         self.sequences_regions = []
#         self.sequences_regions_nv = []
#         self.sequences_regions_gps = []
#
#         self.sequences_labels = []
#         self.sequences_lbl_times = []
#         self.sequences_lbl_time_slots = []
#         self.sequences_lbl_coords = []
#         self.sequences_lbl_regions = []
#         self.sequences_lbl_regions_nv = []
#         self.sequences_lbl_regions_gps = []
#
#         self.sequences_count = []
#         self.Ps = []
#         self.Qs = torch.zeros(loc_count, 1)
#         self.usage = usage
#         self.batch_size = batch_size
#         self.loc_count = loc_count
#         self.custom_seq_count = custom_seq_count
#
#         self.freq = {key: 0 for key in range(loc_count)}
#         self.freq_reg = {key: 0 for key in range(cluster)}
#         self.sequences_freq = []
#         self.sequences_lbl_freq = []
#         self.sequences_freq_reg = []
#         self.sequences_lbl_freq_reg = []
#
#         self.reset()
#
#         # construct region transition graph
#         if not os.path.exists(os.path.join(file_dir, dataset_name + "_graph.pickle")):
#             self.region_graph = np.zeros((cluster, cluster), dtype=np.float32)
#             self.user_region_graph = np.zeros((len(users), cluster), dtype=np.float32)
#             self.user_poi_graph = np.zeros((len(users), loc_count), dtype=np.float32)
#         else:
#             with open(os.path.join(file_dir, dataset_name + "_graph.pickle"), 'rb') as f:
#                 self.region_graph = pickle.load(f)
#                 self.user_poi_graph = pickle.load(f)
#                 self.user_region_graph = pickle.load(f)
#
#         # collect locations:
#         for i in range(loc_count):
#             self.Qs[i, 0] = i
#
#         # align labels to locations (shift by one)
#         for i, loc in enumerate(locs):
#             self.locs[i] = loc[:-1]
#             self.labels.append(loc[1:])
#             self.lbl_times.append(self.times[i][1:])
#             self.lbl_time_slots.append(self.time_slots[i][1:])
#             self.lbl_coords.append(self.coords[i][1:])
#             self.lbl_regions.append(self.regions[i][1:])
#             self.lbl_regions_nv.append(self.regions_nv[i][1:])
#             self.lbl_regions_gps.append(self.regions_gps[i][1:])
#
#             self.times[i] = self.times[i][:-1]
#             self.time_slots[i] = self.time_slots[i][:-1]
#             self.coords[i] = self.coords[i][:-1]
#             self.regions[i] = self.regions[i][:-1]
#             self.regions_nv[i] = self.regions_nv[i][:-1]
#             self.regions_gps[i] = self.regions_gps[i][:-1]
#
#             train_thr = int(len(loc) * 0.8)
#             for j, (location_, region_) in enumerate(zip(loc[:train_thr], regions[i][:train_thr])):
#                 self.freq[location_] += 1
#                 self.freq_reg[region_] += 1
#
#         # split to training / test phase:
#         for i, (user, time, time_slot, coord, loc, region, region_nv, region_gps, label, lbl_time, lbl_time_slot, lbl_coord, lbl_region, lbl_region_nv, lbl_region_gps) in enumerate(
#                 zip(self.users, self.times, self.time_slots, self.coords, self.locs, self.regions, self.regions_nv, self.regions_gps, self.labels, self.lbl_times,
#                     self.lbl_time_slots, self.lbl_coords, self.lbl_regions, self.lbl_regions_nv, self.lbl_regions_gps)):
#             train_thr = int(len(loc) * 0.8)
#             if split == Split.TRAIN:
#                 self.locs[i] = loc[:train_thr]
#                 self.times[i] = time[:train_thr]
#                 self.time_slots[i] = time_slot[:train_thr]
#                 self.coords[i] = coord[:train_thr]
#                 self.regions[i] = region[:train_thr]
#                 self.regions_nv[i] = region_nv[:train_thr]
#                 self.regions_gps[i] = region_gps[:train_thr]
#                 if not os.path.exists(os.path.join(file_dir, dataset_name + "_graph.pickle")):
#                     # construct the region graph
#                     for j in range(len(self.regions[i])-1):
#                         from_, to_ = self.regions[i][j], self.regions[i][j+1]
#                         from_loc, to_loc = self.locs[i][j], self.regions[i][j+1]
#                         self.region_graph[from_][to_] += 1
#                         self.user_poi_graph[user][from_loc] += 1
#                         self.user_region_graph[user][from_] += 1
#                     self.user_poi_graph[user][to_loc] += 1
#                     self.user_region_graph[user][to_] += 1
#
#                 self.labels[i] = label[:train_thr]
#                 self.lbl_times[i] = lbl_time[:train_thr]
#                 self.lbl_time_slots[i] = lbl_time_slot[:train_thr]
#                 self.lbl_coords[i] = lbl_coord[:train_thr]
#                 self.lbl_regions[i] = lbl_region[:train_thr]
#                 self.lbl_regions_nv[i] = lbl_region_nv[:train_thr]
#                 self.lbl_regions_gps[i] = lbl_region_gps[:train_thr]
#
#             if split == Split.TEST:
#                 self.locs[i] = loc[train_thr:]
#                 self.times[i] = time[train_thr:]
#                 self.time_slots[i] = time_slot[train_thr:]
#                 self.coords[i] = coord[train_thr:]
#                 self.regions[i] = region[train_thr:]
#                 self.regions_nv[i] = region_nv[train_thr:]
#                 self.regions_gps[i] = region_gps[train_thr:]
#
#                 self.labels[i] = label[train_thr:]
#                 self.lbl_times[i] = lbl_time[train_thr:]
#                 self.lbl_time_slots[i] = lbl_time_slot[train_thr:]
#                 self.lbl_coords[i] = lbl_coord[train_thr:]
#                 self.lbl_regions[i] = lbl_region[train_thr:]
#                 self.lbl_regions_nv[i] = lbl_region_nv[train_thr:]
#                 self.lbl_regions_gps[i] = lbl_region_gps[train_thr:]
#
#         # split location and labels to sequences:
#         self.max_seq_count = 0
#         self.min_seq_count = 10000000
#         self.capacity = 0
#         for i, (time, time_slot, coord, loc, region, region_nv, region_gps, label, lbl_time, lbl_time_slot, lbl_coord, lbl_region, lbl_region_nv, lbl_region_gps) in enumerate(
#                 zip(self.times, self.time_slots, self.coords, self.locs, self.regions, self.regions_nv, self.regions_gps, self.labels, self.lbl_times,
#                     self.lbl_time_slots, self.lbl_coords, self.lbl_regions, self.lbl_regions_nv, self.lbl_regions_gps)):
#             seq_count = len(loc) // sequence_length
#             assert seq_count > 0, 'fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!'
#             seqs = []
#             seq_times = []
#             seq_time_slots = []
#             seq_coords = []
#             seq_regions = []
#             seq_regions_nv = []
#             seq_regions_gps = []
#             seq_freq = []
#             seq_freq_reg = []
#
#             seq_lbls = []
#             seq_lbl_times = []
#             seq_lbl_time_slots = []
#             seq_lbl_coords = []
#             seq_lbl_regions = []
#             seq_lbl_regions_nv = []
#             seq_lbl_regions_gps = []
#             seq_lbl_freq = []
#             seq_lbl_freq_reg = []
#
#             for j in range(seq_count):
#                 start = j * sequence_length
#                 end = (j + 1) * sequence_length
#                 seqs.append(loc[start:end])
#                 seq_times.append(time[start:end])
#                 seq_time_slots.append(time_slot[start:end])
#                 seq_coords.append(coord[start:end])
#                 seq_regions.append(region[start:end])
#                 seq_regions_nv.append(region_nv[start:end])
#                 seq_regions_gps.append(region_gps[start:end])
#
#                 seq_lbls.append(label[start:end])
#                 seq_lbl_times.append(lbl_time[start:end])
#                 seq_lbl_time_slots.append((lbl_time_slot[start:end]))
#                 seq_lbl_coords.append(lbl_coord[start:end])
#                 seq_lbl_regions.append(lbl_region[start:end])
#                 seq_lbl_regions_nv.append(lbl_region_nv[start:end])
#                 seq_lbl_regions_gps.append(lbl_region_gps[start:end])
#
#                 freq_loc = [self.freq[k] for k in loc[start:end]]
#                 seq_freq.append(freq_loc)
#                 freq_reg = [self.freq_reg[k] for k in region[start:end]]
#                 seq_freq_reg.append(freq_reg)
#                 freq_lbl_loc = [self.freq[k] for k in label[start:end]]
#                 seq_lbl_freq.append(freq_lbl_loc)
#                 freq_lbl_reg = [self.freq_reg[k] for k in lbl_region[start:end]]
#                 seq_lbl_freq_reg.append(freq_lbl_reg)
#
#             self.sequences.append(seqs)
#             self.sequences_times.append(seq_times)
#             self.sequences_time_slots.append(seq_time_slots)
#             self.sequences_coords.append(seq_coords)
#             self.sequences_regions.append(seq_regions)
#             self.sequences_regions_nv.append(seq_regions_nv)
#             self.sequences_regions_gps.append(seq_regions_gps)
#
#             self.sequences_labels.append(seq_lbls)
#             self.sequences_lbl_times.append(seq_lbl_times)
#             self.sequences_lbl_time_slots.append(seq_lbl_time_slots)
#             self.sequences_lbl_coords.append(seq_lbl_coords)
#             self.sequences_lbl_regions.append(seq_lbl_regions)
#             self.sequences_lbl_regions_nv.append(seq_lbl_regions_nv)
#             self.sequences_lbl_regions_gps.append(seq_lbl_regions_gps)
#
#             self.sequences_count.append(seq_count)
#             self.capacity += seq_count
#             self.max_seq_count = max(self.max_seq_count, seq_count)
#             self.min_seq_count = min(self.min_seq_count, seq_count)
#
#             self.sequences_freq.append(seq_freq)
#             self.sequences_lbl_freq.append(seq_lbl_freq)
#             self.sequences_freq_reg.append(seq_freq_reg)
#             self.sequences_lbl_freq_reg.append(seq_lbl_freq_reg)
#
#         # statistics
#         if self.usage == Usage.MIN_SEQ_LENGTH:
#             print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
#         if self.usage == Usage.MAX_SEQ_LENGTH:
#             print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__(), 'cluster:', self.cluster)
#         if self.usage == Usage.CUSTOM:
#             print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:',
#                   self.__len__())
#
#     def sequences_by_user(self, idx):
#         return self.sequences[idx]
#
#     def __len__(self):
#         """ Amount of available batches to process each sequence at least once.
#         """
#
#         if self.usage == Usage.MIN_SEQ_LENGTH:
#             # min times amount_of_user_batches:
#             return self.min_seq_count * (len(self.users) // self.batch_size)
#         if self.usage == Usage.MAX_SEQ_LENGTH:
#             # estimated capacity:
#             estimated = self.capacity // self.batch_size
#             return max(self.max_seq_count, estimated)
#         if self.usage == Usage.CUSTOM:
#             return self.custom_seq_count * (len(self.users) // self.batch_size)
#         raise ValueError()
#
#     def __getitem__(self, idx):
#         """ Against pytorch convention, we directly build a full batch inside __getitem__.
#         Use a batch_size of 1 in your pytorch data loader.
#
#         A batch consists of a list of active users,
#         their next location sequence with timestamps and coordinates.
#
#         y is the target location and y_t, y_s the targets timestamp and coordinates. Provided for
#         possible use.
#
#         reset_h is a flag which indicates when a new user has been replacing a previous user in the
#         batch. You should reset this users hidden state to initial value h_0.
#         """
#
#         seqs = []
#         times = []
#         time_slots = []
#         coords = []
#         regions = []
#         regions_nv = []
#         regions_gps = []
#
#         lbls = []
#         lbl_times = []
#         lbl_time_slots = []
#         lbl_coords = []
#         lbl_regions = []
#         lbl_regions_nv = []
#         lbl_regions_gps = []
#
#         freq = []
#         lbl_freq = []
#         freq_reg = []
#         lbl_freq_reg = []
#
#         reset_h = []
#         for i in range(self.batch_size):
#             i_user = self.active_users[i]
#             j = self.active_user_seq[i]
#             max_j = self.sequences_count[i_user]
#             if self.usage == Usage.MIN_SEQ_LENGTH:
#                 max_j = self.min_seq_count
#             if self.usage == Usage.CUSTOM:
#                 max_j = min(max_j, self.custom_seq_count)
#             if j >= max_j:
#                 i_user = self.user_permutation[self.next_user_idx]
#                 j = 0
#                 self.active_users[i] = i_user
#                 self.active_user_seq[i] = j
#                 self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
#                 while self.user_permutation[self.next_user_idx] in self.active_users:
#                     self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
#                 # TODO: throw exception if wrapped around!
#             # use this user:
#             reset_h.append(j == 0)
#             seqs.append(torch.tensor(self.sequences[i_user][j]))
#             times.append(torch.tensor(self.sequences_times[i_user][j]))
#             time_slots.append(torch.tensor(self.sequences_time_slots[i_user][j]))
#             coords.append(torch.tensor(self.sequences_coords[i_user][j]))
#             regions.append(torch.tensor(self.sequences_regions[i_user][j]))
#             regions_nv.append(torch.tensor(self.sequences_regions_nv[i_user][j]))
#             regions_gps.append(torch.tensor(self.sequences_regions_gps[i_user][j]))
#
#             lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
#             lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
#             lbl_time_slots.append(torch.tensor(self.sequences_lbl_time_slots[i_user][j]))
#             lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
#             lbl_regions.append(torch.tensor(self.sequences_lbl_regions[i_user][j]))
#             lbl_regions_nv.append(torch.tensor(self.sequences_lbl_regions_nv[i_user][j]))
#             lbl_regions_gps.append(torch.tensor(self.sequences_lbl_regions_gps[i_user][j]))
#
#             freq.append(torch.tensor(self.sequences_freq[i_user][j]))
#             lbl_freq.append(torch.tensor(self.sequences_lbl_freq[i_user][j]))
#             freq_reg.append(torch.tensor(self.sequences_freq_reg[i_user][j]))
#             lbl_freq_reg.append(torch.tensor(self.sequences_lbl_freq_reg[i_user][j]))
#
#             self.active_user_seq[i] += 1
#
#         x = torch.stack(seqs, dim=1)
#         t = torch.stack(times, dim=1)
#         t_slot = torch.stack(time_slots, dim=1)
#         s = torch.stack(coords, dim=1)
#         r = torch.stack(regions, dim=1)
#         r_nv = torch.stack(regions_nv, dim=1)
#         r_s = torch.stack(regions_gps, dim=1)
#
#         y = torch.stack(lbls, dim=1)
#         y_t = torch.stack(lbl_times, dim=1)
#         y_t_slot = torch.stack(lbl_time_slots, dim=1)
#         y_s = torch.stack(lbl_coords, dim=1)
#         y_r = torch.stack(lbl_regions, dim=1)
#         y_r_nv = torch.stack(lbl_regions_nv, dim=1)
#         y_r_s = torch.stack(lbl_regions_gps, dim=1)
#
#         f = torch.stack(freq, dim=1)
#         y_f = torch.stack(lbl_freq, dim=1)
#         f_r = torch.stack(freq, dim=1)
#         y_f_r = torch.stack(lbl_freq, dim=1)
#
#         return x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, reset_h, torch.tensor(self.active_users)


import random
from enum import Enum
import torch
import pickle
from torch.utils.data import Dataset
import numpy as np
import os


class Split(Enum):
    """ Defines whether to split for train or test.
    """
    TRAIN = 0
    TEST = 1


class Usage(Enum):
    """
    Each user has a different amount of sequences. The usage defines
    how many sequences are used:

    MAX: each sequence of any user is used (default)
    MIN: only as many as the minimal user has
    CUSTOM: up to a fixed amount if available.

    The unused sequences are discarded. This setting applies after the train/test split.
    """

    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


class PoiDataset(Dataset):
    """
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
    "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users: if an active user
    runs out of locations we replace him with a new one. When there are no unused users available
    we reuse already processed ones. This happens if a single user was way more active than the average user.
    The batch guarantees that each sequence of each user was processed at least once.

    This data management has the implication that some sequences might be processed twice (or more) per epoch.
    During training you should call PoiDataset::shuffle_users before the start of a new epoch. This
    leads to more stochastic as different sequences will be processed twice.
    During testing you *have to* keep track of the already processed users.

    Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user. We
    work with a 80/20 train/test spilt, where test check-ins are strictly after training checkins. To obtain at least
    one test sequence with label we require any user to have at least (5*<sequence-length>+1) checkins in total.
    """

    def reset(self):
        # reset training state:
        self.next_user_idx = 0  # current user index to add
        self.active_users = []  # current active users
        self.active_user_seq = []  # current active users sequences
        self.user_permutation = []  # shuffle users during training

        # set active users:
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)

            # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)

    def shuffle_users(self):
        random.shuffle(self.user_permutation)
        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(self.user_permutation[i])
            self.active_user_seq.append(0)

    def get_region_graph(self):
        return self.region_graph

    def get_user_region_graph(self):
        return self.user_region_graph

    def get_user_poi_graph(self):
        return self.user_poi_graph

    def __init__(self, users, times, time_slots, coords, locs, regions, regions_nv, regions_gps, sequence_length,
                 batch_size, split, usage, loc_count, custom_seq_count, cluster, file_dir, dataset_name):
        self.users = users
        self.locs = locs
        self.times = times
        self.time_slots = time_slots
        self.coords = coords
        self.regions = regions
        self.regions_nv = regions_nv
        self.regions_gps = regions_gps
        self.cluster = cluster

        self.labels = []
        self.lbl_times = []
        self.lbl_time_slots = []
        self.lbl_coords = []
        self.lbl_regions = []
        self.lbl_regions_nv = []
        self.lbl_regions_gps = []

        self.sequences = []
        self.sequences_times = []
        self.sequences_time_slots = []
        self.sequences_coords = []
        self.sequences_regions = []
        self.sequences_regions_nv = []
        self.sequences_regions_gps = []

        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_time_slots = []
        self.sequences_lbl_coords = []
        self.sequences_lbl_regions = []
        self.sequences_lbl_regions_nv = []
        self.sequences_lbl_regions_gps = []

        self.sequences_count = []
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.batch_size = batch_size
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count

        self.freq = {key: 0 for key in range(loc_count)}
        self.freq_reg = {key: 0 for key in range(cluster)}
        self.sequences_freq = []
        self.sequences_lbl_freq = []
        self.sequences_freq_reg = []
        self.sequences_lbl_freq_reg = []

        self.reset()

        # construct region transition graph
        if not os.path.exists(os.path.join(file_dir, dataset_name + "_graph.pickle")):
            self.region_graph = np.zeros((cluster, cluster), dtype=np.float32)
            self.user_region_graph = np.zeros((len(users), cluster), dtype=np.float32)
            self.user_poi_graph = np.zeros((len(users), loc_count), dtype=np.float32)
        else:
            with open(os.path.join(file_dir, dataset_name + "_graph.pickle"), 'rb') as f:
                self.region_graph = pickle.load(f)
                self.user_poi_graph = pickle.load(f)
                self.user_region_graph = pickle.load(f)

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i

        # align labels to locations (shift by one)
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]
            self.labels.append(loc[1:])
            self.lbl_times.append(self.times[i][1:])
            self.lbl_time_slots.append(self.time_slots[i][1:])
            self.lbl_coords.append(self.coords[i][1:])
            self.lbl_regions.append(self.regions[i][1:])
            self.lbl_regions_nv.append(self.regions_nv[i][1:])
            self.lbl_regions_gps.append(self.regions_gps[i][1:])

            self.times[i] = self.times[i][:-1]
            self.time_slots[i] = self.time_slots[i][:-1]
            self.coords[i] = self.coords[i][:-1]
            self.regions[i] = self.regions[i][:-1]
            self.regions_nv[i] = self.regions_nv[i][:-1]
            self.regions_gps[i] = self.regions_gps[i][:-1]

            train_thr = int(len(loc) * 0.8)
            for j, (location_, region_) in enumerate(zip(loc[:train_thr], regions[i][:train_thr])):
                self.freq[location_] += 1
                self.freq_reg[region_] += 1

        # split to training / test phase:
        for i, (
        user, time, time_slot, coord, loc, region, region_nv, region_gps, label, lbl_time, lbl_time_slot, lbl_coord,
        lbl_region, lbl_region_nv, lbl_region_gps) in enumerate(
                zip(self.users, self.times, self.time_slots, self.coords, self.locs, self.regions, self.regions_nv,
                    self.regions_gps, self.labels, self.lbl_times,
                    self.lbl_time_slots, self.lbl_coords, self.lbl_regions, self.lbl_regions_nv, self.lbl_regions_gps)):
            train_thr = int(len(loc) * 0.8)
            if split == Split.TRAIN:
                self.locs[i] = loc[:train_thr]
                self.times[i] = time[:train_thr]
                self.time_slots[i] = time_slot[:train_thr]
                self.coords[i] = coord[:train_thr]
                self.regions[i] = region[:train_thr]
                self.regions_nv[i] = region_nv[:train_thr]
                self.regions_gps[i] = region_gps[:train_thr]
                if not os.path.exists(os.path.join(file_dir, dataset_name + "_graph.pickle")):
                    # construct the region graph
                    for j in range(len(self.regions[i]) - 1):
                        from_, to_ = self.regions[i][j], self.regions[i][j + 1]
                        from_loc, to_loc = self.locs[i][j], self.regions[i][j + 1]
                        self.region_graph[from_][to_] += 1
                        self.user_poi_graph[user][from_loc] += 1
                        self.user_region_graph[user][from_] += 1
                    self.user_poi_graph[user][to_loc] += 1
                    self.user_region_graph[user][to_] += 1

                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_time_slots[i] = lbl_time_slot[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]
                self.lbl_regions[i] = lbl_region[:train_thr]
                self.lbl_regions_nv[i] = lbl_region_nv[:train_thr]
                self.lbl_regions_gps[i] = lbl_region_gps[:train_thr]

            if split == Split.TEST:
                self.locs[i] = loc[train_thr:]
                self.times[i] = time[train_thr:]
                self.time_slots[i] = time_slot[train_thr:]
                self.coords[i] = coord[train_thr:]
                self.regions[i] = region[train_thr:]
                self.regions_nv[i] = region_nv[train_thr:]
                self.regions_gps[i] = region_gps[train_thr:]

                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_time_slots[i] = lbl_time_slot[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]
                self.lbl_regions[i] = lbl_region[train_thr:]
                self.lbl_regions_nv[i] = lbl_region_nv[train_thr:]
                self.lbl_regions_gps[i] = lbl_region_gps[train_thr:]

        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        self.capacity = 0
        for i, (time, time_slot, coord, loc, region, region_nv, region_gps, label, lbl_time, lbl_time_slot, lbl_coord,
                lbl_region, lbl_region_nv, lbl_region_gps) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs, self.regions, self.regions_nv,
                    self.regions_gps, self.labels, self.lbl_times,
                    self.lbl_time_slots, self.lbl_coords, self.lbl_regions, self.lbl_regions_nv, self.lbl_regions_gps)):
            seq_count = len(loc) // sequence_length
            assert seq_count > 0, 'fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!'
            seqs = []
            seq_times = []
            seq_time_slots = []
            seq_coords = []
            seq_regions = []
            seq_regions_nv = []
            seq_regions_gps = []
            seq_freq = []
            seq_freq_reg = []

            seq_lbls = []
            seq_lbl_times = []
            seq_lbl_time_slots = []
            seq_lbl_coords = []
            seq_lbl_regions = []
            seq_lbl_regions_nv = []
            seq_lbl_regions_gps = []
            seq_lbl_freq = []
            seq_lbl_freq_reg = []

            for j in range(seq_count):
                start = j * sequence_length
                end = (j + 1) * sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_time_slots.append(time_slot[start:end])
                seq_coords.append(coord[start:end])
                seq_regions.append(region[start:end])
                seq_regions_nv.append(region_nv[start:end])
                seq_regions_gps.append(region_gps[start:end])

                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_time_slots.append((lbl_time_slot[start:end]))
                seq_lbl_coords.append(lbl_coord[start:end])
                seq_lbl_regions.append(lbl_region[start:end])
                seq_lbl_regions_nv.append(lbl_region_nv[start:end])
                seq_lbl_regions_gps.append(lbl_region_gps[start:end])

                freq_loc = [self.freq[k] for k in loc[start:end]]
                seq_freq.append(freq_loc)
                freq_reg = [self.freq_reg[k] for k in region[start:end]]
                seq_freq_reg.append(freq_reg)
                freq_lbl_loc = [self.freq[k] for k in label[start:end]]
                seq_lbl_freq.append(freq_lbl_loc)
                freq_lbl_reg = [self.freq_reg[k] for k in lbl_region[start:end]]
                seq_lbl_freq_reg.append(freq_lbl_reg)

            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_time_slots.append(seq_time_slots)
            self.sequences_coords.append(seq_coords)
            self.sequences_regions.append(seq_regions)
            self.sequences_regions_nv.append(seq_regions_nv)
            self.sequences_regions_gps.append(seq_regions_gps)

            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_time_slots.append(seq_lbl_time_slots)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_lbl_regions.append(seq_lbl_regions)
            self.sequences_lbl_regions_nv.append(seq_lbl_regions_nv)
            self.sequences_lbl_regions_gps.append(seq_lbl_regions_gps)

            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)

            self.sequences_freq.append(seq_freq)
            self.sequences_lbl_freq.append(seq_lbl_freq)
            self.sequences_freq_reg.append(seq_freq_reg)
            self.sequences_lbl_freq_reg.append(seq_lbl_freq_reg)

        # statistics
        if self.usage == Usage.MIN_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.MAX_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__(),
                  'cluster:', self.cluster)
        if self.usage == Usage.CUSTOM:
            print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:',
                  self.__len__())

    def sequences_by_user(self, idx):
        return self.sequences[idx]

    def __len__(self):
        """ Amount of available batches to process each sequence at least once.
        """

        if self.usage == Usage.MIN_SEQ_LENGTH:
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if self.usage == Usage.MAX_SEQ_LENGTH:
            # estimated capacity:
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(self, idx):
        """ Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.

        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.

        y is the target location and y_t, y_s the targets timestamp and coordinates. Provided for
        possible use.

        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        """

        seqs = []
        times = []
        time_slots = []
        coords = []
        regions = []
        regions_nv = []
        regions_gps = []

        lbls = []
        lbl_times = []
        lbl_time_slots = []
        lbl_coords = []
        lbl_regions = []
        lbl_regions_nv = []
        lbl_regions_gps = []

        freq = []
        lbl_freq = []
        freq_reg = []
        lbl_freq_reg = []

        reset_h = []
        for i in range(self.batch_size):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if self.usage == Usage.MIN_SEQ_LENGTH:
                max_j = self.min_seq_count
            if self.usage == Usage.CUSTOM:
                max_j = min(max_j, self.custom_seq_count)
            if j >= max_j:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            time_slots.append(torch.tensor(self.sequences_time_slots[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            regions.append(torch.tensor(self.sequences_regions[i_user][j]))
            regions_nv.append(torch.tensor(self.sequences_regions_nv[i_user][j]))
            regions_gps.append(torch.tensor(self.sequences_regions_gps[i_user][j]))

            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_time_slots.append(torch.tensor(self.sequences_lbl_time_slots[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            lbl_regions.append(torch.tensor(self.sequences_lbl_regions[i_user][j]))
            lbl_regions_nv.append(torch.tensor(self.sequences_lbl_regions_nv[i_user][j]))
            lbl_regions_gps.append(torch.tensor(self.sequences_lbl_regions_gps[i_user][j]))

            freq.append(torch.tensor(self.sequences_freq[i_user][j]))
            lbl_freq.append(torch.tensor(self.sequences_lbl_freq[i_user][j]))
            freq_reg.append(torch.tensor(self.sequences_freq_reg[i_user][j]))
            lbl_freq_reg.append(torch.tensor(self.sequences_lbl_freq_reg[i_user][j]))

            self.active_user_seq[i] += 1

        x = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        t_slot = torch.stack(time_slots, dim=1)
        s = torch.stack(coords, dim=1)
        r = torch.stack(regions, dim=1)
        r_nv = torch.stack(regions_nv, dim=1)
        r_s = torch.stack(regions_gps, dim=1)

        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_t_slot = torch.stack(lbl_time_slots, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)
        y_r = torch.stack(lbl_regions, dim=1)
        y_r_nv = torch.stack(lbl_regions_nv, dim=1)
        y_r_s = torch.stack(lbl_regions_gps, dim=1)

        f = torch.stack(freq, dim=1)
        y_f = torch.stack(lbl_freq, dim=1)
        f_r = torch.stack(freq, dim=1)
        y_f_r = torch.stack(lbl_freq, dim=1)

        return x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, reset_h, torch.tensor(
            self.active_users)
