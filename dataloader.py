# import os.path
# import sys
# import pickle
# from datetime import datetime, timedelta
# from sklearn.cluster import KMeans
# from dataset import PoiDataset, Usage
# import numpy as np
# from utils import readEmbedFile
# from tqdm import tqdm
# import torch
#
# class PoiDataloader():
#     """ Creates datasets from our prepared Gowalla/Foursquare data files.
#     The file consist of one check-in per line in the following format (tab separated):
#
#     <user-id> <timestamp> <latitude> <longitude> <location-id>
#
#     Check-ins for the same user have to be on continuous lines.
#     Ids for users and locations are recreated and continuous from 0.
#     """
#
#     def __init__(self, max_users=0, min_checkins=0):
#         """ max_users limits the amount of users to load.
#         min_checkins discards users with less than this amount of checkins.
#         """
#
#         self.max_users = max_users
#         self.min_checkins = min_checkins
#
#         self.user2id = {}
#         self.poi2id = {}
#         self.poi2gps = {}
#         self.poi2freq = {}
#         self.region2freq = {}
#
#         self.users = []
#         self.times = []
#         self.time_slots = []
#         self.coords = []
#         self.locs = []
#
#     def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
#         return PoiDataset(self.users.copy(),
#                           self.times.copy(),
#                           self.time_slots.copy(),
#                           self.coords.copy(),
#                           self.locs.copy(),
#                           self.regions.copy(),
#                           self.regions_nv.copy(),
#                           self.regions_gps.copy(),
#                           sequence_length,
#                           batch_size,
#                           split,
#                           usage,
#                           len(self.poi2id),
#                           custom_seq_count,
#                           self.cluster,
#                           self.file_dir,
#                           self.dataset_name
#                           )
#
#     def user_count(self):
#         return len(self.users)
#
#     def locations(self):
#         return len(self.poi2id)
#
#     def checkins_count(self):
#         count = 0
#         for loc in self.locs:
#             count += len(loc)
#         return count
#
#     # def region_count(self):
#     #     return self.cluster
#
#     def read(self, file, offsetfile, cluster, time_slot):
#         if not os.path.isfile(file):
#             print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
#             sys.exit(1)
#         file_dir = os.path.dirname(file)
#         dataset_name = 'gowalla' if 'gowalla' in file else '4sq'
#         file_suf = '{}_{}_{}.pickle'.format('gowalla', str(cluster), str(time_slot)) if 'gowalla' in file else '{}_{}_{}.pickle'.format('4sq', str(cluster), str(time_slot))
#         self.file_dir = file_dir
#         self.dataset_name = dataset_name
#         save_file = os.path.join(file_dir, file_suf)
#         self.cluster = cluster
#         print("Start loading file {}...".format(save_file))
#         try:
#             with open(save_file, 'rb') as f:
#                 self.user2id = pickle.load(f)
#                 self.poi2id = pickle.load(f)
#                 self.poi2gps = pickle.load(f)
#                 self.poi2region = pickle.load(f)
#                 self.poi2region_nv = pickle.load(f)
#                 self.region2gps = pickle.load(f)
#                 self.users = pickle.load(f)
#                 self.times = pickle.load(f)
#                 self.time_slots = pickle.load(f)
#                 self.coords = pickle.load(f)
#                 self.locs = pickle.load(f)
#                 self.regions = pickle.load(f)
#                 self.regions_nv = pickle.load(f)
#                 self.regions_gps = pickle.load(f)
#             print("Successfully loading files...")
#         except:
#             print("Start generating file {}...".format(save_file))
#             # collect all users with min checkins:
#             self.read_users(file)
#             # collect checkins for all collected users:
#             self.read_pois(file, offsetfile, time_slot)
#             self.construct_region()
#             with open(save_file, 'wb') as f:
#                 pickle.dump(self.user2id, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.poi2id, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.poi2gps, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.poi2region, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.poi2region_nv, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.region2gps, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.users, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.times, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.time_slots, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.coords, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.locs, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.regions, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.regions_nv, f, pickle.HIGHEST_PROTOCOL)
#                 pickle.dump(self.regions_gps, f, pickle.HIGHEST_PROTOCOL)
#             print("Successfully generating files...")
#
#
#     def read_users(self, file):
#         f = open(file, 'r')
#         lines = f.readlines()
#
#         prev_user = int(lines[0].split('\t')[0])
#         visit_cnt = 0
#         for i, line in tqdm(enumerate(lines), total=len(lines), desc="read users"):
#             tokens = line.strip().split('\t')
#             user = int(tokens[0])
#             if user == prev_user:
#                 visit_cnt += 1
#             else:
#                 if visit_cnt >= self.min_checkins:
#                     self.user2id[prev_user] = len(self.user2id)
#                 # else:
#                 #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
#                 prev_user = user
#                 visit_cnt = 1
#                 if 0 < self.max_users <= len(self.user2id):
#                     break  # restrict to max users
#
#     def read_pois(self, file, offsetfile, stamp_num):
#         f = open(file, 'r')
#         lines = f.readlines()
#         f2 = open(offsetfile, 'r')
#         offsets = f2.readlines()
#
#         # store location ids
#         user_time = []
#         user_coord = []
#         user_loc = []
#         user_time_slot = []
#
#         prev_user = int(lines[0].split('\t')[0])
#         prev_user = self.user2id.get(prev_user)  # from 0
#         for i, (line, offset) in tqdm(enumerate(zip(lines, offsets)), total=len(lines), desc="read pois"):
#             tokens = line.strip().split('\t')
#             user = int(tokens[0])
#             if self.user2id.get(user) is None:
#                 continue  # user is not of interest(inactive user)
#             user = self.user2id.get(user)  # from 0
#             time_unit = datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=int(offset))
#             time = (time_unit - datetime(1970, 1, 1)).total_seconds()  # unix seconds
#             if stamp_num == 168:
#                 time_slot = time_unit.weekday() * 24 + time_unit.hour
#             elif stamp_num == 96:
#                 time_slot = int(time_unit.weekday() >= 5) * 48 + time_unit.hour * 2 + int(time_unit.minute > 30)
#             elif stamp_num == 48:
#                 time_slot = int(time_unit.weekday() >= 5) * 24 + time_unit.hour
#             elif stamp_num == 24:
#                 time_slot = time_unit.hour
#             lat = float(tokens[2])
#             long = float(tokens[3])
#             coord = (lat, long)
#
#             location = int(tokens[4])  # location nr
#             if self.poi2id.get(location) is None:  # get-or-set locations
#                 self.poi2id[location] = len(self.poi2id)
#                 self.poi2gps[self.poi2id[location]] = coord
#             if self.poi2freq.get(self.poi2id[location]) is None:
#                 self.poi2freq[self.poi2id[location]] = 1
#             else:
#                 self.poi2freq[self.poi2id[location]] += 1
#             location = self.poi2id.get(location)  # from 0
#             # freq =  self.poi2freq.get(location)
#
#             if user == prev_user:
#                 # Because the check-ins for every user is sorted in descending chronological order in the file
#                 user_time.insert(0, time)  # insert in front!
#                 user_time_slot.insert(0, time_slot)
#                 user_coord.insert(0, coord)
#                 user_loc.insert(0, location)
#             else:
#                 self.users.append(prev_user)
#                 self.times.append(user_time)
#                 self.time_slots.append(user_time_slot)
#                 self.coords.append(user_coord)
#                 self.locs.append(user_loc)
#                 # print(len(user_time) == len(user_time_slot) == len(user_loc) == len(user_coord))
#                 # restart:
#                 prev_user = user
#                 user_time = [time]
#                 user_time_slot = [time_slot]
#                 user_coord = [coord]
#                 user_loc = [location]
#
#         # process also the latest user in the for loop
#         self.users.append(prev_user)
#         self.times.append(user_time)
#         self.time_slots.append(user_time_slot)
#         self.coords.append(user_coord)
#         self.locs.append(user_loc)
#
#     def construct_region(self):
#         # construct the mapping from poi to region using KMeans algorithm
#         poi_ids = list(self.poi2gps.keys())
#         gps_coordinates = list(self.poi2gps.values())
#         kmeans = KMeans(n_clusters=self.cluster)
#         kmeans.fit(gps_coordinates)
#         labels = kmeans.labels_
#         centers = kmeans.cluster_centers_
#         self.poi2region = {poi_id: labels[i] for i, poi_id in enumerate(poi_ids)}
#         self.regions = [[self.poi2region[loc] for loc in self.locs[i]] for i in range(len(self.locs))]
#         self.region2gps = {}
#         for region, feature in enumerate(centers):
#             if region not in self.region2gps:
#                 self.region2gps[region] = feature.tolist()
#         self.regions_gps = [[self.region2gps[self.poi2region[loc]] for loc in self.locs[i]] for i in range(len(self.locs))]
#         self.poi2region_nv = self.poi2region
#         self.regions_nv = self.regions
#





import os.path
import sys
import pickle
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from dataset import PoiDataset, Usage
import numpy as np
from utils import readEmbedFile
from tqdm import tqdm
import torch


class PoiDataloader():
    """ Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continuous lines.
    Ids for users and locations are recreated and continuous from 0.
    """

    def __init__(self, max_users=0, min_checkins=0):
        """ max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        """

        self.max_users = max_users
        self.min_checkins = min_checkins

        self.user2id = {}
        self.poi2id = {}
        self.poi2gps = {}
        self.poi2freq = {}
        self.region2freq = {}
        # Sem-DPRL: optional POI semantic/category labels.
        self.poi2semantic = {}
        self.semantic2id = {}
        self.semantic_count = 1
        self.semantics = []
        self._poi_raw_semantic = {}

        self.users = []
        self.times = []
        self.time_slots = []
        self.coords = []
        self.locs = []

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        return PoiDataset(self.users.copy(),
                          self.times.copy(),
                          self.time_slots.copy(),
                          self.coords.copy(),
                          self.locs.copy(),
                          self.regions.copy(),
                          self.regions_nv.copy(),
                          self.regions_gps.copy(),
                          sequence_length,
                          batch_size,
                          split,
                          usage,
                          len(self.poi2id),
                          custom_seq_count,
                          self.cluster,
                          self.file_dir,
                          self.dataset_name
                          )

    def user_count(self):
        return len(self.users)

    def locations(self):
        return len(self.poi2id)

    def checkins_count(self):
        count = 0
        for loc in self.locs:
            count += len(loc)
        return count

    # def region_count(self):
    #     return self.cluster

    def read(self, file, offsetfile, cluster, time_slot, semantic_file='', semantic_fallback='region'):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)
        file_dir = os.path.dirname(file)
        dataset_name = 'gowalla' if 'gowalla' in file else '4sq'
        file_suf = '{}_{}_{}.pickle'.format('gowalla', str(cluster),
                                            str(time_slot)) if 'gowalla' in file else '{}_{}_{}.pickle'.format('4sq',
                                                                                                               str(cluster),
                                                                                                               str(time_slot))
        self.file_dir = file_dir
        self.dataset_name = dataset_name
        save_file = os.path.join(file_dir, file_suf)
        self.cluster = cluster
        print("Start loading file {}...".format(save_file))
        try:
            with open(save_file, 'rb') as f:
                self.user2id = pickle.load(f)
                self.poi2id = pickle.load(f)
                self.poi2gps = pickle.load(f)
                self.poi2region = pickle.load(f)
                self.poi2region_nv = pickle.load(f)
                self.region2gps = pickle.load(f)
                self.users = pickle.load(f)
                self.times = pickle.load(f)
                self.time_slots = pickle.load(f)
                self.coords = pickle.load(f)
                self.locs = pickle.load(f)
                self.regions = pickle.load(f)
                self.regions_nv = pickle.load(f)
                self.regions_gps = pickle.load(f)
                # Newer caches may contain semantic labels; older caches simply skip this part.
                try:
                    self.poi2semantic = pickle.load(f)
                    self.semantic2id = pickle.load(f)
                    self.semantic_count = pickle.load(f)
                    self.semantics = pickle.load(f)
                except EOFError:
                    pass
            self.attach_semantics(semantic_file=semantic_file, fallback=semantic_fallback)
            print("Successfully loading files...")
        except:
            print("Start generating file {}...".format(save_file))
            # collect all users with min checkins:
            self.read_users(file)
            # collect checkins for all collected users:
            self.read_pois(file, offsetfile, time_slot)
            self.construct_region()
            self.attach_semantics(semantic_file=semantic_file, fallback=semantic_fallback)
            with open(save_file, 'wb') as f:
                pickle.dump(self.user2id, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.poi2id, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.poi2gps, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.poi2region, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.poi2region_nv, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.region2gps, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.users, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.times, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.time_slots, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.coords, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.locs, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.regions, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.regions_nv, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.regions_gps, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.poi2semantic, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.semantic2id, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.semantic_count, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.semantics, f, pickle.HIGHEST_PROTOCOL)
            print("Successfully generating files...")

    def read_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in tqdm(enumerate(lines), total=len(lines), desc="read users"):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if 0 < self.max_users <= len(self.user2id):
                    break  # restrict to max users

    def read_pois(self, file, offsetfile, stamp_num):
        f = open(file, 'r')
        lines = f.readlines()
        f2 = open(offsetfile, 'r')
        offsets = f2.readlines()

        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        user_time_slot = []

        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # from 0
        for i, (line, offset) in tqdm(enumerate(zip(lines, offsets)), total=len(lines), desc="read pois"):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue  # user is not of interest(inactive user)
            user = self.user2id.get(user)  # from 0
            time_unit = datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=int(offset))
            time = (time_unit - datetime(1970, 1, 1)).total_seconds()  # unix seconds
            if stamp_num == 168:
                time_slot = time_unit.weekday() * 24 + time_unit.hour
            elif stamp_num == 96:
                time_slot = int(time_unit.weekday() >= 5) * 48 + time_unit.hour * 2 + int(time_unit.minute > 30)
            elif stamp_num == 48:
                time_slot = int(time_unit.weekday() >= 5) * 24 + time_unit.hour
            elif stamp_num == 24:
                time_slot = time_unit.hour
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)

            location = int(tokens[4])  # location nr
            raw_location = location
            raw_semantic = tokens[5] if len(tokens) > 5 else None
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
                self.poi2gps[self.poi2id[location]] = coord
                if raw_semantic is not None:
                    self._poi_raw_semantic[self.poi2id[location]] = raw_semantic
            if self.poi2freq.get(self.poi2id[location]) is None:
                self.poi2freq[self.poi2id[location]] = 1
            else:
                self.poi2freq[self.poi2id[location]] += 1
            location = self.poi2id.get(location)  # from 0
            # freq =  self.poi2freq.get(location)

            if user == prev_user:
                # Because the check-ins for every user is sorted in descending chronological order in the file
                user_time.insert(0, time)  # insert in front!
                user_time_slot.insert(0, time_slot)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.time_slots.append(user_time_slot)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                # print(len(user_time) == len(user_time_slot) == len(user_loc) == len(user_coord))
                # restart:
                prev_user = user
                user_time = [time]
                user_time_slot = [time_slot]
                user_coord = [coord]
                user_loc = [location]

        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.time_slots.append(user_time_slot)
        self.coords.append(user_coord)
        self.locs.append(user_loc)

    def _semantic_to_id(self, semantic_value):
        """Map arbitrary semantic/category text to a compact integer id."""
        key = str(semantic_value).strip()
        if key == '':
            key = 'UNK'
        if key not in self.semantic2id:
            self.semantic2id[key] = len(self.semantic2id)
        return self.semantic2id[key]

    def _load_semantic_file(self, semantic_file):
        """Load external POI semantic labels.

        Supported line formats:
          raw_poi_id<TAB>semantic_id_or_category
          internal_poi_id<TAB>semantic_id_or_category
          raw_poi_id semantic_id_or_category
        """
        if not semantic_file:
            return {}
        path = semantic_file
        if not os.path.isabs(path) and not os.path.isfile(path):
            path = os.path.join(self.file_dir, semantic_file)
        if not os.path.isfile(path):
            print('[Warning]: semantic file not found: {}. Use fallback semantics.'.format(path))
            return {}

        semantic_map = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', '\t').split()
                if len(parts) < 2:
                    continue
                poi_token, sem_token = parts[0], parts[1]
                try:
                    poi_int = int(poi_token)
                except ValueError:
                    continue

                internal_id = None
                # Prefer raw POI id mapping, then fall back to already internal id.
                if poi_int in self.poi2id:
                    internal_id = self.poi2id[poi_int]
                elif 0 <= poi_int < len(self.poi2id):
                    internal_id = poi_int
                if internal_id is not None:
                    semantic_map[internal_id] = self._semantic_to_id(sem_token)
        return semantic_map

    def attach_semantics(self, semantic_file='', fallback='region'):
        """Attach one semantic id to each POI and build semantic sequences.

        Real semantic labels are preferred: external file > raw category column.
        If no semantic data exists, a fallback is used so that the code can still run
        on Gowalla/Foursquare, whose public files usually contain no category text.
        """
        external = self._load_semantic_file(semantic_file)
        if external:
            self.poi2semantic = external.copy()
        elif self._poi_raw_semantic:
            self.poi2semantic = {
                poi: self._semantic_to_id(sem) for poi, sem in self._poi_raw_semantic.items()
            }
        else:
            self.poi2semantic = {}

        loc_count = len(self.poi2id)
        for poi in range(loc_count):
            if poi in self.poi2semantic:
                continue
            if fallback == 'region' and hasattr(self, 'poi2region'):
                self.poi2semantic[poi] = int(self.poi2region.get(poi, 0))
            elif fallback == 'poi':
                self.poi2semantic[poi] = int(poi)
            else:
                self.poi2semantic[poi] = 0

        self.semantic_count = max(self.poi2semantic.values()) + 1 if self.poi2semantic else 1
        self.semantics = [[self.poi2semantic[loc] for loc in seq] for seq in self.locs]
        print('Semantic label count: {}'.format(self.semantic_count))

    def construct_region(self):
        # construct the mapping from poi to region using KMeans algorithm
        poi_ids = list(self.poi2gps.keys())
        gps_coordinates = list(self.poi2gps.values())
        kmeans = KMeans(n_clusters=self.cluster)
        kmeans.fit(gps_coordinates)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        self.poi2region = {poi_id: labels[i] for i, poi_id in enumerate(poi_ids)}
        self.regions = [[self.poi2region[loc] for loc in self.locs[i]] for i in range(len(self.locs))]
        self.region2gps = {}
        for region, feature in enumerate(centers):
            if region not in self.region2gps:
                self.region2gps[region] = feature.tolist()
        self.regions_gps = [[self.region2gps[self.poi2region[loc]] for loc in self.locs[i]] for i in
                            range(len(self.locs))]
        self.poi2region_nv = self.poi2region
        self.regions_nv = self.regions




