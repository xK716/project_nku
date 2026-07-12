# import torch
# import argparse
# import sys
#
# from network import RnnFactory
#
#
# class Setting:
#     """ Defines all settings in a single place using a command line interface.
#     """
#
#     def parse(self):
#         self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])
#
#         parser = argparse.ArgumentParser()
#         if self.guess_foursquare:
#             self.parse_foursquare(parser)
#         else:
#             self.parse_gowalla(parser)
#         self.parse_arguments(parser)
#         args = parser.parse_args()
#
#         ###### settings ######
#         # training
#         self.gpu = args.gpu
#         self.hidden_dim = args.hidden_dim
#         self.weight_decay = args.weight_decay
#         self.learning_rate = args.lr
#         self.epochs = args.epochs
#         self.rnn_factory = RnnFactory(args.rnn)
#         self.is_lstm = self.rnn_factory.is_lstm()
#         self.lambda_t = args.lambda_t
#         self.lambda_s = args.lambda_s
#
#         # data management
#         self.dataset_file = './data/{}'.format(args.dataset)
#         self.offset_file = './data/{}'.format(args.offset)
#         self.friend_file = './data/{}'.format(args.friendship)
#         self.max_users = 0
#         self.sequence_length = 20
#         self.batch_size = args.batch_size
#         self.min_checkins = 101
#         self.cluster = args.cluster
#         self.slot = args.slot
#         self.cl_decay_steps = args.cl_decay_steps
#         self.lambda_r = args.lambda_r
#         self.lambda_m = args.lambda_m
#
#         # evaluation
#         self.validate_epoch = args.validate_epoch
#         self.report_user = args.report_user
#
#         # log
#         self.log_file = args.log_file
#
#         self.trans_loc_file = args.trans_loc_file
#         self.trans_loc_spatial_file = args.trans_loc_spatial_file
#         self.trans_user_file = args.trans_user_file
#         self.trans_interact_file = args.trans_interact_file
#
#         self.lambda_user = args.lambda_user
#         self.lambda_loc = args.lambda_loc
#         self.dropout = args.dropout
#
#         self.use_weight = args.use_weight
#         self.use_graph_user = args.use_graph_user
#         self.use_spatial_graph = args.use_spatial_graph
#
#         ### CUDA Setup ###
#         self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)
#
#     def parse_arguments(self, parser):
#         # training
#         parser.add_argument('--gpu', default=2, type=int, help='the gpu to use')
#         parser.add_argument('--hidden-dim', default=30, type=int, help='hidden dimensions to use')
#         parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay regularization')
#         parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
#         parser.add_argument('--epochs', default=50, type=int, help='amount of epochs')
#         parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')
#
#
#         #////////////////////////////////////////// todo 1
#         parser.add_argument('--sem-dim', default=30, type=int, help='POI语义嵌入维度')
#         parser.add_argument('--use-sem-branch', action='store_true', help='开启语义三分支')
#         parser.add_argument('--sem-weight', default=0.3, type=float, help='语义损失权重')
#         parser.add_argument('--filter-candidate', action='store_true', help='开启语义候选过滤')
#         parser.add_argument('--topm-candidate', default=2000, type=int, help='过滤后候选POI数量')
#         #/////////////////////////////////////////
#         # data management
#         parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
#                             help='the dataset under ./data/<dataset.txt> to load')
#         # evaluation
#         parser.add_argument('--validate-epoch', default=5, type=int,
#                             help='run each validation after this amount of epochs')
#         parser.add_argument('--report-user', default=-1, type=int,
#                             help='report every x user on evaluation (-1: ignore)')
#
#         # log
#         parser.add_argument('--log_file', default='./results/log_gowalla', type=str,
#                             help='存储结果日志')
#         parser.add_argument('--trans_user_file', default='', type=str,
#                             help='使用transe方法构造的user转换图')
#         parser.add_argument('--trans_loc_spatial_file', default='', type=str,
#                             help='使用transe方法构造的空间POI转换图')
#         parser.add_argument('--use_weight', default=False, type=bool, help='应用于GCN的AXW中是否使用W')
#         parser.add_argument('--use_graph_user', default=False, type=bool, help='是否使用user graph')
#         parser.add_argument('--use_spatial_graph', default=False, type=bool, help='是否使用空间POI graph')
#
#     def parse_gowalla(self, parser):
#         # defaults for gowalla dataset
#         parser.add_argument('--batch-size', default=200, type=int,  # 200
#                             help='amount of users to process in one pass (batching)')
#         parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
#         parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
#         parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
#         parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
#         parser.add_argument('--cluster', default=4000, type=int, help='cluster for region')
#         parser.add_argument('--slot', default=48, type=int, help='number of time slot')
#         parser.add_argument('--cl_decay_steps', default=8000, type=int, help='number of cl_decay_steps')
#         parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
#                             help='the friendship file under ../data/<edges.txt> to load')
#         parser.add_argument('--trans_loc_file', default='./KGE/Graphs/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
#                             help='使用transe方法构造的时间POI转换图')
#         parser.add_argument('--trans_interact_file', default='./KGE/Graphs/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
#                             help='使用transe方法构造的用户-POI交互图')
#         # parser.add_argument('--offset', default='checkins_gowalla_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
#         parser.add_argument('--offset', default='checkins_4sq_nyc_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
#
#         parser.add_argument('--dropout', default=0.4, type=float, help='dropout for rnn')
#         parser.add_argument('--lambda_r', default=0.5, type=float, help='factor for region loss')
#         parser.add_argument('--lambda_m', default=0, type=float, help='factor for margin loss')
#
#
#     def parse_foursquare(self, parser):
#         # defaults for foursquare dataset
#         parser.add_argument('--batch-size', default=512, type=int,
#                             help='amount of users to process in one pass (batching)')
#         parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
#         parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
#         parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
#         parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
#         parser.add_argument('--cluster', default=3000, type=int, help='cluster for region')
#         parser.add_argument('--slot', default=48, type=int, help='number of time slot')
#         parser.add_argument('--cl_decay_steps', default=8000, type=int, help='number of cl_decay_steps')
#         parser.add_argument('--friendship', default='4sq_friend.txt', type=str,
#                             help='the friendship file under ../data/<edges.txt> to load')
#         parser.add_argument('--trans_loc_file', default='./KGE/Graphs/foursquare_scheme2_transe_loc_temporal_20.pkl', type=str,
#                             help='使用transe方法构造的时间POI转换图')
#         parser.add_argument('--trans_interact_file', default='./KGE/Graphs/foursquare_scheme2_transe_user-loc_100.pkl', type=str,
#                             help='使用transe方法构造的用户-POI交互图')
#         parser.add_argument('--offset', default='checkins_4sq_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
#         parser.add_argument('--dropout', default=0.1, type=float, help='dropout for rnn')
#         parser.add_argument('--lambda_r', default=0.1, type=float, help='factor for region loss')
#         parser.add_argument('--lambda_m', default=0, type=float, help='factor for margin loss')
#
#
#
#
#
#
#     def __str__(self):
#         return (
#                    'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
#                + 'use device: {}'.format(self.device)






import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.offset_file = './data/{}'.format(args.offset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.max_users = 0
        self.sequence_length = 20
        self.batch_size = args.batch_size
        self.min_checkins = 101
        self.cluster = args.cluster
        self.slot = args.slot
        self.cl_decay_steps = args.cl_decay_steps
        self.lambda_r = args.lambda_r
        self.lambda_m = args.lambda_m

        # evaluation
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user

        # log
        self.log_file = args.log_file

        self.trans_loc_file = args.trans_loc_file
        self.trans_loc_spatial_file = args.trans_loc_spatial_file
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc
        self.dropout = args.dropout

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        # Sem-DPRL options
        self.sem_dim = args.sem_dim
        self.use_sem_branch = args.use_sem_branch
        self.sem_weight = args.sem_weight
        self.semantic_file = args.semantic_file
        self.semantic_fallback = args.semantic_fallback
        self.filter_candidate = args.filter_candidate
        self.topm_candidate = args.topm_candidate
        self.candidate_sem_weight = args.candidate_sem_weight
        self.candidate_region_weight = args.candidate_region_weight
        self.candidate_mode = args.candidate_mode
        self.semantic_fusion_weight = args.semantic_fusion_weight
        self.sem_label_smoothing = args.sem_label_smoothing
        self.sem_warmup_epochs = args.sem_warmup_epochs

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=2, type=int, help='the gpu to use')
        parser.add_argument('--hidden-dim', default=30, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=70, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')


        # Sem-DPRL: semantic branch and semantic-aware candidate filtering
        parser.add_argument('--sem-dim', default=30, type=int, help='POI semantic embedding dimension placeholder')
        parser.add_argument('--use-sem-branch', action='store_true', help='enable semantic branch')
        parser.add_argument('--sem-weight', default=0.3, type=float, help='weight of semantic auxiliary loss')
        parser.add_argument('--semantic-file', default='', type=str,
                            help='optional semantic label file, format: raw_poi_id semantic_id/category')
        parser.add_argument('--semantic-fallback', default='region', choices=['region', 'poi', 'zero'],
                            help='fallback semantic label when semantic-file is unavailable')
        parser.add_argument('--filter-candidate', action='store_true', help='enable semantic-aware candidate filtering')
        parser.add_argument('--topm-candidate', default=2000, type=int, help='number of candidates kept after filtering')
        parser.add_argument('--candidate-sem-weight', default=0.6, type=float, help='semantic score weight in candidate filtering')
        parser.add_argument('--candidate-region-weight', default=0.4, type=float, help='region score weight in candidate filtering')
        parser.add_argument('--candidate-mode', default='score', choices=['score', 'mask'],
                            help='score: soft rerank only; mask: additionally keep only topm candidates')
        parser.add_argument('--semantic-fusion-weight', default=0.5, type=float,
                            help='residual weight for gated semantic fusion into POI prediction')
        parser.add_argument('--sem-label-smoothing', default=0.05, type=float,
                            help='label smoothing for weak semantic labels')
        parser.add_argument('--sem-warmup-epochs', default=5, type=int,
                            help='linearly warm up semantic auxiliary loss for weak semantic labels')
        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        # evaluation
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_gowalla', type=str,
                            help='存储结果日志')
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='使用transe方法构造的user转换图')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='使用transe方法构造的空间POI转换图')
        parser.add_argument('--use_weight', default=False, type=bool, help='应用于GCN的AXW中是否使用W')
        parser.add_argument('--use_graph_user', default=False, type=bool, help='是否使用user graph')
        parser.add_argument('--use_spatial_graph', default=False, type=bool, help='是否使用空间POI graph')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=200, type=int,  # 200
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
        parser.add_argument('--cluster', default=4000, type=int, help='cluster for region')
        parser.add_argument('--slot', default=48, type=int, help='number of time slot')
        parser.add_argument('--cl_decay_steps', default=8000, type=int, help='number of cl_decay_steps')
        parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        parser.add_argument('--trans_loc_file', default='./KGE/Graphs/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='使用transe方法构造的时间POI转换图')
        parser.add_argument('--trans_interact_file', default='./KGE/Graphs/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
                            help='使用transe方法构造的用户-POI交互图')
        # parser.add_argument('--offset', default='checkins_gowalla_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--offset', default='checkins_4sq_nyc_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')

        parser.add_argument('--dropout', default=0.4, type=float, help='dropout for rnn')
        parser.add_argument('--lambda_r', default=0.5, type=float, help='factor for region loss')
        parser.add_argument('--lambda_m', default=0, type=float, help='factor for margin loss')


    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=512, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
        parser.add_argument('--cluster', default=3000, type=int, help='cluster for region')
        parser.add_argument('--slot', default=48, type=int, help='number of time slot')
        parser.add_argument('--cl_decay_steps', default=8000, type=int, help='number of cl_decay_steps')
        parser.add_argument('--friendship', default='4sq_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        parser.add_argument('--trans_loc_file', default='./KGE/Graphs/foursquare_scheme2_transe_loc_temporal_20.pkl', type=str,
                            help='使用transe方法构造的时间POI转换图')
        parser.add_argument('--trans_interact_file', default='./KGE/Graphs/foursquare_scheme2_transe_user-loc_100.pkl', type=str,
                            help='使用transe方法构造的用户-POI交互图')
        # parser.add_argument('--offset', default='checkins_4sq_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--offset', default='checkins_4sq_nyc_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--dropout', default=0.1, type=float, help='dropout for rnn')
        parser.add_argument('--lambda_r', default=0.1, type=float, help='factor for region loss')
        parser.add_argument('--lambda_m', default=0, type=float, help='factor for margin loss')






    def __str__(self):
        return (
                   'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
               + 'use device: {}'.format(self.device)






