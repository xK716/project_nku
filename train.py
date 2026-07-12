# 原代码
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import time, os
# import pickle
# from setting import Setting
# from trainer import FlashbackTrainer
# from dataloader import PoiDataloader
# from dataset import Split
# from utils import *
# from network import create_h0_strategy
# from evaluation import Evaluation
# from tqdm import tqdm
# import shutil, sys
# from scipy.sparse import coo_matrix
#
# set_seed()
# # parse settings
# setting = Setting()
# setting.parse()
# dir_name, file_name = os.path.split(setting.log_file)
# dataset_name = 'foursquare' if setting.guess_foursquare else 'gowalla'
# save_dir = os.path.join(dir_name, dataset_name)
# timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
# save_dir = os.path.join(save_dir, timestring)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir, exist_ok=True)
# if setting.guess_foursquare:
#     pref, sux = file_name.split('_')
#     file_name = pref + '_' + 'foursquare'
# shutil.copy2(sys.argv[0], save_dir)
# shutil.copy2('dataloader.py', save_dir)
# shutil.copy2('dataset.py', save_dir)
# shutil.copy2('evaluation.py', save_dir)
# shutil.copy2('network.py', save_dir)
# shutil.copy2('setting.py', save_dir)
# shutil.copy2('trainer.py', save_dir)
# shutil.copy2('utils.py', save_dir)
# setting.log_file = os.path.join(save_dir, file_name + '_' + timestring)
# log = open(setting.log_file, 'w')
#
# message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
# log_string(log, message)
#
# # load dataset
# poi_loader = PoiDataloader(
#     setting.max_users, setting.min_checkins)
# poi_loader.read(setting.dataset_file, setting.offset_file, setting.cluster, setting.slot)
#
# log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
# log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
# log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))
#
# region_dis = None
# poi2gps = poi_loader.poi2gps
# poi2gps = sorted(poi2gps.items())
# pid_lat_lon = []
# for coord in poi2gps:
#     pid_lat_lon.append(coord[1])
# pid_lat_lon = torch.from_numpy(np.array(pid_lat_lon)).float().to(setting.device)
# dataset = poi_loader.create_dataset(
#     setting.sequence_length, setting.batch_size, Split.TRAIN)
# region_graph = dataset.get_region_graph()
# region_graph = normalize(region_graph)
# user_region_graph = dataset.get_user_region_graph()
# user_poi_graph = dataset.get_user_poi_graph()
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# dataset_test = poi_loader.create_dataset(
#     setting.sequence_length, setting.batch_size, Split.TEST)
# dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
# assert setting.batch_size < poi_loader.user_count(
# ), 'batch size must be lower than the amount of available users'
#
# transition_graph = None
#
# if setting.use_spatial_graph:
#     with open(setting.trans_loc_spatial_file, 'rb') as f:
#         spatial_graph = pickle.load(f)
#     spatial_graph = coo_matrix(spatial_graph)
# else:
#     spatial_graph = None
#
# if setting.use_graph_user:
#     with open(setting.trans_user_file, 'rb') as f:
#         friend_graph = pickle.load(f)
#     friend_graph = coo_matrix(friend_graph)
# else:
#     friend_graph = None
#
# interact_graph = None
#
# log_string(log, 'Successfully load graph')
#
# trainer = FlashbackTrainer(dataset_name, setting.lambda_t, setting.lambda_s, setting.dropout, setting.lambda_loc,
#                            setting.lambda_user,
#                            setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
#                            setting.use_spatial_graph, interact_graph, log)
# h0_strategy = create_h0_strategy(
#     setting.hidden_dim * 2, setting.is_lstm)
# trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.cluster, setting.slot, setting.hidden_dim,
#                 setting.rnn_factory, setting.device, region_graph, region_dis, setting.cl_decay_steps, pid_lat_lon)
# evaluation_test = Evaluation(dataset_test, dataloader_test,
#                              poi_loader.user_count(), poi_loader.locations(), setting.cluster, h0_strategy, trainer,
#                              setting, log)
# print('{} {}'.format(trainer, setting.rnn_factory))
# logits = compute_adjustment(dataset.freq, setting)
# logits_reg = compute_adjustment(dataset.freq_reg, setting)
#
# #  training loop
# optimizer = torch.optim.Adam(trainer.parameters(
# ), lr=setting.learning_rate, weight_decay=setting.weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[20, 40, 60, 80], gamma=0.2)
#
# param_count = trainer.count_parameters()
# log_string(log, f'In total: {param_count} trainable parameters')
#
# bar = tqdm(total=setting.epochs)
# bar.set_description('Training')
# batches_seen = 1
#
# for e in range(setting.epochs):
#     h = h0_strategy.on_init(setting.batch_size, setting.device)
#     dataset.shuffle_users()
#
#     losses = []
#     losses_poi, losses_reg = [], []
#     epoch_start = time.time()
#     total_samples = 0
#     for i, (
#     x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, reset_h, active_users) in enumerate(
#             dataloader):
#         # reset hidden states for newly added users
#         for j, reset in enumerate(reset_h):
#             if reset:
#                 if setting.is_lstm:
#                     hc = h0_strategy.on_reset(active_users[0][j])
#                     h[0][0, j] = hc[0]
#                     h[1][0, j] = hc[1]
#                 else:
#                     h[0, j] = h0_strategy.on_reset(active_users[0][j])
#
#         x = x.squeeze().to(setting.device)
#         t = t.squeeze().to(setting.device)
#         t_slot = t_slot.squeeze().to(setting.device)
#         s = s.squeeze().to(setting.device)
#         r = r.squeeze().long().to(setting.device)
#         r_nv = r_nv.squeeze().long().to(setting.device)
#         r_s = r_s.squeeze().to(setting.device)
#
#         y = y.squeeze().to(setting.device)
#         y_t = y_t.squeeze().to(setting.device)
#         y_t_slot = y_t_slot.squeeze().to(setting.device)
#         y_s = y_s.squeeze().to(setting.device)
#         y_r = y_r.squeeze().long().to(setting.device)
#         y_r_nv = y_r_nv.squeeze().long().to(setting.device)
#         y_r_s = y_r_s.squeeze().to(setting.device)
#
#         active_users = active_users.to(setting.device)
#
#         optimizer.zero_grad()
#         loss_p, loss_r = trainer.loss(x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h,
#                                       active_users, batches_seen, logits, logits_reg)
#         loss = loss_p + loss_r * setting.lambda_r
#         loss = loss
#         loss.backward(retain_graph=True)
#         batches_seen += 1
#
#         total_samples += x.size(0) * x.size(1)
#         losses.append(loss.item())
#         losses_poi.append(loss_p.item())
#         losses_reg.append(loss_r.item())
#         optimizer.step()
#
#     # schedule learning rate:
#     scheduler.step()
#     bar.update(1)
#     epoch_end = time.time()
#     log_string(log, 'One training need {:.2f}s'.format(epoch_end - epoch_start))
#     # statistics:
#     if (e + 1) % 1 == 0:
#         epoch_loss = np.mean(losses)
#         epoch_loss_poi = np.mean(losses_poi)
#         epoch_loss_reg = np.mean(losses_reg)
#         log_string(log, f'Epoch: {e + 1}/{setting.epochs}, Batch_seen: {batches_seen}')
#         log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
#         log_string(log, f'Avg Loss: {epoch_loss}')
#         log_string(log, f'Avg POI Loss: {epoch_loss_poi}')
#         log_string(log, f'Avg Region Loss: {epoch_loss_reg}')
#
#     if (e + 1) % setting.validate_epoch == 0:
#         log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
#         # torch.save(trainer.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
#         evl_start = time.time()
#         evaluation_test.evaluate()
#         evl_end = time.time()
#         log_string(log, 'One evaluate need {:.2f}s'.format(evl_end - evl_start))
#
# bar.close()
#
# # scp -rP 56037 "D:\pythonProject1\DPRL-main" root@connect.bjb1.seetacloud.com:/root/autodl-tmp







import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
import shutil, sys
from scipy.sparse import coo_matrix

set_seed()
# parse settings
setting = Setting()
setting.parse()
dir_name, file_name = os.path.split(setting.log_file)
dataset_name = 'foursquare' if setting.guess_foursquare else 'gowalla'
save_dir = os.path.join(dir_name, dataset_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
save_dir = os.path.join(save_dir, timestring)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
if setting.guess_foursquare:
    pref, sux = file_name.split('_')
    file_name = pref + '_' + 'foursquare'
shutil.copy2(sys.argv[0], save_dir)
shutil.copy2('dataloader.py', save_dir)
shutil.copy2('dataset.py', save_dir)
shutil.copy2('evaluation.py', save_dir)
shutil.copy2('network.py', save_dir)
shutil.copy2('setting.py', save_dir)
shutil.copy2('trainer.py', save_dir)
shutil.copy2('utils.py', save_dir)
setting.log_file = os.path.join(save_dir, file_name + '_' + timestring)
log = open(setting.log_file, 'w')

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset
poi_loader = PoiDataloader(
    setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file, setting.offset_file, setting.cluster, setting.slot,
                semantic_file=setting.semantic_file,
                semantic_fallback=setting.semantic_fallback)

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))
log_string(log, 'Semantic class number:{}'.format(poi_loader.semantic_count))

region_dis = None
poi2gps = poi_loader.poi2gps
poi2gps = sorted(poi2gps.items())
pid_lat_lon = []
for coord in poi2gps:
    pid_lat_lon.append(coord[1])
pid_lat_lon = torch.from_numpy(np.array(pid_lat_lon)).float().to(setting.device)
poi2semantic_vec = np.zeros(poi_loader.locations(), dtype=np.int64)
poi2region_vec = np.zeros(poi_loader.locations(), dtype=np.int64)
for pid in range(poi_loader.locations()):
    poi2semantic_vec[pid] = int(poi_loader.poi2semantic.get(pid, 0))
    poi2region_vec[pid] = int(poi_loader.poi2region.get(pid, 0))
semantic_count = int(poi2semantic_vec.max() + 1) if poi2semantic_vec.size > 0 else 1
dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)
region_graph = dataset.get_region_graph()
region_graph = normalize(region_graph)
user_region_graph = dataset.get_user_region_graph()
user_poi_graph = dataset.get_user_poi_graph()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'

transition_graph = None

if setting.use_spatial_graph:
    with open(setting.trans_loc_spatial_file, 'rb') as f:
        spatial_graph = pickle.load(f)
    spatial_graph = coo_matrix(spatial_graph)
else:
    spatial_graph = None

if setting.use_graph_user:
    with open(setting.trans_user_file, 'rb') as f:
        friend_graph = pickle.load(f)
    friend_graph = coo_matrix(friend_graph)
else:
    friend_graph = None

interact_graph = None

log_string(log, 'Successfully load graph')

trainer = FlashbackTrainer(dataset_name, setting.lambda_t, setting.lambda_s, setting.dropout, setting.lambda_loc,
                           setting.lambda_user,
                           setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
                           setting.use_spatial_graph, interact_graph, log)
h0_strategy = create_h0_strategy(
    setting.hidden_dim * 2, setting.is_lstm)
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.cluster, setting.slot, setting.hidden_dim,
                setting.rnn_factory, setting.device, region_graph, region_dis, setting.cl_decay_steps, pid_lat_lon,
                semantic_count=semantic_count,
                poi_semantic=poi2semantic_vec,
                poi_region=poi2region_vec,
                use_sem_branch=setting.use_sem_branch,
                filter_candidate=setting.filter_candidate,
                topm_candidate=setting.topm_candidate,
                candidate_sem_weight=setting.candidate_sem_weight,
                candidate_region_weight=setting.candidate_region_weight,
                candidate_mode=setting.candidate_mode,
                semantic_fusion_weight=setting.semantic_fusion_weight,
                sem_label_smoothing=setting.sem_label_smoothing)
evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), poi_loader.locations(), setting.cluster, h0_strategy, trainer,
                             setting, log)
print('{} {}'.format(trainer, setting.rnn_factory))
logits = compute_adjustment(dataset.freq, setting)
logits_reg = compute_adjustment(dataset.freq_reg, setting)


# todo 1
def get_semantic_loss_weight(epoch, base_weight, warmup_epochs=5, decay_start=15, decay_epochs=20):
    """
    semantic loss 权重调度：
    1. 前 warmup_epochs 轮线性升高；
    2. warmup 后到 decay_start 前保持 base_weight；
    3. decay_start 后线性衰减；
    4. decay_start + decay_epochs 后降为 0。
    """
    # epoch 是从 0 开始的，所以这里用 epoch + 1 表示真实轮数
    cur_epoch = epoch + 1

    if base_weight <= 0:
        return 0.0

    # warmup 阶段
    if warmup_epochs > 0 and cur_epoch <= warmup_epochs:
        return base_weight * float(cur_epoch) / float(warmup_epochs)

    # 保持阶段
    if cur_epoch <= decay_start:
        return base_weight

    # 衰减阶段
    progress = float(cur_epoch - decay_start) / float(decay_epochs)
    return base_weight * max(0.0, 1.0 - progress)
#############

def is_semantic_param(name):
    """
    判断哪些参数属于 semantic branch。
    兼容你现在用过的 us_encoder / us_short / us_long 两种写法。
    """
    sem_keywords = [
        "semantic_encoder",
        "us_encoder",
        "us_short",
        "us_long",
        "rnn_s",
        "fc_s",
        "sem_residual",
        "sem_pref_residual",
        "sem_gate",
        "cross_proj_s",
        "cross_out_s",
    ]

    return any(k in name for k in sem_keywords)


def set_semantic_trainable(trainer, trainable=True):
    """
    冻结或解冻 semantic branch 参数。
    trainable=False 后，semantic branch 不再接收梯度更新。
    """
    for name, param in trainer.model.named_parameters():
        if is_semantic_param(name):
            param.requires_grad = trainable


def get_semantic_loss_weight_fixed_stop(epoch, base_weight, warmup_epochs=5, stop_epoch=15):
    """
    固定语义权重 + 早停语义梯度：
    1. 前 warmup_epochs 轮线性 warmup；
    2. warmup 后到 stop_epoch 保持 base_weight；
    3. stop_epoch 后 loss_s 不再参与反向传播。
    """
    cur_epoch = epoch + 1

    if base_weight <= 0:
        return 0.0

    if warmup_epochs > 0 and cur_epoch <= warmup_epochs:
        return base_weight * float(cur_epoch) / float(warmup_epochs)

    if cur_epoch <= stop_epoch:
        return base_weight

    return 0.0
def get_semantic_loss_weight_floor(
        epoch,
        base_weight,
        warmup_epochs=5,
        decay_start=15,
        decay_epochs=20,
        floor_ratio=0.1):

    cur_epoch = epoch + 1

    floor_weight = base_weight * floor_ratio

    # warmup
    if cur_epoch <= warmup_epochs:
        return base_weight * cur_epoch / warmup_epochs

    # 保持
    if cur_epoch <= decay_start:
        return base_weight

    # 衰减
    progress = (cur_epoch - decay_start) / decay_epochs

    weight = base_weight * (1 - progress)

    # 不允许低于下限
    return max(weight, floor_weight)
#  training loop
optimizer = torch.optim.Adam(trainer.parameters(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

bar = tqdm(total=setting.epochs)
bar.set_description('Training')
batches_seen = 1

# semantic branch 训练到这个 epoch 后冻结
# 建议先试 10 / 15 / 20
sem_stop_epoch = 25
semantic_frozen = False

# for e in range(setting.epochs):
#     h = h0_strategy.on_init(setting.batch_size, setting.device)
#     dataset.shuffle_users()
for e in range(setting.epochs):

    #到达 stop epoch 之后，冻结 semantic branch 参数
    # if setting.use_sem_branch and (e + 1) > sem_stop_epoch and not semantic_frozen:
    #     set_semantic_trainable(trainer, trainable=False)
    #     semantic_frozen = True
    #     log_string(log, f'[Semantic Early Stop] Freeze semantic branch after epoch {sem_stop_epoch}')

    semantic_frozen = False
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()

    losses = []
    losses_poi, losses_reg, losses_sem = [], [], []
    epoch_start = time.time()
    total_samples = 0
    for i, (
    x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, reset_h, active_users) in enumerate(
            dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        r = r.squeeze().long().to(setting.device)
        r_nv = r_nv.squeeze().long().to(setting.device)
        r_s = r_s.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        y_r = y_r.squeeze().long().to(setting.device)
        y_r_nv = y_r_nv.squeeze().long().to(setting.device)
        y_r_s = y_r_s.squeeze().to(setting.device)

        active_users = active_users.to(setting.device)

        optimizer.zero_grad()
        loss_p, loss_r, loss_s = trainer.loss(x, t, t_slot, s, r, r_nv, r_s, y, y_t, y_t_slot, y_s, y_r, y_r_nv, y_r_s, h,
                                      active_users, batches_seen, logits, logits_reg)
        #todo 2

        # if setting.sem_warmup_epochs > 0:
        #     sem_weight_now = setting.sem_weight * min(1.0, float(e + 1) / float(setting.sem_warmup_epochs))
        # else:
        #     sem_weight_now = setting.sem_weight
        # loss = loss_p + loss_r * setting.lambda_r + loss_s * sem_weight_now

        ######

        # sem_weight_now = get_semantic_loss_weight(
        #     epoch=e,
        #     base_weight=setting.sem_weight,
        #     warmup_epochs=setting.sem_warmup_epochs,
        #     decay_start=15,
        #     decay_epochs=20
        # )
        #
        # loss = loss_p + loss_r * setting.lambda_r + loss_s * sem_weight_now
        #######

        # sem_weight_now = get_semantic_loss_weight_fixed_stop(
        #     epoch=e,
        #     base_weight=setting.sem_weight,
        #     warmup_epochs=setting.sem_warmup_epochs,
        #     stop_epoch=sem_stop_epoch
        # )
        #
        # # stop_epoch 后，semantic loss 不再参与总 loss
        # if sem_weight_now > 0:
        #     sem_loss_term = loss_s * sem_weight_now
        # else:
        #     sem_loss_term = loss_p.new_tensor(0.0)
        #
        # loss = loss_p + loss_r * setting.lambda_r + sem_loss_term
        sem_weight_now = get_semantic_loss_weight_floor(
            epoch=e,
            base_weight=setting.sem_weight,
            warmup_epochs=setting.sem_warmup_epochs,
            decay_start=15,
            decay_epochs=20,
            floor_ratio=0.1
        )

        loss = loss_p + loss_r * setting.lambda_r + loss_s * sem_weight_now

        loss.backward(retain_graph=True)
        batches_seen += 1

        total_samples += x.size(0) * x.size(1)
        losses.append(loss.item())
        losses_poi.append(loss_p.item())
        losses_reg.append(loss_r.item())
        losses_sem.append(loss_s.item())
        optimizer.step()

    # schedule learning rate:
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        epoch_loss_poi = np.mean(losses_poi)
        epoch_loss_reg = np.mean(losses_reg)
        epoch_loss_sem = np.mean(losses_sem)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}, Batch_seen: {batches_seen}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')
        log_string(log, f'Avg POI Loss: {epoch_loss_poi}')
        log_string(log, f'Avg Region Loss: {epoch_loss_reg}')
        log_string(log, f'Avg Semantic Loss: {epoch_loss_sem}')

        # todo 3
        # if setting.use_sem_branch:
        #     if setting.sem_warmup_epochs > 0:
        #         sem_weight_now = setting.sem_weight * min(1.0, float(e + 1) / float(setting.sem_warmup_epochs))
        #     else:
        #         sem_weight_now = setting.sem_weight
        #     log_string(log, f'Current Semantic Loss Weight: {sem_weight_now}')

        ######
        # if setting.use_sem_branch:
        #     sem_weight_now = get_semantic_loss_weight_fixed_stop(
        #         epoch=e,
        #         base_weight=setting.sem_weight,
        #         warmup_epochs=setting.sem_warmup_epochs,
        #         stop_epoch=sem_stop_epoch
        #     )
        #     log_string(log, f'Current Semantic Loss Weight: {sem_weight_now}')
        #     log_string(log, f'Semantic Branch Frozen: {semantic_frozen}')
        #####
        if setting.use_sem_branch:
            sem_weight_now = get_semantic_loss_weight_floor(
                epoch=e,
                base_weight=setting.sem_weight,
                warmup_epochs=setting.sem_warmup_epochs,
                decay_start=15,
                decay_epochs=20,
                floor_ratio=0.1
            )
            log_string(log, f'Current Semantic Loss Weight: {sem_weight_now}')
            log_string(log, f'Semantic Branch Frozen: {semantic_frozen}')
        # if setting.use_sem_branch:
        #     sem_weight_now = get_semantic_loss_weight(
        #         epoch=e,
        #         base_weight=setting.sem_weight,
        #         warmup_epochs=setting.sem_warmup_epochs,
        #         decay_start=15,
        #         decay_epochs=20
        #     )
        #     log_string(log, f'Current Semantic Loss Weight: {sem_weight_now}')

    if (e + 1) % setting.validate_epoch == 0:
        log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        # torch.save(trainer.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        evl_start = time.time()
        evaluation_test.evaluate()
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(evl_end - evl_start))

bar.close()

# scp -rP 56037 "D:\pythonProject1\DPRL-main" root@connect.bjb1.seetacloud.com:/root/autodl-tmp








