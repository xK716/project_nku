# import pickle
# import numpy as np
# import torch
# import random
# import torch.nn.functional as F
# from math import radians, cos, sin, asin, sqrt
# from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
# import scipy.sparse as sp
# import os
# seed = 0
# global_seed = 0
# torch.manual_seed(seed)
#
#
# def load_graph_data(pkl_filename):
#     graph = load_pickle(pkl_filename)  # list
#     # graph = np.array(graph[0])
#     return graph
#
#
# def load_pickle(pickle_file):
#     try:
#         with open(pickle_file, 'rb') as f:
#             pickle_data = pickle.load(f)
#     except UnicodeDecodeError as e:
#         with open(pickle_file, 'rb') as f:
#             pickle_data = pickle.load(f, encoding='latin1')
#     except Exception as e:
#         print('Unable to load data ', pickle_file, ': ', e)
#         raise
#     return pickle_data
#
#
# def calculate_preference_similarity(m1, m2, pref):
#     """
#         m1: (user_len, hidden_size)
#         m2: user_len, seq_len, hidden_size)
#         return: calculate the similarity between user and location, which means user's preference about location
#     """
#     user_len = m1.shape[0]
#     seq_len = m2.shape[1]
#     pref = pref.squeeze()
#     similarity = torch.zeros(user_len, seq_len, dtype=torch.float32)
#     for i in range(user_len):
#         v1 = m1[i]
#         for j in range(seq_len):
#             v2 = m2[i][j]
#             similarity[i][j] = (1 + torch.cosine_similarity(v1 + pref, v2, dim=0).item()) / 2
#
#     return similarity
#
#
# def compute_preference(m1, m2, pref):
#     m1 = (m1 + pref).unsqueeze(1)
#     s = m1 - m2
#     sim = torch.exp(-(torch.norm(s, p=2, dim=-1)))
#     return sim
#
# def get_user_static_preference(pref, locs):
#     """
#         pref: (user_len, seq_len)
#         locs: (user_len, seq_len, hidden_size)
#         return: 返回用户对于所访问POI的全局偏好
#     """
#     user_len, seq_len = pref.shape[0], pref.shape[1]
#     hidden_size = locs.shape[2]
#     user_preference = torch.zeros(user_len, seq_len, hidden_size)
#     for i in range(user_len):
#         for j in range(seq_len):
#             user_preference[i][j] = torch.sum(torch.softmax(pref[i, :j + 1], dim=0).unsqueeze(1) * locs[i, :j + 1],
#                                               dim=0)
#     user_preference = user_preference.permute(1, 0, 2)
#
#     return user_preference
#
#
# def sampling_prob(prob, label, num_neg):
#     num_label, l_m = prob.shape[0], prob.shape[1]
#     init_label = torch.zeros(num_label, dtype=torch.int64)
#     init_prob = torch.zeros(size=(num_label, num_neg + 1))
#
#     for batch in range(num_label):
#         random_ig = random.sample(range(l_m), num_neg)
#         while label[batch].item() in random_ig:
#             random_ig = random.sample(range(l_m), num_neg)
#
#         # place the pos labels ahead and neg samples in the end
#         for i in range(num_neg + 1):
#             if i < 1:
#                 init_prob[batch, i] = prob[batch, label[batch]]
#             else:
#                 init_prob[batch, i] = prob[batch, random_ig[i - 1]]
#
#     global global_seed
#     random.seed(global_seed)
#     global_seed += 1
#
#     return torch.FloatTensor(init_prob), torch.LongTensor(init_label)
#
#
# def bprLoss(pos, neg, target=1.0):
#     loss = - F.logsigmoid(target * (pos - neg))
#     return loss.mean()
#
#
# def haversine(lat1, lon1, lat2, lon2):
#     """
#     Calculate the great circle distance between two points
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#
#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers. Use 3956 for miles
#     return c * r
#
#
# def top_transition_graph(transition_graph):
#     graph = coo_matrix(transition_graph)
#     data = graph.data
#     row = graph.row
#     threshold = 20
#
#     for i in range(0, row.size, threshold):
#         row_data = data[i: i + threshold]
#         norm = row_data.max()
#         row_data = row_data / norm
#         data[i: i + threshold] = row_data
#
#     return graph
#
#
# def sparse_matrix_to_tensor(graph):
#     graph = coo_matrix(graph)
#     vaules = graph.data
#     indices = np.vstack((graph.row, graph.col))
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(vaules)
#     shape = graph.shape
#     graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))
#
#     return graph
#
#
# def calculate_random_walk_matrix(adj_mx):
#     adj_mx = sp.coo_matrix(adj_mx)
#     d = np.array(adj_mx.sum(1))
#     d_inv = np.power(d, -1).flatten()
#     d_inv[np.isinf(d_inv)] = 0.
#     d_mat_inv = sp.diags(d_inv)
#     random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
#
#     return random_walk_mx
#
#
# def calculate_reverse_random_walk_matrix(adj_mx):
#     adj_mx = sp.coo_matrix(adj_mx)
#     return calculate_random_walk_matrix(np.transpose(adj_mx))
#
#
# def log_string(log, string):
#     log.write(string + '\n')
#     log.flush()
#     print(string)
#
# def rotate_batch(head, relation, hidden):
#     pi = 3.14159265358979323846
#     device = head.device
#     re_head, im_head = torch.chunk(head, 2, dim=2)
#
#     #Make phases of relations uniformly distributed in [-pi, pi]
#     embedding_range = torch.nn.Parameter(
#                     torch.Tensor([(24.0 + 2.0) / hidden]),
#                     requires_grad=False
#             ).to(device)
#
#     phase_relation = relation/(embedding_range/pi)
#
#     re_relation = torch.cos(phase_relation)
#     im_relation = torch.sin(phase_relation)
#
#
#     re_score = re_head * re_relation - im_head * im_relation
#     im_score = re_head * im_relation + im_head * re_relation
#
#     score = torch.cat([re_score, im_score], dim = 2)
#     return score
#
# def normalize(mx):
#     '''Row-normalize sparse matrix'''
#     for i in range(mx.shape[0]):
#         b = mx[i, i] + 1
#         mx[i, i] = 0
#         tmp = mx[i].max()
#         if b <= tmp:
#             mx[i, i] = b - 1
#         else:
#             mx[i, i] = tmp * 1.2 if tmp > 0 else tmp + 10
#
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum.astype(float), -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
#
# def calculate_distance(region2gps):
#     adj = np.zeros((len(region2gps), len(region2gps)), dtype=np.float32)
#     for region1, feature1 in region2gps.items():
#         lat1, lng1 = feature1
#         for region2, feature2 in region2gps.items():
#             lat2, lng2 = feature2
#             dis = haversine(lat1, lng1, lat2, lng2)
#             adj[region1][region2] = np.exp(dis * -1)
#
#     rowsum = np.array(adj.sum(1))
#     r_inv = np.power(rowsum.astype(float), -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(adj)
#
#     return mx
#
# def geo_con_loss(preds, targets, pid_lat_lon):
#     log_softmax = torch.nn.functional.log_softmax(preds, dim=-1)
#     l_pred = torch.argmax(log_softmax, dim=-1)
#     l_coor_pred = pid_lat_lon[l_pred]
#     l_coor_tar = pid_lat_lon[targets]
#
#     dlat = l_coor_pred[:, 0] - l_coor_tar[:, 0]
#     dlon = l_coor_pred[:, 1] - l_coor_tar[:, 1]
#     dist = dlat ** 2 + dlon ** 2
#     loc_prob = log_softmax * dist.unsqueeze(-1)
#     loss_geocons = F.nll_loss(loc_prob, targets, reduction='mean')
#
#     return loss_geocons
#
# def maksed_mse_loss(input, target, mask_value=-1):
#     mask = target == mask_value
#     out = (input[~mask] - target[~mask]) ** 2
#     loss = out.mean()
#     return loss
#
# def trajectory_forecasting_loss(pred, true):
#     return F.mse_loss(pred, true, reduction='mean')
#
# def consistency_loss(pred_aux, pred_main, pid_lat_lon):
#     log_softmax = torch.nn.functional.log_softmax(pred_main, dim=-1)
#     l_pred = torch.argmax(log_softmax, dim=-1)
#     pred_main = pid_lat_lon[l_pred]
#     return F.mse_loss(pred_aux.view(-1, 2), pred_main, reduction='mean')
#
# def set_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# def readEmbedFile(embedFile):
#     input = open(embedFile, 'r')
#     lines = []
#     for line in input:
#         lines.append(line)
#
#     embeddings_dict = {}
#     embeddings = []
#     for lineId in range(1, len(lines)):
#         splits = lines[lineId].split(' ')
#         embedId = int(splits[0])
#         embedValue = splits[1:]
#         new_embedValue = [float(x) for x in embedValue]
#         embeddings_dict[embedId] = new_embedValue
#
#     for i in sorted(embeddings_dict):
#         embeddings.append(embeddings_dict[i])
#
#     embeddings = torch.from_numpy(np.array(embeddings)).float()
#
#     return embeddings
#
# def compute_adjustment(label_freq, setting):
#     """compute the base probabilities"""
#
#     label_freq_array = np.array(list(label_freq.values()))
#     max_freq = label_freq_array.max()
#     tau = 1.2
#     adjustments = tau * (1-(np.log(label_freq_array+1e-4) / np.log(max_freq+1e-4)))
#     adjustments = torch.from_numpy(adjustments)
#     adjustments = adjustments.to(setting.device)
#
#     return adjustments
#
# def old_min_max_scale(coords, min_val=None, max_val=None):
#     if min_val is None or max_val is None:
#         min_val = coords.min(0, keepdim=True)[0]
#         max_val = coords.max(0, keepdim=True)[0]
#     scale_coords = (coords - min_val) / (max_val - min_val)
#     return scale_coords, min_val, max_val
#
# def min_max_scale(coords, lat_min, lat_max, lng_min, lng_max):
#     lat_coo, lng_coo = coords[:, :, [0]], coords[:, :, [1]]
#     scale_lat_coo = (lat_coo - lat_min) / (lat_max - lat_min)
#     scale_lng_coo = (lng_coo - lng_min) / (lng_max - lng_min)
#     scale_coords = torch.cat([scale_lat_coo, scale_lng_coo], dim=-1)
#
#     return scale_coords
#
# def focal_loss(pred_reg, y_reg, alpha=0.25, gamma=2.0):
#     """
#     pred_reg: Tensor, shape (T*B, region_count), 模型预测的logits
#     y_reg: Tensor, shape (T*B,), Groundtruth Region类别索引
#     alpha: float, Focal Loss的权重参数
#     gamma: float, Focal Loss的难样本关注参数
#     """
#     probs = torch.softmax(pred_reg, dim=-1)
#     pred_r_values, pred_r_indices = torch.topk(probs, k=10, dim=-1)
#
#     positive_probs = probs[range(len(y_reg)), y_reg]
#
#     focal_loss = -alpha * (1 - positive_probs)**gamma * torch.log(positive_probs + 1e-8)
#     margin_loss = torch.relu(1 + pred_r_values - positive_probs.unsqueeze(-1))
#
#     return focal_loss.mean(), margin_loss.mean()
#
# def compute_class_weights(y_reg, num_classes):
#     class_counts = torch.bincount(y_reg, minlength=num_classes)
#     class_weights = 1.0 / (class_counts + 1e-6)
#     class_weights = class_weights / class_weights.sum()
#
#     return class_weights
#
# if __name__ == '__main__':
#     graph_path = 'data/user_similarity_graph.pkl'
#     user_similarity_matrix = torch.tensor(load_graph_data(pkl_filename=graph_path))
#     print(user_similarity_matrix[1])
#     print('................')
#     print(user_similarity_matrix[1][:10])
#     count = 0
#
#     print('count: ', count)


import pickle
import numpy as np
import torch
import random
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
import scipy.sparse as sp
import os

seed = 0
global_seed = 0
torch.manual_seed(seed)


def load_graph_data(pkl_filename):
    graph = load_pickle(pkl_filename)  # list
    # graph = np.array(graph[0])
    return graph


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ': ', e)
        raise
    return pickle_data


def calculate_preference_similarity(m1, m2, pref):
    """
        m1: (user_len, hidden_size)
        m2: user_len, seq_len, hidden_size)
        return: calculate the similarity between user and location, which means user's preference about location
    """
    user_len = m1.shape[0]
    seq_len = m2.shape[1]
    pref = pref.squeeze()
    similarity = torch.zeros(user_len, seq_len, dtype=torch.float32)
    for i in range(user_len):
        v1 = m1[i]
        for j in range(seq_len):
            v2 = m2[i][j]
            similarity[i][j] = (1 + torch.cosine_similarity(v1 + pref, v2, dim=0).item()) / 2

    return similarity


def compute_preference(m1, m2, pref):
    m1 = (m1 + pref).unsqueeze(1)
    s = m1 - m2
    sim = torch.exp(-(torch.norm(s, p=2, dim=-1)))
    return sim


def get_user_static_preference(pref, locs):
    """
        pref: (user_len, seq_len)
        locs: (user_len, seq_len, hidden_size)
        return: 返回用户对于所访问POI的全局偏好
    """
    user_len, seq_len = pref.shape[0], pref.shape[1]
    hidden_size = locs.shape[2]
    user_preference = torch.zeros(user_len, seq_len, hidden_size)
    for i in range(user_len):
        for j in range(seq_len):
            user_preference[i][j] = torch.sum(torch.softmax(pref[i, :j + 1], dim=0).unsqueeze(1) * locs[i, :j + 1],
                                              dim=0)
    user_preference = user_preference.permute(1, 0, 2)

    return user_preference


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]
    init_label = torch.zeros(num_label, dtype=torch.int64)
    init_prob = torch.zeros(size=(num_label, num_neg + 1))

    for batch in range(num_label):
        random_ig = random.sample(range(l_m), num_neg)
        while label[batch].item() in random_ig:
            random_ig = random.sample(range(l_m), num_neg)

        # place the pos labels ahead and neg samples in the end
        for i in range(num_neg + 1):
            if i < 1:
                init_prob[batch, i] = prob[batch, label[batch]]
            else:
                init_prob[batch, i] = prob[batch, random_ig[i - 1]]

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)


def bprLoss(pos, neg, target=1.0):
    loss = - F.logsigmoid(target * (pos - neg))
    return loss.mean()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def top_transition_graph(transition_graph):
    graph = coo_matrix(transition_graph)
    data = graph.data
    row = graph.row
    threshold = 20

    for i in range(0, row.size, threshold):
        row_data = data[i: i + threshold]
        norm = row_data.max()
        row_data = row_data / norm
        data[i: i + threshold] = row_data

    return graph


def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def rotate_batch(head, relation, hidden):
    pi = 3.14159265358979323846
    device = head.device
    re_head, im_head = torch.chunk(head, 2, dim=2)

    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = torch.nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    phase_relation = relation / (embedding_range / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim=2)
    return score


def normalize(mx):
    '''Row-normalize sparse matrix'''
    for i in range(mx.shape[0]):
        b = mx[i, i] + 1
        mx[i, i] = 0
        tmp = mx[i].max()
        if b <= tmp:
            mx[i, i] = b - 1
        else:
            mx[i, i] = tmp * 1.2 if tmp > 0 else tmp + 10

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def calculate_distance(region2gps):
    adj = np.zeros((len(region2gps), len(region2gps)), dtype=np.float32)
    for region1, feature1 in region2gps.items():
        lat1, lng1 = feature1
        for region2, feature2 in region2gps.items():
            lat2, lng2 = feature2
            dis = haversine(lat1, lng1, lat2, lng2)
            adj[region1][region2] = np.exp(dis * -1)

    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)

    return mx


def geo_con_loss(preds, targets, pid_lat_lon):
    log_softmax = torch.nn.functional.log_softmax(preds, dim=-1)
    l_pred = torch.argmax(log_softmax, dim=-1)
    l_coor_pred = pid_lat_lon[l_pred]
    l_coor_tar = pid_lat_lon[targets]

    dlat = l_coor_pred[:, 0] - l_coor_tar[:, 0]
    dlon = l_coor_pred[:, 1] - l_coor_tar[:, 1]
    dist = dlat ** 2 + dlon ** 2
    loc_prob = log_softmax * dist.unsqueeze(-1)
    loss_geocons = F.nll_loss(loc_prob, targets, reduction='mean')

    return loss_geocons


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def trajectory_forecasting_loss(pred, true):
    return F.mse_loss(pred, true, reduction='mean')


def consistency_loss(pred_aux, pred_main, pid_lat_lon):
    log_softmax = torch.nn.functional.log_softmax(pred_main, dim=-1)
    l_pred = torch.argmax(log_softmax, dim=-1)
    pred_main = pid_lat_lon[l_pred]
    return F.mse_loss(pred_aux.view(-1, 2), pred_main, reduction='mean')


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def readEmbedFile(embedFile):
    input = open(embedFile, 'r')
    lines = []
    for line in input:
        lines.append(line)

    embeddings_dict = {}
    embeddings = []
    for lineId in range(1, len(lines)):
        splits = lines[lineId].split(' ')
        embedId = int(splits[0])
        embedValue = splits[1:]
        new_embedValue = [float(x) for x in embedValue]
        embeddings_dict[embedId] = new_embedValue

    for i in sorted(embeddings_dict):
        embeddings.append(embeddings_dict[i])

    embeddings = torch.from_numpy(np.array(embeddings)).float()

    return embeddings


def compute_adjustment(label_freq, setting):
    """compute the base probabilities"""

    label_freq_array = np.array(list(label_freq.values()))
    max_freq = label_freq_array.max()
    tau = 1.2
    adjustments = tau * (1 - (np.log(label_freq_array + 1e-4) / np.log(max_freq + 1e-4)))
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(setting.device)

    return adjustments


def old_min_max_scale(coords, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val = coords.min(0, keepdim=True)[0]
        max_val = coords.max(0, keepdim=True)[0]
    scale_coords = (coords - min_val) / (max_val - min_val)
    return scale_coords, min_val, max_val


def min_max_scale(coords, lat_min, lat_max, lng_min, lng_max):
    lat_coo, lng_coo = coords[:, :, [0]], coords[:, :, [1]]
    scale_lat_coo = (lat_coo - lat_min) / (lat_max - lat_min)
    scale_lng_coo = (lng_coo - lng_min) / (lng_max - lng_min)
    scale_coords = torch.cat([scale_lat_coo, scale_lng_coo], dim=-1)

    return scale_coords


def focal_loss(pred_reg, y_reg, alpha=0.25, gamma=2.0):
    """
    pred_reg: Tensor, shape (T*B, region_count), 模型预测的logits
    y_reg: Tensor, shape (T*B,), Groundtruth Region类别索引
    alpha: float, Focal Loss的权重参数
    gamma: float, Focal Loss的难样本关注参数
    """
    probs = torch.softmax(pred_reg, dim=-1)
    pred_r_values, pred_r_indices = torch.topk(probs, k=10, dim=-1)

    positive_probs = probs[range(len(y_reg)), y_reg]

    focal_loss = -alpha * (1 - positive_probs) ** gamma * torch.log(positive_probs + 1e-8)
    margin_loss = torch.relu(1 + pred_r_values - positive_probs.unsqueeze(-1))

    return focal_loss.mean(), margin_loss.mean()


def compute_class_weights(y_reg, num_classes):
    class_counts = torch.bincount(y_reg, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    return class_weights


if __name__ == '__main__':
    graph_path = 'data/user_similarity_graph.pkl'
    user_similarity_matrix = torch.tensor(load_graph_data(pkl_filename=graph_path))
    print(user_similarity_matrix[1])
    print('................')
    print(user_similarity_matrix[1][:10])
    count = 0

    print('count: ', count)

