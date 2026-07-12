# import torch
# import numpy as np
# from utils import log_string
# import time
#
#
# class Evaluation:
#     """
#     Handles evaluation on a given POI dataset and loader.
#
#     The two metrics are MAP and recall@n. Our model predicts sequence of
#     next locations determined by the sequence_length at one pass. During evaluation we
#     treat each entry of the sequence as single prediction. One such prediction
#     is the ranked list of all available locations and we can compute the two metrics.
#
#     As a single prediction is of the size of all available locations,
#     evaluation takes its time to compute. The code here is optimized.
#
#     Using the --report_user argument one can access the statistics per user.
#     """
#
#     def __init__(self, dataset, dataloader, user_count, loc_count, region_count, h0_strategy, trainer, setting, log):
#         self.dataset = dataset
#         self.dataloader = dataloader
#         self.user_count = user_count
#         self.loc_count = loc_count
#         self.region_count = region_count
#         self.h0_strategy = h0_strategy
#         self.trainer = trainer
#         self.setting = setting
#         self._log = log
#
#     def calculate_acc(self, output, label, ks):
#         topk_correct_counts = [
#             torch.sum(
#                 (torch.topk(output, k=top_k, dim=1)[1] + 0) == label.unsqueeze(1)
#             ).item()
#             for top_k in ks
#         ]
#         return np.array(topk_correct_counts)
#
#     def calculate_mrr(self, output, true_labels):
#         res = 0.0
#         for i, pred in enumerate(output):
#             sorted_indices = torch.argsort(pred, descending=True)
#             true_index = np.where(true_labels[i].cpu() == sorted_indices.cpu())[0]
#             if len(true_index) > 0:
#                 res += 1.0 / (true_index[0] + 1)
#         return res
#
#     def new_calculate_mrr(self, output, true_labels, output_reg, true_labels_reg):
#         res, res_reg = 0.0, 0.0
#         for i, (pred, label, pred_reg, label_reg) in enumerate(zip(output, true_labels, output_reg, true_labels_reg)):
#             sorted_indices = torch.argsort(pred, descending=True)
#             sorted_indices_reg = torch.argsort(pred_reg, descending=True)
#             true_index = np.where(label.cpu() == sorted_indices.cpu())[0]
#             true_index_reg = np.where(label_reg.cpu() == sorted_indices_reg.cpu())[0]
#             if len(true_index) > 0:
#                 res += 1.0 / (true_index[0] + 1)
#             if len(true_index_reg) > 0:
#                 res_reg += 1.0 / (true_index_reg[0] + 1)
#         return res, res_reg
#
#     def evaluate(self):
#         self.dataset.reset()
#         h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)
#
#         with torch.no_grad():
#             iter_cnt = 0
#             recall1 = 0
#             recall5 = 0
#             recall10 = 0
#             average_precision = 0.
#             recall1_r = 0
#             recall5_r = 0
#             recall10_r = 0
#             average_precision_r = 0.
#
#
#             total_samples = 0
#             top_k_values = [1, 5, 10]
#             top_k_correct_loc = np.array([0 for _ in range(len(top_k_values))])
#             precision_loc = 0
#             top_k_correct_reg = np.array([0 for _ in range(len(top_k_values))])
#             precision_reg = 0
#
#             u_iter_cnt = np.zeros(self.user_count)
#             u_recall1 = np.zeros(self.user_count)
#             u_recall5 = np.zeros(self.user_count)
#             u_recall10 = np.zeros(self.user_count)
#             u_average_precision = np.zeros(self.user_count)
#             reset_count = torch.zeros(self.user_count)
#             u_recall1_r = np.zeros(self.user_count)
#             u_recall5_r = np.zeros(self.user_count)
#             u_recall10_r = np.zeros(self.user_count)
#             u_average_precision_r = np.zeros(self.user_count)
#
#             for i, (x, t, t_slot, s, rg, rg_nv, rg_s, y, y_t, y_t_slot, y_s, y_rg, y_rg_nv, y_rg_s, reset_h, active_users) in enumerate(self.dataloader):
#                 active_users = active_users.squeeze()
#                 for j, reset in enumerate(reset_h):
#                     if reset:
#                         if self.setting.is_lstm:
#                             hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
#                             h[0][0, j] = hc[0]
#                             h[1][0, j] = hc[1]
#                         else:
#                             h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
#                         reset_count[active_users[j]] += 1
#
#                 # squeeze for reasons of "loader-batch-size-is-1"
#                 x = x.squeeze().to(self.setting.device)
#                 t = t.squeeze().to(self.setting.device)
#                 t_slot = t_slot.squeeze().to(self.setting.device)
#                 s = s.squeeze().to(self.setting.device)
#                 rg = rg.squeeze().long().to(self.setting.device)
#                 rg_nv = rg_nv.squeeze().long().to(self.setting.device)
#                 rg_s = rg_s.squeeze().to(self.setting.device)
#
#                 y = y.squeeze().to(self.setting.device)
#                 y_t = y_t.squeeze().to(self.setting.device)
#                 y_t_slot = y_t_slot.squeeze().to(self.setting.device)
#                 y_s = y_s.squeeze().to(self.setting.device)
#                 y_rg = y_rg.squeeze().long().to(self.setting.device)
#                 y_rg_nv = y_rg_nv.squeeze().long().to(self.setting.device)
#                 y_rg_s = y_rg_s.squeeze().to(self.setting.device)
#
#                 active_users = active_users.to(self.setting.device)
#                 out, out_r, h = self.trainer.evaluate(x, t, t_slot, s, rg, rg_nv, rg_s, y_t, y_t_slot, y_s, y_rg, y_rg_nv, y_rg_s, h, active_users)
#
#                 # ratio_samples += y.view(-1).size(0)
#                 valid_indices = [j for j in range(self.setting.batch_size) if reset_count[active_users[j]] <= 1]
#                 valid_indices = torch.tensor(valid_indices, dtype=torch.long)
#                 filtered_out = out[valid_indices]      # Shape: (B-b, T, x)
#                 filtered_out_r = out_r[valid_indices]  # Shape: (B-b, T, x)
#                 out = filtered_out.view(-1, self.loc_count)
#                 out_r = filtered_out_r.view(-1, self.region_count)
#                 poi_y = y.transpose(0, 1)[valid_indices].view(-1)
#                 reg_y = y_rg.transpose(0, 1)[valid_indices].view(-1)
#                 total_samples += poi_y.size(0)
#                 top_k_correct_loc += self.calculate_acc(out, poi_y, top_k_values)
#                 top_k_correct_reg += self.calculate_acc(out_r, reg_y, top_k_values)
#                 curr_mrr_loc, curr_mrr_reg = self.new_calculate_mrr(out, poi_y, out_r, reg_y)
#                 precision_loc += curr_mrr_loc
#                 precision_reg += curr_mrr_reg
#
#             formatter = "{0:.4f}"
#             log_string(self._log, 'POI recall@1: ' + formatter.format(top_k_correct_loc[0] / total_samples))
#             log_string(self._log, 'POI recall@5: ' + formatter.format(top_k_correct_loc[1] / total_samples))
#             log_string(self._log, 'POI recall@10: ' + formatter.format(top_k_correct_loc[2] / total_samples))
#             log_string(self._log, 'POI MAP: ' + formatter.format(precision_loc / total_samples))
#
#             log_string(self._log, 'Region recall@1: ' + formatter.format(top_k_correct_reg[0] / total_samples))
#             log_string(self._log, 'Region recall@5: ' + formatter.format(top_k_correct_reg[1] / total_samples))
#             log_string(self._log, 'Region recall@10: ' + formatter.format(top_k_correct_reg[2] / total_samples))
#             log_string(self._log, 'Region MAP: ' + formatter.format(precision_reg / total_samples))
#             log_string(self._log, 'predictions:' + formatter.format(total_samples))
#
#             #     for j in range(self.setting.batch_size):
#             #         # o contains a per user list of votes for all locations for each sequence entry
#             #         o = out[j]
#             #         o_r = out_r[j]
#
#             #         # partition elements
#             #         o_n = np.array(o.cpu().detach().numpy())
#             #         ind = np.argpartition(o_n, -10, axis=1)[:, -10:]  # top 10 elements
#             #         o_n_r = np.array(o_r.cpu().detach().numpy())
#             #         ind_r = np.argpartition(o_n_r, -10, axis=1)[:, -10:]  # top 10 elements
#
#             #         y_j = y[:, j]
#             #         y_j_r = y_rg[:, j]
#
#             #         for k in range(len(y_j)):
#             #             if reset_count[active_users[j]] > 1:
#             #                 continue  # skip already evaluated users.
#
#             #             # resort indices for k:
#             #             ind_k = ind[k]
#             #             r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]  # sort top 10 elements descending
#             #             ind_k_r = ind_r[k]
#             #             r_r = ind_k_r[np.argsort(-o_n_r[k, ind_k_r], axis=0)]  # sort top 10 elements descending
#
#             #             t = y_j[k].item()
#             #             t_r = y_j_r[k].item()
#
#             #             # compute MAP:
#             #             r_kj = o_n[k, :]
#             #             t_val = r_kj[t]
#             #             upper = np.where(r_kj > t_val)[0]
#             #             precision = 1. / (1 + len(upper))
#             #             r_kj_r = o_n_r[k, :]
#             #             t_val_r = r_kj_r[t_r]
#             #             upper_r = np.where(r_kj_r > t_val_r)[0]
#             #             precision_r = 1. / (1 + len(upper_r))
#
#             #             # store
#             #             u_iter_cnt[active_users[j]] += 1
#             #             u_recall1[active_users[j]] += t in r[:1]
#             #             u_recall5[active_users[j]] += t in r[:5]
#             #             u_recall10[active_users[j]] += t in r[:10]
#             #             u_average_precision[active_users[j]] += precision
#
#             #             u_recall1_r[active_users[j]] += t_r in r_r[:1]
#             #             u_recall5_r[active_users[j]] += t_r in r_r[:5]
#             #             u_recall10_r[active_users[j]] += t_r in r_r[:10]
#             #             u_average_precision_r[active_users[j]] += precision_r
#
#             # formatter = "{0:.4f}"
#             # for j in range(self.user_count):
#             #     iter_cnt += u_iter_cnt[j]
#             #     recall1 += u_recall1[j]
#             #     recall5 += u_recall5[j]
#             #     recall10 += u_recall10[j]
#             #     average_precision += u_average_precision[j]
#             #     recall1_r += u_recall1_r[j]
#             #     recall5_r += u_recall5_r[j]
#             #     recall10_r += u_recall10_r[j]
#             #     average_precision_r += u_average_precision_r[j]
#
#             #     if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
#             #         print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
#             #               formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
#             #               formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')
#
#             # log_string(self._log, 'POI recall@1: ' + formatter.format(recall1 / iter_cnt))
#             # log_string(self._log, 'POI recall@5: ' + formatter.format(recall5 / iter_cnt))
#             # log_string(self._log, 'POI recall@10: ' + formatter.format(recall10 / iter_cnt))
#             # log_string(self._log, 'POI MAP: ' + formatter.format(average_precision / iter_cnt))
#
#             # log_string(self._log, 'Region recall@1: ' + formatter.format(recall1_r / iter_cnt))
#             # log_string(self._log, 'Region recall@5: ' + formatter.format(recall5_r / iter_cnt))
#             # log_string(self._log, 'Region recall@10: ' + formatter.format(recall10_r / iter_cnt))
#             # log_string(self._log, 'Region MAP: ' + formatter.format(average_precision_r / iter_cnt))
#             # log_string(self._log, 'predictions: ' + formatter.format(iter_cnt))









import torch
import numpy as np
from utils import log_string
import time


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MAP and recall@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, loc_count, region_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.loc_count = loc_count
        self.region_count = region_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def calculate_acc(self, output, label, ks):
        topk_correct_counts = [
            torch.sum(
                (torch.topk(output, k=top_k, dim=1)[1] + 0) == label.unsqueeze(1)
            ).item()
            for top_k in ks
        ]
        return np.array(topk_correct_counts)

    def calculate_mrr(self, output, true_labels):
        res = 0.0
        for i, pred in enumerate(output):
            sorted_indices = torch.argsort(pred, descending=True)
            true_index = np.where(true_labels[i].cpu() == sorted_indices.cpu())[0]
            if len(true_index) > 0:
                res += 1.0 / (true_index[0] + 1)
        return res

    def new_calculate_mrr(self, output, true_labels, output_reg, true_labels_reg):
        res, res_reg = 0.0, 0.0
        for i, (pred, label, pred_reg, label_reg) in enumerate(zip(output, true_labels, output_reg, true_labels_reg)):
            sorted_indices = torch.argsort(pred, descending=True)
            sorted_indices_reg = torch.argsort(pred_reg, descending=True)
            true_index = np.where(label.cpu() == sorted_indices.cpu())[0]
            true_index_reg = np.where(label_reg.cpu() == sorted_indices_reg.cpu())[0]
            if len(true_index) > 0:
                res += 1.0 / (true_index[0] + 1)
            if len(true_index_reg) > 0:
                res_reg += 1.0 / (true_index_reg[0] + 1)
        return res, res_reg

    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.
            recall1_r = 0
            recall5_r = 0
            recall10_r = 0
            average_precision_r = 0.


            total_samples = 0
            top_k_values = [1, 5, 10]
            top_k_correct_loc = np.array([0 for _ in range(len(top_k_values))])
            precision_loc = 0
            top_k_correct_reg = np.array([0 for _ in range(len(top_k_values))])
            precision_reg = 0

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)
            u_recall1_r = np.zeros(self.user_count)
            u_recall5_r = np.zeros(self.user_count)
            u_recall10_r = np.zeros(self.user_count)
            u_average_precision_r = np.zeros(self.user_count)

            for i, (x, t, t_slot, s, rg, rg_nv, rg_s, y, y_t, y_t_slot, y_s, y_rg, y_rg_nv, y_rg_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)
                rg = rg.squeeze().long().to(self.setting.device)
                rg_nv = rg_nv.squeeze().long().to(self.setting.device)
                rg_s = rg_s.squeeze().to(self.setting.device)

                y = y.squeeze().to(self.setting.device)
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                y_rg = y_rg.squeeze().long().to(self.setting.device)
                y_rg_nv = y_rg_nv.squeeze().long().to(self.setting.device)
                y_rg_s = y_rg_s.squeeze().to(self.setting.device)

                active_users = active_users.to(self.setting.device)
                out, out_r, out_s, h = self.trainer.evaluate(x, t, t_slot, s, rg, rg_nv, rg_s, y_t, y_t_slot, y_s, y_rg, y_rg_nv, y_rg_s, h, active_users)

                # ratio_samples += y.view(-1).size(0)
                valid_indices = [j for j in range(self.setting.batch_size) if reset_count[active_users[j]] <= 1]
                valid_indices = torch.tensor(valid_indices, dtype=torch.long)
                filtered_out = out[valid_indices]      # Shape: (B-b, T, x)
                filtered_out_r = out_r[valid_indices]  # Shape: (B-b, T, x)
                filtered_out_s = out_s[valid_indices] if out_s is not None else None
                out = filtered_out.view(-1, self.loc_count)
                out_r = filtered_out_r.view(-1, self.region_count)
                out_s = filtered_out_s.view(-1, self.trainer.semantic_count) if filtered_out_s is not None else None
                out = self.trainer.filter_candidates(out, out_r, out_s)
                poi_y = y.transpose(0, 1)[valid_indices].view(-1)
                reg_y = y_rg.transpose(0, 1)[valid_indices].view(-1)
                total_samples += poi_y.size(0)
                top_k_correct_loc += self.calculate_acc(out, poi_y, top_k_values)
                top_k_correct_reg += self.calculate_acc(out_r, reg_y, top_k_values)
                curr_mrr_loc, curr_mrr_reg = self.new_calculate_mrr(out, poi_y, out_r, reg_y)
                precision_loc += curr_mrr_loc
                precision_reg += curr_mrr_reg

            formatter = "{0:.4f}"
            log_string(self._log, 'POI recall@1: ' + formatter.format(top_k_correct_loc[0] / total_samples))
            log_string(self._log, 'POI recall@5: ' + formatter.format(top_k_correct_loc[1] / total_samples))
            log_string(self._log, 'POI recall@10: ' + formatter.format(top_k_correct_loc[2] / total_samples))
            log_string(self._log, 'POI MAP: ' + formatter.format(precision_loc / total_samples))

            log_string(self._log, 'Region recall@1: ' + formatter.format(top_k_correct_reg[0] / total_samples))
            log_string(self._log, 'Region recall@5: ' + formatter.format(top_k_correct_reg[1] / total_samples))
            log_string(self._log, 'Region recall@10: ' + formatter.format(top_k_correct_reg[2] / total_samples))
            log_string(self._log, 'Region MAP: ' + formatter.format(precision_reg / total_samples))
            log_string(self._log, 'predictions:' + formatter.format(total_samples))

            #     for j in range(self.setting.batch_size):
            #         # o contains a per user list of votes for all locations for each sequence entry
            #         o = out[j]
            #         o_r = out_r[j]

            #         # partition elements
            #         o_n = np.array(o.cpu().detach().numpy())
            #         ind = np.argpartition(o_n, -10, axis=1)[:, -10:]  # top 10 elements
            #         o_n_r = np.array(o_r.cpu().detach().numpy())
            #         ind_r = np.argpartition(o_n_r, -10, axis=1)[:, -10:]  # top 10 elements

            #         y_j = y[:, j]
            #         y_j_r = y_rg[:, j]

            #         for k in range(len(y_j)):
            #             if reset_count[active_users[j]] > 1:
            #                 continue  # skip already evaluated users.

            #             # resort indices for k:
            #             ind_k = ind[k]
            #             r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]  # sort top 10 elements descending
            #             ind_k_r = ind_r[k]
            #             r_r = ind_k_r[np.argsort(-o_n_r[k, ind_k_r], axis=0)]  # sort top 10 elements descending

            #             t = y_j[k].item()
            #             t_r = y_j_r[k].item()

            #             # compute MAP:
            #             r_kj = o_n[k, :]
            #             t_val = r_kj[t]
            #             upper = np.where(r_kj > t_val)[0]
            #             precision = 1. / (1 + len(upper))
            #             r_kj_r = o_n_r[k, :]
            #             t_val_r = r_kj_r[t_r]
            #             upper_r = np.where(r_kj_r > t_val_r)[0]
            #             precision_r = 1. / (1 + len(upper_r))

            #             # store
            #             u_iter_cnt[active_users[j]] += 1
            #             u_recall1[active_users[j]] += t in r[:1]
            #             u_recall5[active_users[j]] += t in r[:5]
            #             u_recall10[active_users[j]] += t in r[:10]
            #             u_average_precision[active_users[j]] += precision

            #             u_recall1_r[active_users[j]] += t_r in r_r[:1]
            #             u_recall5_r[active_users[j]] += t_r in r_r[:5]
            #             u_recall10_r[active_users[j]] += t_r in r_r[:10]
            #             u_average_precision_r[active_users[j]] += precision_r

            # formatter = "{0:.4f}"
            # for j in range(self.user_count):
            #     iter_cnt += u_iter_cnt[j]
            #     recall1 += u_recall1[j]
            #     recall5 += u_recall5[j]
            #     recall10 += u_recall10[j]
            #     average_precision += u_average_precision[j]
            #     recall1_r += u_recall1_r[j]
            #     recall5_r += u_recall5_r[j]
            #     recall10_r += u_recall10_r[j]
            #     average_precision_r += u_average_precision_r[j]

            #     if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
            #         print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
            #               formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
            #               formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            # log_string(self._log, 'POI recall@1: ' + formatter.format(recall1 / iter_cnt))
            # log_string(self._log, 'POI recall@5: ' + formatter.format(recall5 / iter_cnt))
            # log_string(self._log, 'POI recall@10: ' + formatter.format(recall10 / iter_cnt))
            # log_string(self._log, 'POI MAP: ' + formatter.format(average_precision / iter_cnt))

            # log_string(self._log, 'Region recall@1: ' + formatter.format(recall1_r / iter_cnt))
            # log_string(self._log, 'Region recall@5: ' + formatter.format(recall5_r / iter_cnt))
            # log_string(self._log, 'Region recall@10: ' + formatter.format(recall10_r / iter_cnt))
            # log_string(self._log, 'Region MAP: ' + formatter.format(average_precision_r / iter_cnt))
            # log_string(self._log, 'predictions: ' + formatter.format(iter_cnt))



# --dataset
# checkins-4sq-nyc.txt
# --hidden-dim
# 30
# --weight_decay
# 1e-5
# --gpu
# 0
# --validate-epoch
# 5
# --validate-epoch
# 5
# --use-sem-branch
# --semantic-file
# poi_semantics_4sq_nyc.txt
# --sem-weight
# 0.3
# --filter-candidate
# --topm-candidate
# 5000
# --semantic-fusion-weight
# 0.5
# --sem-label-smoothing
# 0.05
# --sem-warmup-epochs
# 5
# --candidate-mode
# score
# --candidate-sem-weight
# 0.2
# --candidate-region-weight
# 0.1

