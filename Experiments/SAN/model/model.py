import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from model.untils import matrix_mul, element_wise_mul


def get_hit_info_cwe(b, scores, result, max_node_num):
    q3_fd_t3, q3_fd_t1, q3_fd_t50, q3_fd_t30, q3_fd_total = 0, 0, 0, 0, 0
    q3_bav_t50, q3_bav_t30, q3_bav_total = 0, 0, 0
    q3_bse_t3, q3_bse_t1, q3_bse_t50, q3_bse_t30, q3_bse_total = 0, 0, 0, 0, 0
    q3_bsk_t3, q3_bsk_t1, q3_bsk_t50, q3_bsk_t30, q3_bsk_total = 0, 0, 0, 0, 0
    q3_sesk_t3, q3_sesk_t1, q3_sesk_t50, q3_sesk_t30, q3_sesk_total = 0, 0, 0, 0, 0

    scores = scores.detach().cpu().numpy().tolist()

    real_pos_idx = [idx for idx, e in enumerate(b) if e['target'] == 1]
    predict_pos_idx = [idx for idx, r in enumerate(result) if r >= 0.5]

    real_neg_idx = [idx for idx, e in enumerate(b) if e['target'] == 0]
    predict_neg_idx = [idx for idx, r in enumerate(result) if r < 0.5]

    hit_pos_idx = []
    for i in real_pos_idx:
        if i in predict_pos_idx:
            # 依次为图索引、图实际结点数(扩充前)、[该图漏洞发生结点索引]
            be_order = [node_order for node_order, n in enumerate(b[i]['nodes']) if 'bad_source' in n.keys()]
            bk_order = [node_order for node_order, n in enumerate(b[i]['nodes']) if 'bad_sink' in n.keys()]
            be_bk_order = [node_order for node_order, n in enumerate(b[i]['nodes'])
                           if 'bad_source' in n.keys() or 'bad_sink' in n.keys()]
            hit_pos_idx.append([i, len(b[i]['nodes']), be_order, bk_order, be_bk_order])

    hit_neg_idx = []

    for i in real_neg_idx:
        if i in predict_neg_idx:
            hit_neg_idx.append([i, len(b[i]['nodes']), [node_order for node_order, n in enumerate(b[i]['nodes'])
                                                        if 'fixed' in n.keys()]])

    for h in hit_pos_idx:
        # 如果结点数大于最大结点数，不参与计算，该类型比例很小，不会影响整体结果
        if h[1] > max_node_num:
            continue
        # 获得该图的所有实际结点得分(即去除扩充的结点)
        needed_scores = scores[h[0]][:h[1]]
        # 获得这些得分的降序索引序列
        sorted_id = sorted(range(len(needed_scores)), key=lambda k: needed_scores[k], reverse=True)
        # 若数据没有bad_source和bad_sink,则跳过
        if len(h[4]) == 0:
            continue
        # 若数据结点数小于5，也跳过
        if h[1] < 5:
            continue
        q3_sesk_total += 1
        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 1:
                q3_sesk_t1 += 1
                break
        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 3:
                q3_sesk_t3 += 1
                break
        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.5:
                q3_sesk_t50 += 1
                break
        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.3:
                q3_sesk_t30 += 1
                break

        if len(h[2]) != 0:
            q3_bse_total += 1
            for order in h[2]:
                idx_idx = sorted_id.index(order)
                if idx_idx + 1 <= 1:
                    q3_bse_t1 += 1
                    break
            for order in h[2]:
                idx_idx = sorted_id.index(order)
                if idx_idx + 1 <= 3:
                    q3_bse_t3 += 1
                    break
            for order in h[2]:
                idx_idx = sorted_id.index(order)
                if (idx_idx + 1) / h[1] <= 0.5:
                    q3_bse_t50 += 1
                    break
            for order in h[2]:
                idx_idx = sorted_id.index(order)
                if (idx_idx + 1) / h[1] <= 0.3:
                    q3_bse_t30 += 1
                    break
        if len(h[3]) != 0:
            q3_bsk_total += 1
            for order in h[3]:
                idx_idx = sorted_id.index(order)
                if idx_idx + 1 <= 1:
                    q3_bsk_t1 += 1
                    break
            for order in h[3]:
                idx_idx = sorted_id.index(order)
                if idx_idx + 1 <= 3:
                    q3_bsk_t3 += 1
                    break
            for order in h[3]:
                idx_idx = sorted_id.index(order)
                if (idx_idx + 1) / h[1] <= 0.5:
                    q3_bsk_t50 += 1
                    break
            for order in h[3]:
                idx_idx = sorted_id.index(order)
                if (idx_idx + 1) / h[1] <= 0.3:
                    q3_bsk_t30 += 1
                    break
        if len(h[2]) == 1 and len(h[3]) == 1:
            q3_bav_total += 1
            se_idx = sorted_id.index(h[2][0]) + 1
            sk_idx = sorted_id.index(h[3][0]) + 1
            if ((se_idx + sk_idx) / 2) / h[1] <= 0.5:
                q3_bav_t50 += 1
            if ((se_idx + sk_idx) / 2) / h[1] <= 0.3:
                q3_bav_t30 += 1

    for h in hit_neg_idx:
        # 如果结点数大于最大结点数，不参与计算，该类型比例很小，不会影响整体结果
        if h[1] > max_node_num:
            continue
        # 获得该图的所有实际结点得分(即去除扩充的结点)
        needed_scores = scores[h[0]][:h[1]]
        # 获得这些得分的降序索引序列
        sorted_id = sorted(range(len(needed_scores)), key=lambda k: needed_scores[k], reverse=True)
        if len(h[2]) == 0:
            continue
        if h[1] < 5:
            continue
        q3_fd_total += 1
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 1:
                q3_fd_t1 += 1
                break
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 3:
                q3_fd_t3 += 1
                break
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.5:
                q3_fd_t50 += 1
                break
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.3:
                q3_fd_t30 += 1
                break
    q31 = [q3_fd_t3, q3_fd_t1, q3_fd_t50, q3_fd_t30, q3_fd_total]
    q32 = [q3_bav_t50, q3_bav_t30, q3_bav_total]
    q33 = [q3_bse_t3, q3_bse_t1, q3_bse_t50, q3_bse_t30, q3_bse_total]
    q34 = [q3_bsk_t3, q3_bsk_t1, q3_bsk_t50, q3_bsk_t30, q3_bsk_total]
    q35 = [q3_sesk_t3, q3_sesk_t1, q3_sesk_t50, q3_sesk_t30, q3_sesk_total]
    q3 = [q31, q32, q33, q34, q35]
    return q3


def get_hit_info(b, scores, result, max_node_num):
    q1_fd_t3, q1_fd_t1, q1_fd_t50, q1_fd_t30, q1_fd_total = 0, 0, 0, 0, 0
    q1_bav_t50, q1_bav_t30, q1_bav_total = 0, 0, 0
    q1_sesk_t3, q1_sesk_t1, q1_sesk_t50, q1_sesk_t30, q1_sesk_total = 0, 0, 0, 0, 0

    q2_0_fd_t5, q2_0_fd_t3, q2_0_fd_t1, q2_0_fd_t50, q2_0_fd_t30, q2_0_fd_total = 0, 0, 0, 0, 0, 0
    q2_0_bav_t50, q2_0_bav_t30, q2_0_bav_total = 0, 0, 0
    q2_0_sesk_t5, q2_0_sesk_t3, q2_0_sesk_t1, q2_0_sesk_t50, q2_0_sesk_t30, q2_0_sesk_total = 0, 0, 0, 0, 0, 0

    q2_1_fd_t10, q2_1_fd_t5, q2_1_fd_t3, q2_1_fd_t1, q2_1_fd_t50, q2_1_fd_t30, q2_1_fd_total = 0, 0, 0, 0, 0, 0, 0
    q2_1_bav_t50, q2_1_bav_t30, q2_1_bav_total = 0, 0, 0
    q2_1_sesk_t10, q2_1_sesk_t5, q2_1_sesk_t3, q2_1_sesk_t1, q2_1_sesk_t50, q2_1_sesk_t30, q2_1_sesk_total = 0, 0, 0, 0, 0, 0, 0

    q2_2_fd_t10, q2_2_fd_t5, q2_2_fd_t3, q2_2_fd_t1, q2_2_fd_t50, q2_2_fd_t30, q2_2_fd_total = 0, 0, 0, 0, 0, 0, 0
    q2_2_bav_t50, q2_2_bav_t30, q2_2_bav_total = 0, 0, 0
    q2_2_sesk_t10, q2_2_sesk_t5, q2_2_sesk_t3, q2_2_sesk_t1, q2_2_sesk_t50, q2_2_sesk_t30, q2_2_sesk_total = 0, 0, 0, 0, 0, 0, 0

    # total_valid_node_pos = 0
    # total_valid_node_pos_happen = 0
    #
    # total_valid_node_neg = 0
    # total_valid_node_neg_happen = 0

    scores = scores.detach().cpu().numpy().tolist()

    real_pos_idx = [idx for idx, e in enumerate(b) if e['target'] == 1]
    predict_pos_idx = [idx for idx, r in enumerate(result) if r >= 0.5]

    real_neg_idx = [idx for idx, e in enumerate(b) if e['target'] == 0]
    predict_neg_idx = [idx for idx, r in enumerate(result) if r < 0.5]

    hit_pos_idx = []
    for i in real_pos_idx:
        if i in predict_pos_idx:
            # 依次为图索引、图实际结点数(扩充前)、[该图漏洞发生结点索引]
            be_order = [node_order for node_order, n in enumerate(b[i]['nodes']) if 'bad_source' in n.keys()]
            bk_order = [node_order for node_order, n in enumerate(b[i]['nodes']) if 'bad_sink' in n.keys()]
            be_bk_order = [node_order for node_order, n in enumerate(b[i]['nodes'])
                           if 'bad_source' in n.keys() or 'bad_sink' in n.keys()]
            hit_pos_idx.append([i, len(b[i]['nodes']), be_order, bk_order, be_bk_order])

    hit_neg_idx = []

    for i in real_neg_idx:
        if i in predict_neg_idx:
            hit_neg_idx.append([i, len(b[i]['nodes']), [node_order for node_order, n in enumerate(b[i]['nodes'])
                                                        if 'fixed' in n.keys()]])
    for h in hit_pos_idx:
        # 如果结点数大于最大结点数，不参与计算，该类型比例很小，不会影响整体结果
        if h[1] > max_node_num:
            continue
        # 获得该图的所有实际结点得分(即去除扩充的结点)
        needed_scores = scores[h[0]][:h[1]]
        # 获得这些得分的降序索引序列
        sorted_id = sorted(range(len(needed_scores)), key=lambda k: needed_scores[k], reverse=True)
        # 若数据没有bad_source和bad_sink,则跳过
        if len(h[4]) == 0:
            continue
        # 若数据结点数小于5，也跳过
        if h[1] < 5:
            continue

        # total_valid_node_pos += h[1]
        # total_valid_node_pos_happen += len(h[4])

        # 计算q1
        q1_sesk_total += 1
        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 1:
                q1_sesk_t1 += 1
                break  # 只要有一个结点被hit该样例就算被hit

        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 3:
                q1_sesk_t3 += 1
                break

        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.5:
                q1_sesk_t50 += 1
                break

        for order in h[4]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.3:
                q1_sesk_t30 += 1
                break
        # 计算bav
        if len(h[2]) == 1 and len(h[3]) == 1:
            q1_bav_total += 1
            se_idx = sorted_id.index(h[2][0]) + 1
            sk_idx = sorted_id.index(h[3][0]) + 1
            if ((se_idx + sk_idx) / 2) / h[1] <= 0.5:
                q1_bav_t50 += 1
            if ((se_idx + sk_idx) / 2) / h[1] <= 0.3:
                q1_bav_t30 += 1

        # 计算q2
        if 5 <= h[1] < 10:
            q2_0_sesk_total += 1
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 1:
                    q2_0_sesk_t1 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 3:
                    q2_0_sesk_t3 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 5:
                    q2_0_sesk_t5 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.5:
                    q2_0_sesk_t50 += 1
                    break

            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.3:
                    q2_0_sesk_t30 += 1
                    break
            if len(h[2]) == 1 and len(h[3]) == 1:
                q2_0_bav_total += 1
                se_index = sorted_id.index(h[2][0]) + 1
                sk_index = sorted_id.index(h[3][0]) + 1
                if ((se_index + sk_index) / 2) / h[1] <= 0.5:
                    q2_0_bav_t50 += 1
                if ((se_index + sk_index) / 2) / h[1] <= 0.3:
                    q2_0_bav_t30 += 1
        elif 10 <= h[1] < 20:
            q2_1_sesk_total += 1
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 1:
                    q2_1_sesk_t1 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 3:
                    q2_1_sesk_t3 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 5:
                    q2_1_sesk_t5 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 10:
                    q2_1_sesk_t10 += 1
                    break

            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.5:
                    q2_1_sesk_t50 += 1
                    break

            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.3:
                    q2_1_sesk_t30 += 1
                    break

            if len(h[2]) == 1 and len(h[3]) == 1:
                q2_1_bav_total += 1
                se_index = sorted_id.index(h[2][0]) + 1
                sk_index = sorted_id.index(h[3][0]) + 1
                if ((se_index + sk_index) / 2) / h[1] <= 0.5:
                    q2_1_bav_t50 += 1
                if ((se_index + sk_index) / 2) / h[1] <= 0.3:
                    q2_1_bav_t30 += 1
        elif h[1] >= 20:
            q2_2_sesk_total += 1
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 1:
                    q2_2_sesk_t1 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 3:
                    q2_2_sesk_t3 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 5:
                    q2_2_sesk_t5 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 10:
                    q2_2_sesk_t10 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.5:
                    q2_2_sesk_t50 += 1
                    break
            for n_order in h[4]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.3:
                    q2_2_sesk_t30 += 1
                    break
            if len(h[2]) == 1 and len(h[3]) == 1:
                q2_2_bav_total += 1
                se_index = sorted_id.index(h[2][0]) + 1
                sk_index = sorted_id.index(h[3][0]) + 1
                if ((se_index + sk_index) / 2) / h[1] <= 0.5:
                    q2_2_bav_t50 += 1
                if ((se_index + sk_index) / 2) / h[1] <= 0.3:
                    q2_2_bav_t30 += 1

    for h in hit_neg_idx:
        # 如果结点数大于最大结点数，不参与计算，该类型比例很小，不会影响整体结果
        if h[1] > max_node_num:
            continue
        # 获得该图的所有实际结点得分(即去除扩充的结点)
        needed_scores = scores[h[0]][:h[1]]
        # 获得这些得分的降序索引序列
        sorted_id = sorted(range(len(needed_scores)), key=lambda k: needed_scores[k], reverse=True)
        if len(h[2]) == 0:
            continue
        if h[1] < 5:
            continue

        q1_fd_total += 1
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 1:
                q1_fd_t1 += 1
                break
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if idx_idx + 1 <= 3:
                q1_fd_t3 += 1
                break
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.5:
                q1_fd_t50 += 1
                break
        for order in h[2]:
            idx_idx = sorted_id.index(order)
            if (idx_idx + 1) / h[1] <= 0.3:
                q1_fd_t30 += 1
                break
        if 5 <= h[1] < 10:
            q2_0_fd_total += 1
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 1:
                    q2_0_fd_t1 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 3:
                    q2_0_fd_t3 += 1
                    break

            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 5:
                    q2_0_fd_t5 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.5:
                    q2_0_fd_t50 += 1
                    break

            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.3:
                    q2_0_fd_t30 += 1
                    break
        elif 10 <= h[1] < 20:
            q2_1_fd_total += 1
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 1:
                    q2_1_fd_t1 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 3:
                    q2_1_fd_t3 += 1
                    break

            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 5:
                    q2_1_fd_t5 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 10:
                    q2_1_fd_t10 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.5:
                    q2_1_fd_t50 += 1
                    break

            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.3:
                    q2_1_fd_t30 += 1
                    break
        elif h[1] >= 20:
            q2_2_fd_total += 1
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 1:
                    q2_2_fd_t1 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 3:
                    q2_2_fd_t3 += 1
                    break

            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 5:
                    q2_2_fd_t5 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if index_index + 1 <= 10:
                    q2_2_fd_t10 += 1
                    break
            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.5:
                    q2_2_fd_t50 += 1
                    break

            for n_order in h[2]:
                index_index = sorted_id.index(n_order)
                if (index_index + 1) / h[1] <= 0.3:
                    q2_2_fd_t30 += 1
                    break
    q11 = [q1_fd_t3, q1_fd_t1, q1_fd_t50, q1_fd_t30, q1_fd_total]
    q12 = [q1_bav_t50, q1_bav_t30, q1_bav_total]
    q13 = [q1_sesk_t3, q1_sesk_t1, q1_sesk_t50, q1_sesk_t30, q1_sesk_total]
    q201 = [q2_0_fd_t5, q2_0_fd_t3, q2_0_fd_t1, q2_0_fd_t50, q2_0_fd_t30, q2_0_fd_total]
    q202 = [q2_0_bav_t50, q2_0_bav_t30, q2_0_bav_total]
    q203 = [q2_0_sesk_t5, q2_0_sesk_t3, q2_0_sesk_t1, q2_0_sesk_t50, q2_0_sesk_t30, q2_0_sesk_total]
    q211 = [q2_1_fd_t10, q2_1_fd_t5, q2_1_fd_t3, q2_1_fd_t1, q2_1_fd_t50, q2_1_fd_t30, q2_1_fd_total]
    q212 = [q2_1_bav_t50, q2_1_bav_t30, q2_1_bav_total]
    q213 = [q2_1_sesk_t10, q2_1_sesk_t5, q2_1_sesk_t3, q2_1_sesk_t1, q2_1_sesk_t50, q2_1_sesk_t30, q2_1_sesk_total]
    q221 = [q2_2_fd_t10, q2_2_fd_t5, q2_2_fd_t3, q2_2_fd_t1, q2_2_fd_t50, q2_2_fd_t30, q2_2_fd_total]
    q222 = [q2_2_bav_t50, q2_2_bav_t30, q2_2_bav_total]
    q223 = [q2_2_sesk_t10, q2_2_sesk_t5, q2_2_sesk_t3, q2_2_sesk_t1, q2_2_sesk_t50, q2_2_sesk_t30, q2_2_sesk_total]
    q1 = [q11, q12, q13]
    q2 = [[q201, q202, q203], [q211, q212, q213], [q221, q222, q223]]
    return q1, q2


class GraphWithSAN(nn.Module):
    def __init__(self, word2vec, node_type_dic, max_sen_len, conv_layer_info, sen_att_layer_info):
        super(GraphWithSAN, self).__init__()

        self.max_sen_len = max_sen_len
        self.node_type_dic = node_type_dic
        # 结点信息嵌入层
        self.word_embed_size = word2vec.shape[1]
        word_size = word2vec.shape[0]
        unknown_word = np.zeros((1, self.word_embed_size))
        self.word2vec = torch.from_numpy(np.concatenate([unknown_word, word2vec], axis=0).astype(np.float32)).cuda()
        self.lookup = nn.Embedding(num_embeddings=word_size + 1,
                                   embedding_dim=self.word_embed_size).from_pretrained(self.word2vec, freeze=True)

        self.cov1d = nn.Conv1d(in_channels=conv_layer_info[0], out_channels=conv_layer_info[1],
                               kernel_size=conv_layer_info[2])

        # 句子注意力层
        self.sent_weight = nn.Parameter(torch.Tensor(2 * sen_att_layer_info[0], 2 * sen_att_layer_info[0]))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sen_att_layer_info[0]))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sen_att_layer_info[0], 1))
        self.gru = nn.GRU(conv_layer_info[1] + len(self.node_type_dic), sen_att_layer_info[0], bidirectional=True,
                          batch_first=True)
        self._create_weights(mean=0.0, std=0.05)

        # 线性分类
        self.classifier = nn.Linear(in_features=2 * sen_att_layer_info[0], out_features=1)
        self.sigmoid = nn.Sigmoid()

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, batch, if_count_hit=False, if_cwe=False):
        all_graph = []
        for g in batch:
            # 每个样本有x个结点，每个结点有K个词，每个词的长度是50
            input_ = torch.tensor(g['nodes'][0]['tokens_order'], dtype=torch.long).cuda()
            nodes_tensor = self.lookup(input_)
            nodes_tensor = nodes_tensor.unsqueeze(dim=0)
            # 这里结点个数应该都是大于1的
            for n in g['nodes'][1:]:
                input_ = torch.tensor(n['tokens_order'], dtype=torch.long).cuda()
                node_tensor = self.lookup(input_).unsqueeze(dim=0)
                nodes_tensor = torch.cat((nodes_tensor, node_tensor), dim=0)
            # print(nodes_tensor.shape)

            # 卷积
            g_info = self.cov1d(nodes_tensor.permute(0, 2, 1))
            # 每个卷积核得到的值进行取平均
            g_info = torch.mean(g_info, dim=-1)
            # 获得每个结点的类型信息，并创建对应的one-hot向量，并和其语义信息连接
            n_type_idx = []
            for n in g['nodes']:
                n_type_idx.append(self.node_type_dic[n['type']])
            n_type_one_hot = F.one_hot(torch.tensor(n_type_idx, dtype=torch.long).cuda(),
                                       num_classes=len(self.node_type_dic)).float()
            g_info = torch.cat((g_info, n_type_one_hot), dim=1)
            all_graph.append(g_info)
        for i, v in enumerate(all_graph):
            if v.shape[0] < self.max_sen_len:
                all_graph[i] = torch.cat(
                    (v, torch.zeros(size=(self.max_sen_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                    device=v.device)), dim=0)
            elif v.shape[0] > self.max_sen_len:
                all_graph[i] = all_graph[i][:self.max_sen_len]
        sen_input = torch.stack(all_graph, 0)
        # 进入句子级注意力层
        gru_output, _ = self.gru(sen_input)
        # print(gru_output.shape)
        output = matrix_mul(gru_output, self.sent_weight, self.sent_bias)
        # print(output.shape)
        output = matrix_mul(output, self.context_weight)
        # scores1 = output.clone().detach()
        # print(output.shape)
        # 这里就是得分
        output = F.softmax(output, dim=1)

        scores = output.clone().detach()
        # true_pos_num, batch_hit_rate_sum = self.get_hit_info(batch, output)

        # print(output.shape)
        output = element_wise_mul(gru_output, output)
        # print(output.shape)

        # 最后的线性分类层
        output = self.classifier(output)
        result = self.sigmoid(output).squeeze(dim=-1)
        if if_count_hit:
            q1, q2 = get_hit_info(batch, scores, result, self.max_sen_len)
            return result, q1, q2

        if if_cwe:
            q3 = get_hit_info_cwe(batch, scores, result, self.max_sen_len)
            return q3
        return result
