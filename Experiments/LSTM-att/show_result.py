import copy
import json as js
import warnings

from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from torch.nn import BCELoss
from torch.optim import Adam
import torch
from model.model import Att_BLSTM
import os
import shutil
import csv

warnings.filterwarnings("ignore")

result_dir = "show_result"

trained_model_path = "E:/COMPSAC/final paper materials/experiments/Att_BLSTM/result_save_new/model_0/SAN.bin"


def get_batches_idx(total, if_shuffle):
    indices = np.arange(0, total - 1, 1)
    if if_shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    end = len(indices)
    curr = 0
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])
        curr = c_end
    return batch_indices


with open('data/node_type_dic.json', 'r') as f:
    node_type_dic = js.load(f)
    f.close()

# only label the vulnerability-related code lines for test data
with open('data/labeled_processed_test_data.json', 'r') as f:
    test_data = np.array(js.load(f))
    f.close()

w2v = Word2Vec.load('data/w2v.model')
print("data loaded")

batch_size = 256
shuffle = True

# Conv para
conv_in_fea_size = w2v.wv.vectors.shape[1]  # same as the size of w2v' word vector
conv_out_fea_size = 50
kernel_size = 2

# the max number of nods. Actually, almost all samples' number of nodes is less than 50
max_sen_size = 50

hidden_size = 50

model = Att_BLSTM(w2v.wv.vectors, node_type_dic, max_sen_size,
                  [conv_in_fea_size, conv_out_fea_size, kernel_size],
                  [hidden_size])
model.cuda()
model.load_state_dict(torch.load(trained_model_path))
model.eval()
with torch.no_grad():
    # level_0: >=5<10  level_1: >=10<20 level_3 >=20
    Q1 = [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0]]
    Q2 = [[[0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]]
    Q3 = [[[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
    all_predictions, all_targets = [], []
    test_batch_index = get_batches_idx(len(test_data), False)
    print('total test batch {}'.format(len(test_batch_index)))
    for test_it, test_idx in enumerate(test_batch_index):
        print(test_it)
        targets = [t['target'] for t in test_data[test_idx]]
        targets = torch.tensor(targets, dtype=torch.float).cuda()
        predictions, q1, q2 = model(test_data[test_idx], True)

        # total_node_number_of_valid_pos += happen_rate_info[0]
        # total_node_number_of_valid_pos_happen += happen_rate_info[1]
        for i in range(len(q1)):
            for j in range(len(q1[i])):
                Q1[i][j] += q1[i][j]
        for i in range(len(q2)):
            for j in range(len(q2[i])):
                for k in range(len(q2[i][j])):
                    Q2[i][j][k] += q2[i][j][k]

        predictions = predictions.detach().cpu()
        all_predictions.extend(
            predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                dtype=torch.int32).numpy().tolist()
        )
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    with open(result_dir + '/' + 'b_sesk_and_fd_level_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pos_l0_number', 'pos_l1_number', 'pos_l2_number', 'neg_l0_number', 'neg_l1_number',
                         'neg_l2_number'])
        writer.writerow([Q2[0][-1][-1], Q2[1][-1][-1], Q2[2][-1][-1], Q2[0][0][-1], Q2[1][0][-1], Q2[2][0][-1]])
        f.close()

    with open(result_dir + '/' + 'Q1_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hit_type', 'Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
        if Q1[0][-1] == 0:
            writer.writerow(['fixed', '-', '-', '-', '-'])
        else:
            writer.writerow(['fixed', Q1[0][0] / Q1[0][-1], Q1[0][1] / Q1[0][-1], Q1[0][2] / Q1[0][-1],
                             Q1[0][3] / Q1[0][-1]])
        if Q1[1][-1] == 0:
            writer.writerow(['b_av', '-', '-', '-', '-'])
        else:
            writer.writerow(['b_av', '-', '-', Q1[1][0] / Q1[1][-1], Q1[1][1] / Q1[1][-1]])
        if Q1[2][-1] == 0:
            writer.writerow(['b_sesk', '-', '-', '-', '-'])
        else:
            writer.writerow(['b_sesk', Q1[2][0] / Q1[2][-1], Q1[2][1] / Q1[2][-1], Q1[2][2] / Q1[2][-1],
                             Q1[2][3] / Q1[2][-1]])
        f.close()

    with open(result_dir + '/' + 'Q2_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_number', 'hit_type', 'Hit10', 'Hit5', 'Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
        if Q2[0][0][-1] == 0:
            writer.writerow(['>=5<10', 'fixed', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=5<10', 'fixed', '-', Q2[0][0][0] / Q2[0][0][-1], Q2[0][0][1] / Q2[0][0][-1],
                             Q2[0][0][2] / Q2[0][0][-1], Q2[0][0][3] / Q2[0][0][-1],
                             Q2[0][0][4] / Q2[0][0][-1]])

        if Q2[0][1][-1] == 0:
            writer.writerow(['>=5<10', 'b_av', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(
                ['>=5<10', 'b_av', '-', '-', '-', '-', Q2[0][1][0] / Q2[0][1][-1], Q2[0][1][1] / Q2[0][1][-1]])

        if Q2[0][2][-1] == 0:
            writer.writerow(['>=5<10', 'b_sesk', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['<=5<10', 'b_sesk', '-', Q2[0][2][0] / Q2[0][2][-1], Q2[0][2][1] / Q2[0][2][-1],
                             Q2[0][2][2] / Q2[0][2][-1], Q2[0][2][3] / Q2[0][2][-1],
                             Q2[0][2][4] / Q2[0][2][-1]])

        if Q2[1][0][-1] == 0:
            writer.writerow(['>=10<20', 'fixed', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=10<20', 'fixed', Q2[1][0][0] / Q2[1][0][-1], Q2[1][0][1] / Q2[1][0][-1],
                             Q2[1][0][2] / Q2[1][0][-1], Q2[1][0][3] / Q2[1][0][-1], Q2[1][0][4] / Q2[1][0][-1],
                             Q2[1][0][5] / Q2[1][0][-1]])

        if Q2[1][1][-1] == 0:
            writer.writerow(['>=10<20', 'b_av', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(
                ['>=10<20', 'b_av', '-', '-', '-', '-', Q2[1][1][0] / Q2[1][1][-1], Q2[1][1][1] / Q2[1][1][-1]])

        if Q2[1][2][-1] == 0:
            writer.writerow(['>=10<20', 'b_sesk', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=10<20', 'b_sesk', Q2[1][2][0] / Q2[1][2][-1], Q2[1][2][1] / Q2[1][2][-1],
                             Q2[1][2][2] / Q2[1][2][-1], Q2[1][2][3] / Q2[1][2][-1], Q2[1][2][4] / Q2[1][2][-1],
                             Q2[1][2][5] / Q2[1][2][-1]])

        if Q2[2][0][-1] == 0:
            writer.writerow(['>=20', 'fixed', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=20', 'fixed', Q2[2][0][0] / Q2[2][0][-1], Q2[2][0][1] / Q2[2][0][-1],
                             Q2[2][0][2] / Q2[2][0][-1], Q2[2][0][3] / Q2[2][0][-1], Q2[2][0][4] / Q2[2][0][-1],
                             Q2[2][0][5] / Q2[2][0][-1]])
        if Q2[2][1][-1] == 0:
            writer.writerow(['>=20', 'b_av', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(
                ['>=20', 'b_av', '-', '-', '-', '-', Q2[2][1][0] / Q2[2][1][-1], Q2[2][1][1] / Q2[2][1][-1]])

        if Q2[2][2][-1] == 0:
            writer.writerow(['>=20', 'b_sesk', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=20', 'b_sesk', Q2[2][2][0] / Q2[2][2][-1], Q2[2][2][1] / Q2[2][2][-1],
                             Q2[2][2][2] / Q2[2][2][-1], Q2[2][2][3] / Q2[2][2][-1], Q2[2][2][4] / Q2[2][2][-1],
                             Q2[2][2][5] / Q2[2][2][-1]])
        f.close()

    test_acc = accuracy_score(all_targets, all_predictions) * 100
    test_pre = precision_score(all_targets, all_predictions) * 100
    test_recall = recall_score(all_targets, all_predictions) * 100
    test_f1 = f1_score(all_targets, all_predictions) * 100

    con_m = confusion_matrix(all_targets, all_predictions)
    tn, fp, fn, tp = con_m.ravel()
    with open(result_dir + '/' + 'prediction_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pre', 'acc', 'recall', 'f1', 'FPR', 'FNR'])
        writer.writerow([test_pre, test_acc, test_recall, test_f1, fp / (fp + tn), fn / (tp + fn)])
        f.close()

    # the attention score on different cwe:
    cwe = ['22', '79', '89', '190', '191']
    with open(result_dir + '/' + 'Q3_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cwe', 'hit_type', 'Hit3', 'Hit1', 'Hit50', 'Hit30'])
        for idx, c in enumerate(cwe):
            with open('data/CWE{}.json'.format(c)) as cwe_f:
                cwe_data = np.array(js.load(cwe_f))
                cwe_f.close()
            cwe_test_batch = get_batches_idx(len(cwe_data), False)
            for cwe_test_it, cwe_test_idx in enumerate(cwe_test_batch):
                q3 = model(cwe_data[cwe_test_idx], False, True)
                for i in range(len(q3)):
                    for j in range(len(q3[i])):
                        Q3[idx][i][j] += q3[i][j]

            if Q3[idx][0][-1] == 0:
                writer.writerow([c, 'fixed', '-', '-', '-', '-'])
            else:
                writer.writerow([c, 'fixed', Q3[idx][0][0] / Q3[idx][0][-1], Q3[idx][0][1] / Q3[idx][0][-1],
                                 Q3[idx][0][2] / Q3[idx][0][-1], Q3[idx][0][3] / Q3[idx][0][-1]])

            if Q3[idx][1][-1] == 0:
                writer.writerow([c, 'b_av', '-', '-', '-', '-'])
            else:
                writer.writerow(
                    [c, 'b_av', '-', '-', Q3[idx][1][0] / Q3[idx][1][-1], Q3[idx][1][1] / Q3[idx][1][-1]])

            if Q3[idx][2][-1] == 0:
                writer.writerow([c, 'b_se', '-', '-', '-', '-'])
            else:
                writer.writerow([c, 'b_se', Q3[idx][2][0] / Q3[idx][2][-1], Q3[idx][2][1] / Q3[idx][2][-1],
                                 Q3[idx][2][2] / Q3[idx][2][-1], Q3[idx][2][3] / Q3[idx][2][-1]])

            if Q3[idx][3][-1] == 0:
                writer.writerow([c, 'b_sk', '-', '-', '-', '-'])
            else:
                writer.writerow([c, 'b_sk', Q3[idx][3][0] / Q3[idx][3][-1], Q3[idx][3][1] / Q3[idx][3][-1],
                                 Q3[idx][3][2] / Q3[idx][3][-1], Q3[idx][3][3] / Q3[idx][3][-1]])

            if Q3[idx][4][-1] == 0:
                writer.writerow([c, 'b_sesk', '-', '-', '-', '-'])
            else:
                writer.writerow(
                    [c, 'b_sesk', Q3[idx][4][0] / Q3[idx][4][-1], Q3[idx][4][1] / Q3[idx][4][-1],
                     Q3[idx][4][2] / Q3[idx][4][-1], Q3[idx][4][3] / Q3[idx][4][-1]])

        f.close()

    with open(result_dir + '/' + 'Q1_value_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hit_type', 'Hit3', 'Hit1', 'Hit50%', 'Hit30%', 'total'])
        if Q1[0][-1] == 0:
            writer.writerow(['fixed', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['fixed', Q1[0][0], Q1[0][1], Q1[0][2], Q1[0][3], Q1[0][-1]])
        if Q1[1][-1] == 0:
            writer.writerow(['b_av', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['b_av', '-', '-', Q1[1][0], Q1[1][1], Q1[1][-1]])
        if Q1[2][-1] == 0:
            writer.writerow(['b_sesk', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['b_sesk', Q1[2][0], Q1[2][1], Q1[2][2], Q1[2][3], Q1[2][-1]])
        f.close()

    with open(result_dir + '/' + 'Q2_value_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_number', 'hit_type', 'Hit10', 'Hit5', 'Hit3', 'Hit1', 'Hit50%', 'Hit30%', 'total'])
        if Q2[0][0][-1] == 0:
            writer.writerow(['>=5<10', 'fixed', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=5<10', 'fixed', '-', Q2[0][0][0], Q2[0][0][1], Q2[0][0][2], Q2[0][0][3],
                             Q2[0][0][4], Q2[0][0][-1]])

        if Q2[0][1][-1] == 0:
            writer.writerow(['>=5<10', 'b_av', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(
                ['>=5<10', 'b_av', '-', '-', '-', '-', Q2[0][1][0], Q2[0][1][1], Q2[0][1][-1]])

        if Q2[0][2][-1] == 0:
            writer.writerow(['>=5<10', 'b_sesk', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=5<10', 'b_sesk', '-', Q2[0][2][0], Q2[0][2][1], Q2[0][2][2], Q2[0][2][3],
                             Q2[0][2][4], Q2[0][2][-1]])

        if Q2[1][0][-1] == 0:
            writer.writerow(['>=10<20', 'fixed', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=10<20', 'fixed', Q2[1][0][0], Q2[1][0][1], Q2[1][0][2], Q2[1][0][3], Q2[1][0][4],
                             Q2[1][0][5], Q2[1][0][-1]])

        if Q2[1][1][-1] == 0:
            writer.writerow(['>=10<20', 'b_av', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(
                ['>=10<20', 'b_av', '-', '-', '-', '-', Q2[1][1][0], Q2[1][1][1], Q2[1][1][-1]])

        if Q2[1][2][-1] == 0:
            writer.writerow(['>=10<20', 'b_sesk', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=10<20', 'b_sesk', Q2[1][2][0], Q2[1][2][1], Q2[1][2][2], Q2[1][2][3], Q2[1][2][4],
                             Q2[1][2][5], Q2[1][2][-1]])

        if Q2[2][0][-1] == 0:
            writer.writerow(['>=20', 'fixed', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=20', 'fixed', Q2[2][0][0], Q2[2][0][1], Q2[2][0][2], Q2[2][0][3], Q2[2][0][4],
                             Q2[2][0][5], Q2[2][0][-1]])
        if Q2[2][1][-1] == 0:
            writer.writerow(['>=20', 'b_av', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(
                ['>=20', 'b_av', '-', '-', '-', '-', Q2[2][1][0], Q2[2][1][1], Q2[2][1][-1]])

        if Q2[2][2][-1] == 0:
            writer.writerow(['>=20', 'b_sesk', '-', '-', '-', '-', '-', '-', '-'])
        else:
            writer.writerow(['>=20', 'b_sesk', Q2[2][2][0], Q2[2][2][1], Q2[2][2][2], Q2[2][2][3], Q2[2][2][4],
                             Q2[2][2][5], Q2[2][2][-1]])
        f.close()

    with open(result_dir + '/' + 'Q3_value_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cwe', 'hit_type', 'Hit3', 'Hit1', 'Hit50', 'Hit30', 'total'])
        for idx in range(len(Q3)):
            if idx == 0:
                c = '22'
            elif idx == 1:
                c = '79'
            elif idx == 2:
                c = '89'
            elif idx == 3:
                c = '190'
            elif idx == 4:
                c = '191'
            if Q3[idx][0][-1] == 0:
                writer.writerow([c, 'fixed', '-', '-', '-', '-', '-'])
            else:
                writer.writerow([c, 'fixed', Q3[idx][0][0], Q3[idx][0][1],
                                 Q3[idx][0][2], Q3[idx][0][3], Q3[idx][0][-1]])

            if Q3[idx][1][-1] == 0:
                writer.writerow([c, 'b_av', '-', '-', '-', '-', '-'])
            else:
                writer.writerow(
                    [c, 'b_av', '-', '-', Q3[idx][1][0], Q3[idx][1][1], Q3[idx][1][-1]])

            if Q3[idx][2][-1] == 0:
                writer.writerow([c, 'b_se', '-', '-', '-', '-', '-'])
            else:
                writer.writerow([c, 'b_se', Q3[idx][2][0], Q3[idx][2][1],
                                 Q3[idx][2][2], Q3[idx][2][3], Q3[idx][2][-1]])

            if Q3[idx][3][-1] == 0:
                writer.writerow([c, 'b_sk', '-', '-', '-', '-', '-'])
            else:
                writer.writerow([c, 'b_sk', Q3[idx][3][0], Q3[idx][3][1],
                                 Q3[idx][3][2], Q3[idx][3][3], Q3[idx][3][-1]])

            if Q3[idx][4][-1] == 0:
                writer.writerow([c, 'b_sesk', '-', '-', '-', '-', '-'])
            else:
                writer.writerow(
                    [c, 'b_sesk', Q3[idx][4][0], Q3[idx][4][1],
                     Q3[idx][4][2], Q3[idx][4][3], Q3[idx][4][-1]])

        f.close()
