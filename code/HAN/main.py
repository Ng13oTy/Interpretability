import copy
import json as js
import warnings

from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from torch.nn import BCELoss
from torch.optim import Adam
import torch
from model.model import GraphWithHAN
import os
import shutil

warnings.filterwarnings("ignore")


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


# 训练100个模型
model_number = 1
result_dir = 'result_save/'
for m_number in range(model_number):
    print('current_model', model_number)
    cur_m = 'model_{}'.format(m_number)
    if cur_m not in os.listdir(result_dir):
        os.mkdir(result_dir + cur_m)
    else:
        for ff in os.listdir(result_dir + cur_m):
            if os.path.isdir(ff):
                shutil.rmtree(result_dir + cur_m + '/' + ff)
            else:
                os.remove(result_dir + cur_m + '/' + ff)

    print("loading data")
    with open('data/node_type_dic.json', 'r')as f:
        node_type_dic = js.load(f)
        f.close()
    with open('data/labeled_processed_test_data.json', 'r')as f:
        test_data = np.array(js.load(f))
        f.close()
    with open('data/processed_train_data.json', 'r')as f:
        train_data = np.array(js.load(f))
        f.close()

    w2v = Word2Vec.load('data/w2v.model')
    print("data loaded")
    # 超参数
    epochs = 100
    batch_size = 256
    num_steps = 6

    # 这里根据你们的情况替换CFG\DDG\CDG都是3  PDG和CFG_DDG换成4
    max_edge_type = 3

    shuffle = True

    # 卷积层参数
    conv_in_fea_size = w2v.wv.vectors.shape[1]  # 和单词嵌入长度一样
    conv_out_fea_size = 50  # 卷积后每个结点的向量表示
    kernel_size = 2
    # # ggnn 参数
    # graph_in_features_size = conv_out_fea_size + len(node_type_dic.keys())
    # graph_out_features_size = 100

    # 句子长度阈值，即每个图的结点个数要修剪为该大小
    max_sen_size = 50
    # 单词注意力参数
    word_hidden_size = 50
    # 句子注意力层参数
    sen_hidden_size = 50

    model = GraphWithHAN(w2v.wv.vectors, node_type_dic, max_sen_size,
                         [conv_in_fea_size, conv_out_fea_size, kernel_size],
                         [sen_hidden_size], [word_hidden_size])
    model.cuda()
    loss_func = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    model.train()

    show_train_batch_count = 5
    max_patience = 5

    # from_scratch_train = False
    # if not from_scratch_train:
    #     model_.load_state_dict(torch.load('data/GGNNHAN-model_.bin'))

    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0

    stop = False
    for epoch in range(epochs):
        if stop:
            break
        print('epoch {}'.format(epoch) + '\n')
        batches_idx = get_batches_idx(len(train_data), shuffle)
        for it, batch_idx in enumerate(batches_idx):
            optim.zero_grad()
            batch_train_data = train_data[batch_idx]
            targets = [t['target'] for t in train_data[batch_idx]]
            targets = torch.FloatTensor(targets).cuda()
            predictions = model(batch_train_data)
            batch_loss = loss_func(predictions, targets)
            batch_loss.backward()
            optim.step()

            train_losses.append(batch_loss.detach().cpu().item())

            if (it + 1) % show_train_batch_count == 0:
                targets = targets.detach().cpu().numpy().tolist()
                predictions = predictions.detach().cpu()
                predictions = predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                    dtype=torch.int32).numpy().tolist()

                print('Batches %d\t\tTrain Loss %10.3f' % (it + 1, batch_loss.detach().cpu().item()))
                print('f1: %.2f  acc: %.2f  prec: %.2f  ' % (
                    f1_score(targets, predictions) * 100, accuracy_score(targets, predictions) * 100,
                    precision_score(targets, predictions) * 100))
                # tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
                print(confusion_matrix(targets, predictions))
                # print(tn, fp, fn, tp)

        print("validation...")

        model.eval()
        with torch.no_grad():
            # level_0: >=5<10  level_1: >=10<20 level_3 >=20
            Q1 = [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0]]
            Q2 = [[[0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]]
            # Q1 = {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #       'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #       'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}}
            # Q2 = {'level_0': {'fixed': {'top5': 0, 'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                   'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                   'b_se_or_b_sk': {'top5': 0, 'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0,
            #                                    'total': 0}},
            #       'level_1': {'fixed': {'top10': 0, 'top5': 0, 'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                   'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                   'b_se_or_b_sk': {'top10': 0, 'top5': 0, 'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0,
            #                                    'total': 0}},
            #       'level_2': {'fixed': {'top10': 0, 'top5': 0, 'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                   'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                   'b_se_or_b_sk': {'top10': 0, 'top5': 0, 'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0,
            #                                    'total': 0}}}
            Q3 = [[[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
            # Q3 = {'CWE22': {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_se': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}},
            #       'CWE79': {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_se': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}},
            #       'CWE89': {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_se': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                 'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}},
            #       'CWE190': {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_se': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}},
            #       'CWE191': {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_se': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                  'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}},
            #       'CWE190_191': {'fixed': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                      'b_av': {'top%50': 0, 'top%30': 0, 'total': 0},
            #                      'b_se': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                      'b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0},
            #                      'b_se_or_b_sk': {'top3': 0, 'top1': 0, 'top%50': 0, 'top%30': 0, 'total': 0}}
            #       }

            # total_node_number_of_valid_pos = 0
            # total_node_number_of_valid_pos_happen = 0

            _loss = []
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

                batch_loss = loss_func(predictions, targets)
                _loss.append(batch_loss.detach().cpu().item())
                predictions = predictions.detach().cpu()
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
                all_targets.extend(targets.detach().cpu().numpy().tolist())

            print('pos_total_level_0_number: {}'.format(Q2[0][-1][-1]))
            print('pos_total_level_1_number: {}'.format(Q2[1][-1][-1]))
            print('pos_total_level_2_number: {}'.format(Q2[2][-1][-1]))

            print('neg_total_level_0_number: {}'.format(Q2[0][0][-1]))
            print('neg_total_level_1_number: {}'.format(Q2[1][0][-1]))
            print('neg_total_level_2_number: {}'.format(Q2[2][0][-1]))

            print('q1 info:')

            print('fixed:')
            if Q1[0][-1] == 0:
                print('fixed number == 0')
            else:
                print('top3:', Q1[0][0] / Q1[0][-1], 'top1:', Q1[0][1] / Q1[0][-1], 'top50%', Q1[0][2] / Q1[0][-1],
                      'top30%',
                      Q1[0][3] / Q1[0][-1])

            print('b_av:')
            if Q1[1][-1] == 0:
                print('bav number == 0')
            else:
                print('top50%', Q1[1][0] / Q1[1][-1], 'top30%', Q1[1][1] / Q1[1][-1])

            print('b_sesk:')
            if Q1[2][-1] == 0:
                print('b_sesk number == 0')
            else:
                print('top3:', Q1[2][0] / Q1[2][-1], 'top1:', Q1[2][1] / Q1[2][-1], 'top50%', Q1[2][2] / Q1[2][-1],
                      'top30%',
                      Q1[2][3] / Q1[2][-1])

            print('\nq2 info:')
            print('level_0:')
            print('fixed:')
            if Q2[0][0][-1] == 0:
                print('fixed number == 0')
            else:
                print('top5', Q2[0][0][0] / Q2[0][0][-1], 'top3', Q2[0][0][1] / Q2[0][0][-1], 'top1',
                      Q2[0][0][2] / Q2[0][0][-1],
                      'top50%', Q2[0][0][3] / Q2[0][0][-1], 'top30%', Q2[0][0][4] / Q2[0][0][-1])

            print('b_av:')
            if Q2[0][1][-1] == 0:
                print('b_av == 0')
            else:
                print('top50%', Q2[0][1][0] / Q2[0][1][-1], 'top30%', Q2[0][1][1] / Q2[0][1][-1])

            print('b_sesk:')
            if Q2[0][2][-1] == 0:
                print('b_sesk number == 0')
            else:
                print('top5', Q2[0][2][0] / Q2[0][2][-1], 'top3', Q2[0][2][1] / Q2[0][2][-1], 'top1',
                      Q2[0][2][2] / Q2[0][2][-1],
                      'top50%', Q2[0][2][3] / Q2[0][2][-1], 'top30%', Q2[0][2][4] / Q2[0][2][-1])

            print('level_1:')
            print('fixed:')
            if Q2[1][0][-1] == 0:
                print('fixed number == 0')
            else:
                print('top10', Q2[1][0][0] / Q2[1][0][-1], 'top5', Q2[1][0][1] / Q2[1][0][-1], 'top3',
                      Q2[1][0][2] / Q2[1][0][-1],
                      'top1', Q2[1][0][3] / Q2[1][0][-1], 'top50%', Q2[1][0][4] / Q2[1][0][-1], 'top30%',
                      Q2[1][0][5] / Q2[1][0][-1])

            print('b_av:')
            if Q2[1][1][-1] == 0:
                print('b_av number == 0')
            else:
                print('top50%', Q2[1][1][0] / Q2[1][1][-1], 'top30%', Q2[1][1][1] / Q2[1][1][-1])

            print('b_sesk:')
            if Q2[1][2][-1] == 0:
                print('b_sesk number == 0')
            else:
                print('top10', Q2[1][2][0] / Q2[1][2][-1], 'top5', Q2[1][2][1] / Q2[1][2][-1], 'top3',
                      Q2[1][2][2] / Q2[1][2][-1],
                      'top1', Q2[1][2][3] / Q2[1][2][-1], 'top50%', Q2[1][2][4] / Q2[1][2][-1], 'top30%',
                      Q2[1][2][5] / Q2[1][2][-1])

            print('level_2:')
            print('fixed:')
            if Q2[2][0][-1] == 0:
                print('fixed number == 0')
            else:
                print('top10', Q2[2][0][0] / Q2[2][0][-1], 'top5', Q2[2][0][1] / Q2[2][0][-1], 'top3',
                      Q2[2][0][2] / Q2[2][0][-1],
                      'top1', Q2[2][0][3] / Q2[2][0][-1], 'top50%', Q2[2][0][4] / Q2[2][0][-1], 'top30%',
                      Q2[2][0][5] / Q2[2][0][-1])

            print('b_av:')
            if Q2[2][1][-1] == 0:
                print('b_av number == 0')
            else:
                print('top50%', Q2[2][1][0] / Q2[2][1][-1], 'top30%', Q2[2][1][1] / Q2[2][1][-1])

            print('b_sesk:')
            if Q2[2][2][-1] == 0:
                print('b_sesk number == 0')
            else:
                print('top10', Q2[2][2][0] / Q2[2][2][-1], 'top5', Q2[2][2][1] / Q2[2][2][-1], 'top3',
                      Q2[2][2][2] / Q2[2][2][-1],
                      'top1', Q2[2][2][3] / Q2[2][2][-1], 'top50%', Q2[2][2][4] / Q2[2][2][-1], 'top30%',
                      Q2[2][2][5] / Q2[2][2][-1])

            test_loss = np.mean(_loss).item()
            test_acc = accuracy_score(all_targets, all_predictions) * 100
            test_pre = precision_score(all_targets, all_predictions) * 100
            test_recall = recall_score(all_targets, all_predictions) * 100
            test_f1 = f1_score(all_targets, all_predictions) * 100

            con_m = confusion_matrix(all_targets, all_predictions)
            print("confusion matrix :")
            tn, fp, fn, tp = con_m.ravel()
            print(con_m)
            print(tn, fp, fn, tp)
            print("FPR: {}".format(fp / (fp + tn)))
            print("FNR: {}".format(fn / (tp + fn)))

            # 计算各个CWE上的表现情况:
            cwe = ['22', '79', '89', '190', '191']
            for idx, c in enumerate(cwe):
                print(c)
                with open('data/CWE{}.json'.format(c))as cwe_f:
                    cwe_data = np.array(js.load(cwe_f))
                    cwe_f.close()
                cwe_test_batch = get_batches_idx(len(cwe_data), False)
                for cwe_test_it, cwe_test_idx in enumerate(cwe_test_batch):
                    print(cwe_test_it)
                    q3 = model(cwe_data[cwe_test_idx], False, True)
                    for i in range(len(q3)):
                        for j in range(len(q3[i])):
                            Q3[idx][i][j] += q3[i][j]

                print('fixed:')
                if Q3[idx][0][-1] == 0:
                    print('fixed number == 0')
                else:
                    print('top3', Q3[idx][0][0] / Q3[idx][0][-1], 'top1', Q3[idx][0][1] / Q3[idx][0][-1], 'top50%',
                          Q3[idx][0][2] / Q3[idx][0][-1], 'top30%', Q3[idx][0][3] / Q3[idx][0][-1])

                print('bav:')
                if Q3[idx][1][-1] == 0:
                    print('bav number == 0')
                else:
                    print('top50%', Q3[idx][1][0] / Q3[idx][1][-1], 'top30%', Q3[idx][1][1] / Q3[idx][1][-1])

                print('bse:')
                if Q3[idx][2][-1] == 0:
                    print('bse number == 0')
                else:
                    print('top3', Q3[idx][2][0] / Q3[idx][2][-1], 'top1', Q3[idx][2][1] / Q3[idx][2][-1], 'top50%',
                          Q3[idx][2][2] / Q3[idx][2][-1], 'top30%', Q3[idx][2][3] / Q3[idx][2][-1])

                print('bsk')
                if Q3[idx][3][-1] == 0:
                    print('bsk number == 0')
                else:
                    print('top3', Q3[idx][3][0] / Q3[idx][3][-1], 'top1', Q3[idx][3][1] / Q3[idx][3][-1], 'top50%',
                          Q3[idx][3][2] / Q3[idx][3][-1], 'top30%', Q3[idx][3][3] / Q3[idx][3][-1])

                print('b_sesk:')
                if Q3[idx][4][-1] == 0:
                    print('b_sesk number == 0')
                else:
                    print('top3', Q3[idx][4][0] / Q3[idx][4][-1], 'top1', Q3[idx][4][1] / Q3[idx][4][-1], 'top50%',
                          Q3[idx][4][2] / Q3[idx][4][-1], 'top30%', Q3[idx][4][3] / Q3[idx][4][-1])

            if test_f1 > best_f1:
                patience_counter = 0
                best_f1 = test_f1
                best_model = copy.deepcopy(model.state_dict())
                _save_file = open('data/GGNNHAN' + '-model_.bin', 'wb')
                torch.save(model.state_dict(), _save_file)
                _save_file.close()
            else:
                patience_counter += 1
            print('epoch %d  Train Loss %0.3f  Valid Loss %0.3f f1: %.2f  acc: %.2f  prec: %.2f  Patience %d' % (
                epoch, np.mean(train_losses).item(), test_loss, test_f1, test_acc, test_pre, patience_counter))
            model.train()
            if patience_counter == max_patience:
                stop = True
                break

    print("上面最后一个patience为0的epoch的结果就是最终结果")
