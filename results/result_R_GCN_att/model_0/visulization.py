import copy
import json as js
import warnings

from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from torch.nn import BCELoss
from torch.optim import Adam
import torch
from model.model import GraphWithAtt
import os
import shutil
import csv

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


# 训练10个模型

result_dir = 'result_save_new/'

print("loading data")
with open('../../data/node_type_dic.json', 'r')as f:
    node_type_dic = js.load(f)
    f.close()

w2v = Word2Vec.load('../../data/w2v.model')
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
# gnn 参数
graph_in_features_size = conv_out_fea_size + len(node_type_dic.keys())
graph_out_features_size = 50


# 句子长度阈值，即每个图的结点个数要修剪为该大小
max_sen_size = 50


model = GraphWithAtt(w2v.wv.vectors, max_edge_type, node_type_dic, max_sen_size,
                     [conv_in_fea_size, conv_out_fea_size, kernel_size],
                     [graph_in_features_size, graph_out_features_size])
model.cuda()
loss_func = BCELoss(reduction='sum')
optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

# model.train()

show_train_batch_count = 5
max_patience = 10

model.load_state_dict(torch.load('Att_GNN.bin'))

model.eval()

with open('../../data/labeled_processed_test_data.json', 'r')as f:
    test_data = np.array(js.load(f))
    f.close()
cwe_190_data = []
for d in test_data:
    if d['cwe'] == 'CWE190':
        cwe_190_data.append(d)
with torch.no_grad():
    all_predictions, all_targets = [], []
    test_batch_index = get_batches_idx(len(test_data), False)
    for test_it, test_idx in enumerate(test_batch_index):
        targets = [t['target'] for t in test_data[test_idx]]
        targets = torch.tensor(targets, dtype=torch.float).cuda()
        predictions, q1, q2 = model(test_data[test_idx], True)