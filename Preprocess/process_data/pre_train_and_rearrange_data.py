import json as js
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import re
from random import sample


def get_max_len(l):
    max_len = 0
    for ll in l:
        if len(ll['tokens']) > max_len:
            max_len = len(ll['tokens'])
    return max_len


def process_example(cwe_, d, train):
    nodes = d['nodes']
    edges = d['edges']
    if len(nodes) == 0:
        return
    node_sen_size = get_max_len(nodes)
    for n in nodes:
        number_exp_node = [0] * node_sen_size
        for order, t in enumerate(n['new_tokens']):
            if t in wv.vocab.keys():
                # todo 注意这里加1了，后续要在wv张量前加一组全0，代表不在字典的字符
                idx = wv.vocab[t].index + 1
                number_exp_node[order] = idx
        n['tokens_order'] = number_exp_node

    # 所有的结点都已经按规则排好序，接下来需要给结点换成从0开始的id，对应的边也需要换，以便后续DGL输入
    try:
        old2new = {}
        for idx, n in enumerate(nodes):
            old2new[n['id']] = idx
            n['id'] = idx
        new_deges = []
        for idx, e in enumerate(edges):
            # 由于在创建CFG时可能加了一个虚拟的return结点，并且前面的处理将这个虚拟结点去除了，但对应的边没去除，这里对边进行筛选
            if e['source'] not in old2new.keys() or e['target'] not in old2new.keys():
                continue
            e['source'] = old2new[e['source']]
            e['target'] = old2new[e['target']]
            new_deges.append(e)

        if train:
            processed_train_data.append(
                {'cwe': cwe_, 'fun_info': d['fun_info'], 'nodes': nodes, 'edges': new_deges, 'target': d['target']})
        else:
            processed_test_data.append(
                {'cwe': cwe_, 'fun_info': d['fun_info'], 'nodes': nodes, 'edges': new_deges, 'target': d['target']})

    except:
        print(111)
        exit(-2)


# graph_type = ['CFG', 'CDG', 'DDG', 'PDG', 'CFG_DDG']
graph_type = ['DDG']

for g_type in graph_type:
    print(g_type)
    with open('data/corpus.json', 'r')as f:
        corpus = js.load(f)
        corpus = [c[0] for c in corpus]
        f.close()
    w2vmodel = Word2Vec(min_count=5, size=50)

    w2vmodel.build_vocab(sentences=corpus)

    wv = w2vmodel.wv

    with open("data/mapped_final_data.json", 'r')as f:
        data = js.load(f)
        f.close()

    processed_train_data = []
    processed_test_data = []

    total_example_size = 0
    total_pos_size = 0
    total_neg_size = 0

    # 在这里就把数据划分了，每个CWE都按照5：1划分 每个CWE的train、test的样本正负比例保持一样
    for cwe in data.keys():
        # if cwe == 'CWE80':
        #     print(111)
        x = data[cwe]
        print("{} num: {}".format(cwe, len(x)))
        y = [example['target'] for example in x]
        # 对于少于5个样本的cwe单独处理
        if len(x) < 5:
            pos = [ddd for ddd in x if ddd['target'] == 1]
            neg = [ddd for ddd in x if ddd['target'] == 0]
            sample_pos = sample(pos, len(pos) // 2)
            sample_neg = sample(neg, len(neg) // 2)

            for dd in pos:
                if dd in sample_pos:
                    process_example(cwe, dd, False)
                else:
                    process_example(cwe, dd, True)

            for dd in neg:
                if dd in sample_neg:
                    process_example(cwe, dd, False)
                else:
                    process_example(cwe, dd, True)
            continue
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)
        for dd in x_train:
            process_example(cwe, dd, True)

        for dd in x_test:
            process_example(cwe, dd, False)

    print("total_example_size: ", len(processed_train_data) + len(processed_test_data))
    for dd in processed_train_data:
        if dd['target']:
            total_pos_size += 1
        else:
            total_neg_size += 1

    for dd in processed_test_data:
        if dd['target']:
            total_pos_size += 1
        else:
            total_neg_size += 1
    print("total_pos_size: ", total_pos_size)
    print("total_neg_size", total_neg_size)

    print("start train corpus...")
    w2vmodel.train(corpus, total_examples=w2vmodel.corpus_count, epochs=50)
    print("train w2v done")
    w2vmodel.save('data/w2v.model')
    with open('data/processed_train_data.json', 'w')as f:
        js.dump(processed_train_data, f)
        f.close()
    with open('data/processed_test_data.json', 'w')as f:
        js.dump(processed_test_data, f)
        f.close()
    print(111)
