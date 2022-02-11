import json as js
import re


def get_func_name(tokens):
    start = ""
    for tkn in tokens:
        start = start + " " + tkn
    start = re.sub(" \(.*\).*", "", start, re.DOTALL)
    name = start.split(" ")[-1]
    return name


# graph_type = ['CDG', 'CFG', 'DDG', 'PDG', 'CFG_DDG']
graph_type = ['DDG']

for g_type in graph_type:
    with open('data/final_data.json', 'r')as f:
        final_data = js.load(f)
        f.close()

    corpus = []
    new_final_data = {}
    node_type_dic = {}
    for cwe in final_data.keys():
        print(cwe)
        data = final_data[cwe]
        new_final_data[cwe] = []
        for d in data:
            print(d['fun_info'])
            # 重新排列函数之间的结点
            arranged_nodes = []
            nodes = d['nodes']
            edges = d['edges']
            if len(nodes) == 0:
                continue

            temp_func_nodes_dic = {}
            # 将每个函数的结点存放在一起，每个函数的结点前面已经按行号排列
            for n in nodes:
                func_id = re.sub("_[0-9]+$", "", n['id'])
                if func_id not in temp_func_nodes_dic:
                    temp_func_nodes_dic[func_id] = [n]
                else:
                    temp_func_nodes_dic[func_id].append(n)
                if n['type'] not in node_type_dic:
                    node_type_dic[n['type']] = len(node_type_dic)

            # 按照深度和广度排列函数，即深度靠前的先排列，同一深度的函数按广度(发生的先后顺序)来排列结点
            visited_func = {0: [0]}
            for e in edges:
                if e['id'] == '-1':
                    source_func_id = re.sub("_[0-9]+$", "", e['source'])
                    depth = int(source_func_id.split('_')[0])
                    breadth = int(source_func_id.split('_')[1])
                    if depth not in visited_func:
                        visited_func[depth] = []
                    if breadth not in visited_func[depth]:
                        visited_func[depth].append(breadth)

                    target_func_id = re.sub("_[0-9]+$", "", e['target'])
                    depth = int(target_func_id.split('_')[0])
                    breadth = int(target_func_id.split('_')[1])
                    if depth not in visited_func:
                        visited_func[depth] = []
                    if breadth not in visited_func[depth]:
                        visited_func[depth].append(breadth)
            sorted_depth = sorted(visited_func.keys())

            for depth_ in sorted_depth:
                visited_func[depth_].sort()
                for breadth_ in visited_func[depth_]:
                    func_id = str(depth_) + '_' + str(breadth_)
                    arranged_nodes.extend(temp_func_nodes_dic[func_id])

            # 这里进行normalization以及数据的去重
            # 首先找到所有的class_name, 然后对每个函数结点进行函数名和变量名对的map
            class_name = []
            func_map = {}
            for n in arranged_nodes:
                if len(n['mtdCallInfo']) > 0:
                    for mtd_call_info in n['mtdCallInfo']:
                        cls_name = mtd_call_info['className']
                        if cls_name not in class_name:
                            class_name.append(cls_name)

                depth_breadth_order = n['id'].split('_')
                func_id = depth_breadth_order[0] + '_' + depth_breadth_order[1]
                if func_id not in func_map:
                    func_map[func_id] = {'variables': [], 'func_names': []}

                try:
                    if n['type'] == 'MethodDeclaration':
                        func_map[func_id]['func_names'].append(get_func_name(n['tokens']))
                    if len(n['varibles']) > 0:
                        for v in n['varibles']:
                            if v not in func_map[func_id]['variables']:
                                func_map[func_id]['variables'].append(v)
                    if len(n['mtdCallInfo']) > 0:
                        for mtd_call_info in n['mtdCallInfo']:
                            mtd_name = mtd_call_info['mtdSignature']
                            mtd_name_pure = re.sub("\(.*\)", "", mtd_name, re.DOTALL)
                            if mtd_name_pure not in func_map[func_id]['func_names']:
                                func_map[func_id]['func_names'].append(mtd_name_pure)
                except:
                    print('somethong wrong')
                    exit(-2)

            # map 按照出现的顺序
            example_corpus = []
            func_order = 0
            v_order = 0
            cls_order = 0

            func_name_map = {}
            v_map = {}
            cls_map = {}

            for n in arranged_nodes:
                depth_breadth_order = n['id'].split('_')
                func_id = depth_breadth_order[0] + '_' + depth_breadth_order[1]
                new_tokens = []
                for t in n['tokens']:
                    if t in class_name:
                        if t not in cls_map:
                            cls_map[t] = 'CLS' + str(cls_order)
                            new_tokens.append('CLS' + str(cls_order))
                            cls_order += 1
                        else:
                            new_tokens.append(cls_map[t])
                    elif t in func_map[func_id]['variables']:
                        # 不同函数的变量即使同名也是不同的变量，其实对于不同类的method，即使名字一样也要区分开，这里暂时不做处理
                        if func_id + t not in v_map:
                            v_map[func_id + t] = 'VAR' + str(v_order)
                            new_tokens.append('VAR' + str(v_order))
                            v_order += 1
                        else:
                            new_tokens.append(v_map[func_id + t])
                    elif t in func_map[func_id]['func_names']:
                        if t not in func_name_map:
                            func_name_map[t] = 'FUN' + str(func_order)
                            new_tokens.append('FUN' + str(func_order))
                            func_order += 1
                        else:
                            new_tokens.append(func_name_map[t])
                    elif 'bad' in t or 'good' in t:
                        new_tokens.append('VAR')
                    else:
                        new_tokens.append(t)
                n['new_tokens'] = new_tokens
                example_corpus.extend(new_tokens)
            # 如果已经有此数据
            if [example_corpus, d['target']] in corpus:
                # print(corpus.index([example_corpus, d['target']]))
                # print(example_corpus)
                continue
            else:
                corpus.append([example_corpus, d['target']])
            new_final_data[cwe].append({'fun_info': d['fun_info'], 'nodes': arranged_nodes, 'edges': d['edges'],
                                        'target': d['target']})

    with open('data/corpus.json', 'w')as f:
        js.dump(corpus, f)
        f.close()

    with open('data/mapped_final_data.json', 'w')as f:
        js.dump(new_final_data, f)
        f.close()

    with open('data/node_type_dic.json', 'w')as f:
        js.dump(node_type_dic, f)
        f.close()
    print(111)
