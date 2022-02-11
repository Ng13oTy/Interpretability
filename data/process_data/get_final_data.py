import copy
import json as js
import re


# graph_type = ['CDG', 'CFG', 'DDG', 'PDG', 'CFG_DDG']
graph_type = ['DDG']  # 这是无用的，只是为了省事，没改
for g_type in graph_type:
    # # IO信息是通用的
    # with open('data/before_cross_fun_all_data/IO_{}.json'.format(g_type), 'r')as f:
    #     common_IO_info = js.load(f)
    #     f.close()

    with open('data/before_cross_fun_all_data.json', 'r')as f:
        data = js.load(f)
        f.close()

    final_data = {}
    # todo 下面的步骤可以集成到java处理的步骤
    # 得到所有的函数，以类名做key，值为dic，两个键值对分别为函数名：name、graph:图信息
    mtd_info = {}
    for cwe in data.keys():
        cwe_data = data[cwe]
        for java_file_name in cwe_data.keys():
            java_file_data = cwe_data[java_file_name]
            for complete_fun_name in java_file_data.keys():
                func_name = complete_fun_name.split('____')[-1]
                cls_name = complete_fun_name.replace('____' + func_name, '')
                if cls_name not in mtd_info:
                    mtd_info[cls_name] = {}
                mtd_info[cls_name][func_name] = {'nodes': java_file_data[complete_fun_name]['nodes'],
                                                 'edges': java_file_data[complete_fun_name]['edges']}

    for key in mtd_info:
        if key == 'CWE369_Divide_by_Zero__float_random_divide_68a':
            aa = mtd_info[key]
        if key == 'CWE369_Divide_by_Zero__float_random_divide_68b':
            bb = mtd_info[key]
    # for mtd in common_IO_info:
    #     for k in mtd.keys():
    #         func_name = k.split('____')[-1]
    #         cls_name = k.replace('____' + func_name, '')
    #         if cls_name not in mtd_info:
    #             mtd_info[cls_name] = {}
    #         mtd_info[cls_name][func_name] = {'nodes': mtd[k]['nodes'], 'edges': mtd[k]['edges']}

    # 在这里对每个函数的结点按照行号和列号进行排序，这样做是为了后续在访问同一深度的函数调用时有个先后，然后再排列不同函数之间的结点时也按照这个先后顺序
    # 注意 添加的return语句行号为-1要特殊处理
    for k in mtd_info.keys():
        for kk in mtd_info[k].keys():
            nodes = mtd_info[k][kk]['nodes']
            nodes.sort(key=lambda xx: (xx['line'], xx['column']))
            for idx, n in enumerate(nodes):
                if n['line'] == -1:
                    nodes.remove(nodes[idx])
                    break

    # a = mtd_info['IO']

    def cross_func(depth, callee_func_info, graph, checked_graph):

        if depth not in breadth_dic:
            breadth_dic[depth] = 0

        # print("called " + callee_func + "\n")
        for n in checked_graph['nodes']:
            if len(n['mtdCallInfo']) > 0:
                # 生成新的函数依赖边
                for mtd_call_info in n['mtdCallInfo']:
                    called_cls_name = mtd_call_info['className']
                    called_func_name = mtd_call_info['mtdSignature']

                    # 首先找到被调用函数，并找到该函数数据流图的起始点和返回点
                    if called_cls_name not in mtd_info.keys():
                        # todo 由于目前这里javaparser只能解析到抽象类，不能定位到具体的实现类，对于SARD数据就先手工设置一下
                        if re.search("_base$", called_cls_name):
                            # todo 这里是为了获取没有参数信息的纯函数名，可以在java解析时保存这个信息
                            pure_callee_func_name = re.sub("\(.*\)", "", callee_func_info[1], re.DOTALL)
                            called_cls_name = called_cls_name[:-5] + "_" + pure_callee_func_name
                            if called_cls_name not in mtd_info.keys():
                                print("don't have called class {}".format(called_cls_name))
                                continue
                        else:
                            print("don't have called class {}".format(called_cls_name))
                            continue
                    if called_func_name not in mtd_info[called_cls_name].keys():
                        print("class {} don't have called func {}".format(called_cls_name, called_func_name))
                        continue

                    called_func_info = copy.deepcopy(mtd_info[called_cls_name][called_func_name])

                    start_node = None
                    return_nodes = []

                    if [called_cls_name, called_func_name] in already_traverse_func_info:
                        for nn in already_traverse_func_info_graph[called_cls_name + called_func_name]['nodes']:
                            if nn['type'] == 'MethodDeclaration':
                                start_node = nn
                            if nn['type'] == 'ReturnStmt':
                                return_nodes.append(nn)

                        # 若被调用函数既没有参数，即数据流入，也没有返回语句，即数据流出，则该函数不会被划入整体数据图中
                        if start_node is None and len(return_nodes) == 0:
                            continue
                        # 找到被调用函数的函数声明结点，若存在，则说明数据流从函数开头进入
                        if start_node is not None:
                            # 这里由于被调用函数的结点ID和边的source、target结点ID都已经被改变过了，只需要简单加一条边就可以了
                            new_func_call_in_edge = {'id': '-1', 'source': n['id'], 'target': start_node['id'],
                                                     'type': 'func_call_out'}
                            graph['edges'].append(new_func_call_in_edge)
                            # 对所有的有返回值的被调用函数，会有若干条数据流返回边流回调用函数的调用结点
                        if len(return_nodes) > 0:
                            for return_node in return_nodes:
                                new_func_call_out_edge = {'id': '-1', 'source': return_node['id'], 'target': n['id'],
                                                          'type': 'func_call_in'}
                                graph['edges'].append(new_func_call_out_edge)

                    else:
                        # 下面的代码是被调用函数没有没遍历过的处理方式
                        for nn in called_func_info['nodes']:
                            nn['id'] = str(depth) + '_' + str(breadth_dic[depth]) + '_' + str(nn['id'])
                            if nn['type'] == 'MethodDeclaration':
                                start_node = nn
                            if nn['type'] == 'ReturnStmt':
                                return_nodes.append(nn)

                        for ee in called_func_info['edges']:
                            ee['source'] = str(depth) + '_' + str(breadth_dic[depth]) + '_' + str(ee['source'])
                            ee['target'] = str(depth) + '_' + str(breadth_dic[depth]) + '_' + str(ee['target'])

                        # 若被调用函数既没有参数，即数据流入，也没有返回语句，即数据流出，则该函数不会被划入整体数据图中
                        if start_node is None and len(return_nodes) == 0:
                            continue
                        # 找到被调用函数的函数声明结点，若存在，则说明数据流从函数开头进入
                        if start_node is not None:
                            # 这里由于被调用函数的结点ID和边的source、target结点ID都已经被改变过了，只需要简单加一条边就可以了
                            new_func_call_in_edge = {'id': '-1', 'source': n['id'], 'target': start_node['id'],
                                                     'type': 'func_call_out'}
                            graph['edges'].append(new_func_call_in_edge)
                            # 对所有的有返回值的被调用函数，会有若干条数据流返回边流回调用函数的调用结点
                        if len(return_nodes) > 0:
                            for return_node in return_nodes:
                                new_func_call_out_edge = {'id': '-1', 'source': return_node['id'], 'target': n['id'],
                                                          'type': 'func_call_in'}
                                graph['edges'].append(new_func_call_out_edge)

                        graph['nodes'].extend(called_func_info['nodes'])
                        graph['edges'].extend(called_func_info['edges'])

                        already_traverse_func_info.append([called_cls_name, called_func_name])
                        already_traverse_func_info_graph[called_cls_name + called_func_name] = called_func_info

                        breadth_dic[depth] += 1

                        depth += 1  # 表示函数调用的深度
                        cross_func(depth, called_func_name, graph, called_func_info)
                        depth -= 1

        return graph


    total_testcase_number = 0
    for cwe in data.keys():
        # print("\n" + cwe + "\n")

        cwe_data = data[cwe]
        # if cwe == 'CWE197':
        #     print(111)
        final_data[cwe] = []

        # 找到主文件，即起始的地方
        for java_file_name in cwe_data.keys():

            if re.search("_[0-9]{1,2}\.java$", java_file_name) or re.search("_[0-9]{1,2}a\.java$", java_file_name):
                if java_file_name == 'CWE369_Divide_by_Zero__float_random_divide_68a.java':
                    print(111)
                total_testcase_number += 1
                print(java_file_name)
                # 找到主函数，即流开始的地方
                java_file_data = cwe_data[java_file_name]
                for complete_fun_name in java_file_data.keys():

                    # breadth 表示同一深度被调用函数的广度，深度加广度可以帮助重新为结点ID赋值
                    breadth_dic = {}
                    # if complete_fun_name == 'CWE113_HTTP_Response_Splitting__connect_tcp_addCookieServlet_52a_bad(javax.servlet.http.HttpServletRequest, javax.servlet.http.HttpServletResponse)':
                    #     print(111)
                    #     print(123)

                    func_name = complete_fun_name.split('____')[-1]
                    # todo 这里是为了获取没有参数信息的纯函数名，可以在java解析时保存这个信息
                    pure_func_name = re.sub("\(.*\)", "", func_name, re.DOTALL)
                    # 匹配主函数
                    if (not re.search("^bad$", pure_func_name)) and (
                            not re.search("^good(\d+|G2B\d*|B2G\d*)$", pure_func_name)):
                        continue

                    cls_name = complete_fun_name.replace('____' + func_name, '')

                    func_graph_info = copy.deepcopy(mtd_info[cls_name][func_name])

                    # 重构数据，实现跨函数、打标签
                    for node in func_graph_info['nodes']:
                        node['id'] = str(0) + '_' + str(0) + '_' + str(node['id'])
                    for edge in func_graph_info['edges']:
                        edge['source'] = str(0) + '_' + str(0) + '_' + str(edge['source'])
                        edge['target'] = str(0) + '_' + str(0) + '_' + str(edge['target'])

                    copy_data = copy.deepcopy(func_graph_info)  # 所有和主函数数据流相关的函数将会在主函数基础上扩建，所以这里预先深复制一份，只是为了不改变初始的信息。

                    already_traverse_func_info = [[cls_name, func_name]]  # 这是为了防止自我调用和交叉调用，记录已经遍历过的函数
                    already_traverse_func_info_graph = {cls_name + func_name: func_graph_info}

                    breadth_dic[0] = 0
                    new_data = {'fun_info': cls_name + '____' + pure_func_name}
                    temp = cross_func(1, [cls_name, func_name], func_graph_info, copy_data)
                    new_data['nodes'] = temp['nodes']
                    new_data['edges'] = temp['edges']
                    if 'bad' in func_name:
                        new_data['target'] = 1
                    elif 'good' in func_name:
                        new_data['target'] = 0

                    final_data[cwe].append(new_data)

    # with open('data/final_data.json', 'w') as f:
    #     js.dump(final_data, f)
    #     f.close()
    # print('total_testcase: {}'.format(total_testcase_number))
    # print(111)
