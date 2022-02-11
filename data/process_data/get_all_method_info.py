import json as js
import os

"目的是得到这样一个json文件：dic，其key是cwe名字，value是一个dic1，该dic1的key是java file 文件名，value是一个dic2，" \
"该dic2的key是函数的标识，即类名加函数名，value是图信息"

main_dic = {}

base_dir = 'D:/java_vul_detect/using_slice_juliet/all_new/data_from_xuedi/huizong'


for enter_dir in os.listdir(base_dir):
    if not enter_dir.startswith('CWE'):
        continue

    cur_cwe_name = enter_dir.split('_')[0]

    if cur_cwe_name not in main_dic:
        main_dic[cur_cwe_name] = {}

    now = False
    d = base_dir + '/' + enter_dir
    for d_ in os.listdir(d):
        if d_ == 'antbuild':
            now = True
            break

    if now:
        js_java_files = []
        for dd in os.listdir(d):
            if dd.endswith("_DDG.json") and not dd.endswith("_CFG_DDG.json"):
                js_java_files.append(d + '/' + dd)

        for js_java_file in js_java_files:

            java_file_name = js_java_file.split('/')[-1][:-9] + '.java'
            main_dic[cur_cwe_name][java_file_name] = {}

            with open(js_java_file, 'r')as f:
                print(js_java_file)
                cur_java_file_info = js.load(f)
                for func_infos in cur_java_file_info:
                    for key in func_infos.keys():
                        func_info = func_infos[key]
                        # 此处为每个结点都表明所在的java文件名，以便后续标注漏洞发生处和fixed处
                        for nn in func_info['nodes']:
                            nn['java_file'] = java_file_name[:-5]
                        main_dic[cur_cwe_name][java_file_name][key] = func_info

    else:
        for dd_ in os.listdir(d):
            cur_base_dir = d + '/' + dd_
            js_java_files = []
            for dd in os.listdir(cur_base_dir):
                if dd.endswith("_DDG.json") and not dd.endswith("_CFG_DDG.json"):
                    js_java_files.append(cur_base_dir + '/' + dd)

            for js_java_file in js_java_files:
                java_file_name = js_java_file.split('/')[-1][:-9] + '.java'
                main_dic[cur_cwe_name][java_file_name] = {}

                with open(js_java_file, 'r')as f:
                    print(js_java_file)
                    cur_java_file_info = js.load(f)
                    for func_infos in cur_java_file_info:
                        for key in func_infos.keys():
                            func_info = func_infos[key]
                            # 此处为每个结点都表明所在的java文件名，以便后续标注漏洞发生处和fixed处
                            for nn in func_info['nodes']:
                                nn['java_file'] = java_file_name[:-5]
                            main_dic[cur_cwe_name][java_file_name][key] = func_info

with open('data/before_cross_fun_all_data.json', 'w')as f:
    js.dump(main_dic, f)
    f.close()
print(111)